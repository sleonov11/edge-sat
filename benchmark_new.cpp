// benchmark_new.cpp
#include "core/Image.h"                     // для генерации и scaleToGray
#include "core/TileGenerator.h"
#include "core/TileBuffers.h"
#include "filters/rgb_to_gray.h"
#include "filters/mirror_pad.h"
#include "filters/gauss_tile.h"
#include "dwt/dwt_tile.h"
#include "filters/gauss_kernel.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>
#include <omp.h>
#include <atomic>
#include <random>
#include <cmath>

// ---------- Счётчик аллокаций (глобальный) ----------
static std::atomic<size_t> allocations{0};
static std::atomic<size_t> deallocations{0};
static std::atomic<long long> total_new_ns{0};
static std::atomic<long long> total_delete_ns{0};

void* operator new(size_t size) {
    auto start = std::chrono::steady_clock::now();
    allocations++;
    void* ptr = malloc(size);
    if (!ptr) throw std::bad_alloc();
    auto end = std::chrono::steady_clock::now();
    total_new_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return ptr;
}

void operator delete(void* ptr) noexcept {
    auto start = std::chrono::steady_clock::now();
    deallocations++;
    free(ptr);
    auto end = std::chrono::steady_clock::now();
    total_delete_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

// ---------- Генерация случайного RGB-изображения ----------
Image<uint8_t> generateRandomImage(size_t w, size_t h, size_t ch = 3) {
    Image<uint8_t> img(w, h, ch);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < img.size(); ++i) {
        img[i] = static_cast<uint8_t>(dist(gen));
    }
    return img;
}

// ---------- Измерение времени (среднее по 3 прогонам) ----------
template <typename Func>
double measureAvgMs(Func&& func, int runs = 3) {
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::steady_clock::now();
        func();
        auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    return std::accumulate(times.begin(), times.end(), 0.0) / runs;
}

// ---------- Функция вычисления статистик для 4 субполос (для TILE=128) ----------
template<int TILE>
struct FeatureStats {
    float mean, variance, energy, entropy;
};

template<int SIZE>
FeatureStats<SIZE> compute_stats_subband(const float* data) {
    constexpr int N = SIZE * SIZE;
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < N; ++i) {
        double v = data[i];
        sum += v;
        sum_sq += v * v;
    }
    double mean = sum / N;
    double variance = sum_sq / N - mean * mean;
    double energy = sum_sq;
    double entropy = std::log(energy + 1e-12);
    return {static_cast<float>(mean),
            static_cast<float>(variance),
            static_cast<float>(energy),
            static_cast<float>(entropy)};
}

// Извлекает 16 признаков (4 subbands × 4 stats) из dwt_buf
template<int TILE>
std::array<float, 16> extract_features_from_dwt(const float* dwt_buf) {
    constexpr int HALF = TILE / 2;
    const float* ll = dwt_buf;
    const float* lh = dwt_buf + HALF;
    const float* hl = dwt_buf + HALF * TILE;
    const float* hh = dwt_buf + HALF * TILE + HALF;

    auto s_ll = compute_stats_subband<HALF>(ll);
    auto s_lh = compute_stats_subband<HALF>(lh);
    auto s_hl = compute_stats_subband<HALF>(hl);
    auto s_hh = compute_stats_subband<HALF>(hh);

    return {
        s_ll.mean, s_ll.variance, s_ll.energy, s_ll.entropy,
        s_lh.mean, s_lh.variance, s_lh.energy, s_lh.entropy,
        s_hl.mean, s_hl.variance, s_hl.energy, s_hl.entropy,
        s_hh.mean, s_hh.variance, s_hh.energy, s_hh.entropy
    };
}

// ---------- Основной пайплайн для одного тайла (все этапы) ----------
template<int TILE>
void process_tile(const TileView& view, TileBuffers<TILE>& buf, 
                  const GaussKernel<2>& kernel, std::array<float, 16>& features) {
    // 1. RGB → Gray
    rgb_to_gray<TILE>(view, buf.gray);

    // 2. Mirror pad
    mirror_pad<TILE, 2>(buf.gray, buf.padded);

    // 3. Гаусс H
    gauss_h_tile<TILE, 2>(buf.padded, buf.gauss_h, kernel.k);

    // 4. Гаусс V
    gauss_v_tile<TILE, 2>(buf.gauss_h, buf.gauss_v, kernel.k);

    // 5. DWT
    std::copy(buf.gauss_v, buf.gauss_v + TILE * TILE, buf.dwt_buf);
    dwt_2d_haar<TILE>(buf.dwt_buf, buf.dwt_tmp);

    // 6. Извлечение признаков
    features = extract_features_from_dwt<TILE>(buf.dwt_buf);
}

// ---------- Бенчмарк для заданного размера тайла ----------
template<int TILE>
double benchmark_tile_size(const Image<uint8_t>& gray, 
                           const GaussKernel<2>& kernel,
                           int num_threads, bool parallel) {
    // Сброс счётчиков аллокаций (для этого прогона)
    allocations = 0;
    deallocations = 0;
    total_new_ns = 0;
    total_delete_ns = 0;

    // Создаём генератор тайлов (полные тайлы)
    TileGenerator gen(gray.data(), gray.width() * sizeof(uint8_t),
                      static_cast<int>(gray.width()),
                      static_cast<int>(gray.height()),
                      TILE, TILE, 16);

    // Собираем все View (разовое выделение, не в горячем пути)
    std::vector<TileView> views;
    views.reserve(gen.count());
    TileView view;
    while (gen.next(view)) {
        views.push_back(view);
    }

    // Подготовка буферов для каждого потока (используем thread_local static, чтобы не аллоцировать)
    // В данном простом варианте – создаём буфер на стеке внутри параллельной области.

    auto work = [&]() {
        #pragma omp parallel for num_threads(num_threads) if(parallel) schedule(dynamic)
        for (size_t i = 0; i < views.size(); ++i) {
            // Каждый поток получает свой стековый буфер
            TileBuffers<TILE> buf;
            std::array<float, 16> features;
            process_tile<TILE>(views[i], buf, kernel, features);
            // Чтобы компилятор не выбросил вычисления
            volatile float dummy = features[0];
            (void)dummy;
        }
    };

    double time_ms = measureAvgMs(work, 3);

    // Вывод статистики аллокаций для этого прогона
    double new_ms = total_new_ns.load() / 1e6;
    double delete_ms = total_delete_ns.load() / 1e6;
    double alloc_total = new_ms + delete_ms;
    std::cout << "    Аллокаций: " << allocations.load() 
              << " (new: " << new_ms << " ms, delete: " << delete_ms << " ms, итого: " << alloc_total << " ms)" 
              << std::endl;

    return time_ms;
}

// ---------- main ----------
int main() {
    constexpr size_t W = 3840, H = 2160;
    constexpr float SIGMA = 1.5f;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Генерация 4K RGB-изображения...\n";
    Image<uint8_t> rgb = generateRandomImage(W, H, 3);
    Image<uint8_t> gray = rgb.scaleToGray();  // переводим в чб один раз

    // Предварительный расчёт ядра Гаусса (константа времени компиляции)
    constexpr GaussKernel<2> KERNEL(SIGMA);  // RADIUS=2, т.к. sigma=1.5

    // Тестируем разные размеры тайлов
    std::vector<int> tile_sizes = {64, 128, 256, 512};
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};

    std::cout << "\n=== Бенчмарк нового пайплайна (TileView + стековые буферы) ===\n";
    std::cout << "Изображение: " << W << "x" << H << "x1 (Grayscale)\n";

    // Для каждого размера тайла – прогон с 8 потоками (оптимально)
    std::cout << "\n--- Влияние размера тайла (8 потоков) ---\n";
    std::cout << std::setw(10) << "Тайл" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(15) << "Аллокаций" 
              << "\n";
    for (int tile : tile_sizes) {
        double time = 0.0;
        if (tile == 64) time = benchmark_tile_size<64>(gray, KERNEL, 8, true);
        else if (tile == 128) time = benchmark_tile_size<128>(gray, KERNEL, 8, true);
        else if (tile == 256) time = benchmark_tile_size<256>(gray, KERNEL, 8, true);
        else if (tile == 512) time = benchmark_tile_size<512>(gray, KERNEL, 8, true);
        std::cout << std::setw(10) << tile 
                  << std::setw(15) << time 
                  << std::setw(15) << allocations.load() 
                  << "\n";
    }

    // Масштабирование по числу потоков (фиксируем 128×128)
    std::cout << "\n--- Масштабирование по числу потоков (тайл 128×128) ---\n";
    std::cout << std::setw(10) << "Потоки" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(15) << "Ускорение" 
              << "\n";
    double base_time = 0.0;
    for (int threads : thread_counts) {
        double time = benchmark_tile_size<128>(gray, KERNEL, threads, true);
        if (threads == 1) base_time = time;
        double speedup = base_time / time;
        std::cout << std::setw(10) << threads 
                  << std::setw(15) << time 
                  << std::setw(15) << speedup 
                  << "\n";
    }

    // Дополнительно: сравнение последовательной и параллельной версии для 128×128
    std::cout << "\n--- Последовательно vs параллельно (128×128) ---\n";
    double seq = benchmark_tile_size<128>(gray, KERNEL, 1, false);  // 1 поток, без распараллеливания
    double par = benchmark_tile_size<128>(gray, KERNEL, 8, true);
    std::cout << "Последовательно: " << seq << " ms\n";
    std::cout << "Параллельно (8 потоков): " << par << " ms\n";
    std::cout << "Ускорение: " << seq/par << "x\n";

    return 0;
}