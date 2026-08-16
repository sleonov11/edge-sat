#include "core/Image.h"
#include "filters/GaussianFilter.h"
#include "dwt/WaveletTransform.h"
#include "core/tiling.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>
#include <omp.h>
#include <atomic>

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

// ---------- Вспомогательные функции ----------
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

template <typename Func>
double measureMs(Func&& func) {
    auto start = std::chrono::steady_clock::now();
    func();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename Func>
double measureAvgMs(Func&& func, int runs = 3) {
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        times.push_back(measureMs(std::forward<Func>(func)));
    }
    return std::accumulate(times.begin(), times.end(), 0.0) / runs;
}

// ---------- Тестирование ----------
void testImage(const Image<uint8_t>& img, const std::string& name,
               const GaussianFilter& filter, int runs = 3) {
    size_t W = img.width(), H = img.height(), CH = img.channels();
    std::cout << "\n=== " << name << " (" << W << "x" << H << "x" << CH << ") ===\n";

    // ---- 1. Целое изображение ----
    double wholeFilter = measureAvgMs([&]() {
        auto filtered = filter.apply(img);
    }, runs);
    double wholeTotal = measureAvgMs([&]() {
        auto filtered = filter.apply(img);
        auto features = extract_features(filtered);
    }, runs);
    double wholeDWT = wholeTotal - wholeFilter;

    std::cout << "  Целое: фильтрация = " << wholeFilter << " ms, DWT = " << wholeDWT
              << " ms, итого = " << wholeTotal << " ms\n";

    // ---- 2. Тайловые конфигурации ----
    struct TileConfig {
        size_t w, h, ox, oy;
    };
    std::vector<TileConfig> configs = {
        {64, 64, 8, 8},
        {128, 128, 16, 16},
        {256, 256, 32, 32},
        {512, 512, 64, 64}
    };

    std::cout << "\n  Тайлы (фильтрация / DWT отдельно):\n";
    std::cout << std::setw(10) << "Размер"
              << std::setw(10) << "Перекр."
              << std::setw(12) << "Кол-во"
              << std::setw(14) << "Фильтр(п)"
              << std::setw(14) << "DWT(п)"
              << std::setw(14) << "Фильтр(о)"
              << std::setw(14) << "DWT(о)"
              << std::setw(12) << "Уск.(п/о)"
              << "\n";
    std::cout << std::string(110, '-') << "\n";

    for (const auto& cfg : configs) {
        auto tiles = splitToTile(img, cfg.w, cfg.h, cfg.ox, cfg.oy);
        size_t nt = tiles.size();
        if (!nt) continue;

        // ---- Параллельная обработка ----
        double parFilter = measureAvgMs([&]() {
            #pragma omp parallel for
            for (size_t i = 0; i < nt; ++i) {
                auto filtered = filter.apply(tiles[i]);
            }
        }, runs);
        double parTotal = measureAvgMs([&]() {
            #pragma omp parallel for
            for (size_t i = 0; i < nt; ++i) {
                auto filtered = filter.apply(tiles[i]);
                auto features = extract_features(filtered);
            }
        }, runs);
        double parDWT = parTotal - parFilter;

        // ---- Последовательная обработка ----
        double seqFilter = measureAvgMs([&]() {
            for (size_t i = 0; i < nt; ++i) {
                auto filtered = filter.apply(tiles[i]);
            }
        }, runs);
        double seqTotal = measureAvgMs([&]() {
            for (size_t i = 0; i < nt; ++i) {
                auto filtered = filter.apply(tiles[i]);
                auto features = extract_features(filtered);
            }
        }, runs);
        double seqDWT = seqTotal - seqFilter;

        double speedup = seqTotal / parTotal;

        std::cout << std::setw(10) << std::to_string(cfg.w) + "x" + std::to_string(cfg.h)
                  << std::setw(10) << std::to_string(cfg.ox) + "x" + std::to_string(cfg.oy)
                  << std::setw(12) << nt
                  << std::setw(14) << std::fixed << std::setprecision(2) << parFilter
                  << std::setw(14) << parDWT
                  << std::setw(14) << seqFilter
                  << std::setw(14) << seqDWT
                  << std::setw(12) << speedup
                  << "\n";
    }
}

// ---------- main ----------
int main() {
    constexpr size_t W = 3840, H = 2160;
    constexpr float SIGMA = 1.5f;
    constexpr int RUNS = 3;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Генерация 4K RGB ...\n";
    Image<uint8_t> rgb = generateRandomImage(W, H, 3);
    GaussianFilter filter(SIGMA, EdgeHandling::Mirror);

    // ---- Сброс счётчика аллокаций ----
    allocations = 0;
    deallocations = 0;

    // ---- Основной тест (Grayscale) ----
    std::cout << "\nПреобразование в Grayscale ...\n";
    Image<uint8_t> gray = rgb.scaleToGray();

    // Сброс счётчика перед обработкой (чтобы замерить аллокации внутри пайплайна)
    allocations = 0;
    deallocations = 0;

    testImage(gray, "Grayscale", filter, RUNS);

    std::cout << "\nАллокаций (new) в ходе теста: " << allocations.load() << "\n";
    std::cout << "Деаллокаций (delete): " << deallocations.load() << "\n";

    double new_ms = total_new_ns.load() / 1e6;
    double delete_ms = total_delete_ns.load() / 1e6;
    double alloc_total_ms = new_ms + delete_ms;
    std::cout << "\nВремя в new: " << new_ms << " ms\n";
    std::cout << "Время в delete: " << delete_ms << " ms\n";
    std::cout << "Итого время аллокаций: " << alloc_total_ms << " ms\n";

    // ---- Дополнительно: тест с разным числом потоков для 128×128 ----
    std::cout << "\n=== Влияние числа потоков (тайл 128×128) ===\n";
    auto tiles128 = splitToTile(gray, 128, 128, 16, 16);
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    std::cout << std::setw(10) << "Потоки" << std::setw(15) << "Время (мс)" << std::setw(15) << "Ускорение\n";
    double baseTime = 0.0;
    for (int t : thread_counts) {
        double time = measureAvgMs([&]() {
            #pragma omp parallel for num_threads(t)
            for (size_t i = 0; i < tiles128.size(); ++i) {
                auto filtered = filter.apply(tiles128[i]);
                auto features = extract_features(filtered);
            }
        }, RUNS);
        if (t == 1) baseTime = time;
        double speedup = baseTime / time;
        std::cout << std::setw(10) << t << std::setw(15) << time << std::setw(15) << speedup << "\n";
    }

    return 0;
}