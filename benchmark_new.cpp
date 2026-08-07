#include "core/Image.h"
#include "core/TileGenerator.h"
#include "core/TileBuffers.h"
#include "filters/gauss_kernel.h"
#include "filters/gauss_tile.h"
#include "filters/mirror_pad.h"
#include "dwt/dwt_tile.h"
#include "dwt/dwt_features.h"
#include "core/TileView.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <cmath>
#include <algorithm>

using namespace std::chrono;

// ---------- Генерация случайного RGB-изображения ----------
Image<uint8_t> generateRandomImage(size_t w, size_t h) {
    Image<uint8_t> img(w, h, 3);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < img.size(); ++i) {
        img[i] = static_cast<uint8_t>(dist(gen));
    }
    return img;
}

// ---------- Обработка одного grayscale-тайла ----------
template <int TILE>
void process_gray_tile(const TileView& view, TileBuffers<TILE>& buf,
                       const GaussKernel<2>& kernel, std::array<float, 16>& features) {
    // 1. Копируем grayscale данные в buf.gray (view.channels == 1)
    for (int y = 0; y < view.h; ++y) {
        const uint8_t* src_row = view.row(y, 0);
        float* dst_row = buf.gray + y * TILE;
        for (int x = 0; x < view.w; ++x) {
            dst_row[x] = static_cast<float>(src_row[x]);
        }
    }

    // 2. Mirror pad
    mirror_pad<TILE, 2>(buf.gray, buf.padded);

    // 3. Гаусс H
    gauss_h_tile<TILE, 2>(buf.padded, buf.gauss_h, kernel.k);

    // 4. Гаусс V
    gauss_v_tile<TILE, 2>(buf.gauss_h, buf.gauss_v, kernel.k);

    // 5. DWT (in-place)
    std::copy(buf.gauss_v, buf.gauss_v + TILE * TILE, buf.dwt_buf);
    dwt_2d_haar<TILE>(buf.dwt_buf, buf.dwt_tmp);

    // 6. Признаки
    features = extract_features_from_dwt_optimized<TILE>(buf.dwt_buf);
}

// ---------- Замер времени ----------
template <typename Func>
double measureTimeMs(Func&& func, int repeats = 3) {
    std::vector<double> times;
    times.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        auto start = steady_clock::now();
        func();
        auto end = steady_clock::now();
        times.push_back(duration<double, std::milli>(end - start).count());
    }
    std::sort(times.begin(), times.end());
    double sum = 0.0;
    for (size_t i = 1; i < times.size() - 1; ++i) sum += times[i];
    return sum / (times.size() - 2);
}

// ---------- Шаблонная функция для конкретного размера ----------
template <int TILE>
void run_benchmark_for_size(const Image<uint8_t>& gray, const GaussKernel<2>& kernel,
                            const std::vector<int>& thread_counts, int repeats = 5) {
    TileGenerator gen(gray.data(), gray.width() * sizeof(uint8_t),
                      static_cast<int>(gray.width()),
                      static_cast<int>(gray.height()),
                      TILE, TILE, 0);

    std::vector<TileView> views;
    TileView view;
    while (gen.next(view)) {
        views.push_back(view);
    }
    size_t num_tiles = views.size();
    if (num_tiles == 0) return;

    // Прогрев
    {
        #pragma omp parallel for num_threads(8) schedule(dynamic)
        for (size_t i = 0; i < views.size(); ++i) {
            TileBuffers<TILE> buf;
            std::array<float, 16> features;
            process_gray_tile<TILE>(views[i], buf, kernel, features);
            volatile float dummy = features[0];
            (void)dummy;
        }
    }

    // Основные замеры для каждого числа потоков
    for (int num_threads : thread_counts) {
        double time_ms = measureTimeMs([&]() {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
            for (size_t i = 0; i < views.size(); ++i) {
                TileBuffers<TILE> buf;
                std::array<float, 16> features;
                process_gray_tile<TILE>(views[i], buf, kernel, features);
                volatile float dummy = features[0];
                (void)dummy;
            }
        }, repeats);

        double ms_per_tile = time_ms / num_tiles;
        std::cout << std::setw(10) << TILE
                  << std::setw(10) << num_threads
                  << std::setw(15) << time_ms
                  << std::setw(15) << num_tiles
                  << std::setw(15) << ms_per_tile
                  << "\n";
    }
}

// ---------- Дополнительный замер только Гаусса для TILE=128 ----------
template <int TILE>
void run_gauss_only(const Image<uint8_t>& gray, const GaussKernel<2>& kernel,
                    int num_threads = 8, int repeats = 5) {
    TileGenerator gen(gray.data(), gray.width() * sizeof(uint8_t),
                      static_cast<int>(gray.width()),
                      static_cast<int>(gray.height()),
                      TILE, TILE, 0);
    std::vector<TileView> views;
    TileView view;
    while (gen.next(view)) views.push_back(view);

    // Прогрев
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t i = 0; i < views.size(); ++i) {
        TileBuffers<TILE> buf;
        for (int y = 0; y < views[i].h; ++y) {
            const uint8_t* src = views[i].row(y, 0);
            float* dst = buf.gray + y * TILE;
            for (int x = 0; x < views[i].w; ++x) dst[x] = src[x];
        }
        mirror_pad<TILE, 2>(buf.gray, buf.padded);
        gauss_h_tile<TILE, 2>(buf.padded, buf.gauss_h, kernel.k);
        gauss_v_tile<TILE, 2>(buf.gauss_h, buf.gauss_v, kernel.k);
    }

    double time_gauss = measureTimeMs([&]() {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (size_t i = 0; i < views.size(); ++i) {
            TileBuffers<TILE> buf;
            for (int y = 0; y < views[i].h; ++y) {
                const uint8_t* src = views[i].row(y, 0);
                float* dst = buf.gray + y * TILE;
                for (int x = 0; x < views[i].w; ++x) dst[x] = src[x];
            }
            mirror_pad<TILE, 2>(buf.gray, buf.padded);
            gauss_h_tile<TILE, 2>(buf.padded, buf.gauss_h, kernel.k);
            gauss_v_tile<TILE, 2>(buf.gauss_h, buf.gauss_v, kernel.k);
        }
    }, repeats);

    double time_full = measureTimeMs([&]() {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (size_t i = 0; i < views.size(); ++i) {
            TileBuffers<TILE> buf;
            std::array<float, 16> features;
            process_gray_tile<TILE>(views[i], buf, kernel, features);
            volatile float dummy = features[0];
            (void)dummy;
        }
    }, repeats);

    std::cout << "\n=== Только Гаусс (тайл " << TILE << ", " << num_threads << " потоков) ===\n";
    std::cout << "Гаусс: " << time_gauss << " ms\n";
    std::cout << "Полный пайплайн: " << time_full << " ms\n";
    std::cout << "DWT + признаки: " << (time_full - time_gauss) << " ms\n";
}

// ---------- Основная программа ----------
int main() {
    constexpr int W = 3840, H = 2160;
    constexpr float SIGMA = 1.5f;
    constexpr int REPEATS = 5;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Генерация 4K RGB-изображения...\n";
    Image<uint8_t> rgb = generateRandomImage(W, H);
    std::cout << "Преобразование в Grayscale...\n";
    Image<uint8_t> gray = rgb.scaleToGray();

    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    GaussKernel<2> kernel(SIGMA);

    std::cout << "\n=== Бенчмарк тайловой обработки (Grayscale) ===\n";
    std::cout << "Размер изображения: " << W << "x" << H << "\n";
    std::cout << "Сигма Гаусса: " << SIGMA << "\n";
    std::cout << "Перекрытие: 0 (полные тайлы)\n\n";

    std::cout << std::setw(10) << "Тайл" 
              << std::setw(10) << "Потоки" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(15) << "Тайлов" 
              << std::setw(15) << "Мс/тайл" 
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    // Запуск для размеров 64, 128, 256
    run_benchmark_for_size<64>(gray, kernel, thread_counts, REPEATS);
    std::cout << std::string(70, '-') << "\n";
    run_benchmark_for_size<128>(gray, kernel, thread_counts, REPEATS);
    std::cout << std::string(70, '-') << "\n";
    run_benchmark_for_size<256>(gray, kernel, thread_counts, REPEATS);

    // Дополнительно: только Гаусс для 128
    run_gauss_only<128>(gray, kernel, 8, REPEATS);

    return 0;
}