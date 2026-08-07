// benchmark_tile_profile.cpp
#include "core/Image.h"
#include "core/TileBuffers.h"
#include "filters/gauss_kernel.h"
#include "filters/gauss_tile.h"
#include "filters/mirror_pad.h"
#include "dwt/dwt_tile.h"
#include "dwt/dwt_features.h"
#include "core/TileView.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <omp.h>
#include <algorithm>
#include <array>

using namespace std::chrono;

// ---------- Генерация случайного RGB 128x128 ----------
void generateRandomTileRGB(uint8_t* data, int stride) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);
    for (int y = 0; y < 128; ++y) {
        for (int x = 0; x < 128; ++x) {
            data[y * stride + 3*x + 0] = dist(gen);
            data[y * stride + 3*x + 1] = dist(gen);
            data[y * stride + 3*x + 2] = dist(gen);
        }
    }
}

// ---------- Замер времени (усреднение) ----------
template <typename Func>
double measureMs(Func&& func, int repeats = 10000) {
    auto start = steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        func();
    }
    auto end = steady_clock::now();
    return duration<double, std::milli>(end - start).count() / repeats;
}

int main() {
    constexpr int TILE = 128;
    constexpr float SIGMA = 1.5f;
    constexpr int REPEATS = 10000;

    // Подготовка данных
    alignas(64) uint8_t rgb_data[TILE * TILE * 3];
    generateRandomTileRGB(rgb_data, TILE * 3);

    TileView view(rgb_data, TILE * 3, 0, 0, TILE, TILE, 3);
    TileBuffers<TILE> buf;
    GaussKernel<2> kernel(SIGMA);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Детальный профиль для одного тайла " << TILE << "×" << TILE << " ===\n";
    std::cout << "Количество итераций: " << REPEATS << " (усреднение)\n\n";

    // 1) RGB->Gray (копирование grayscale из view в buf.gray)
    double t_gray = measureMs([&]() {
        for (int y = 0; y < view.h; ++y) {
            const uint8_t* src = view.row(y, 0);
            float* dst = buf.gray + y * TILE;
            for (int x = 0; x < view.w; ++x) {
                dst[x] = static_cast<float>(src[x]);
            }
        }
    }, REPEATS);

    // 2) mirror_pad
    double t_pad = measureMs([&]() {
        mirror_pad<TILE, 2>(buf.gray, buf.padded);
    }, REPEATS);

    // 3) gauss_h
    double t_h = measureMs([&]() {
        gauss_h_tile<TILE, 2>(buf.padded, buf.gauss_h, kernel.k);
    }, REPEATS);

    // 4) gauss_v
    double t_v = measureMs([&]() {
        gauss_v_tile<TILE, 2>(buf.gauss_h, buf.gauss_v, kernel.k);
    }, REPEATS);

    // 5) copy to dwt_buf
    double t_copy = measureMs([&]() {
        std::copy(buf.gauss_v, buf.gauss_v + TILE * TILE, buf.dwt_buf);
    }, REPEATS);

    // 6) DWT (полностью)
    double t_dwt = measureMs([&]() {
        std::copy(buf.gauss_v, buf.gauss_v + TILE * TILE, buf.dwt_buf);
        dwt_2d_haar<TILE>(buf.dwt_buf, buf.dwt_tmp);
    }, REPEATS);
    double t_dwt_clean = t_dwt - t_copy; // вычитаем копирование

    // 7) Извлечение признаков (после выполнения DWT один раз)
    std::copy(buf.gauss_v, buf.gauss_v + TILE * TILE, buf.dwt_buf);
    dwt_2d_haar<TILE>(buf.dwt_buf, buf.dwt_tmp);
    double t_features = measureMs([&]() {
        std::array<float, 16> features = extract_features_from_dwt_optimized<TILE>(buf.dwt_buf);
        volatile float dummy = features[0];
        (void)dummy;
    }, REPEATS);

    // 8) Полный пайплайн (для проверки суммы)
    double t_full = measureMs([&]() {
        for (int y = 0; y < view.h; ++y) {
            const uint8_t* src = view.row(y, 0);
            float* dst = buf.gray + y * TILE;
            for (int x = 0; x < view.w; ++x) dst[x] = src[x];
        }
        mirror_pad<TILE, 2>(buf.gray, buf.padded);
        gauss_h_tile<TILE, 2>(buf.padded, buf.gauss_h, kernel.k);
        gauss_v_tile<TILE, 2>(buf.gauss_h, buf.gauss_v, kernel.k);
        std::copy(buf.gauss_v, buf.gauss_v + TILE * TILE, buf.dwt_buf);
        dwt_2d_haar<TILE>(buf.dwt_buf, buf.dwt_tmp);
        std::array<float, 16> features = extract_features_from_dwt_optimized<TILE>(buf.dwt_buf);
        volatile float dummy = features[0];
        (void)dummy;
    }, REPEATS);

    // Вывод
    double total = t_gray + t_pad + t_h + t_v + t_copy + t_dwt_clean + t_features;
    std::cout << "Этап                     Время (μs)    Доля\n";
    std::cout << "------------------------------------------------\n";
    auto print = [&](const char* name, double t) {
        std::cout << std::setw(25) << name << std::setw(15) << t * 1000 << std::setw(10) << (t/total)*100 << "%\n";
    };
    print("RGB->Gray", t_gray);
    print("mirror_pad", t_pad);
    print("gauss_h", t_h);
    print("gauss_v", t_v);
    print("std::copy", t_copy);
    print("dwt_2d_haar (clean)", t_dwt_clean);
    print("extract_features", t_features);
    std::cout << "------------------------------------------------\n";
    print("Сумма частей", total);
    print("Полный pipeline (замер)", t_full);

    return 0;
}