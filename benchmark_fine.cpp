#include "core/Image.h"
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
#include <random>
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>

using namespace std::chrono;

// ---------- Вспомогательные структуры и функции для статистик ----------
template<int SIZE>
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

// ---------- Измерение времени ----------
template <typename Func>
double measureMs(Func&& func, int repeats = 10000) {
    auto start = steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        func();
    }
    auto end = steady_clock::now();
    return duration<double, std::milli>(end - start).count();
}

// ---------- Главная ----------
int main() {
    constexpr int TILE = 128;
    constexpr float SIGMA = 1.5f;

    // 1. Генерируем изображение и берём первый тайл
    Image<uint8_t> rgb = generateRandomImage(3840, 2160, 3);
    Image<uint8_t> gray = rgb.scaleToGray();

    TileGenerator gen(gray.data(), gray.width() * sizeof(uint8_t),
                      static_cast<int>(gray.width()),
                      static_cast<int>(gray.height()),
                      TILE, TILE, 16);
    TileView view;
    if (!gen.next(view)) {
        std::cerr << "No tiles!\n";
        return 1;
    }

    // 2. Буферы (один раз)
    TileBuffers<TILE> buf;
    GaussKernel<2> kernel(SIGMA);

    const int REPEATS = 10000;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Fine-grained benchmark for one " << TILE << "x" << TILE << " tile\n";
    std::cout << "Repeats: " << REPEATS << "\n\n";

    // ----- 1) rgb_to_gray -----
    double t1 = measureMs([&]() {
        rgb_to_gray<TILE>(view, buf.gray);
    }, REPEATS);
    std::cout << "rgb_to_gray:       " << t1 << " ms\n";

    // ----- 2) mirror_pad -----
    double t2 = measureMs([&]() {
        mirror_pad<TILE, 2>(buf.gray, buf.padded);
    }, REPEATS);
    std::cout << "mirror_pad:        " << t2 << " ms\n";

    // ----- 3) gauss_h -----
    double t3 = measureMs([&]() {
        gauss_h_tile<TILE, 2>(buf.padded, buf.gauss_h, kernel.k);
    }, REPEATS);
    std::cout << "gauss_h:           " << t3 << " ms\n";

    // ----- 4) gauss_v -----
    double t4 = measureMs([&]() {
        gauss_v_tile<TILE, 2>(buf.gauss_h, buf.gauss_v, kernel.k);
    }, REPEATS);
    std::cout << "gauss_v:           " << t4 << " ms\n";

    // ----- 5) std::copy (gauss_v → dwt_buf) -----
    double t5 = measureMs([&]() {
        std::copy(buf.gauss_v, buf.gauss_v + TILE * TILE, buf.dwt_buf);
    }, REPEATS);
    std::cout << "std::copy:         " << t5 << " ms\n";

    // ----- 6) dwt_2d_haar (in-place) -----
    // Замеряем вместе с копированием внутрь, чтобы не искажать
    double t6 = measureMs([&]() {
        std::copy(buf.gauss_v, buf.gauss_v + TILE * TILE, buf.dwt_buf);
        dwt_2d_haar<TILE>(buf.dwt_buf, buf.dwt_tmp);
    }, REPEATS);
    double t6_clean = t6 - t5;  // вычитаем время копирования
    std::cout << "dwt_2d_haar (incl copy): " << t6 << " ms\n";
    std::cout << "dwt_2d_haar (clean):     " << t6_clean << " ms\n";

    // ----- 7) extract_features_from_dwt -----
    // Выполняем один раз dwt, чтобы подготовить данные
    std::copy(buf.gauss_v, buf.gauss_v + TILE * TILE, buf.dwt_buf);
    dwt_2d_haar<TILE>(buf.dwt_buf, buf.dwt_tmp);
    double t7 = measureMs([&]() {
        std::array<float, 16> features = extract_features_from_dwt<TILE>(buf.dwt_buf);
        volatile float dummy = features[0];
        (void)dummy;
    }, REPEATS);
    std::cout << "extract_features:  " << t7 << " ms\n";

    // ----- Суммарное время (sum of parts) -----
    double total_measured = t1 + t2 + t3 + t4 + t5 + t6_clean + t7;
    std::cout << "\nTotal (sum of parts): " << total_measured << " ms\n";

    // ----- Замер всего пайплайна целиком (без промежуточных копий) -----
    double total_full = measureMs([&]() {
        rgb_to_gray<TILE>(view, buf.gray);
        mirror_pad<TILE, 2>(buf.gray, buf.padded);
        gauss_h_tile<TILE, 2>(buf.padded, buf.gauss_h, kernel.k);
        gauss_v_tile<TILE, 2>(buf.gauss_h, buf.gauss_v, kernel.k);
        std::copy(buf.gauss_v, buf.gauss_v + TILE * TILE, buf.dwt_buf);
        dwt_2d_haar<TILE>(buf.dwt_buf, buf.dwt_tmp);
        std::array<float, 16> features = extract_features_from_dwt<TILE>(buf.dwt_buf);
        volatile float dummy = features[0];
        (void)dummy;
    }, REPEATS);
    std::cout << "Full pipeline (measured once): " << total_full << " ms\n";

    // ----- Проценты -----
    std::cout << "\nPercentages:\n";
    std::cout << "rgb_to_gray:       " << (t1 / total_measured) * 100 << "%\n";
    std::cout << "mirror_pad:        " << (t2 / total_measured) * 100 << "%\n";
    std::cout << "gauss_h:           " << (t3 / total_measured) * 100 << "%\n";
    std::cout << "gauss_v:           " << (t4 / total_measured) * 100 << "%\n";
    std::cout << "std::copy:         " << (t5 / total_measured) * 100 << "%\n";
    std::cout << "dwt_2d_haar:       " << (t6_clean / total_measured) * 100 << "%\n";
    std::cout << "extract_features:  " << (t7 / total_measured) * 100 << "%\n";

    return 0;
}