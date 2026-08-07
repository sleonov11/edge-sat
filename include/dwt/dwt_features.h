#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <algorithm>

template <int TILE>
std::array<float, 16> extract_features_from_dwt_optimized(const float* __restrict buf) {
    constexpr int HALF = TILE / 2;
    constexpr int N = HALF * HALF;
    constexpr float EPS = 1e-12f;

    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_sq[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma omp simd reduction(+:sum, sum_sq)
    for (int i = 0; i < N; ++i) {
        float v = buf[i];  // LL
        sum[0] += v;
        sum_sq[0] += v * v;
    }
    #pragma omp simd reduction(+:sum, sum_sq)
    for (int i = 0; i < N; ++i) {
        float v = buf[HALF + i];  // LH (сдвиг по x на HALF)
        sum[1] += v;
        sum_sq[1] += v * v;
    }
    #pragma omp simd reduction(+:sum, sum_sq)
    for (int i = 0; i < N; ++i) {
        float v = buf[HALF * TILE + i];  // HL (сдвиг по y)
        sum[2] += v;
        sum_sq[2] += v * v;
    }
    #pragma omp simd reduction(+:sum, sum_sq)
    for (int i = 0; i < N; ++i) {
        float v = buf[HALF * TILE + HALF + i];  // HH
        sum[3] += v;
        sum_sq[3] += v * v;
    }

    std::array<float, 16> f{};

    for (int b = 0; b < 4; ++b) {
        float mean = sum[b] / N;
        float energy = sum_sq[b];
        float variance = energy / N - mean * mean;

        float entropy = 0.0f;
        if (energy > EPS) {
            const float* subband = buf + (b / 2) * HALF * TILE + (b % 2) * HALF;
            #pragma omp simd reduction(+:entropy)
            for (int i = 0; i < N; ++i) {
                float p = (subband[i] * subband[i]) / energy;
                entropy -= p * std::logf(p + EPS);
            }
        }

        f[b * 4 + 0] = mean;
        f[b * 4 + 1] = variance;
        f[b * 4 + 2] = energy;
        f[b * 4 + 3] = entropy;
    }

    return f;
}