#pragma once
#include <cmath>

template <int RADIUS>
struct GaussKernel {
    static constexpr int SIZE = 2 * RADIUS + 1;
    float k[SIZE];

    constexpr GaussKernel(float sigma) : k{} {
        float sum = 0;
        for (int i = 0; i < SIZE; ++i) {
            int d = i - RADIUS;
            k[i] = std::exp(-(d*d) / (2.0f * sigma * sigma)); // c++26 constexpr
            sum += k[i];
        }
        for (int i = 0; i < SIZE; ++i) k[i] /= sum;
    }
};

inline constexpr GaussKernel<2> GAUSS_K2_15(1.5f);