#pragma once

// вход PADDED x PADDED 136х136
// выход TILE x PADDED 128x136
#include <cmath>
#include <algorithm>

template <int TILE, int RADIUS>
void gauss_h_tile (
    const float* __restrict src, // in padded[PADDED*PADDED]
    float* __restrict dst, // out gauss_h[TILE*PADDED]
    const float* __restrict kernel // kernel[2 * RADIUS + 1]
) {
    constexpr int PAD = RADIUS;
    constexpr int PADDED = TILE + 2 * PAD;

    for (int y = PAD; y < PADDED - PAD; ++y) {
        const float* src_row = src + y * PADDED + PAD;
        float* dst_row = dst + (y - PAD) * PADDED;
        
        for (int x = 0; x < TILE; ++x) {
            float sum = 0.0f;

            #pragma unroll
            for (int k = -RADIUS; k <= RADIUS; ++k) {
                sum+= src_row[x + k] * kernel[k + RADIUS];
            }
            dst_row[x] = sum; 
        }
    }
}

template <int TILE, int RADIUS>
void gauss_v_tile (
    const float* __restrict src,
    float* __restrict dst,
    const float* __restrict kernel
) {
   constexpr int PAD = RADIUS;
   constexpr int PADDED = TILE + 2 * PAD;

   for (int y = 0; y < TILE; ++y) {
        for (int x = 0; x < TILE; ++x) {
            float sum = 0.0f;

            #pragma unroll
            for (int k = -RADIUS; k <= RADIUS; ++k) {
                sum += src[(y + PAD + k) * PADDED + x] * kernel[k + RADIUS];        
            }
            dst[y*TILE + x] = sum; 
        }
   }
}
