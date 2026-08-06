#pragma once

template <int TILE, int RADIUS>
void gauss_h_tile (
    const float* __restrict src,   // [PADDED*PADDED]
    float* __restrict dst,         // [PADDED*PADDED]   
    const float* __restrict kernel
) {
    constexpr int PAD = RADIUS;
    constexpr int PADDED = TILE + 2 * PAD;

    for (int y = 0; y < PADDED; ++y) {          
        const float* src_row = src + y * PADDED;
        float* dst_row = dst + y * PADDED;

        #pragma omp simd
        for (int x = 0; x < TILE; ++x) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = -RADIUS; k <= RADIUS; ++k) {
                sum += src_row[x + PAD + k] * kernel[k + RADIUS];
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
        #pragma omp simd
        for (int x = 0; x < TILE; ++x) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = -RADIUS; k <= RADIUS; ++k) {
                sum += src[(y + PAD + k) * PADDED + x] * kernel[k + RADIUS];
            }
            dst[y * TILE + x] = sum;
        }
    }
}