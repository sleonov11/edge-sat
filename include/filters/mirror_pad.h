#pragma once

inline int reflect101 (int i, int len) {
    if (i < 0) {
        return -i - 1;
    } else if (i >= len) {
        return 2 * len - i - 1;
    } else {
        return i;
    }
}

template <int TILE, int PAD>
void mirror_pad (const float* __restrict src, float* __restrict dst) {
    constexpr int PADDED = TILE + 2 * PAD;

    for (int y = 0; y < PADDED; ++y) {
        int src_y = reflect101(y - PAD, TILE);
        float* dst_row = dst + y * PADDED;
        const float* src_row = src + src_y * TILE;

        for (int x = 0; x < PADDED; ++x) {
            int src_x = reflect101 (x - PAD, TILE);
            dst_row[x] = src_row[src_x];
        }
    }
}

/*
template<int TILE, int PAD>
void mirror_pad(const float* __restrict src, float* __restrict dst) {
    constexpr int PADDED = TILE + 2 * PAD;
    
    // цент
    for (int y = 0; y < TILE; ++y) {
        for (int x = 0; x < TILE; ++x) {
            dst[(y + PAD) * PADDED + (x + PAD)] = src[y * TILE + x];
        }
    }
    
    // лево и право
    for (int y = 0; y < TILE; ++y) {
        for (int i = 1; i <= PAD; ++i) {
            dst[(y + PAD) * PADDED + (PAD - i)] = src[y * TILE + (i - 1)];
            dst[(y + PAD) * PADDED + (PADDED - i)] = src[y * TILE + (TILE - i)];
        }
    }
    
    // верх и низ
    for (int x = 0; x < PADDED; ++x) {
        for (int i = 1; i <= PAD; ++i) {
            dst[(PAD - i) * PADDED + x] = dst[(PAD + i - 1) * PADDED + x];
            dst[(PADDED - i) * PADDED + x] = dst[(PADDED - i - 1) * PADDED + x];
        }
    }
}

*/