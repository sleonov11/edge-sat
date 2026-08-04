template <int N> 
void dwt_1d_haar (const float* __restrict src, float* __restrict dst) {
    constexpr float INV_SQRT2 = 1 / std::sqrt(2);

    for (int i = 0; i < N/2; ++i) {
        float a = src[2 * i];
        float b = src [2 * i + 1];
        dst[i] = (a + b) * INV_SQRT2;
        dst[N/2 + i] = (a - b) * INV_SQRT2;
    }
}

template <int N, int M>
void transpose (const float* __restrict src, float* __restrict dst) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            dst [j * N + i] = src [i * M + j];
        }
    }
}

template<int N, int M, int BLOCK = 8>
void transpose_block(const float* __restrict src, float* __restrict dst) {
    for (int bi = 0; bi < N; bi += BLOCK) {
        for (int bj = 0; bj < M; bj += BLOCK) {
            for (int i = bi; i < std::min(bi + BLOCK, N); ++i) {
                for (int j = bj; j < std::min(bj + BLOCK, M); ++j) {
                    dst[j * N + i] = src[i * M + j];
                }
            }
        }
    }
}

template <int TILE>
void dwt_2d_haar (float* __restrict buf, float* __restrict tmp) {
    for (int y = 0; y < TILE; ++y) {
        dwt_1d_haar<TILE>(buf + y * TILE, tmp + y * TILE);
    }

    transpose_block<TILE,TILE> (tmp, buf);

    for (int y = 0; y < TILE; ++y) {
        dwt_1d_haar<TILE> (buf + y * TILE, tmp + y * TILE);
    }

}