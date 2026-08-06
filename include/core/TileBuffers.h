#pragma once

#include <cstddef>

template <int TILE>
struct TileBuffers {
    static constexpr int PAD = 2;
    static constexpr int PADDED = TILE + 2 * PAD;

    alignas(64) float gray [TILE * TILE]; // для чб
    alignas(64) float padded[PADDED * PADDED]; // для миррор пад, чтобы не вылезать за рамки.
    alignas(64) float gauss_h [PADDED * PADDED]; // для гаусса горизонт.
    alignas(64) float gauss_v [TILE * TILE]; //  для гаусса верт.
    alignas(64) float dwt_buf [TILE * TILE]; // двт на месте
    alignas(64) float dwt_tmp [TILE * TILE]; // двт транспонирование.

    static constexpr size_t total_bytes () {
        return sizeof(gray) + sizeof(padded) + sizeof(gauss_h) + sizeof(gauss_v) +
        sizeof(dwt_buf) + sizeof(dwt_tmp);
    }
};

using TileBuffers128 = TileBuffers<128>;