#pragma once 

#include "core/TileView.h"

template <int TILE>
void rgb_to_gray (const ImageView& view, float* __restrict gray) {
    // view - rgb тайл из ориг изображения
    // gray - выход, буффер TILE * TILE float; 

    for (int y = 0; y < view.h; ++y) {
        const uint8_t* row_r = view.row(y, 0); // весь канал R
        float* dst = gray + y * TILE;
        for (int x = 0; x < view.w; ++x) {
            uint8_t r = row_r[3 * x];
            uint8_t g = row_r[3 * x + 1];
            uint8_t b = row_r[3 * x + 2];
            dst[x] = 0.299f * r + 0.587f * g + 0.114f * b;
        }

    }

}