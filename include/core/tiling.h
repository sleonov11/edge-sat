#include "Image.h"

#include <vector>
#include <functional>
#include <algorithm>

template <typename T>
std::vector<Image<T>> splitToTile (const Image<T>& img,size_t tileW,size_t tileH,size_t overlap_x=0,size_t overlap_y=0) {
    std::vector<Image<T>> tiles;
    size_t w = img.width();
    size_t h = img.height();
    size_t step_x = tileW - overlap_x;
    size_t step_y = tileH - overlap_y;

    for (size_t y = 0; y < h; y += step_y) {
        for (size_t x = 0; x < w; x += step_x) {
            size_t endX = std::min(x + tileW, w);
            size_t endY = std::min(y + tileH, h);
            if ((endX - x) < tileW / 2 || (endY - y) < tileH / 2) continue;

            Image<T> tile (endX - x, endY - y, img.channels());
            for (size_t yy = 0; yy < tile.height(); ++yy) {
                const T* src = &img (x, y + yy, 0);
                T* dst = &tile (0, yy, 0);
                std::copy (src, src + tile.width() * tile.channels(), dst);
            }
            tiles.push_back(std::move(tile));
        }
    }
    return tiles;
}

