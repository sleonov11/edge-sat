#pragma once

#include <cstdint>
#include <cstddef>

struct TileView {
    const uint8_t* data; // указатель на начала ОРИГИНАЛА изображения
    size_t stride_bytes; // шаг между строками изображения
    int x, y; // координаты тайла в ИСХОДНИКЕ
    int w, h; // ширина и высота тайла
    int channels; // кол-во каналов (3 ргб, 1 чб)

    TileView() = default;
    TileView(const uint8_t* d, size_t stride, int x_, int y_, int w_, int h_, int ch_) :
    data(d), stride_bytes(stride), x(x_), y(y_), w(w_), h(h_), channels(ch_) {}

    const uint8_t* pixel (int tx, int ty, int c) const {
        return data + (y + ty) * stride_bytes + (x + tx) * channels + c;
    }

    const uint8_t* row (int ty, int c) const {
        return data + (y + ty) * stride_bytes + x * channels + c;
    }

    // для чб (channels == 1);
    const uint8_t at(int tx, int ty) const {
        return data[(y + ty) * stride_bytes + (x + tx)];
    }

    size_t size_bytes () const {
        return static_cast<size_t>(w) * h * channels;
    }

};