#pragma once

#include <vector>
#include <cstdint>
#include <cassert>
#include <algorithm>

struct YoloTask {
    int frame_id;
    int offset_x, offset_y;
    int width, height, channels;
    std::vector<uint8_t> pixels;

    YoloTask (int fid, int x, int y, int w, int h, int ch,
            const uint8_t* frame_data, int frame_stride, int frame_w, int frame_h) :
            frame_id(fid), offset_x(x), offset_y(y),
          width(w), height(h), channels(ch) {

        assert(frame_data != nullptr);
        assert(w > 0 && h > 0 && ch > 0);
        assert(x >= 0 && y >= 0);
        assert(x + w <= frame_w);   
        assert(y + h <= frame_h);   
        pixels.resize(w * h * ch);
        for (int row = 0; row < h; ++row) {
            const uint8_t* src = frame_data + (y + row) * frame_stride + x * ch;
            uint8_t* dst = pixels.data() + w * row * ch;
            std::copy(src, src + w * ch, dst);
        }
    }        
};