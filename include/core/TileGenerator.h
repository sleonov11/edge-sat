#pragma once

#include "TileView.h"

class TileGenerator {
public:
    TileGenerator (const uint8_t* data, size_t stride, int img_w, int img_h,
                     int tile_w, int tile_h, int overlap) :
                     data_(data), stride_(stride), img_w_(img_w), img_h_(img_h),
                     tile_w_(tile_w), tile_h_(tile_h), step_x_(tile_w - overlap),
                     step_y_(tile_h - overlap) {}
    
    bool next (TileView& view) {
        if (current_y_ >= img_h_) return false;
        
        int w = std::min(tile_w_, img_w_ - current_x_);
        int h = std::min (tile_h_, img_h_ - current_y_);

        if (w < tile_w_ / 2 || h < tile_h_ / 2) {
            advance();
            return next(view);
        }

        view = TileView(data_, stride_, current_x_, current_y_, w, h, 3);

        advance();
        return true;
    }

    size_t count() const {
        return static_cast<size_t>((img_w_ * img_h_ )/ (tile_w_ * tile_h_));
    }

private:
    void advance() {
        current_x_ += step_x_;
        if (current_x_ >= img_w_) {
            current_x_ = 0;
            current_y_ += step_y_;
        }
    }

    const uint8_t* data_;
    size_t stride_;
    int img_w_, img_h_; 
    int tile_w_, tile_h_;
    int step_x_, step_y_;
    int current_x_ = 0;
    int current_y_ = 0;

};