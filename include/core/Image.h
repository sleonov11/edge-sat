#pragma once

#include <vector>
#include <span>
#include <algorithm>
#include <cstddef>

template <typename T> 
class Image {
public:
    Image () : width_(0) , height_(0), channels_(0) {}
    explicit Image (size_t width, size_t height, size_t channels) : 
        width_(width), height_(height), channels_(channels), data_(width*height*channels) {}
    ~Image() noexcept = default;

    Image(Image&& other) noexcept :
        width_(other.width_), height_(other.height_), channels_(other.channels_), data_(std::move(other.data_)) {
            other.width_ = 0;
            other.height_ = 0;
            other.channels_ = 0;
        }
    
    Image& operator =(Image&& other) noexcept {
        std::swap(width_, other.width_);
        std::swap(height_, other.height_);
        std::swap(channels_, other.channels_);
        std::swap(data_, other.data_); 
        return *this;
    }

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    
    size_t width() const noexcept {return width_;}
    size_t height() const noexcept {return height_;}
    size_t channels() const noexcept {return channels_;}
    size_t size() const noexcept {return data_.size();}
    bool empty() const noexcept {return data_.empty();}

    void reset() noexcept {
        width_ = 0;
        height_ = 0;
        channels_ = 0;
        data_.clear();
    }

    T* data() noexcept {return data_.data();}
    const T* data() const noexcept {return data_.data();}
    
    std::span<T> span_data() noexcept {
        return std::span<T> (data_.data(), data_.size());
    }
    std::span<const T> span_data() const noexcept {
        return std::span<const T> (data_.data(), data_.size());
    }

    T& operator() (size_t x, size_t y, size_t c) {
        return data_[(y*width_ + x) * channels_ + c];
    }

    const T& operator() (size_t x, size_t y, size_t c) const {
        return data_[(y*width_ + x) * channels_ + c];
    }

    T& operator[](size_t idx) {return data_[idx];}
    const T& operator[](size_t idx) const {return data_[idx];}

    T& at(size_t x, size_t y, size_t c) {
        return data_.at((y * width_ + x) * channels_ + c);
    }
    const T& at(size_t x, size_t y, size_t c) const {
        return data_.at((y * width_ + x) * channels_ + c);
    }
    // good way to work with rows in dwt
    std::span<T> row(size_t y) {
        size_t row_start = y * width_ * channels_;
        return std::span<T>(data_.data() + row_start, width_ * channels_);
    }

    std::span<const T> row(size_t y) const {
        size_t row_start = y * width_ * channels_;
        return std::span<const T>(data_.data() + row_start, width_ * channels_);
    }

    // it can make dwt faster (i believe in it);
    Image transpose() const {
        Image<T> result (height_, width_, channels_);
        constexpr size_t BLOCK = 64;
        const T* src = data();
        T* dst = result.data();

        for (size_t by = 0; by < height_; by += BLOCK) {
            size_t max_y = std::min(by + BLOCK, height_);

            for(size_t bx = 0; bx < width_; bx += BLOCK) {
                size_t max_x = std::min(bx + BLOCK, width_);
                
                for (size_t y = by; y < max_y; ++y) {
                    for(size_t x = bx; x < max_x; ++x) {
                        size_t src_idx = (y * width_ + x) * channels_;
                        size_t dst_idx = (x * height_ + y) * channels_;
                        if (channels_ == 3) { // for rgb;
                            dst[dst_idx] = src[src_idx];
                            dst[dst_idx + 1] = src[src_idx + 1];
                            dst[dst_idx + 2] = src[src_idx + 2];
                        } else {
                            for (size_t c = 0; c < channels_; ++c) {dst[dst_idx + c] = src[src_idx + c];}
                        }
                    }
                }
            }
        }
        return result;
    }

    template <typename T>
    Image<T> scaleToGray (Image<T>& src) {
        size_t w = src.widht(), h = src.height();
        Image<T> gray(src.widht(), src.height(), 1);
        for (size_t y = 0; y < h; ++y) {
            for (size_t x = 0; x < w; ++x) {
                T r = src(x,y,0);
                T g = src(x,y,1);
                T b = src(x,y,2);
                gray(x,y,0) = static_cast<T> (0.299 * r + 0.587 * g + 0.114 * b);
            }
        }
        return gray;
    }

private: 
    size_t width_, height_, channels_;
    std::vector<T> data_;
};