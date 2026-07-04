#pragma once

#include <vector>

template <typename T> 
class Image {
public:
    Image () : width_(0) , height_(0), channels_(0) {}
    explicit Image (size_t width, size_t height, size_t channels) : 
        width_(width), height_(height), channels_(channels), data(width*height*channels) {}
    ~Image() noexcept = default;

    Image(Image&& other) noexcept :
        width_(other.width_), height_(other.height_), channels_(other.channels_), data(std::move(other.data)) {
            other.width_ = 0;
            other.height_ = 0;
            other.channels_ = 0;
        }
    Image& operator =(Image&&) noexcept = default;

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    
    size_t width() const noexcept {return width_;}
    size_t height() const noexcept {return height_;}
    size_t channels() const noexcept {return channels_;}
    size_t size() const noexcept {return data.size();}
    bool empty() const noexcept {return data.empty();}

    void reset() noexcept {
        width_ = 0;
        height_ = 0;
        channels_ = 0;
        data.clear();
    }

    T* data() noexcept {return data.data();}
    const T* data() const noexcept {return data.data();}
    
    T& operator[](size_t idx) {return data[idx];}
    const T& operator[](size_t idx) const {return data[idx];}

    T& at(size_t x, size_t y, size_t c) {
        return data.at((y * width_ + x) * channels_ + c);
    }
    const T& at(size_t x, size_t y, size_t c) const {
        return data.at((y * width_ + x) * channels_ + c);
    }

private: 
    size_t width_, height_, channels_;
    std::vector<T> data;
};