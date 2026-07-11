#include "core/Image.h"
#include <vector>
#include <span>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iomanip>

template <typename T>
void dwt_1d (std::span<T> src, std::span<T> res) {
    size_t n = src.size();
    const T inv_sqrt2 = T(1) / std::sqrt(T(2));
    for (size_t i = 0; i < n / 2; ++i) {
        T mean = (src[2*i] + src[2*i + 1]) * inv_sqrt2;
        T detail = (src[2*i] - src[2*i + 1]) * inv_sqrt2;
        res[i] = mean;
        res[n/2+i] = detail;
    }
}

template <typename T>
void dwt_2d (Image<T>& img) {
    size_t w = img.width(), h = img.height();

    for (size_t y = 0; y < h; ++y) {
        auto ry = img.row(y);
        std::vector<T> temp(w);
        dwt_1d(ry, std::span<T>(temp));
        std::copy(temp.begin(), temp.end(), ry.begin());
    }

    auto transposed = img.transpose();
    for (size_t x = 0; x < h; ++x) {
        auto span_x = transposed.row(x);
        std::vector<T> tmp(h);
        dwt_1d (span_x, std::span<T>(tmp));
        std::copy (tmp.begin(), tmp.end(), span_x.begin());
    }
    img = transposed.transpose();
} 

int main() {
    std::vector<double> data = {2.0, 3.0, 4.0, 1.0, 7.0, 9.0};
    std::vector<double> result(data.size());

    std::span<double> src(data);
    std::span<double> res(result);

    dwt_1d(src, res);

    for(auto v : res) {
        std::cout << v << " ";
    }
    
        const size_t W = 4, H = 4;
    Image<float> img(W, H, 1);

    float val = 0.0f;
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            img(x, y, 0) = val++;
        }
    }

    std::cout << "Original:\n";
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            std::cout << std::setw(6) << img(x, y, 0) << " ";
        }
        std::cout << "\n";
    }

    dwt_2d(img);

    std::cout << "After 2D DWT:\n";
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            std::cout << std::setw(6) << img(x, y, 0) << " ";
        }
        std::cout << "\n";
    }
};