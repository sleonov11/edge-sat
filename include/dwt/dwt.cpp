// #include "core/Image.h"
#include <vector>
#include <span>
#include <cmath>
#include <iostream>

template <typename T>
void dwt_1d (std::span<T> src, std::span<T> res) {
    size_t n = src.size();
    const T inv_sqrt2 = T(1) / std::sqrt(T(2));
    for (size_t i = 0; i < n / 2; ++i) {
        T mean = (src[2*i] + src[2*i + 1]) * inv_sqrt2;
        T approx = (src[2*i] - src[2*i + 1]) * inv_sqrt2;
        res[i] = mean;
        res[n/2+i] = approx;
    }

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
    return 0;
};