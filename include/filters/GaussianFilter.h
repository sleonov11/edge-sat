#pragma once

#include "core/Image.h"
#include <cmath>

// стратегии обхода краёев изображения (пока не используется)
enum class EdgeHandling {
    Mirror, // Зеркально отражение (по умолчанию)
    Clamp, // ограничение крайним пикселем 
    Zero, // нулевое дополнение
    Extend // повторение крайнего
};

class GaussianFilter {
public:
    explicit GaussianFilter(float sigma, EdgeHandling border_mode_=EdgeHandling::Mirror);
    
    template <typename U>
    Image<U> apply(const Image<U>& input) const;

    float sigma() const {return sigma_;}
    int radius() const {return radius_;}
    EdgeHandling border_mode() const {return border_mode_;}
    const std::vector<float>& kernel() const { return kernel_; }
private:
    float sigma_;
    std::vector<float> kernel_;
    int radius_;
    EdgeHandling border_mode_;

    void build_kernel();
    
    template <typename U>
    U getPixel(const Image<U>& img, int x, int y, int c) const;

    void convolveHorizontal(const Image<float>& src, Image<float>& dst) const;

    void convolveVertical(const Image<float>& src, Image<float>& dst) const;
};


#include "GaussianFilter.inl"