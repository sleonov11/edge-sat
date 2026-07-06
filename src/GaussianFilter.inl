#include "include/filters/GaussianFilter.h"

GaussianFilter::GaussianFilter(float sigma, EdgeHandling border_mode) 
        : sigma_(sigma), border_mode_(border_mode) {build_kernel();} 

void GaussianFilter::build_kernel() {
    int r = static_cast<int>(3.0f * sigma_);
    radius_ = r < 1 ? 1 : r;
    int size = 2 * radius_ + 1;
    kernel_.resize(size);
    
    float sum = 0.0f;
    for (int i = 0; i <= 2*radius_; ++i) {
        int d = i - radius_;
        float val = std::exp(-(d*d) / (2.0f*sigma_*sigma_));
        kernel_[i] = val;
        sum += val;
    }

    for (float& v : kernel_) v /= sum;    
}

template <typename U>
U GaussianFilter::getPixel (const Image<U>& img, int x, int y, int c) const {
    int w = static_cast<int> (img.width());
    int h = static_cast<int> (img.height());

    if (x < 0) {
        x = -x - 1;
    } else if (x >= w) {
        x = 2*w - x - 1;
    }

    if (y < 0) {
        y = -y - 1;
    } else if (y >= h) {
        y = 2*h - y - 1;
    }

    return img(static_cast<size_t>(x), static_cast<size_t>(y),static_cast<size_t>(c));
}

template <typename U>
void GaussianFilter::convolveHorizontal(const Image<U>& src, Image<float>& dst) const {
    size_t w = src.width();
    size_t h = src.height();
    size_t ch = src.channels();

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            for (size_t c = 0; c < ch; ++c) {
                float sum = 0;
                for (int dx = -radius_ ; dx <= radius_ ; ++dx) {
                    int neighbor_x =static_cast<int>(x) + dx;
                    float pixel_val = getPixel(src, neighbor_x, static_cast<int>(y), static_cast<int>(c));
                    float weight = kernel_[dx + radius_];
                    sum += pixel_val * weight;
                }
                dst(x,y,c) = sum;
            }
        }
    }
}