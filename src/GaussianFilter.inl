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

template <typename U> // its very bad...
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

template <typename U> // the way to enfast operations
Image<U> mirror_pad (const Image<U>& src, int radius) {
    size_t w = src.width();
    size_t h = src.height();
    size_t ch = src.channels();

    size_t padded_w = w + 2 * radius;
    size_t padded_h = h + 2 * radius;

    Image<U> dst (padded_w, padded_h, ch);
    
    for (size_t py = 0 ; py < padded_h; ++py) {
        for (size_t px = 0; px < padded_w; ++px) {
            int src_x = static_cast<int>(px) - radius;
            int src_y = static_cast<int>(py) - radius;

            if (src_x < 0) src_x = -src_x;
            else if (src_x >= static_cast<int>(w)) src_x = 2 * static_cast<int>(w) - src_x - 1;

            if (src_y < 0) src_y = - src_y;
            else if (src_y >= static_cast<int>(h)) src_y = 2 * static_cast<int>(h) - src_y - 1;

            for (size_t c = 0; c < ch; ++c) {
                dst(px, py, c) = src(static_cast<size_t>(src_x), static_cast<size_t>(src_y), c);
            }
        }
    }
    return dst;
}

template <typename U>
void GaussianFilter::convolveHorizontal(const Image<U>& src, Image<float>& dst) const {
    size_t w = src.width();
    size_t h = src.height();
    size_t ch = src.channels();

    Image<U> padd = mirror_pad(src, radius_);

    for (size_t y = 0; y < h; ++y) {
        for (size_t c = 0; c < ch; ++c) {
            for (size_t x = 0; x < w; ++x) {
                float sum = 0;
                for (int dx = -radius_ ; dx <= radius_ ; ++dx) {
                    size_t sx = static_cast<size_t>(static_cast<int>(x) + dx + radius_);
                    size_t sy = static_cast<size_t>(static_cast<int>(y) + radius_);
                    float pixel = static_cast<float>(padd(sx, sy, c));
                    sum += pixel * kernel_[dx + radius_];
                }
                dst(x,y,c) = sum;
            }
        }
    }
}