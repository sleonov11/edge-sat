#pragma once

#include "core/Image.h"
#include <span>
#include <cmath>
#include <algorithm>

struct FeatureStats {
    double mean;
    double variance;
    double energy;
    double entropy;
    FeatureStats (double mean, double variance, double energy, double entropy) : mean(mean), variance(variance), energy(energy), entropy(entropy) {}
};

template<typename T>
void dwt_1d (std::span<T> src, std::span<T> res);

template <typename T>
void dwt_2d (Image<T>& src);

template <typename T>
std::vector<float> extract_features(const Image<T>& src);


template <typename T>
Image<T> extract_subband (const Image<T>& src, size_t x0, size_t y0, size_t w, size_t h);

template <typename T>
FeatureStats compute_stats (const Image<T>& subband);

#include "WaveletTransform.inl"