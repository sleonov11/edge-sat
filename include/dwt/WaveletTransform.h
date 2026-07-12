#pragma once

#include "core/Image.h"
#include <span>
#include <cmath>
#include <algorithm>

template<typename T>
void dwt_1d (std::span<T> src, std::span<T> res);

template <typename T>
void dwt_2d (Image<T>& src);

template <typename T>
std::vector<float> extract_features(Image<T>& src);

#include "WaveletTransform.cpp"

