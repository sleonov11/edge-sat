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
    for (size_t x = 0; x < transposed.height(); ++x) {
        auto span_x = transposed.row(x);
        std::vector<T> tmp(h);
        dwt_1d (span_x, std::span<T>(tmp));
        std::copy (tmp.begin(), tmp.end(), span_x.begin());
    }
    img = std::move(transposed);
}

template <typename T>
Image<T> extract_subband (const Image<T>& src, size_t x0, size_t y0, size_t w, size_t h) {
    Image<T> sub (w, h, 1);
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            sub(x, y, 0) = src(x + x0, y + y0, 0);
        }
    }
    return sub;
}

template <typename T>
FeatureStats compute_stats (const Image<T>& subband) {
    size_t N = subband.size();
    double sum = 0, sumSq = 0, sumLog = 0;

    for (size_t i = 0; i < N; ++i) {
        double v = static_cast<double> (subband[i]);
        sum += v;
        sumSq += v * v;
        if (v > 1e-12) sumLog += std::log(std::abs(v));
    }

    double mean = sum / N;
    FeatureStats res(mean, (sumSq / N) - mean*mean, sumSq, -sumLog/N);
    return res;
}

template <typename T>
std::vector<float> extract_features(const Image<T>& src) {
    Image<T> copy = src.scaleToGray();

    dwt_2d(copy);

    size_t w = copy.width() / 2;
    size_t h = copy.height() / 2;

    auto LL = extract_subband (copy, 0, 0, w, h);
    auto LH = extract_subband (copy, w, 0, w, h);
    auto HL = extract_subband (copy, 0, h, w, h);
    auto HH = extract_subband (copy, w, h, w, h);

    auto sLL = compute_stats(LL);
    auto sLH = compute_stats(LH);
    auto sHL = compute_stats(HL);
    auto sHH = compute_stats(HH);

    std::vector<float> features;
    features.reserve(16);

    auto pushStats = [&](const FeatureStats& s) {
        features.push_back(s.mean);
        features.push_back(s.variance);
        features.push_back(s.energy);
        features.push_back(s.entropy);
    };

    pushStats(sLL);
    pushStats(sLH);
    pushStats(sHL);
    pushStats(sHH);

    return features;
}












;