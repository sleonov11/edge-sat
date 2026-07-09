#include "filters/GaussianFilter.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <numeric>
#include <cstdint>

// -------------------------------------------------------------------
// Вспомогательные функции
// -------------------------------------------------------------------

// Сравнение двух изображений с допуском (абсолютная погрешность)
template <typename T>
bool imagesAlmostEqual(const Image<T>& a, const Image<T>& b, float eps = 1.0f) {
    if (a.width() != b.width() || a.height() != b.height() || a.channels() != b.channels())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i])) > eps)
            return false;
    }
    return true;
}

// Генерация изображения с постоянным значением
template <typename T>
Image<T> constantImage(size_t w, size_t h, size_t ch, T value) {
    Image<T> img(w, h, ch);
    std::fill(img.data(), img.data() + img.size(), value);
    return img;
}

// Генерация изображения с единичным импульсом в центре
template <typename T>
Image<T> impulseImage(size_t w, size_t h, size_t ch, T peak = 255) {
    Image<T> img(w, h, ch);
    std::fill(img.data(), img.data() + img.size(), T(0));
    if (w > 0 && h > 0) {
        size_t cx = w / 2;
        size_t cy = h / 2;
        for (size_t c = 0; c < ch; ++c)
            img(cx, cy, c) = peak;
    }
    return img;
}

// Сохранение одноканального изображения в формат PGM (P5, серый)
bool savePGM(const Image<uint8_t>& img, const std::string& filename) {
    if (img.channels() != 1) return false;
    std::ofstream f(filename, std::ios::binary);
    if (!f) return false;
    f << "P5\n" << img.width() << " " << img.height() << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data()), img.size());
    return true;
}

// -------------------------------------------------------------------
// Тесты
// -------------------------------------------------------------------

void testKernelNormalization() {
    const size_t W = 16, H = 16;
    auto img = constantImage<float>(W, H, 1, 1.0f);
    GaussianFilter filter(1.0f);
    auto res = filter.apply(img);
    float mean = std::accumulate(res.data(), res.data() + res.size(), 0.0f) / res.size();
    assert(std::abs(mean - 1.0f) < 1e-4);
    std::cout << "[PASS] Kernel normalization (via constant image)\n";
}

void testConstantImage() {
    const size_t W = 8, H = 8, CH = 1;
    auto img = constantImage<uint8_t>(W, H, CH, 128);
    GaussianFilter filter(1.0f);
    auto filtered = filter.apply(img);
    assert(imagesAlmostEqual(img, filtered, 1.0f));
    std::cout << "[PASS] Constant image preservation\n";
}

void testImpulseResponse() {
    const size_t W = 9, H = 9, CH = 1;
    auto img = impulseImage<float>(W, H, CH, 1.0f);
    GaussianFilter filter(1.0f);
    auto res = filter.apply(img);

    // Сумма пикселей результата должна быть ≈ 1 (сохранение энергии)
    float sum = std::accumulate(res.data(), res.data() + res.size(), 0.0f);
    assert(std::abs(sum - 1.0f) < 1e-4);

    // Пик должен быть в центре
    size_t cx = W / 2, cy = H / 2;
    float maxVal = 0.0f;
    size_t maxX = 0, maxY = 0;
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            float v = res(x, y, 0);
            if (v > maxVal) { maxVal = v; maxX = x; maxY = y; }
        }
    }
    assert(maxX == cx && maxY == cy);
    std::cout << "[PASS] Impulse response (energy conservation, peak at center)\n";
}

void testGradientMean() {
    const size_t W = 16, H = 16, CH = 1;
    Image<float> img(W, H, CH);
    for (size_t y = 0; y < H; ++y)
        for (size_t x = 0; x < W; ++x)
            img(x, y, 0) = static_cast<float>(x) / W * 255.0f;

    GaussianFilter filter(1.0f);
    auto filtered = filter.apply(img);

    float meanOrig = std::accumulate(img.data(), img.data() + img.size(), 0.0f) / img.size();
    float meanFilt = std::accumulate(filtered.data(), filtered.data() + filtered.size(), 0.0f) / filtered.size();
    assert(std::abs(meanOrig - meanFilt) < 1.0f);
    std::cout << "[PASS] Gradient mean preservation\n";
}

void testVisual() {
    const size_t W = 64, H = 64, CH = 1;
    auto img = constantImage<uint8_t>(W, H, CH, 128);
    // Добавим чёрный квадрат в центре
    for (size_t y = 20; y < 44; ++y)
        for (size_t x = 20; x < 44; ++x)
            img(x, y, 0) = 0;

    GaussianFilter filter(2.0f);
    auto filtered = filter.apply(img);

    savePGM(img, "input.pgm");
    savePGM(filtered, "filtered.pgm");
    std::cout << "[INFO] Visual test: saved input.pgm and filtered.pgm. Check them manually.\n";
}

// -------------------------------------------------------------------
// Основная функция
// -------------------------------------------------------------------
int main() {
    std::cout << "Running GaussianFilter tests...\n";
    try {
        testKernelNormalization();
        testConstantImage();
        testImpulseResponse();
        testGradientMean();
        testVisual();
        std::cout << "\nAll tests PASSED.\n";
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Test FAILED with unknown exception.\n";
        return 1;
    }
    return 0;
}