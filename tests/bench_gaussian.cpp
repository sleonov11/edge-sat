#include "filters/GaussianFilter.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstdint>
#include <vector>
#include <numeric>

using namespace std::chrono;

// Генерация случайного изображения с плавающими значениями (чтобы избежать преобразований)
template <typename T>
Image<T> randomImage(size_t w, size_t h, size_t ch, T min = 0, T max = 255) {
    Image<T> img(w, h, ch);
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min, max);
        for (size_t i = 0; i < img.size(); ++i) img[i] = dist(gen);
    } else {
        std::uniform_int_distribution<T> dist(min, max);
        for (size_t i = 0; i < img.size(); ++i) img[i] = dist(gen);
    }
    return img;
}

// Один прогон бенчмарка
void benchmark(size_t w, size_t h, size_t ch, float sigma, int repeat = 5) {
    auto img = randomImage<float>(w, h, ch, 0.0f, 255.0f);
    GaussianFilter filter(sigma);

    // Прогрев (чтобы кэш и аллокации устаканились)
    auto warm = filter.apply(img);

    std::vector<double> times;
    times.reserve(repeat);
    for (int i = 0; i < repeat; ++i) {
        auto start = steady_clock::now();
        auto result = filter.apply(img);
        auto end = steady_clock::now();
        duration<double> elapsed = end - start;
        times.push_back(elapsed.count());
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double minTime = *std::min_element(times.begin(), times.end());

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Size: " << w << "x" << h << " ch=" << ch 
              << " sigma=" << sigma
              << " avg: " << avg * 1000 << " ms"
              << " min: " << minTime * 1000 << " ms\n";
}

int main() {
    std::cout << "=== GaussianFilter Benchmark ===\n";

    // Комбинации размеров и каналов
    std::vector<std::tuple<size_t, size_t, size_t>> sizes = {
        {256, 256, 1},
        {256, 256, 3},
        {512, 512, 1},
        {512, 512, 3},
        {1024, 1024, 1},
        {1024, 1024, 3},
        {2048, 2048, 1},
        {2048, 2048, 3}
    };

    std::vector<float> sigmas = {0.5f, 1.0f, 2.0f, 5.0f};

    for (auto [w, h, ch] : sizes) {
        for (auto sigma : sigmas) {
            benchmark(w, h, ch, sigma);
        }
        std::cout << "------------------------\n";
    }
    return 0;
}