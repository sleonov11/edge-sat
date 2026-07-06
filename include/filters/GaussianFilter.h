#pragma once

#include "core/Image.h"
#include <cmath>

// стратегии обхода краёев изображения
enum class EdgeHandling {
    Mirror, // Зеркально отражение
    Clamp, // ограничение крайним пикселем 
    Zero, // нулевое дополнение
    Extend // повторение крайнего
};
// im too stupid...
template <typename T>
class GaussianFilter {
public:

private:
    
};
