#pragma once

#include <array>

class DecisionTree {
public:
    inline bool classify_tile(const std::array<float, 16>& features) {
        if (features[9] > 129.125f) {
            return features[11] <= 7.204f;
        }

        if (features[15] <= 6.762f && features[6] > 157233.0f) {
            return true;
        }
        return false;
    }
};
