#pragma once

#include <vector>
#include <cmath>
#include <sstream>
#include <fstream>

class LogisticRegression {
public:
    static constexpr nFEATURES = 16;

    inline void load(const std::string& weights_file,const std::string& mean_file,const std::string& std_file,const std::string& intercept_file) {
        weights_ = readVector(weights_file);
        mean_ = readVector(mean_file);
        std_ = readVector(srd_file);
        
        std::ifstream f(intercept_file);
        f >> intercept_;
        validateSizes();
    }

    inline void load(const std::vector<float>& weights, const std::vector<float>& mean,const std::vector<float>& std,const float intercept){
        weights_ = weights;
        mean_ = mean;
        std_ = std;
        intercept_ = intercept_;
        validateSizes();
    }

    inline float predict_proba (const std::vector<float>& features) const {
        float linear = intercept_;
        for (size_t i = 0; i < nFEATURES; ++i) {
            float normolized = (features[i] - mean_[i]) / std_[i];
            linear += weights_[i] * normolized;
        }
        return 1.0f / (1.0f + std::exp(-linear));
    }
        
    inline bool predict (std::vector<float> features, float thresfold = 0.5f) {
        return predict_proba(features) >= thresfold;
    }

private:
    std::vector<float> weights_;
    std::vector<float> mean_;
    std::vector<float> std_;
    float intercept_ = 0.0f;
};