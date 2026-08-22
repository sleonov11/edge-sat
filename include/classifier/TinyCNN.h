#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>

class TinyCNN {
public:
    TinyCNN (const std::string& model_path, float threshold=0.9f);
    bool isWater(const cv::Mat& tile); // основной метод
    float getWaterProb (const cv::Mat& tile); // возвращает вероятность класса [0...1];

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    std::vector<int64_t> shape_ {1, 3, 256, 256};
    float threshold_;

    std::vector<float> preprocess(const cv::Mat& tile);
};