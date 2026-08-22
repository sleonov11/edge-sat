#include "TinyCNN.h"
#include <algorithm>

static Ort::SessionOptions MakeSessionOptions () {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(8);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}

TinyCNN::TinyCNN(const std::string& model_path, float threshold) 
    : env_(ORT_LOGGING_LEVEL_WARNING, "tinycnn"),
    session_(env_, model_path.c_str(), MakeSessionOptions()),
    memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
    threshold_(threshold) {}

std::vector<float> TinyCNN::preprocess(const cv::Mat& tile) {
    cv::Mat resized;

    cv::resize(tile, resized, cv::Size(256,256));

    cv::Mat float_mat;
    resized.convertTo(float_mat, CV_32F, 1.0/255.0);

    std::vector<float> input_data(3 * 256 * 256);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 256; ++h) {
            for (int w = 0; w < 256; ++w) {
                float val = float_mat.at<cv::Vec3f>(h,w)[c];
                input_data[c * 256 * 256 + h * 256 + w] = (val - 0.5f) / 0.5f;
            }
        }
    }
    return input_data;
}

float TinyCNN::getWaterProb(const cv::Mat& tile) {
    std::vector<float> input_data = preprocess(tile);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_data.data(),
        input_data.size(),
        shape_.data(),
        shape_.size()
    );

    // запуск модели
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    float* out_data = outputs[0].GetTensorMutableData<float>();
    // out_data[0] - суша || out_data[1] - вода/корабль

    float max_logit = std::max(out_data[0], out_data[1]);
    float exp0 = std::exp(out_data[0] - max_logit);
    float exp1 = std::exp(out_data[1] - max_logit);
    float prob_water = exp1 / (exp0 + exp1);
    
    return prob_water;
}

bool TinyCNN::isWater(const cv::Mat& tile) {
    return getWaterProb(tile) > threshold_;
}