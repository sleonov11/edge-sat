#include "YoloWorker.h"
#include <iostream>
#include <opencv2/opencv.hpp>

static Ort::SessionOptions MakeSessionOptions () {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}

YoloWorker::YoloWorker(const std::string& model_path, TaskQueue<YoloTask>& queue) :
    queue_(queue),
    env_ (ORT_LOGGING_LEVEL_WARNING, "yolo"),
    session_(env_, model_path.c_str(), MakeSessionOptions()),
    memory_info_ (Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
    input_shape_{1,3,640,640},
    conf_threshold_(0.25f),
    nms_threshold_(0.45f)
{

}

void YoloWorker::run() {
    while (true) {
        auto task = queue_.pop();
        if (!task) break;

        auto input = preprocess(*task);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            input.data(),
            input.size(),
            input_shape_.data(),
            input_shape_.size()
        );

        const char* input_names[] = {"images"};
        const char* output_names[] = {"output0"};

        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1
        );
    }
}

std::vector<float> YoloWorker::preprocess(const YoloTask& task) {
    cv::Mat roi(task.height, task.width, CV_8UC(task.channels), const_cast<uint8_t*>(task.pixels.data()));

    cv::Mat rgb;
    if (task.channels == 1) {
        cv::cvtColor (roi, rgb, cv::COLOR_GRAY2RGB);
    } else {
        rgb = roi;
    }
    cv::Mat resized;
    cv::resize (rgb, resized, cv::Size(640, 640));

    cv::Mat float_mat;
    resized.convertTo(float_mat, CV_32F, 1.0 / 255.0);

    std::vector<float> input (3 * 640 * 640);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 640; ++h) {
            for (int w = 0; w < 640; ++w) {
                input[c * 640 * 640 + h * 640 + w] = float_mat.at<cv::Vec3f>(h,w)[c];
            }
        }
    }
    return input;
}