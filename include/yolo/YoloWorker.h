#pragma once

#include "TaskQueue.h"
#include "YoloTask.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class YoloWorker {
    TaskQueue<YoloTask>& queue_; // ссылка тк очередь общая.
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    std::vector<int64_t> input_shape_ = {1, 3, 640, 640};
    float conf_threshold_ = 0.25f;
    float nms_threshold_ = 0.45f;

public:
    YoloWorker(const std::string& model_path, TaskQueue<YoloTask>& queue);
    void run();
};