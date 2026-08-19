#pragma once

#include "TaskQueue.h"
#include "YoloTask.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

struct YoloBox {
    int x, y, w, h;
    int class_id;
    float conf;
};

struct DetectionResult {
    int frame_id;
    int offset_x, offset_y;
    std::vector<YoloBox> boxes;

    explicit DetectionResult (const YoloTask& task)
        : frame_id(task.frame_id), offset_x(task.offset_x), offset_y(task.offset_y) {}
};

class YoloWorker {
    TaskQueue<YoloTask>& queue_; // ссылка тк очередь общая.
    TaskQueue<DetectionResult>& result_queue_;
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    std::vector<int64_t> input_shape_ = {1, 3, 640, 640};
    float conf_threshold_ = 0.6f;
    float nms_threshold_ = 0.45f;

public:
    YoloWorker(const std::string& model_path, TaskQueue<YoloTask>& queue, TaskQueue<DetectionResult>& result_queue);
    void run();
    std::vector<float> preprocess (const YoloTask& task);
};