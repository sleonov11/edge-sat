#include "YoloWorker.h"
#include <iostream>
#include <opencv2/opencv.hpp>

static Ort::SessionOptions MakeSessionOptions () {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}

YoloWorker::YoloWorker(const std::string& model_path, TaskQueue<YoloTask>& queue, TaskQueue<DetectionResult>& result_queue) :
    queue_(queue),
    result_queue_(result_queue),
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

        float* data = outputs[0].GetTensorMutableData<float>();
        const int num_classes = 10;
        const int num_boxes = 8400;

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        for (int i = 0; i < num_boxes; ++i) {
            // Координаты центра и размеры (относительно 640×640)
            float x = data[0 * num_boxes + i];
            float y = data[1 * num_boxes + i];
            float w = data[2 * num_boxes + i];  
            float h = data[3 * num_boxes + i];
            // ищем максимум среди всех классов
            float max_score = 0.0f;
            int class_id = -1;
            for (int c = 0; c < num_classes; ++c) {
                float score = data[(4 + c)*num_boxes + i];
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
            }

            if (max_score > conf_threshold_) {
                int x1 = static_cast<int>(x - w/2);
                int y1 = static_cast<int>(y - h/2);
                int x2 = static_cast<int>(w);
                int y2 = static_cast<int>(h);

                boxes.push_back(cv::Rect(x1, y1, x2, y2));
                scores.push_back(max_score);
                class_ids.push_back(class_id);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, indices);

        float scale_x = task->width / 640.0f;
        float scale_y = task->height / 640.0f;

        DetectionResult result(*task);
        for (int idx : indices) {
            YoloBox box;
            box.x = static_cast<int>(boxes[idx].x * scale_x) + task->offset_x;
            box.y = static_cast<int>(boxes[idx].y * scale_y) + task->offset_y;
            box.w = static_cast<int>(boxes[idx].width * scale_x);
            box.h = static_cast<int>(boxes[idx].height * scale_y);
            box.class_id = class_ids[idx];
            box.conf = scores[idx];
            
            result.boxes.push_back(box);
        }
        result_queue_.push(std::move(result));
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