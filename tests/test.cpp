#include "yolo/TaskQueue.h"
#include "yolo/YoloTask.h"
#include "yolo/YoloWorker.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    std::string model_path = (argc > 1) ? argv[1] : "include/yolo/models/yolov8n.onnx";
    std::string image_path = (argc > 2) ? argv[2] : "test.jpg";

    TaskQueue<YoloTask> task_queue;
    TaskQueue<DetectionResult> result_queue;

    YoloWorker worker(model_path, task_queue, result_queue);

    std::thread worker_thread([&]{ worker.run(); });
    
    // Consumer: забирает результаты и печатает
    std::thread result_thread([&]{
        while (auto res = result_queue.pop()) {
            std::cout << "[Result] Frame " << res->frame_id
                      << " objects: " << res->boxes.size() << "\n";
            for (const auto& b : res->boxes) {
                std::cout << "  class=" << b.class_id
                          << " conf=" << b.conf
                          << " box=[" << b.x << "," << b.y
                          << "," << b.w << "," << b.h << "]\n";
            }
        }
        std::cout << "[Result] consumer exiting\n";
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cout << "WARNING: " << image_path << " not found. Using synthetic.\n";
        img = cv::Mat(480, 640, CV_8UC3);
        cv::randu(img, cv::Scalar(0,0,0), cv::Scalar(255,255,255));
    }

    YoloTask task(42, 0, 0, img.cols, img.rows, img.channels(),
                  img.data, static_cast<int>(img.step), img.cols, img.rows);
    task_queue.push(std::move(task));

    std::this_thread::sleep_for(std::chrono::seconds(2));

    // ВАЖНЫЙ ПОРЯДОК:
    task_queue.stop();      // 1. Worker'ы выходят из pop(), завершают run()
    worker_thread.join();   // 2. Ждём, пока worker закончит
    
    result_queue.stop();    // 3. Будим consumer результатов
    result_thread.join();   // 4. Ждём consumer

    std::cout << "Test finished\n";
    return 0;
}