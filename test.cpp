#include "yolo/TaskQueue.h"
#include "yolo/YoloTask.h"
#include "yolo/YoloWorker.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    // Пути можно переопределить аргументами:
    // ./test path/to/model.onnx path/to/image.jpg
    std::string model_path = (argc > 1) ? argv[1] : "include/yolo/models/yolov8n.onnx";
    std::string image_path = (argc > 2) ? argv[2] : "test.jpg";

    TaskQueue<YoloTask> queue;
    YoloWorker worker(model_path, queue);

    std::thread t([&]{ worker.run(); });

    // Даём воркеру заснуть в pop()
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Загружаем изображение
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cout << "WARNING: " << image_path << " not found. "
                  << "Creating synthetic 640x480 image...\n";
        // Синтетика — шум, детекций не будет, но pipeline проверим
        img = cv::Mat(480, 640, CV_8UC3);
        cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    }

    std::cout << "Loaded: " << img.cols << "x" << img.rows 
              << " channels=" << img.channels() << "\n";

    // Создаём задачу на весь кадр
    // img.step — stride в байтах (width * channels + возможный padding)
    YoloTask task(42, 0, 0, img.cols, img.rows, img.channels(),
                  img.data, static_cast<int>(img.step), img.cols, img.rows);

    queue.push(std::move(task));

    // Даём время на inference (YOLOv8n ~30-100ms на CPU, но пусть будет запас)
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "Stopping...\n";
    queue.stop();
    t.join();

    std::cout << "Test finished\n";
    return 0;
}