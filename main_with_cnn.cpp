// ВЕРСИЯ CNN вместо gauss -> dwt -> decision tree;

#include "include/core/TileBuffers.h"
#include "include/core/TileGenerator.h"
#include "include/classifier/TinyCNN.h"
#include "include/yolo/TaskQueue.h"
#include "include/yolo/YoloTask.h"
#include "include/yolo/YoloWorker.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <omp.h>
#include <chrono>

constexpr int TILE = 256;
constexpr int OVERLAP = 16;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image.bmp> [model.onnx] [output.bmp]\n";
        return 1;
    }

    std::string image_path = argv[1];
    std::string model_path = (argc >= 3) ? argv[2] : "include/yolo/models/ships_yolo11l.onnx";
    std::string output_path = (argc >= 4) ? argv[3] : "output.bmp";

    cv::Mat frame = cv::imread(image_path, cv::IMREAD_COLOR);
    if (frame.empty()) {
        std::cerr << "Cannot read image: " << image_path << "\n";
        return 1;
    }
    std::cout << "Image: " << frame.cols << "x" << frame.rows << "\n";

    TaskQueue<YoloTask> task_queue;
    TaskQueue<DetectionResult> result_queue;

    YoloWorker worker(model_path, task_queue, result_queue);
    std::thread worker_thread([&] {worker.run(); });

    std::vector<DetectionResult> all_results;

    std::thread result_thread([&] {
        while (true) {
            auto res = result_queue.pop();
            if (!res) break;  // stop() вызван
            all_results.push_back(std::move(*res));
        }
    });

    TinyCNN classifier("include/classifier/water_classifier.onnx");

    TileGenerator gen(frame.data, frame.step,
                      frame.cols, frame.rows,
                      TILE, TILE, OVERLAP);

    std::vector<TileView> views;
    TileView view;
    while (gen.next(view)) {
        views.push_back(view);
    }

    int tile_count = 0;
    int positive_count = 0;

    auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic) reduction(+:tile_count,positive_count)
    for (size_t i = 0; i < views.size(); ++i) {
        const TileView& v = views[i];
        cv::Mat tile_rgb(frame, cv::Rect(v.x, v.y, v.w, v.h)); 
        ++tile_count;
        if (classifier.isWater(tile_rgb)) {
            ++positive_count;
            const int ctx_size = 640;
            int cx = v.x + v.w / 2;
            int cy = v.y + v.h / 2;

            int ctx_x = cx - ctx_size / 2;
            int ctx_y = cy - ctx_size / 2;

            // Проверяем, что изображение достаточно велико
            if (frame.cols >= ctx_size && frame.rows >= ctx_size) {
                ctx_x = std::max(0, std::min(ctx_x, frame.cols - ctx_size));
                ctx_y = std::max(0, std::min(ctx_y, frame.rows - ctx_size));

                YoloTask task(
                    0,
                    ctx_x, ctx_y,
                    ctx_size, ctx_size,
                    3,
                    frame.data,
                    static_cast<int>(frame.step),
                    frame.cols,
                    frame.rows
                );
                // Отправка в очередь потокобезопасна
                task_queue.push(std::move(task));
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Tiles: " << tile_count
              << ", positive: " << positive_count
              << ", time: " << ms << " ms\n";

    // ---------- Shutdown (порядок критичен) ----------
    task_queue.stop();
    worker_thread.join();      // ждём, пока worker закончит все push

    result_queue.stop();
    result_thread.join();

    // ---------- Объединяем боксы ----------
    std::vector<YoloBox> all_boxes;
    for (const auto& res : all_results) {
        for (const auto& box : res.boxes) {
            all_boxes.push_back(box);
        }
    }
    std::cout << "Total detections: " << all_boxes.size() << "\n";

    std::vector<cv::Rect> rects;
    std::vector<float> confs;
    std::vector<int> class_ids;
    std::vector<YoloBox> filtered_boxes;  

    for (const auto& box : all_boxes) {
        if (box.w < 20 || box.h < 20) continue;
        if (box.w > frame.cols * 0.9 && box.h > frame.rows * 0.9) continue;

        rects.push_back(cv::Rect(box.x, box.y, box.w, box.h));
        confs.push_back(box.conf);
        class_ids.push_back(box.class_id);
        filtered_boxes.push_back(box);   
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(rects, confs, 0.4f, 0.4f, indices);

    // ---------- Рисуем только прошедшие NMS ----------
    cv::Mat output = frame.clone();
    for (int idx : indices) {
        const auto& box = filtered_boxes[idx];
        cv::rectangle(output,
                    cv::Point(box.x, box.y),
                    cv::Point(box.x + box.w, box.y + box.h),
                    cv::Scalar(0, 255, 0), 2);
        std::string label = "cls:" + std::to_string(box.class_id) +
                            " " + std::to_string(int(box.conf * 100)) + "%";
        cv::putText(output, label,
                    cv::Point(box.x, std::max(box.y - 5, 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1);
    }

    cv::imwrite(output_path, output);
    std::cout << "Saved: " << output_path << "\n";

    return 0;
}