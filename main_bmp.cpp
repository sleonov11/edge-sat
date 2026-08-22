#include "include/core/TileView.h"
#include "include/core/TileBuffers.h"
#include "include/core/TileGenerator.h"
#include "include/dwt/dwt_features.h"
#include "include/dwt/dwt_tile.h"
#include "include/filters/mirror_pad.h"
#include "include/filters/gauss_kernel.h"
#include "include/filters/gauss_tile.h"
#include "include/classifier/DecisionTree.h"
#include "include/yolo/TaskQueue.h"
#include "include/yolo/YoloTask.h"
#include "include/yolo/YoloWorker.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <omp.h>

// ---------- Константы ----------
constexpr int TILE = 128;
constexpr int OVERLAP = 16;
constexpr float SIGMA = 1.5f;

// ---------- Обработка одного тайла ----------
template <int TILE>
void process_gray_tile(const TileView& view, TileBuffers<TILE>& buf,
                       const GaussKernel<2>& kernel, std::array<float, 16>& features) {
    for (int y = 0; y < view.h; ++y) {
        const uint8_t* src_row = view.row(y, 0);
        float* dst_row = buf.gray + y * TILE;
        for (int x = 0; x < view.w; ++x) {
            dst_row[x] = static_cast<float>(src_row[x]);
        }
    }

    mirror_pad<TILE, 2>(buf.gray, buf.padded);
    gauss_h_tile<TILE, 2>(buf.padded, buf.gauss_h, kernel.k);
    gauss_v_tile<TILE, 2>(buf.gauss_h, buf.gauss_v, kernel.k);

    std::copy(buf.gauss_v, buf.gauss_v + TILE * TILE, buf.dwt_buf);
    dwt_2d_haar<TILE>(buf.dwt_buf, buf.dwt_tmp);

    features = extract_features_from_dwt_optimized<TILE>(buf.dwt_buf);
}

// ---------- Главная функция ----------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image.bmp> [model.onnx] [output.bmp]\n";
        return 1;
    }

    std::string image_path = argv[1];
    std::string model_path = (argc >= 3) ? argv[2] : "include/yolo/models/ships_yolo11l.onnx";
    std::string output_path = (argc >= 4) ? argv[3] : "output.bmp";

    // Загружаем изображение (BGR)
    cv::Mat frame = cv::imread(image_path, cv::IMREAD_COLOR);
    if (frame.empty()) {
        std::cerr << "Cannot read image: " << image_path << "\n";
        return 1;
    }
    std::cout << "Image: " << frame.cols << "x" << frame.rows << "\n";

    // Grayscale для классификатора
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // ---------- Очереди ----------
    TaskQueue<YoloTask> task_queue;
    TaskQueue<DetectionResult> result_queue;

    // ---------- YOLO Worker ----------
    YoloWorker worker(model_path, task_queue, result_queue);
    std::thread worker_thread([&] { worker.run(); });

    // ---------- Сбор результатов ----------
    std::vector<DetectionResult> all_results;

    std::thread result_thread([&] {
        while (true) {
            auto res = result_queue.pop();
            if (!res) break;  // stop() вызван
            all_results.push_back(std::move(*res));
        }
    });

    // ---------- Пайплайн ----------
    GaussKernel<2> kernel(SIGMA);
    DecisionTree classifier;

    // Сначала собираем все тайлы в вектор
    TileGenerator gen(gray.data, gray.step,
                      gray.cols, gray.rows,
                      TILE, TILE, OVERLAP);

    std::vector<TileView> views;
    TileView view;
    while (gen.next(view)) {
        views.push_back(view);
    }

    int tile_count = 0;
    int positive_count = 0;

    auto t0 = std::chrono::steady_clock::now();

    // Параллельная обработка тайлов
    #pragma omp parallel for schedule(dynamic) reduction(+:tile_count, positive_count)
    for (size_t i = 0; i < views.size(); ++i) {
        const TileView& v = views[i];

        // Каждый поток создаёт свой буфер на стеке
        TileBuffers<TILE> buf;
        std::array<float, 16> features;

        process_gray_tile<TILE>(v, buf, kernel, features);

        // Увеличиваем счётчик тайлов (reduction уже обеспечивает безопасность)
        ++tile_count;

        if (classifier.classify_tile(features)) {
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