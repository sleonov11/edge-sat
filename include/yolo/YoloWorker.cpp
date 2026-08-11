#include "YoloWorker.h"
#include <iostream>

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
        auto task = queue_.pop();   // спит здесь, если очередь пуста
        
        if (!task) {                // queue.stop() был вызван
            std::cout << "Worker: stopping\n";
            break;
        }
        
        // ЗАГЛУШКА
        std::cout << "Worker got frame " << task->frame_id 
                  << ", pixels: " << task->pixels.size() << "\n";
    }
}