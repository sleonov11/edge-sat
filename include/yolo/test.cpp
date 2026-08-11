#include "TaskQueue.h"
#include "YoloTask.h"
#include "YoloWorker.h"
#include <thread>
#include <chrono>

int main() {
    TaskQueue<YoloTask> queue;
    YoloWorker worker("include/yolo/models/yolov8n.onnx", queue);
    
    std::thread t([&]{ worker.run(); });
    
    // Даём воркеру заснуть в pop()
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Создаём фейковую задачу (весь кадр 100x100 RGB)
    std::vector<uint8_t> fake_frame(100 * 100 * 3);
    for (int i = 0; i < fake_frame.size(); ++i) fake_frame[i] = i % 256;
    
    YoloTask task(42, 0, 0, 100, 100, 3, 
                  fake_frame.data(), 100*3, 100, 100);
    queue.push(std::move(task));
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    queue.stop();
    t.join();
    
    return 0;
}