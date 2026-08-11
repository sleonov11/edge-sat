#include "TaskQueue.h"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    TaskQueue<std::string> queue;

    std::thread worker([&]{
        while (auto task = queue.pop()) {
            std::cout << "Worker got" << *task << '\n';
        }
        std::cout << "worker exiting\n";
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "pushing task";
    queue.push("task1");
    queue.push("task2");
    queue.push("task3");

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "stopping... \n";
    queue.stop();
    worker.join();

    std::cout << "rabotaet";
}