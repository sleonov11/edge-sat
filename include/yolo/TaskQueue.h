#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template <typename T>
class TaskQueue {
private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_ = false;

public:
    explicit TaskQueue() = default;
    ~TaskQueue() = default;

    void push(T new_value) {
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push(std::move(new_value));
        cv_.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] {return stop_ || !queue_.empty();});
        
        if (stop_ && queue_.empty()) {
            return std::nullopt;
        }
        
        T value = std::move(queue_.front());
        queue_.pop();
        return std::move(value); 
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
        cv_.notify_all();
    }

    bool empty() {
        return queue_.empty();
    }

};