#include <cstddef>
#include <new>
#include <cstdint>

class Arena {
private:
    size_t size_;
    uint8_t* start_;
    uint8_t* current_;

public:
    explicit Arena (size_t size) : start_(new (std::nothrow) uint8_t[size]), size_(size), current_(start_) {
        if (start_ == nullptr) {
            current_ = nullptr;
            size_ = 0;
        }
    } 
    Arena(const Arena&) = delete;
    Arena& operator= (const Arena&) = delete;
    Arena& operator (Arena&&) = delete;
    Arena(Arena&&) = delete;
    ~Arena() {delete[] start_;}

    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) {
        if (start_ == nullptr) return nullptr;

        uintptr_t cur = reinterpret_cast<uintptr_t>(current_); 

        uintptr_t aligned = (cur + alignment - 1) & ~(alignment - 1);

        uintptr_t end = reinterpret_cast<uintptr_t> (start_) + size_;
        if (aligned > end) return nullptr;
        
        size_t available = end - aligned;
        if (size > available) return nullptr;

        void* ptr = reinterpret_cast<void*>(aligned);
        current_ = reinterpret_cast<uint8_t*>(aligned + size);  
        return ptr;
    }
    
    void reset() {
        current_ = start_;
    }

    size_t capacity() const {return size_;}
    size_t used() const {return static_cast<size_t>(current_ - start_);}
};