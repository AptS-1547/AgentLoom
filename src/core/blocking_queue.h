#pragma once

#include "result.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace core {

template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(std::size_t capacity = 0)
        : capacity_(capacity) {}

    BlockingQueue(const BlockingQueue&) = delete;
    BlockingQueue& operator=(const BlockingQueue&) = delete;

    Status Push(T value) {
        std::unique_lock lock(mutex_);
        not_full_.wait(lock, [this] {
            return closed_ || capacity_ == 0 || queue_.size() < capacity_;
        });

        if (closed_) {
            return Status::Error(ErrorCode::Cancelled, "queue is closed");
        }

        queue_.push_back(std::move(value));
        not_empty_.notify_one();
        return Status::Ok();
    }

    Status TryPush(T value) {
        {
            std::lock_guard lock(mutex_);
            if (closed_) {
                return Status::Error(ErrorCode::Cancelled, "queue is closed");
            }
            if (capacity_ > 0 && queue_.size() >= capacity_) {
                return Status::Error(ErrorCode::ResourceExhausted, "queue is full");
            }
            queue_.push_back(std::move(value));
        }
        not_empty_.notify_one();
        return Status::Ok();
    }

    Result<T> WaitPop() {
        std::unique_lock lock(mutex_);
        not_empty_.wait(lock, [this] {
            return closed_ || !queue_.empty();
        });

        if (queue_.empty()) {
            return Status::Error(ErrorCode::Cancelled, "queue is closed");
        }

        T value = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return value;
    }

    template <typename Rep, typename Period>
    Result<T> WaitPopFor(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock lock(mutex_);
        const bool ready = not_empty_.wait_for(lock, timeout, [this] {
            return closed_ || !queue_.empty();
        });

        if (!ready) {
            return Status::Error(ErrorCode::Timeout, "queue wait timed out");
        }
        if (queue_.empty()) {
            return Status::Error(ErrorCode::Cancelled, "queue is closed");
        }

        T value = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return value;
    }

    Result<T> TryPop() {
        std::lock_guard lock(mutex_);
        if (queue_.empty()) {
            return Status::Error(ErrorCode::NotFound, "queue is empty");
        }

        T value = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return value;
    }

    void Close(bool discard = false) {
        {
            std::lock_guard lock(mutex_);
            closed_ = true;
            if (discard) {
                queue_.clear();
            }
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    bool closed() const {
        std::lock_guard lock(mutex_);
        return closed_;
    }

    std::size_t size() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

    std::size_t capacity() const noexcept {
        return capacity_;
    }

    bool empty() const {
        std::lock_guard lock(mutex_);
        return queue_.empty();
    }

private:
    std::size_t capacity_ = 0;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> queue_;
    bool closed_ = false;
};

} // namespace core
