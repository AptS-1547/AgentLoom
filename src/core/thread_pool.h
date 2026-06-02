#pragma once

#include "blocking_queue.h"
#include "result.h"
#include "shared_memory_block.h"
#include "trace_context.h"

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace core {

struct ThreadPoolOptions {
    std::size_t worker_count = 0;
    std::size_t queue_capacity = 0;
    std::string name = "core-thread-pool";
};

struct ThreadPoolStats {
    std::size_t worker_count = 0;
    std::size_t queued_tasks = 0;
    std::size_t active_workers = 0;
    std::size_t submitted_tasks = 0;
    std::size_t completed_tasks = 0;
    std::size_t failed_tasks = 0;
    std::size_t rejected_tasks = 0;
};

enum class WorkerState {
    Starting,
    Idle,
    Running,
    Stopping,
    Stopped
};

struct WorkerStatus {
    std::size_t worker_index = 0;
    WorkerState state = WorkerState::Stopped;
    std::string current_task;
    std::size_t completed_tasks = 0;
    std::size_t failed_tasks = 0;
    Status last_status = Status::Ok();
};

class ThreadPoolContext {
public:
    std::size_t worker_index() const noexcept {
        return worker_index_;
    }

    const std::string& pool_name() const noexcept {
        return pool_name_;
    }

    std::stop_token stop_token() const noexcept {
        return stop_token_;
    }

    bool stop_requested() const noexcept {
        return stop_token_.stop_requested();
    }

    SharedMemoryBlock& payload() noexcept {
        return payload_;
    }

    const SharedMemoryBlock& payload() const noexcept {
        return payload_;
    }

    SharedMemoryBlock take_payload() noexcept {
        return std::move(payload_);
    }

private:
    friend class ThreadPool;

    ThreadPoolContext(std::size_t worker_index, std::string pool_name, std::stop_token stop_token)
        : worker_index_(worker_index),
          pool_name_(std::move(pool_name)),
          stop_token_(std::move(stop_token)) {}

    void set_payload(SharedMemoryBlock payload) noexcept {
        payload_ = std::move(payload);
    }

    void clear_payload() noexcept {
        payload_.reset();
    }

    std::size_t worker_index_ = 0;
    std::string pool_name_;
    std::stop_token stop_token_;
    SharedMemoryBlock payload_;
};

class ThreadPool {
public:
    using TaskFunction = std::function<Status(ThreadPoolContext&)>;

    explicit ThreadPool(ThreadPoolOptions options = {});
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    Status Start();
    void Shutdown(bool drain = true);

    Status SubmitTask(TaskFunction task, SharedMemoryBlock payload = {}, std::string name = {});

    template <typename Fn>
    Status Submit(Fn&& fn, SharedMemoryBlock payload = {}, std::string name = {}) {
        TaskFunction task = [func = std::forward<Fn>(fn)](ThreadPoolContext& context) mutable -> Status {
            if constexpr (std::is_invocable_v<Fn&, ThreadPoolContext&>) {
                using ReturnType = std::invoke_result_t<Fn&, ThreadPoolContext&>;
                if constexpr (std::is_same_v<ReturnType, Status>) {
                    return std::invoke(func, context);
                } else if constexpr (std::is_void_v<ReturnType>) {
                    std::invoke(func, context);
                    return Status::Ok();
                } else {
                    static_assert(std::is_void_v<ReturnType>, "thread pool task must return void or core::Status");
                }
            } else if constexpr (std::is_invocable_v<Fn&>) {
                using ReturnType = std::invoke_result_t<Fn&>;
                if constexpr (std::is_same_v<ReturnType, Status>) {
                    return std::invoke(func);
                } else if constexpr (std::is_void_v<ReturnType>) {
                    std::invoke(func);
                    return Status::Ok();
                } else {
                    static_assert(std::is_void_v<ReturnType>, "thread pool task must return void or core::Status");
                }
            } else {
                static_assert(std::is_invocable_v<Fn&, ThreadPoolContext&>, "thread pool task is not invocable");
            }
        };
        return SubmitTask(std::move(task), std::move(payload), std::move(name));
    }

    ThreadPoolStats Stats() const;
    std::vector<WorkerStatus> WorkerStatuses() const;
    bool running() const noexcept;

private:
    struct QueuedTask {
        TaskFunction task;
        SharedMemoryBlock payload;
        std::string name;
        std::string trace_id;
    };

    struct WorkerRuntime {
        mutable std::mutex mutex;
        WorkerStatus status;
    };

    void WorkerLoop(std::stop_token stop_token, std::size_t worker_index);
    void InitializeWorkerStatuses(std::size_t worker_count);
    void SetWorkerState(std::size_t worker_index, WorkerState state, std::string current_task = {});
    void FinishWorkerTask(std::size_t worker_index, Status status);
    static std::size_t ResolveWorkerCount(std::size_t requested) noexcept;

    ThreadPoolOptions options_;
    BlockingQueue<QueuedTask> queue_;
    mutable std::mutex lifecycle_mutex_;
    std::vector<std::jthread> workers_;
    std::vector<std::unique_ptr<WorkerRuntime>> worker_statuses_;
    std::atomic<bool> running_{false};
    std::atomic<std::size_t> active_workers_{0};
    std::atomic<std::size_t> submitted_tasks_{0};
    std::atomic<std::size_t> completed_tasks_{0};
    std::atomic<std::size_t> failed_tasks_{0};
    std::atomic<std::size_t> rejected_tasks_{0};
};

} // namespace core
