#pragma once

#include "result.h"
#include "shared_memory_block.h"
#include "task_group.h"
#include "thread_pool_scheduler.h"
#include "trace_context.h"

#include <atomic>
#include <condition_variable>
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

namespace detail {
class DeferredTaskCompletionState;
struct ThreadPoolCompletionCounters {
    std::atomic<std::size_t> completed{0};
    std::atomic<std::size_t> failed{0};
    std::atomic<std::size_t> deferred_outstanding{0};
    mutable std::mutex mutex;
    std::condition_variable drained_cv;
};
}

/// 将线程池任务的 scheduler 完成时刻延后到异步 continuation。
/// token 可移动且恰好完成一次；未显式完成便析构时按 Cancelled 释放 lane。
class DeferredTaskCompletion final {
public:
    DeferredTaskCompletion() = default;
    ~DeferredTaskCompletion();

    DeferredTaskCompletion(const DeferredTaskCompletion&) = delete;
    DeferredTaskCompletion& operator=(const DeferredTaskCompletion&) = delete;
    DeferredTaskCompletion(DeferredTaskCompletion&& other) noexcept;
    DeferredTaskCompletion& operator=(DeferredTaskCompletion&& other) noexcept;

    void Complete(Status status = Status::Ok()) noexcept;
    bool valid() const noexcept;

private:
    explicit DeferredTaskCompletion(
        std::shared_ptr<detail::DeferredTaskCompletionState> state) noexcept;

    std::shared_ptr<detail::DeferredTaskCompletionState> state_;
    bool active_ = false;

    friend class ThreadPoolContext;
};

struct ThreadPoolOptions {
    std::size_t worker_count = 0;
    std::size_t queue_capacity = 0;
    std::string name = "core-thread-pool";
    std::shared_ptr<IThreadPoolTaskScheduler> scheduler;
};

struct ThreadPoolStats {
    std::size_t worker_count = 0;
    std::size_t queued_tasks = 0;
    std::size_t active_workers = 0;
    std::size_t submitted_tasks = 0;
    std::size_t completed_tasks = 0;
    std::size_t failed_tasks = 0;
    std::size_t rejected_tasks = 0;
    ThreadPoolConcurrencySnapshot scheduler;
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

    Result<TaskGroupToken> AcquireChildTask() const {
        if (!task_group_token_) {
            return Status::Error(ErrorCode::FailedPrecondition, "current task is not tracked by a task group");
        }
        return task_group_token_->AcquireChild();
    }

    /// 延后当前 work item 的 scheduler 完成；worker 会立即返回池中继续处理其他任务。
    /// 仅允许调用一次，返回 token 必须由异步 continuation 持有并最终完成。
    Result<DeferredTaskCompletion> DeferCompletion();

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

    void set_task_group_token(TaskGroupToken* token) noexcept {
        task_group_token_ = token;
    }

    void set_deferred_completion_state(
        std::shared_ptr<detail::DeferredTaskCompletionState> state) noexcept {
        deferred_completion_state_ = std::move(state);
    }

    std::size_t worker_index_ = 0;
    std::string pool_name_;
    std::stop_token stop_token_;
    SharedMemoryBlock payload_;
    TaskGroupToken* task_group_token_ = nullptr;
    std::shared_ptr<detail::DeferredTaskCompletionState> deferred_completion_state_;
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

    Status SubmitTask(TaskFunction task,
                      SharedMemoryBlock payload = {},
                      std::string name = {},
                      ThreadPoolTaskMetadata metadata = {});
    Status SubmitTask(TaskGroup& group,
                      TaskFunction task,
                      SharedMemoryBlock payload = {},
                      std::string name = {},
                      ThreadPoolTaskMetadata metadata = {});
    Status SubmitTask(TaskGroupToken token,
                      TaskFunction task,
                      SharedMemoryBlock payload = {},
                      std::string name = {},
                      ThreadPoolTaskMetadata metadata = {});

    template <typename Fn>
    Status Submit(Fn&& fn,
                  SharedMemoryBlock payload = {},
                  std::string name = {},
                  ThreadPoolTaskMetadata metadata = {}) {
        return SubmitTask(MakeTask(std::forward<Fn>(fn)),
                          std::move(payload),
                          std::move(name),
                          std::move(metadata));
    }

    template <typename Fn>
    Status Submit(TaskGroup& group,
                  Fn&& fn,
                  SharedMemoryBlock payload = {},
                  std::string name = {},
                  ThreadPoolTaskMetadata metadata = {}) {
        return SubmitTask(group,
                          MakeTask(std::forward<Fn>(fn)),
                          std::move(payload),
                          std::move(name),
                          std::move(metadata));
    }

    template <typename Fn>
    Status Submit(TaskGroupToken token,
                  Fn&& fn,
                  SharedMemoryBlock payload = {},
                  std::string name = {},
                  ThreadPoolTaskMetadata metadata = {}) {
        return SubmitTask(
            std::move(token),
            MakeTask(std::forward<Fn>(fn)),
            std::move(payload),
            std::move(name),
            std::move(metadata));
    }

    ThreadPoolStats Stats() const;
    std::vector<WorkerStatus> WorkerStatuses() const;
    bool running() const noexcept;
    bool serializes_concurrency_key_until_completion() const noexcept;

private:
    template <typename Fn>
    static TaskFunction MakeTask(Fn&& fn) {
        return [func = std::forward<Fn>(fn)](ThreadPoolContext& context) mutable -> Status {
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
    }
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
    std::shared_ptr<IThreadPoolTaskScheduler> scheduler_;
    mutable std::mutex lifecycle_mutex_;
    std::vector<std::jthread> workers_;
    std::vector<std::unique_ptr<WorkerRuntime>> worker_statuses_;
    std::atomic<bool> running_{false};
    std::atomic<std::size_t> active_workers_{0};
    std::atomic<std::size_t> submitted_tasks_{0};
    std::shared_ptr<detail::ThreadPoolCompletionCounters> completion_counters_ =
        std::make_shared<detail::ThreadPoolCompletionCounters>();
    std::atomic<std::size_t> rejected_tasks_{0};
};

} // namespace core
