#pragma once

#include "result.h"
#include "shared_memory_block.h"
#include "task_group.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <stop_token>
#include <string>

namespace core {

namespace detail {
class DeferredTaskCompletionState;
}

class ThreadPool;
class ThreadPoolContext;

struct ThreadPoolTaskMetadata {
    std::string concurrency_key;
    std::string fairness_key;
    std::string tenant_key;
    std::string execution_id;
};

struct ThreadPoolSchedulerOptions {
    std::size_t worker_count = 0;
    std::size_t queue_capacity = 0;
    std::string pool_name;
};

struct ThreadPoolConcurrencySnapshot {
    std::size_t active_keys = 0;
    std::size_t ready_keys = 0;
    std::size_t queued_tasks = 0;
    std::size_t running_tasks = 0;
    std::size_t rejected_tasks = 0;
    std::size_t rejected_global = 0;
    std::size_t rejected_per_key = 0;
    std::size_t rejected_per_fairness_key = 0;
    std::size_t rejected_per_tenant = 0;
    std::size_t max_lane_depth = 0;
};

class ThreadPoolWorkItem final {
public:
    using TaskFunction = std::function<Status(ThreadPoolContext&)>;

    ThreadPoolWorkItem(const ThreadPoolWorkItem&) = delete;
    ThreadPoolWorkItem& operator=(const ThreadPoolWorkItem&) = delete;
    ThreadPoolWorkItem(ThreadPoolWorkItem&&) = delete;
    ThreadPoolWorkItem& operator=(ThreadPoolWorkItem&&) = delete;
    ~ThreadPoolWorkItem() = default;

    const ThreadPoolTaskMetadata& metadata() const noexcept {
        return metadata_;
    }

    const std::string& name() const noexcept {
        return name_;
    }

private:
    friend class ThreadPool;
    friend class detail::DeferredTaskCompletionState;

    ThreadPoolWorkItem(TaskFunction task,
                       SharedMemoryBlock payload,
                       std::string name,
                       std::string trace_id,
                       ThreadPoolTaskMetadata metadata,
                       std::shared_ptr<TaskGroupToken> task_group_token)
        : task_(std::move(task)),
          payload_(std::move(payload)),
          name_(std::move(name)),
          trace_id_(std::move(trace_id)),
          metadata_(std::move(metadata)),
          task_group_token_(std::move(task_group_token)) {}

    TaskFunction task_;
    SharedMemoryBlock payload_;
    std::string name_;
    std::string trace_id_;
    ThreadPoolTaskMetadata metadata_;
    std::shared_ptr<TaskGroupToken> task_group_token_;
};

class IThreadPoolTaskScheduler {
public:
    virtual ~IThreadPoolTaskScheduler() = default;

    virtual Status Start(const ThreadPoolSchedulerOptions& options) = 0;
    virtual Status TryEnqueue(const std::shared_ptr<ThreadPoolWorkItem>& item) = 0;
    virtual Result<std::shared_ptr<ThreadPoolWorkItem>> WaitDequeue(
        std::size_t worker_index,
        std::stop_token stop_token) = 0;
    virtual void Complete(const ThreadPoolWorkItem& item,
                          std::size_t worker_index,
                          const Status& status) noexcept = 0;
    virtual void Close(bool discard) noexcept = 0;
    virtual bool closed() const noexcept = 0;
    virtual std::size_t QueuedTaskCount() const noexcept = 0;
    virtual ThreadPoolConcurrencySnapshot Snapshot() const = 0;
    /// 当前 work item 未 Complete 前，是否保证相同 concurrency_key 不会再次出队。
    virtual bool SerializesConcurrencyKeyUntilCompletion() const noexcept { return false; }
};

class DefaultFifoThreadPoolTaskScheduler final : public IThreadPoolTaskScheduler {
public:
    DefaultFifoThreadPoolTaskScheduler();
    ~DefaultFifoThreadPoolTaskScheduler() override;

    DefaultFifoThreadPoolTaskScheduler(const DefaultFifoThreadPoolTaskScheduler&) = delete;
    DefaultFifoThreadPoolTaskScheduler& operator=(const DefaultFifoThreadPoolTaskScheduler&) = delete;

    Status Start(const ThreadPoolSchedulerOptions& options) override;
    Status TryEnqueue(const std::shared_ptr<ThreadPoolWorkItem>& item) override;
    Result<std::shared_ptr<ThreadPoolWorkItem>> WaitDequeue(
        std::size_t worker_index,
        std::stop_token stop_token) override;
    void Complete(const ThreadPoolWorkItem& item,
                  std::size_t worker_index,
                  const Status& status) noexcept override;
    void Close(bool discard) noexcept override;
    bool closed() const noexcept override;
    std::size_t QueuedTaskCount() const noexcept override;
    ThreadPoolConcurrencySnapshot Snapshot() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace core
