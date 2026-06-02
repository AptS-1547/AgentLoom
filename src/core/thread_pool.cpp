#include "thread_pool.h"

#include "exception.h"

#include <algorithm>
#include <exception>
#include <memory>

namespace core {

ThreadPool::ThreadPool(ThreadPoolOptions options)
    : options_(std::move(options)),
      queue_(options_.queue_capacity) {}

ThreadPool::~ThreadPool() {
    Shutdown(false);
}

Status ThreadPool::Start() {
    std::lock_guard lock(lifecycle_mutex_);
    if (running_.load(std::memory_order_acquire)) {
        return Status::Ok();
    }
    if (queue_.closed()) {
        return Status::Error(ErrorCode::Unavailable, "thread pool has been shut down");
    }

    const auto worker_count = ResolveWorkerCount(options_.worker_count);
    try {
        workers_.reserve(worker_count);
        InitializeWorkerStatuses(worker_count);
        for (std::size_t i = 0; i < worker_count; ++i) {
            workers_.emplace_back([this, i](std::stop_token stop_token) {
                WorkerLoop(stop_token, i);
            });
        }
    } catch (const std::exception& e) {
        const auto status = Status::Error(ErrorCode::InternalError, e.what());
        queue_.Close(true);
        for (auto& worker : workers_) {
            worker.request_stop();
        }
        workers_.clear();
        for (auto& runtime : worker_statuses_) {
            std::lock_guard status_lock(runtime->mutex);
            runtime->status.state = WorkerState::Stopped;
            runtime->status.current_task.clear();
            runtime->status.last_status = status;
        }
        return status;
    }

    running_.store(true, std::memory_order_release);
    return Status::Ok();
}

void ThreadPool::Shutdown(bool drain) {
    std::vector<std::jthread> workers;
    {
        std::lock_guard lock(lifecycle_mutex_);
        if (!running_.exchange(false, std::memory_order_acq_rel) && workers_.empty()) {
            return;
        }

        if (!drain) {
            for (auto& worker : workers_) {
                worker.request_stop();
            }
        }

        queue_.Close(!drain);
        workers = std::move(workers_);
    }

    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

Status ThreadPool::SubmitTask(TaskFunction task, SharedMemoryBlock payload, std::string name) {
    if (!task) {
        rejected_tasks_.fetch_add(1, std::memory_order_relaxed);
        return Status::Error(ErrorCode::InvalidArgument, "task is empty");
    }
    if (!running_.load(std::memory_order_acquire)) {
        rejected_tasks_.fetch_add(1, std::memory_order_relaxed);
        return Status::Error(ErrorCode::Unavailable, "thread pool is not running");
    }

    std::string trace_id;
    if (current_trace && !current_trace->trace_id.empty()) {
        trace_id = current_trace->trace_id;
    }

    auto status = queue_.TryPush(QueuedTask{
        std::move(task), std::move(payload), std::move(name), std::move(trace_id)});
    if (!status.ok()) {
        rejected_tasks_.fetch_add(1, std::memory_order_relaxed);
        return status;
    }

    submitted_tasks_.fetch_add(1, std::memory_order_relaxed);
    return Status::Ok();
}

ThreadPoolStats ThreadPool::Stats() const {
    std::lock_guard lock(lifecycle_mutex_);
    ThreadPoolStats stats;
    stats.worker_count = worker_statuses_.size();
    stats.queued_tasks = queue_.size();
    stats.active_workers = active_workers_.load(std::memory_order_relaxed);
    stats.submitted_tasks = submitted_tasks_.load(std::memory_order_relaxed);
    stats.completed_tasks = completed_tasks_.load(std::memory_order_relaxed);
    stats.failed_tasks = failed_tasks_.load(std::memory_order_relaxed);
    stats.rejected_tasks = rejected_tasks_.load(std::memory_order_relaxed);
    return stats;
}

std::vector<WorkerStatus> ThreadPool::WorkerStatuses() const {
    std::lock_guard lock(lifecycle_mutex_);
    std::vector<WorkerStatus> statuses;
    statuses.reserve(worker_statuses_.size());
    for (const auto& runtime : worker_statuses_) {
        std::lock_guard status_lock(runtime->mutex);
        statuses.push_back(runtime->status);
    }
    return statuses;
}

bool ThreadPool::running() const noexcept {
    return running_.load(std::memory_order_acquire);
}

void ThreadPool::WorkerLoop(std::stop_token stop_token, std::size_t worker_index) {
    ThreadPoolContext context(worker_index, options_.name, stop_token);
    SetWorkerState(worker_index, WorkerState::Idle);

    while (!stop_token.stop_requested()) {
        auto task_result = queue_.WaitPop();
        if (!task_result) {
            SetWorkerState(worker_index, WorkerState::Stopping);
            break;
        }

        auto queued = std::move(task_result).value();
        active_workers_.fetch_add(1, std::memory_order_relaxed);
        SetWorkerState(worker_index, WorkerState::Running, std::move(queued.name));
        context.set_payload(std::move(queued.payload));

        TraceContext trace_ctx;
        trace_ctx.trace_id = std::move(queued.trace_id);
        TraceScope trace_scope(trace_ctx);

        Status status = Status::Ok();
        try {
            status = queued.task(context);
        } catch (const AppException& e) {
            status = e.status();
        } catch (const std::exception& e) {
            status = Status::Error(ErrorCode::InternalError, e.what());
        } catch (...) {
            status = Status::Error(ErrorCode::Unknown, "unknown task error");
        }

        context.clear_payload();
        active_workers_.fetch_sub(1, std::memory_order_relaxed);
        const auto task_ok = status.ok();
        FinishWorkerTask(worker_index, std::move(status));

        if (task_ok) {
            completed_tasks_.fetch_add(1, std::memory_order_relaxed);
        } else {
            failed_tasks_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    SetWorkerState(worker_index, WorkerState::Stopped);
}

void ThreadPool::InitializeWorkerStatuses(std::size_t worker_count) {
    worker_statuses_.clear();
    worker_statuses_.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
        auto runtime = std::make_unique<WorkerRuntime>();
        runtime->status.worker_index = i;
        runtime->status.state = WorkerState::Starting;
        worker_statuses_.push_back(std::move(runtime));
    }
}

void ThreadPool::SetWorkerState(std::size_t worker_index, WorkerState state, std::string current_task) {
    if (worker_index >= worker_statuses_.size()) {
        return;
    }

    auto& runtime = *worker_statuses_[worker_index];
    std::lock_guard lock(runtime.mutex);
    runtime.status.state = state;
    runtime.status.current_task = std::move(current_task);
}

void ThreadPool::FinishWorkerTask(std::size_t worker_index, Status status) {
    if (worker_index >= worker_statuses_.size()) {
        return;
    }

    auto& runtime = *worker_statuses_[worker_index];
    std::lock_guard lock(runtime.mutex);
    if (status.ok()) {
        ++runtime.status.completed_tasks;
    } else {
        ++runtime.status.failed_tasks;
    }
    runtime.status.last_status = std::move(status);
    runtime.status.current_task.clear();
    runtime.status.state = WorkerState::Idle;
}

std::size_t ThreadPool::ResolveWorkerCount(std::size_t requested) noexcept {
    if (requested > 0) {
        return requested;
    }
    const auto hardware = std::thread::hardware_concurrency();
    return std::max<std::size_t>(1, hardware == 0 ? 1 : hardware);
}

} // namespace core
