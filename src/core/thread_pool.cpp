#include "thread_pool.h"

#include "exception.h"

#include <algorithm>
#include <exception>
#include <memory>

namespace core {

namespace detail {

class DeferredTaskCompletionState final {
public:
    DeferredTaskCompletionState(
        std::shared_ptr<IThreadPoolTaskScheduler> scheduler,
        std::shared_ptr<ThreadPoolWorkItem> item,
        std::size_t worker_index,
        std::shared_ptr<ThreadPoolCompletionCounters> counters)
        : scheduler_(std::move(scheduler)),
          item_(std::move(item)),
          worker_index_(worker_index),
          counters_(std::move(counters)) {}

    bool TryDefer() noexcept {
        bool expected = false;
        if (!deferred_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            return false;
        }
        counters_->deferred_outstanding.fetch_add(1, std::memory_order_acq_rel);
        return true;
    }

    bool deferred() const noexcept {
        return deferred_.load(std::memory_order_acquire);
    }

    void Complete(Status status) noexcept {
        if (completed_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        scheduler_->Complete(*item_, worker_index_, status);
        if (item_->task_group_token_) {
            item_->task_group_token_->Complete(status);
        }
        if (status.ok()) {
            counters_->completed.fetch_add(1, std::memory_order_relaxed);
        } else {
            counters_->failed.fetch_add(1, std::memory_order_relaxed);
        }
        if (deferred_.load(std::memory_order_acquire) &&
            counters_->deferred_outstanding.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard lock(counters_->mutex);
            counters_->drained_cv.notify_all();
        }
        item_.reset();
    }

private:
    std::shared_ptr<IThreadPoolTaskScheduler> scheduler_;
    std::shared_ptr<ThreadPoolWorkItem> item_;
    std::size_t worker_index_ = 0;
    std::shared_ptr<ThreadPoolCompletionCounters> counters_;
    std::atomic<bool> deferred_{false};
    std::atomic<bool> completed_{false};
};

} // namespace detail

DeferredTaskCompletion::DeferredTaskCompletion(
    std::shared_ptr<detail::DeferredTaskCompletionState> state) noexcept
    : state_(std::move(state)), active_(state_ != nullptr) {}

DeferredTaskCompletion::~DeferredTaskCompletion() {
    if (active_) {
        Complete(Status::Error(
            ErrorCode::Cancelled,
            "deferred thread pool task released without completion"));
    }
}

DeferredTaskCompletion::DeferredTaskCompletion(DeferredTaskCompletion&& other) noexcept
    : state_(std::move(other.state_)),
      active_(std::exchange(other.active_, false)) {}

DeferredTaskCompletion& DeferredTaskCompletion::operator=(
    DeferredTaskCompletion&& other) noexcept {
    if (this != &other) {
        if (active_) {
            Complete(Status::Error(
                ErrorCode::Cancelled,
                "deferred thread pool task replaced without completion"));
        }
        state_ = std::move(other.state_);
        active_ = std::exchange(other.active_, false);
    }
    return *this;
}

void DeferredTaskCompletion::Complete(Status status) noexcept {
    if (!active_ || !state_) {
        return;
    }
    active_ = false;
    state_->Complete(std::move(status));
    state_.reset();
}

bool DeferredTaskCompletion::valid() const noexcept {
    return active_ && state_ != nullptr;
}

Result<DeferredTaskCompletion> ThreadPoolContext::DeferCompletion() {
    if (!deferred_completion_state_) {
        return Status::Error(
            ErrorCode::FailedPrecondition,
            "current task does not support deferred completion");
    }
    if (!deferred_completion_state_->TryDefer()) {
        return Status::Error(
            ErrorCode::AlreadyExists,
            "current task completion is already deferred");
    }
    return DeferredTaskCompletion(deferred_completion_state_);
}

ThreadPool::ThreadPool(ThreadPoolOptions options)
    : options_(std::move(options)),
      scheduler_(options_.scheduler
                     ? options_.scheduler
                     : std::make_shared<DefaultFifoThreadPoolTaskScheduler>()) {}

ThreadPool::~ThreadPool() {
    Shutdown(false);
}

Status ThreadPool::Start() {
    std::lock_guard lock(lifecycle_mutex_);
    if (running_.load(std::memory_order_acquire)) {
        return Status::Ok();
    }
    if (!scheduler_ || scheduler_->closed()) {
        return Status::Error(ErrorCode::Unavailable, "thread pool has been shut down");
    }

    const auto worker_count = ResolveWorkerCount(options_.worker_count);
    auto scheduler_status = scheduler_->Start(ThreadPoolSchedulerOptions{
        .worker_count = worker_count,
        .queue_capacity = options_.queue_capacity,
        .pool_name = options_.name,
    });
    if (!scheduler_status.ok()) {
        return scheduler_status;
    }
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
        scheduler_->Close(true);
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

        scheduler_->Close(!drain);
        workers = std::move(workers_);
    }

    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    if (drain) {
        std::unique_lock lock(completion_counters_->mutex);
        completion_counters_->drained_cv.wait(lock, [this] {
            return completion_counters_->deferred_outstanding.load(
                       std::memory_order_acquire) == 0;
        });
    }
}

Status ThreadPool::SubmitTask(TaskFunction task,
                              SharedMemoryBlock payload,
                              std::string name,
                              ThreadPoolTaskMetadata metadata) {
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

    auto item = std::shared_ptr<ThreadPoolWorkItem>(new ThreadPoolWorkItem(
        std::move(task),
        std::move(payload),
        std::move(name),
        std::move(trace_id),
        std::move(metadata),
        {}));
    auto status = scheduler_->TryEnqueue(item);
    if (!status.ok()) {
        rejected_tasks_.fetch_add(1, std::memory_order_relaxed);
        return status;
    }

    submitted_tasks_.fetch_add(1, std::memory_order_relaxed);
    return Status::Ok();
}

Status ThreadPool::SubmitTask(TaskGroup& group,
                              TaskFunction task,
                              SharedMemoryBlock payload,
                              std::string name,
                              ThreadPoolTaskMetadata metadata) {
    auto token = group.AcquireRoot();
    if (!token.ok()) {
        rejected_tasks_.fetch_add(1, std::memory_order_relaxed);
        return token.status();
    }
    return SubmitTask(std::move(token).value(),
                      std::move(task),
                      std::move(payload),
                      std::move(name),
                      std::move(metadata));
}

Status ThreadPool::SubmitTask(TaskGroupToken token,
                              TaskFunction task,
                              SharedMemoryBlock payload,
                              std::string name,
                              ThreadPoolTaskMetadata metadata) {
    if (!token.valid()) {
        rejected_tasks_.fetch_add(1, std::memory_order_relaxed);
        return Status::Error(ErrorCode::InvalidArgument, "task group token is inactive");
    }
    if (!task) {
        rejected_tasks_.fetch_add(1, std::memory_order_relaxed);
        const auto status = Status::Error(ErrorCode::InvalidArgument, "task is empty");
        token.Complete(status);
        return status;
    }
    if (!running_.load(std::memory_order_acquire)) {
        rejected_tasks_.fetch_add(1, std::memory_order_relaxed);
        const auto status = Status::Error(ErrorCode::Unavailable, "thread pool is not running");
        token.Complete(status);
        return status;
    }

    std::string trace_id;
    if (current_trace && !current_trace->trace_id.empty()) {
        trace_id = current_trace->trace_id;
    }

    auto tracking_token = std::make_shared<TaskGroupToken>(std::move(token));
    auto item = std::shared_ptr<ThreadPoolWorkItem>(new ThreadPoolWorkItem(
        std::move(task),
        std::move(payload),
        std::move(name),
        std::move(trace_id),
        std::move(metadata),
        tracking_token));
    auto status = scheduler_->TryEnqueue(item);
    if (!status.ok()) {
        rejected_tasks_.fetch_add(1, std::memory_order_relaxed);
        tracking_token->Complete(status);
        return status;
    }

    submitted_tasks_.fetch_add(1, std::memory_order_relaxed);
    return Status::Ok();
}

ThreadPoolStats ThreadPool::Stats() const {
    std::lock_guard lock(lifecycle_mutex_);
    ThreadPoolStats stats;
    stats.worker_count = worker_statuses_.size();
    stats.queued_tasks = scheduler_ ? scheduler_->QueuedTaskCount() : 0;
    stats.active_workers = active_workers_.load(std::memory_order_relaxed);
    stats.submitted_tasks = submitted_tasks_.load(std::memory_order_relaxed);
    stats.completed_tasks = completion_counters_->completed.load(std::memory_order_relaxed);
    stats.failed_tasks = completion_counters_->failed.load(std::memory_order_relaxed);
    stats.rejected_tasks = rejected_tasks_.load(std::memory_order_relaxed);
    if (scheduler_) {
        stats.scheduler = scheduler_->Snapshot();
    }
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

bool ThreadPool::serializes_concurrency_key_until_completion() const noexcept {
    return scheduler_ && scheduler_->SerializesConcurrencyKeyUntilCompletion();
}

void ThreadPool::WorkerLoop(std::stop_token stop_token, std::size_t worker_index) {
    ThreadPoolContext context(worker_index, options_.name, stop_token);
    SetWorkerState(worker_index, WorkerState::Idle);

    while (!stop_token.stop_requested()) {
        auto task_result = scheduler_->WaitDequeue(worker_index, stop_token);
        if (!task_result) {
            SetWorkerState(worker_index, WorkerState::Stopping);
            break;
        }

        auto queued = std::move(task_result).value();
        active_workers_.fetch_add(1, std::memory_order_relaxed);
        SetWorkerState(worker_index, WorkerState::Running, queued->name_);
        context.set_payload(std::move(queued->payload_));
        context.set_task_group_token(queued->task_group_token_.get());
        auto completion = std::make_shared<detail::DeferredTaskCompletionState>(
            scheduler_, queued, worker_index, completion_counters_);
        context.set_deferred_completion_state(completion);

        TraceContext trace_ctx;
        trace_ctx.trace_id = queued->trace_id_;
        TraceScope trace_scope(trace_ctx);

        Status status = Status::Ok();
        try {
            status = queued->task_(context);
        } catch (const AppException& e) {
            status = e.status();
        } catch (const std::exception& e) {
            status = Status::Error(ErrorCode::InternalError, e.what());
        } catch (...) {
            status = Status::Error(ErrorCode::Unknown, "unknown task error");
        }

        context.clear_payload();
        context.set_task_group_token(nullptr);
        context.set_deferred_completion_state(nullptr);
        active_workers_.fetch_sub(1, std::memory_order_relaxed);
        if (!completion->deferred() || !status.ok()) {
            completion->Complete(status);
        }
        FinishWorkerTask(worker_index, std::move(status));
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
