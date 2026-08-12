#include "thread_pool_scheduler.h"

#include "blocking_queue.h"

#include <atomic>
#include <mutex>

namespace core {

struct DefaultFifoThreadPoolTaskScheduler::Impl {
    mutable std::mutex mutex;
    std::unique_ptr<BlockingQueue<std::shared_ptr<ThreadPoolWorkItem>>> queue;
    std::atomic<std::size_t> running_tasks{0};
    std::atomic<std::size_t> rejected_tasks{0};
};

DefaultFifoThreadPoolTaskScheduler::DefaultFifoThreadPoolTaskScheduler()
    : impl_(std::make_unique<Impl>()) {}

DefaultFifoThreadPoolTaskScheduler::~DefaultFifoThreadPoolTaskScheduler() = default;

Status DefaultFifoThreadPoolTaskScheduler::Start(const ThreadPoolSchedulerOptions& options) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->queue) {
        if (impl_->queue->closed()) {
            return Status::Error(ErrorCode::Unavailable, "thread pool scheduler has been shut down");
        }
        return Status::Ok();
    }
    impl_->queue = std::make_unique<BlockingQueue<std::shared_ptr<ThreadPoolWorkItem>>>(
        options.queue_capacity);
    return Status::Ok();
}

Status DefaultFifoThreadPoolTaskScheduler::TryEnqueue(
    const std::shared_ptr<ThreadPoolWorkItem>& item) {
    if (!item) {
        impl_->rejected_tasks.fetch_add(1, std::memory_order_relaxed);
        return Status::Error(ErrorCode::InvalidArgument, "thread pool work item is empty");
    }

    std::lock_guard lock(impl_->mutex);
    if (!impl_->queue) {
        impl_->rejected_tasks.fetch_add(1, std::memory_order_relaxed);
        return Status::Error(ErrorCode::Unavailable, "thread pool scheduler is not running");
    }
    auto status = impl_->queue->TryPush(item);
    if (!status.ok()) {
        impl_->rejected_tasks.fetch_add(1, std::memory_order_relaxed);
    }
    return status;
}

Result<std::shared_ptr<ThreadPoolWorkItem>> DefaultFifoThreadPoolTaskScheduler::WaitDequeue(
    std::size_t,
    std::stop_token) {
    BlockingQueue<std::shared_ptr<ThreadPoolWorkItem>>* queue = nullptr;
    {
        std::lock_guard lock(impl_->mutex);
        queue = impl_->queue.get();
    }
    if (!queue) {
        return Status::Error(ErrorCode::Unavailable, "thread pool scheduler is not running");
    }
    auto result = queue->WaitPop();
    if (result.ok()) {
        impl_->running_tasks.fetch_add(1, std::memory_order_relaxed);
    }
    return result;
}

void DefaultFifoThreadPoolTaskScheduler::Complete(
    const ThreadPoolWorkItem&,
    std::size_t,
    const Status&) noexcept {
    impl_->running_tasks.fetch_sub(1, std::memory_order_relaxed);
}

void DefaultFifoThreadPoolTaskScheduler::Close(bool discard) noexcept {
    std::lock_guard lock(impl_->mutex);
    if (impl_->queue) {
        impl_->queue->Close(discard);
    }
}

bool DefaultFifoThreadPoolTaskScheduler::closed() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->queue && impl_->queue->closed();
}

std::size_t DefaultFifoThreadPoolTaskScheduler::QueuedTaskCount() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->queue ? impl_->queue->size() : 0;
}

ThreadPoolConcurrencySnapshot DefaultFifoThreadPoolTaskScheduler::Snapshot() const {
    ThreadPoolConcurrencySnapshot snapshot;
    snapshot.queued_tasks = QueuedTaskCount();
    snapshot.running_tasks = impl_->running_tasks.load(std::memory_order_relaxed);
    snapshot.rejected_tasks = impl_->rejected_tasks.load(std::memory_order_relaxed);
    return snapshot;
}

} // namespace core
