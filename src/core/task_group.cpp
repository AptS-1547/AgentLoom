#include "task_group.h"

#include <condition_variable>
#include <exception>
#include <mutex>
#include <utility>
#include <vector>

namespace core {

struct TaskGroupToken::SharedState {
    mutable std::mutex mutex;
    std::condition_variable drained_cv;
    std::size_t outstanding_tasks = 0;
    std::size_t completed_tasks = 0;
    std::size_t failed_tasks = 0;
    bool sealed = false;
    Status first_failure = Status::Ok();
    std::vector<TaskGroup::DrainedCallback> callbacks;
};

namespace {

TaskGroupSnapshot SnapshotLocked(const TaskGroupToken::SharedState& state) {
    return {
        .outstanding_tasks = state.outstanding_tasks,
        .completed_tasks = state.completed_tasks,
        .failed_tasks = state.failed_tasks,
        .sealed = state.sealed,
        .drained = state.sealed && state.outstanding_tasks == 0,
        .first_failure = state.first_failure,
    };
}

void InvokeCallbacks(std::vector<TaskGroup::DrainedCallback> callbacks) noexcept {
    for (auto& callback : callbacks) {
        try {
            callback();
        } catch (const std::exception&) {
        } catch (...) {
        }
    }
}

} // namespace

TaskGroupToken::TaskGroupToken(std::shared_ptr<SharedState> state) noexcept
    : state_(std::move(state)), active_(state_ != nullptr) {}

TaskGroupToken::~TaskGroupToken() {
    ReleaseWithoutCompletion();
}

TaskGroupToken::TaskGroupToken(TaskGroupToken&& other) noexcept
    : state_(std::move(other.state_)), active_(std::exchange(other.active_, false)) {}

TaskGroupToken& TaskGroupToken::operator=(TaskGroupToken&& other) noexcept {
    if (this != &other) {
        ReleaseWithoutCompletion();
        state_ = std::move(other.state_);
        active_ = std::exchange(other.active_, false);
    }
    return *this;
}

Result<TaskGroupToken> TaskGroupToken::AcquireChild() const {
    if (!active_ || !state_) {
        return Status::Error(ErrorCode::FailedPrecondition, "task group token is inactive");
    }
    std::lock_guard lock(state_->mutex);
    if (!active_) {
        return Status::Error(ErrorCode::FailedPrecondition, "task group token is inactive");
    }
    ++state_->outstanding_tasks;
    return TaskGroupToken(state_);
}

void TaskGroupToken::Complete(Status status) noexcept {
    if (!active_ || !state_) {
        return;
    }

    std::vector<TaskGroup::DrainedCallback> callbacks;
    {
        std::lock_guard lock(state_->mutex);
        if (!active_) {
            return;
        }
        active_ = false;
        if (state_->outstanding_tasks > 0) {
            --state_->outstanding_tasks;
        }
        if (status.ok()) {
            ++state_->completed_tasks;
        } else {
            ++state_->failed_tasks;
            if (state_->first_failure.ok()) {
                state_->first_failure = std::move(status);
            }
        }
        if (state_->sealed && state_->outstanding_tasks == 0) {
            callbacks.swap(state_->callbacks);
            state_->drained_cv.notify_all();
        }
    }
    InvokeCallbacks(std::move(callbacks));
}

bool TaskGroupToken::valid() const noexcept {
    return active_ && state_ != nullptr;
}

void TaskGroupToken::ReleaseWithoutCompletion() noexcept {
    if (active_) {
        Complete(Status::Error(
            ErrorCode::Cancelled,
            "task group token released without completion"));
    }
}

TaskGroup::TaskGroup()
    : state_(std::make_shared<TaskGroupToken::SharedState>()) {}

Result<TaskGroupToken> TaskGroup::AcquireRoot() {
    std::lock_guard lock(state_->mutex);
    if (state_->sealed) {
        return Status::Error(ErrorCode::FailedPrecondition, "task group is sealed");
    }
    ++state_->outstanding_tasks;
    return TaskGroupToken(state_);
}

Status TaskGroup::Seal() {
    std::vector<DrainedCallback> callbacks;
    {
        std::lock_guard lock(state_->mutex);
        state_->sealed = true;
        if (state_->outstanding_tasks == 0) {
            callbacks.swap(state_->callbacks);
            state_->drained_cv.notify_all();
        }
    }
    InvokeCallbacks(std::move(callbacks));
    return Status::Ok();
}

Result<TaskGroupSnapshot> TaskGroup::WaitFor(std::chrono::milliseconds timeout) const {
    if (timeout.count() < 0) {
        return Status::Error(ErrorCode::InvalidArgument, "task group wait timeout must not be negative");
    }
    std::unique_lock lock(state_->mutex);
    if (!state_->drained_cv.wait_for(lock, timeout, [this] {
            return state_->sealed && state_->outstanding_tasks == 0;
        })) {
        return Status::Error(ErrorCode::Timeout, "task group wait timed out");
    }
    return SnapshotLocked(*state_);
}

Status TaskGroup::OnDrained(DrainedCallback callback) {
    if (!callback) {
        return Status::Error(ErrorCode::InvalidArgument, "task group drained callback is empty");
    }

    bool invoke_now = false;
    {
        std::lock_guard lock(state_->mutex);
        if (state_->sealed && state_->outstanding_tasks == 0) {
            invoke_now = true;
        } else {
            state_->callbacks.push_back(std::move(callback));
        }
    }
    if (invoke_now) {
        InvokeCallbacks({std::move(callback)});
    }
    return Status::Ok();
}

TaskGroupSnapshot TaskGroup::Snapshot() const {
    std::lock_guard lock(state_->mutex);
    return SnapshotLocked(*state_);
}

} // namespace core
