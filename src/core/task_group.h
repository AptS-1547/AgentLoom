#pragma once

#include "result.h"

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>

namespace core {

struct TaskGroupSnapshot {
    std::size_t outstanding_tasks = 0;
    std::size_t completed_tasks = 0;
    std::size_t failed_tasks = 0;
    bool sealed = false;
    bool drained = false;
    Status first_failure = Status::Ok();
};

class TaskGroup;

class TaskGroupToken {
public:
    struct SharedState;

    TaskGroupToken() = default;
    ~TaskGroupToken();

    TaskGroupToken(const TaskGroupToken&) = delete;
    TaskGroupToken& operator=(const TaskGroupToken&) = delete;
    TaskGroupToken(TaskGroupToken&& other) noexcept;
    TaskGroupToken& operator=(TaskGroupToken&& other) noexcept;

    Result<TaskGroupToken> AcquireChild() const;
    void Complete(Status status = Status::Ok()) noexcept;
    bool valid() const noexcept;

private:
    explicit TaskGroupToken(std::shared_ptr<SharedState> state) noexcept;
    void ReleaseWithoutCompletion() noexcept;

    std::shared_ptr<SharedState> state_;
    bool active_ = false;

    friend class TaskGroup;
    friend class ThreadPool;
};

class TaskGroup final {
public:
    using DrainedCallback = std::function<void()>;

    TaskGroup();
    ~TaskGroup() = default;

    TaskGroup(const TaskGroup&) = delete;
    TaskGroup& operator=(const TaskGroup&) = delete;
    TaskGroup(TaskGroup&&) = delete;
    TaskGroup& operator=(TaskGroup&&) = delete;

    Result<TaskGroupToken> AcquireRoot();
    Status Seal();
    Result<TaskGroupSnapshot> WaitFor(std::chrono::milliseconds timeout) const;
    Status OnDrained(DrainedCallback callback);
    TaskGroupSnapshot Snapshot() const;

private:
    std::shared_ptr<TaskGroupToken::SharedState> state_;
};

} // namespace core
