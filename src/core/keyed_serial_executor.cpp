#include "keyed_serial_executor.h"

#include "exception.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace core {

struct KeyedSerialExecutor::Impl {
    struct QueuedTask {
        Task task;
        std::string name;
    };

    struct Lane {
        std::deque<QueuedTask> tasks;
        bool scheduled = false;
    };

    Impl(KeyedSerialExecutor& owner,
         std::shared_ptr<ThreadPool> target_pool,
         KeyedSerialExecutorOptions executor_options)
        : parent(owner), pool(std::move(target_pool)), options(std::move(executor_options)) {}

    Status Submit(std::string_view key, Task task, std::string task_name) {
        if (key.empty()) {
            return Status::Error(ErrorCode::InvalidArgument, "serial executor key is required");
        }
        if (!task) {
            return Status::Error(ErrorCode::InvalidArgument, "serial executor task is empty");
        }
        if (!pool) {
            return Status::Error(ErrorCode::FailedPrecondition, "serial executor pool is missing");
        }

        std::lock_guard lock(mutex);
        auto lane_it = lanes.find(std::string(key));
        if (lane_it == lanes.end()) {
            if (lanes.size() >= options.max_keys) {
                return Status::Error(ErrorCode::ResourceExhausted, "serial executor key limit reached");
            }
            lane_it = lanes.emplace(std::string(key), Lane{}).first;
        }
        auto& lane = lane_it->second;
        if (lane.tasks.size() >= options.queue_capacity_per_key) {
            return Status::Error(ErrorCode::ResourceExhausted, "serial executor lane capacity reached");
        }
        lane.tasks.push_back({std::move(task), std::move(task_name)});
        if (lane.scheduled) {
            return Status::Ok();
        }

        lane.scheduled = true;
        auto self = parent.shared_from_this();
        auto status = pool->Submit(
            [self, lane_key = lane_it->first]() mutable -> Status {
                return self->impl_->RunOne(std::move(lane_key));
            },
            {},
            PoolTaskName(lane.tasks.front().name));
        if (!status.ok()) {
            lane.tasks.pop_back();
            lane.scheduled = false;
            if (lane.tasks.empty()) {
                lanes.erase(lane_it);
            }
        }
        return status;
    }

    Status RunOne(std::string key) {
        Status first_failure = Status::Ok();
        for (;;) {
            QueuedTask queued;
            {
                std::lock_guard lock(mutex);
                auto lane_it = lanes.find(key);
                if (lane_it == lanes.end() || lane_it->second.tasks.empty()) {
                    return Status::Error(ErrorCode::FailedPrecondition, "serial executor lane is empty");
                }
                queued = std::move(lane_it->second.tasks.front());
                lane_it->second.tasks.pop_front();
            }

            Status task_status = Status::Ok();
            try {
                task_status = queued.task();
            } catch (const AppException& error) {
                task_status = error.status();
            } catch (const std::exception& error) {
                task_status = Status::Error(ErrorCode::InternalError, error.what());
            } catch (...) {
                task_status = Status::Error(ErrorCode::Unknown, "unknown serial executor task error");
            }
            if (first_failure.ok() && !task_status.ok()) {
                first_failure = task_status;
            }

            std::lock_guard lock(mutex);
            auto lane_it = lanes.find(key);
            if (lane_it == lanes.end()) {
                return first_failure;
            }
            if (lane_it->second.tasks.empty()) {
                lanes.erase(lane_it);
                idle_cv.notify_all();
                return first_failure;
            }

            auto self = parent.shared_from_this();
            auto schedule_status = pool->Submit(
                [self, lane_key = key]() mutable -> Status {
                    return self->impl_->RunOne(std::move(lane_key));
                },
                {},
                PoolTaskName(lane_it->second.tasks.front().name));
            if (schedule_status.ok()) {
                return first_failure;
            }
            if (schedule_status.code() != ErrorCode::ResourceExhausted) {
                lane_it->second.scheduled = false;
                return schedule_status;
            }
        }
    }

    Status WaitIdle(std::string_view key, std::chrono::milliseconds timeout) {
        if (key.empty()) {
            return Status::Error(ErrorCode::InvalidArgument, "serial executor key is required");
        }
        std::unique_lock lock(mutex);
        if (!idle_cv.wait_for(lock, timeout, [&] { return !lanes.contains(std::string(key)); })) {
            return Status::Error(ErrorCode::Timeout, "serial executor lane did not become idle");
        }
        return Status::Ok();
    }

    std::string PoolTaskName(std::string_view task_name) const {
        if (task_name.empty()) {
            return options.task_name_prefix;
        }
        return options.task_name_prefix + ":" + std::string(task_name);
    }

    KeyedSerialExecutor& parent;
    std::shared_ptr<ThreadPool> pool;
    KeyedSerialExecutorOptions options;
    std::mutex mutex;
    std::condition_variable idle_cv;
    std::unordered_map<std::string, Lane> lanes;
};

KeyedSerialExecutor::KeyedSerialExecutor(
    std::shared_ptr<ThreadPool> pool,
    KeyedSerialExecutorOptions options)
    : impl_(std::make_unique<Impl>(*this, std::move(pool), std::move(options))) {}

KeyedSerialExecutor::~KeyedSerialExecutor() = default;

Status KeyedSerialExecutor::Submit(std::string_view key, Task task, std::string task_name) {
    return impl_->Submit(key, std::move(task), std::move(task_name));
}

Status KeyedSerialExecutor::WaitIdle(std::string_view key, std::chrono::milliseconds timeout) {
    return impl_->WaitIdle(key, timeout);
}

} // namespace core
