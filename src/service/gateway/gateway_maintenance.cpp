#include "gateway_maintenance.h"

#include <algorithm>
#include <exception>
#include <utility>

namespace agent::service::gateway {

RuntimeMaintenanceService::RuntimeMaintenanceService(core::LoggerAdapter logger)
    : logger_(std::move(logger)) {}

RuntimeMaintenanceService::~RuntimeMaintenanceService() {
    Stop();
}

core::Status RuntimeMaintenanceService::RegisterTask(
    std::shared_ptr<IRuntimeMaintenanceTask> task) {
    if (!task) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "maintenance task is required");
    }
    if (task->Name().empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "maintenance task name is required");
    }
    if (task->Interval().count() <= 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "maintenance task interval must be positive");
    }
    if (running_.load(std::memory_order_acquire)) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                   "maintenance service is running");
    }

    auto slot = std::make_shared<TaskSlot>();
    slot->health.name = std::string(task->Name());
    slot->task = std::move(task);

    std::lock_guard lock(mutex_);
    const auto duplicate = std::any_of(
        tasks_.begin(), tasks_.end(),
        [&](const auto& existing) { return existing->health.name == slot->health.name; });
    if (duplicate) {
        return core::Status::Error(core::ErrorCode::AlreadyExists,
                                   "maintenance task already registered");
    }
    tasks_.push_back(std::move(slot));
    return core::Status::Ok();
}

core::Status RuntimeMaintenanceService::Start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return core::Status::Ok();
    }

    std::vector<std::shared_ptr<TaskSlot>> tasks;
    {
        std::lock_guard lock(mutex_);
        tasks = tasks_;
    }
    for (const auto& slot : tasks) {
        slot->worker = std::jthread([this, slot](std::stop_token stop_token) {
            RunTaskLoop(slot, stop_token);
        });
        logger_.info("[maintenance] started task={} interval_ms={}",
                     slot->health.name, slot->task->Interval().count());
    }
    return core::Status::Ok();
}

void RuntimeMaintenanceService::Stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }

    std::vector<std::shared_ptr<TaskSlot>> tasks;
    {
        std::lock_guard lock(mutex_);
        tasks = tasks_;
    }
    for (const auto& slot : tasks) {
        if (slot->worker.joinable()) {
            slot->worker.request_stop();
        }
    }
    for (auto& slot : tasks) {
        if (slot->worker.joinable()) {
            slot->worker.join();
        }
    }
    logger_.info("[maintenance] stopped");
}

std::vector<MaintenanceTaskHealth> RuntimeMaintenanceService::SnapshotHealth() const {
    std::vector<std::shared_ptr<TaskSlot>> tasks;
    {
        std::lock_guard lock(mutex_);
        tasks = tasks_;
    }
    std::vector<MaintenanceTaskHealth> snapshots;
    snapshots.reserve(tasks.size());
    for (const auto& slot : tasks) {
        std::lock_guard health_lock(slot->mutex);
        snapshots.push_back(slot->health);
    }
    return snapshots;
}

bool RuntimeMaintenanceService::running() const noexcept {
    return running_.load(std::memory_order_acquire);
}

void RuntimeMaintenanceService::RunTaskLoop(const std::shared_ptr<TaskSlot>& slot,
                                            std::stop_token stop_token) {
    while (!stop_token.stop_requested()) {
        {
            std::lock_guard lock(slot->mutex);
            slot->health.running = true;
            slot->health.last_started_at = std::chrono::steady_clock::now();
        }

        core::Status status = core::Status::Ok();
        try {
            status = slot->task->Tick(stop_token);
        } catch (const std::exception& error) {
            status = core::Status::Error(core::ErrorCode::InternalError, error.what());
        } catch (...) {
            status = core::Status::Error(core::ErrorCode::InternalError,
                                         "maintenance task threw unknown exception");
        }

        {
            std::lock_guard lock(slot->mutex);
            slot->health.running = false;
            slot->health.last_finished_at = std::chrono::steady_clock::now();
            slot->health.last_status = status;
            if (status.ok()) {
                ++slot->health.success_count;
                slot->health.last_success_at = slot->health.last_finished_at;
            } else {
                ++slot->health.failure_count;
            }
        }
        if (!status.ok()) {
            logger_.warn("[maintenance] task failed name={} code={} reason={}",
                         slot->health.name,
                         static_cast<int>(status.code()),
                         status.message());
        }

        const auto interval = slot->task->Interval();
        std::condition_variable_any condition;
        std::mutex wait_mutex;
        std::unique_lock wait_lock(wait_mutex);
        condition.wait_for(wait_lock, stop_token, interval, [] { return false; });
    }
}

SessionMaintenanceTask::SessionMaintenanceTask(persona::ISessionManager& sessions,
                                               std::chrono::milliseconds interval,
                                               core::LoggerAdapter logger)
    : sessions_(sessions), interval_(interval), logger_(std::move(logger)) {}

std::string_view SessionMaintenanceTask::Name() const noexcept {
    return "session_cleanup";
}

std::chrono::milliseconds SessionMaintenanceTask::Interval() const noexcept {
    return interval_;
}

core::Status SessionMaintenanceTask::Tick(std::stop_token stop_token) {
    if (stop_token.stop_requested()) {
        return core::Status::Ok();
    }
    const auto expired = sessions_.CleanupExpired();
    if (!expired.empty()) {
        logger_.info("[maintenance] session cleanup expired_count={}", expired.size());
    }
    return core::Status::Ok();
}

}
