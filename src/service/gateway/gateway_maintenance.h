#pragma once

#include "logger_adapter.h"
#include "result.h"
#include "session_manager.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace agent::service::gateway {

class IRuntimeMaintenanceTask {
public:
    virtual ~IRuntimeMaintenanceTask() = default;

    virtual std::string_view Name() const noexcept = 0;
    virtual std::chrono::milliseconds Interval() const noexcept = 0;
    virtual core::Status Tick(std::stop_token stop_token) = 0;
};

struct MaintenanceTaskHealth {
    std::string name;
    bool running = false;
    std::chrono::steady_clock::time_point last_started_at{};
    std::chrono::steady_clock::time_point last_finished_at{};
    std::chrono::steady_clock::time_point last_success_at{};
    std::uint64_t success_count = 0;
    std::uint64_t failure_count = 0;
    core::Status last_status = core::Status::Ok();
};

class RuntimeMaintenanceService final {
public:
    explicit RuntimeMaintenanceService(
        core::LoggerAdapter logger = core::LoggerAdapter::ForModule("gateway"));
    ~RuntimeMaintenanceService();

    RuntimeMaintenanceService(const RuntimeMaintenanceService&) = delete;
    RuntimeMaintenanceService& operator=(const RuntimeMaintenanceService&) = delete;

    core::Status RegisterTask(std::shared_ptr<IRuntimeMaintenanceTask> task);
    core::Status Start();
    void Stop();
    std::vector<MaintenanceTaskHealth> SnapshotHealth() const;
    bool running() const noexcept;

private:
    struct TaskSlot {
        std::shared_ptr<IRuntimeMaintenanceTask> task;
        mutable std::mutex mutex;
        MaintenanceTaskHealth health;
        std::jthread worker;
    };

    void RunTaskLoop(const std::shared_ptr<TaskSlot>& slot, std::stop_token stop_token);

    core::LoggerAdapter logger_;
    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<TaskSlot>> tasks_;
    std::atomic_bool running_{false};
};

class SessionMaintenanceTask final : public IRuntimeMaintenanceTask {
public:
    SessionMaintenanceTask(persona::ISessionManager& sessions,
                           std::chrono::milliseconds interval,
                           core::LoggerAdapter logger =
                               core::LoggerAdapter::ForModule("gateway"));

    std::string_view Name() const noexcept override;
    std::chrono::milliseconds Interval() const noexcept override;
    core::Status Tick(std::stop_token stop_token) override;

private:
    persona::ISessionManager& sessions_;
    std::chrono::milliseconds interval_;
    core::LoggerAdapter logger_;
};

}
