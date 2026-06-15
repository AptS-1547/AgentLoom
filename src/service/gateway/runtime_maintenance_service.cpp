#include "runtime_maintenance_service.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <utility>

namespace agent::service::gateway {
namespace {

std::int64_t NowUnixMs() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

std::tm LocalTime(std::time_t value) {
    std::tm out{};
#ifdef _WIN32
    localtime_s(&out, &value);
#else
    localtime_r(&value, &out);
#endif
    return out;
}

} // namespace

RuntimeMaintenanceService::RuntimeMaintenanceService(core::LoggerAdapter logger)
    : logger_(std::move(logger)) {}

RuntimeMaintenanceService::~RuntimeMaintenanceService() {
    Stop();
}

core::Status RuntimeMaintenanceService::RegisterTask(std::shared_ptr<IRuntimeMaintenanceTask> task) {
    if (!task) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "maintenance task is required");
    }
    if (task->Name().empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "maintenance task name is required");
    }
    if (task->Interval().count() <= 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "maintenance task interval must be positive");
    }
    if (running_.load(std::memory_order_acquire)) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "maintenance service is running");
    }

    auto slot = std::make_shared<TaskSlot>();
    slot->health.name = std::string(task->Name());
    slot->task = std::move(task);

    std::lock_guard lock(mutex_);
    const auto duplicate = std::any_of(tasks_.begin(), tasks_.end(), [&](const auto& existing) {
        return existing->health.name == slot->health.name;
    });
    if (duplicate) {
        return core::Status::Error(core::ErrorCode::AlreadyExists, "maintenance task already registered");
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
                     slot->health.name,
                     slot->task->Interval().count());
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

void RuntimeMaintenanceService::RunTaskLoop(const std::shared_ptr<TaskSlot>& slot, std::stop_token stop_token) {
    while (!stop_token.stop_requested()) {
        {
            std::lock_guard lock(slot->mutex);
            slot->health.running = true;
            slot->health.last_started_at = std::chrono::steady_clock::now();
        }

        core::Status status = core::Status::Ok();
        try {
            status = slot->task->Tick(stop_token);
        } catch (const std::exception& ex) {
            status = core::Status::Error(core::ErrorCode::InternalError, ex.what());
        } catch (...) {
            status = core::Status::Error(core::ErrorCode::InternalError, "maintenance task threw unknown exception");
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
        if (interval.count() <= 0) {
            break;
        }
        std::condition_variable_any cv;
        std::mutex wait_mutex;
        std::unique_lock wait_lock(wait_mutex);
        cv.wait_for(wait_lock, stop_token, interval, [] {
            return false;
        });
    }
}

SessionMaintenanceTask::SessionMaintenanceTask(persona::ISessionManager& sessions,
                                               std::chrono::milliseconds interval,
                                               core::LoggerAdapter logger)
    : sessions_(sessions),
      interval_(interval),
      logger_(std::move(logger)) {}

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

DocumentRetentionMaintenanceTask::DocumentRetentionMaintenanceTask(
    std::shared_ptr<document::DocumentAnalysisService> documents,
    std::chrono::milliseconds interval)
    : documents_(std::move(documents)),
      interval_(interval) {}

std::string_view DocumentRetentionMaintenanceTask::Name() const noexcept {
    return "document_retention_cleanup";
}

std::chrono::milliseconds DocumentRetentionMaintenanceTask::Interval() const noexcept {
    return interval_;
}

core::Status DocumentRetentionMaintenanceTask::Tick(std::stop_token stop_token) {
    if (stop_token.stop_requested() || !documents_) {
        return core::Status::Ok();
    }
    return documents_->RunRetentionCleanupOnce(NowUnixMs());
}

L3MemoryFlushMaintenanceTask::L3MemoryFlushMaintenanceTask(
    std::shared_ptr<memory::LongTermMemoryCompressor> compressor,
    const persona::ISessionManager& sessions,
    L3MemoryFlushMaintenanceOptions options,
    core::LoggerAdapter logger)
    : compressor_(std::move(compressor)),
      sessions_(sessions),
      options_(std::move(options)),
      logger_(std::move(logger)) {}

std::string_view L3MemoryFlushMaintenanceTask::Name() const noexcept {
    return "l3_memory_flush";
}

std::chrono::milliseconds L3MemoryFlushMaintenanceTask::Interval() const noexcept {
    return options_.interval;
}

bool L3MemoryFlushMaintenanceTask::IsDue() const {
    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    const auto local = LocalTime(now_time);
    if (local.tm_hour > options_.flush_hour) {
        return true;
    }
    if (local.tm_hour == options_.flush_hour && local.tm_min >= options_.flush_minute) {
        return true;
    }
    return false;
}

std::string L3MemoryFlushMaintenanceTask::TargetDate() const {
    auto now = std::chrono::system_clock::now();
    now -= std::chrono::hours(24 * options_.flush_date_offset_days);
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    const auto local = LocalTime(now_time);
    std::ostringstream out;
    out << std::put_time(&local, "%Y-%m-%d");
    return out.str();
}

core::Status L3MemoryFlushMaintenanceTask::Tick(std::stop_token stop_token) {
    if (stop_token.stop_requested() || !compressor_) {
        return core::Status::Ok();
    }
    if (!IsDue()) {
        return core::Status::Ok();
    }
    if (options_.defer_when_sessions_active && sessions_.SessionCount() > 0) {
        logger_.info("[maintenance] l3 flush deferred active_sessions={}", sessions_.SessionCount());
        return core::Status::Ok();
    }

    const auto date = TargetDate();
    {
        std::lock_guard lock(mutex_);
        if (flushed_dates_.contains(date)) {
            return core::Status::Ok();
        }
    }

    std::vector<std::string> users = options_.user_uuids;
    auto registered_users = compressor_->GetRegisteredUsers();
    if (!registered_users.ok()) {
        return registered_users.status();
    }
    users.insert(users.end(), registered_users.value().begin(), registered_users.value().end());
    std::sort(users.begin(), users.end());
    users.erase(std::unique(users.begin(), users.end()), users.end());
    users.erase(std::remove_if(users.begin(), users.end(), [](const std::string& user) {
                    return user.empty();
                }),
                users.end());
    if (users.empty()) {
        return core::Status::Ok();
    }

    std::size_t success_count = 0;
    std::size_t failure_count = 0;
    std::string first_failure;
    for (const auto& user_uuid : users) {
        if (stop_token.stop_requested()) {
            break;
        }
        if (user_uuid.empty()) {
            continue;
        }
        auto result = compressor_->CompressDailyMemory(user_uuid, date);
        if (!result.ok()) {
            ++failure_count;
            if (first_failure.empty()) {
                first_failure = result.status().message();
            }
            logger_.warn("[maintenance] l3 flush failed user={} date={} reason={}",
                         user_uuid,
                         date,
                         result.status().message());
            continue;
        }
        ++success_count;
        logger_.info("[maintenance] l3 flush user={} date={} source_records={}",
                     user_uuid,
                     date,
                     result.value());
    }

    if (failure_count == 0) {
        std::lock_guard lock(mutex_);
        flushed_dates_.insert(date);
    }
    if (failure_count > 0) {
        return core::Status::Error(
            core::ErrorCode::InternalError,
            "l3 flush failed for " + std::to_string(failure_count) +
                " user(s): " + first_failure);
    }
    if (success_count > 0) {
        logger_.info("[maintenance] l3 flush completed date={} users={}", date, success_count);
    }
    return core::Status::Ok();
}

} // namespace agent::service::gateway
