#include "runtime_maintenance_service.h"

#include "inference_frame_ipc_control.h"

#if defined(AGENTLOOM_HAS_MEDIA)
#include "webrtc_session_registry.h"
#endif

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

AuthSessionMaintenanceTask::AuthSessionMaintenanceTask(
    std::shared_ptr<IAuthSessionStore> store,
    std::chrono::milliseconds interval,
    std::size_t batch_size,
    core::LoggerAdapter logger)
    : store_(std::move(store)),
      interval_(interval),
      batch_size_(batch_size),
      logger_(std::move(logger)) {}

std::string_view AuthSessionMaintenanceTask::Name() const noexcept {
    return "gateway_auth_session_cleanup";
}

std::chrono::milliseconds AuthSessionMaintenanceTask::Interval() const noexcept {
    return interval_;
}

core::Status AuthSessionMaintenanceTask::Tick(std::stop_token stop_token) {
    if (stop_token.stop_requested() || !store_) {
        return core::Status::Ok();
    }
    auto cleanup = store_->CleanupExpired(std::chrono::system_clock::now(), batch_size_);
    if (!cleanup.ok()) {
        return cleanup.status();
    }
    if (cleanup.value() > 0) {
        logger_.info("[maintenance] gateway auth session cleanup removed_count={}", cleanup.value());
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

WebRtcSessionMaintenanceTask::WebRtcSessionMaintenanceTask(
    std::shared_ptr<media::WebRtcSessionRegistry> registry,
    std::chrono::milliseconds interval,
    core::LoggerAdapter logger)
    : registry_(std::move(registry)),
      interval_(interval),
      logger_(std::move(logger)) {}

std::string_view WebRtcSessionMaintenanceTask::Name() const noexcept {
    return "webrtc_session_cleanup";
}

std::chrono::milliseconds WebRtcSessionMaintenanceTask::Interval() const noexcept {
    return interval_;
}

core::Status WebRtcSessionMaintenanceTask::Tick(std::stop_token stop_token) {
    if (stop_token.stop_requested() || !registry_) {
        return core::Status::Ok();
    }
#if defined(AGENTLOOM_HAS_MEDIA)
    const auto expired = registry_->CleanupExpired(stop_token);
    if (expired > 0) {
        logger_.info("[maintenance] rtc session cleanup expired_count={}", expired);
    }
    return core::Status::Ok();
#else
    return core::Status::Error(
        core::ErrorCode::Unimplemented,
        "WebRTC media support is not available in this build");
#endif
}

SkillSessionMaintenanceTask::SkillSessionMaintenanceTask(
    std::shared_ptr<persona::ISkillSessionManager> manager,
    std::chrono::milliseconds interval,
    core::LoggerAdapter logger)
    : manager_(std::move(manager)),
      interval_(interval),
      logger_(std::move(logger)) {}

std::string_view SkillSessionMaintenanceTask::Name() const noexcept {
    return "skill_session_cleanup";
}

std::chrono::milliseconds SkillSessionMaintenanceTask::Interval() const noexcept {
    return interval_;
}

core::Status SkillSessionMaintenanceTask::Tick(std::stop_token stop_token) {
    if (stop_token.stop_requested() || !manager_) {
        return core::Status::Ok();
    }
    const auto expired = manager_->CleanupExpired(stop_token);
    if (expired > 0) {
        logger_.info("[maintenance] skill session cleanup expired_count={}", expired);
    }
    return core::Status::Ok();
}

InferenceFrameIpcPeerMaintenanceTask::InferenceFrameIpcPeerMaintenanceTask(
    std::shared_ptr<ipc::media::IInferenceFrameIpcLeaseCoordinator> coordinator,
    std::chrono::milliseconds interval,
    bool auto_recover)
    : coordinator_(std::move(coordinator)),
      interval_(interval),
      auto_recover_(auto_recover) {}

std::string_view InferenceFrameIpcPeerMaintenanceTask::Name() const noexcept {
    return "inference_frame_ipc_peer";
}

std::chrono::milliseconds InferenceFrameIpcPeerMaintenanceTask::Interval() const noexcept {
    return interval_;
}

core::Status InferenceFrameIpcPeerMaintenanceTask::Tick(std::stop_token stop_token) {
    if (stop_token.stop_requested() || !coordinator_) {
        return core::Status::Ok();
    }
    using State = ipc::media::InferenceFrameIpcControlState;
    switch (coordinator_->Snapshot().state) {
    case State::Idle:
        return coordinator_->Start();
    case State::Granted:
        return coordinator_->CheckPeer();
    case State::Fenced:
    case State::Failed:
        return auto_recover_
            ? coordinator_->Recover()
            : core::Status::Error(core::ErrorCode::Unavailable, "inference frame IPC peer is fenced");
    case State::Recovering:
        return core::Status::Error(core::ErrorCode::Unavailable, "inference frame IPC peer is recovering");
    case State::Shutdown:
        return core::Status::Ok();
    }
    return core::Status::Error(core::ErrorCode::InternalError, "unknown inference frame IPC control state");
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
