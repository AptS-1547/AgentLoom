#include "skill_session_manager.h"

#include <algorithm>
#include <utility>

namespace agent::service::persona {
namespace {

bool EmptyId(std::string_view value) {
    return value.empty();
}

} // namespace

SkillSessionManager::SkillSessionManager(SkillSessionOptions options, core::LoggerAdapter logger)
    : options_(std::move(options)), logger_(std::move(logger)) {}

SkillSessionObservationSink::SkillSessionObservationSink(std::shared_ptr<ISkillSessionManager> manager)
    : manager_(std::move(manager)) {}

core::Status SkillSessionObservationSink::Publish(SkillObservation observation) {
    if (!manager_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill session manager is not configured");
    }
    return manager_->RecordObservation(observation);
}

core::Result<SkillSessionSnapshot> SkillSessionManager::Start(const SkillSessionStartRequest& request) {
    if (EmptyId(request.session_id) || EmptyId(request.skill_id)) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "skill_id and session_id are required");
    }
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    const auto key = Key(request.session_id, request.skill_id);
    auto& session = sessions_[key];
    if (!session.session_id.empty() && !Terminal(session.state)) {
        if (!request.user_uuid.empty() && session.user_uuid != request.user_uuid) {
            return core::Status::Error(core::ErrorCode::PermissionDenied, "skill session does not belong to authenticated user");
        }
        if (session.state == SkillSessionState::Closing) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill session is closing");
        }
        session.last_activity_at = now;
        session.trace_id = request.trace_id.empty() ? session.trace_id : request.trace_id;
        return SnapshotLocked(session);
    }

    session = SkillSessionSnapshot{};
    session.skill_id = request.skill_id;
    session.execution_id = request.execution_id.empty()
        ? request.session_id + ":" + request.skill_id + ":" + std::to_string(next_execution_id_++)
        : request.execution_id;
    session.session_id = request.session_id;
    session.user_uuid = request.user_uuid;
    session.persona_id = request.persona_id;
    session.trace_id = request.trace_id;
    session.state = SkillSessionState::Starting;
    session.status_text = request.reason.empty() ? "skill session starting" : request.reason;
    session.started_at = now;
    session.last_activity_at = now;
    session.max_duration = request.max_duration.count() > 0 ? request.max_duration : options_.max_duration;
    logger_.info("[skill] start skill={} session={} source={} trace_id={}",
                 session.skill_id,
                 session.session_id,
                 request.source,
                 session.trace_id);
    return SnapshotLocked(session);
}

core::Result<SkillSessionSnapshot> SkillSessionManager::Stop(const SkillSessionStopRequest& request) {
    if (EmptyId(request.session_id) || EmptyId(request.skill_id)) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "skill_id and session_id are required");
    }
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(Key(request.session_id, request.skill_id));
    if (it == sessions_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "skill session not found");
    }
    auto& session = it->second;
    if (!request.authenticated_user_uuid.empty() && session.user_uuid != request.authenticated_user_uuid) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "skill session does not belong to authenticated user");
    }
    session.trace_id = request.trace_id.empty() ? session.trace_id : request.trace_id;
    session.close_reason = request.reason;
    session.status_text = request.reason.empty() ? "skill session closing" : request.reason;
    session.last_activity_at = now;
    session.closing_started_at = now;
    if (!request.execution_id.empty() && request.execution_id != session.execution_id) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill execution is stale");
    }
    session.state = SkillSessionState::Closing;
    logger_.info("[skill] stop skill={} session={} source={} trace_id={}",
                 session.skill_id,
                 session.session_id,
                 request.source,
                 session.trace_id);
    return SnapshotLocked(session);
}

core::Result<std::optional<SkillSessionSnapshot>> SkillSessionManager::Get(std::string_view session_id,
                                                                           std::string_view skill_id) const {
    if (EmptyId(session_id) || EmptyId(skill_id)) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "skill_id and session_id are required");
    }
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(Key(session_id, skill_id));
    if (it == sessions_.end()) {
        return std::optional<SkillSessionSnapshot>{};
    }
    return std::optional<SkillSessionSnapshot>{SnapshotLocked(it->second)};
}

core::Result<std::vector<SkillSessionSnapshot>> SkillSessionManager::List(
    std::string_view session_id, std::string_view user_uuid) const {
    if (EmptyId(session_id)) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "session_id is required");
    }
    std::vector<SkillSessionSnapshot> result;
    std::lock_guard lock(mutex_);
    const std::string prefix = std::string(session_id) + "\n";
    for (const auto& [key, session] : sessions_) {
        if (key.rfind(prefix, 0) != 0 ||
            (!user_uuid.empty() && session.user_uuid != user_uuid) ||
            Terminal(session.state)) {
            continue;
        }
        result.push_back(SnapshotLocked(session));
    }
    return result;
}

core::Status SkillSessionManager::MarkReady(std::string_view session_id,
                                            std::string_view skill_id,
                                            std::string status_text,
                                            std::string_view trace_id) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(Key(session_id, skill_id));
    if (it == sessions_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "skill session not found");
    }
    auto& session = it->second;
    if (Terminal(session.state)) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill session is terminal");
    }
    if (session.state == SkillSessionState::Closing) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill session is closing");
    }
    session.state = SkillSessionState::Ready;
    session.status_text = std::move(status_text);
    session.trace_id = trace_id.empty() ? session.trace_id : std::string(trace_id);
    session.last_activity_at = now;
    return core::Status::Ok();
}

core::Status SkillSessionManager::MarkFailed(std::string_view session_id,
                                             std::string_view skill_id,
                                             std::string error,
                                             std::string_view trace_id,
                                             std::string_view execution_id) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(Key(session_id, skill_id));
    if (it == sessions_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "skill session not found");
    }
    auto& session = it->second;
    if (!execution_id.empty() && execution_id != session.execution_id) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill execution is stale");
    }
    session.state = SkillSessionState::Failed;
    session.last_error = std::move(error);
    session.status_text = session.last_error;
    session.trace_id = trace_id.empty() ? session.trace_id : std::string(trace_id);
    session.last_activity_at = now;
    logger_.warn("[skill] failed skill={} session={} trace_id={} reason={}",
                 session.skill_id,
                 session.session_id,
                 session.trace_id,
                 session.last_error);
    return core::Status::Ok();
}

core::Status SkillSessionManager::BeginClosing(std::string_view session_id,
                                               std::string_view skill_id,
                                               std::string_view execution_id,
                                               std::string reason,
                                               std::string_view trace_id) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(Key(session_id, skill_id));
    if (it == sessions_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "skill session not found");
    }
    auto& session = it->second;
    if (session.execution_id != execution_id) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill execution is stale");
    }
    if (Terminal(session.state)) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill session is terminal");
    }
    if (session.state != SkillSessionState::Closing) {
        session.closing_started_at = now;
    }
    session.state = SkillSessionState::Closing;
    session.close_reason = std::move(reason);
    session.status_text = session.close_reason.empty() ? "skill session closing" : session.close_reason;
    session.trace_id = trace_id.empty() ? session.trace_id : std::string(trace_id);
    session.last_activity_at = now;
    return core::Status::Ok();
}

core::Status SkillSessionManager::CompleteClosing(std::string_view session_id,
                                                  std::string_view skill_id,
                                                  std::string_view execution_id,
                                                  std::string status_text,
                                                  std::string_view trace_id) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(Key(session_id, skill_id));
    if (it == sessions_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "skill session not found");
    }
    auto& session = it->second;
    if (session.execution_id != execution_id) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill execution is stale");
    }
    if (session.state == SkillSessionState::Closed) {
        return core::Status::Ok();
    }
    if (session.state != SkillSessionState::Closing) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill session is not closing");
    }
    session.state = SkillSessionState::Closed;
    session.status_text = status_text.empty() ? "skill session closed" : std::move(status_text);
    session.trace_id = trace_id.empty() ? session.trace_id : std::string(trace_id);
    session.last_activity_at = now;
    return core::Status::Ok();
}

core::Status SkillSessionManager::RecordObservation(const SkillObservation& observation) {
    if (EmptyId(observation.session_id) || EmptyId(observation.skill_id)) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "skill_id and session_id are required");
    }
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(Key(observation.session_id, observation.skill_id));
    if (it == sessions_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "skill session not found");
    }
    auto& session = it->second;
    if (!observation.execution_id.empty() && observation.execution_id != session.execution_id) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill execution is stale");
    }
    if (Terminal(session.state)) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill session is terminal");
    }
    if (session.state != SkillSessionState::Closing) {
        session.state = SkillSessionState::Running;
    }
    session.trace_id = observation.trace_id.empty() ? session.trace_id : observation.trace_id;
    session.last_activity_at = now;
    if (!observation.stale && observation.should_inject_prompt) {
        session.last_observation = observation.summary;
    }
    session.recent_observations.push_back(observation);
    while (session.recent_observations.size() > options_.max_recent_observations) {
        session.recent_observations.erase(session.recent_observations.begin());
    }
    return core::Status::Ok();
}

std::size_t SkillSessionManager::CleanupExpired(std::stop_token stop_token) {
    const auto now = std::chrono::steady_clock::now();
    std::size_t expired = 0;
    std::lock_guard lock(mutex_);
    for (auto& [_, session] : sessions_) {
        if (stop_token.stop_requested()) {
            break;
        }
        if (Terminal(session.state)) {
            continue;
        }
        if (session.state == SkillSessionState::Starting && now - session.started_at > options_.startup_timeout) {
            ExpireLocked(session, "startup_timeout", now);
            ++expired;
            continue;
        }
        if ((session.state == SkillSessionState::Ready ||
             session.state == SkillSessionState::Running ||
             session.state == SkillSessionState::WaitingInput) &&
            now - session.started_at > session.max_duration) {
            ExpireLocked(session, "max_duration", now);
            ++expired;
            continue;
        }
        if ((session.state == SkillSessionState::Ready ||
             session.state == SkillSessionState::Running ||
             session.state == SkillSessionState::WaitingInput) &&
            now - session.last_activity_at > options_.idle_timeout) {
            ExpireLocked(session, "idle_timeout", now);
            ++expired;
            continue;
        }
        if (session.state == SkillSessionState::Closing &&
            now - session.closing_started_at > options_.closing_timeout) {
            ExpireLocked(session, "closing_timeout", now);
            ++expired;
        }
    }
    return expired;
}

std::string SkillSessionManager::Key(std::string_view session_id, std::string_view skill_id) {
    return std::string(session_id) + "\n" + std::string(skill_id);
}

std::string SkillSessionManager::StateName(SkillSessionState state) {
    switch (state) {
    case SkillSessionState::Idle:
        return "idle";
    case SkillSessionState::Starting:
        return "starting";
    case SkillSessionState::Ready:
        return "ready";
    case SkillSessionState::Running:
        return "running";
    case SkillSessionState::WaitingInput:
        return "waiting_input";
    case SkillSessionState::Closing:
        return "closing";
    case SkillSessionState::Closed:
        return "closed";
    case SkillSessionState::Failed:
        return "failed";
    case SkillSessionState::Expired:
        return "expired";
    }
    return "unknown";
}

bool SkillSessionManager::Terminal(SkillSessionState state) noexcept {
    return state == SkillSessionState::Closed ||
           state == SkillSessionState::Failed ||
           state == SkillSessionState::Expired;
}

SkillSessionSnapshot SkillSessionManager::SnapshotLocked(const SkillSessionSnapshot& session) const {
    return session;
}

void SkillSessionManager::ExpireLocked(SkillSessionSnapshot& session,
                                       std::string reason,
                                       std::chrono::steady_clock::time_point now) {
    session.state = SkillSessionState::Expired;
    session.last_error = reason;
    session.status_text = "skill session expired: " + reason;
    session.last_activity_at = now;
    logger_.warn("[skill] expired skill={} session={} trace_id={} reason={}",
                 session.skill_id,
                 session.session_id,
                 session.trace_id,
                 reason);
}

} // namespace agent::service::persona
