#include "session_manager.h"

#include <atomic>
#include <sstream>
#include <utility>

namespace agent::service::persona {
namespace {

std::string NonEmptyOrGeneratedTrace(std::string trace_id) {
    if (!trace_id.empty()) {
        return trace_id;
    }
    if (auto current = core::CurrentTraceId(); current != "-") {
        return std::string(current);
    }
    return core::GenerateTraceId();
}

std::string TaskName(std::string_view role, const DispatchOptions& options) {
    std::string name;
    name.reserve(role.size() + options.module.size() + options.operation.size() + 3);
    name.append(role);
    name.push_back(':');
    name.append(options.module.empty() ? "session" : options.module);
    name.push_back(':');
    name.append(options.operation.empty() ? "task" : options.operation);
    return name;
}

} // namespace

SessionManager::SessionManager(core::ThreadPool& compute_pool,
                               core::ThreadPool& io_pool,
                               SessionOptions options,
                               core::LoggerAdapter logger)
    : compute_pool_(compute_pool),
      io_pool_(io_pool),
      options_(options),
      logger_(std::move(logger)) {}

core::Result<SessionSnapshot> SessionManager::CreateSession(CreateSessionRequest request) {
    if (request.user_uuid.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "user_uuid is required");
    }
    if (request.persona_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "persona_id is required");
    }
    if (request.personality.name.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "personality.name is required");
    }

    core::TraceContext trace;
    trace.trace_id = NonEmptyOrGeneratedTrace(std::move(request.trace_id));
    trace.user_uuid = request.user_uuid;
    core::TraceScope trace_scope(trace);

    auto slot = std::make_shared<SessionSlot>();
    {
        std::lock_guard lock(slot->mutex);
        auto now = std::chrono::steady_clock::now();
        slot->state.session_id = request.session_id.empty() ? MakeSessionId() : std::move(request.session_id);
        slot->state.tenant_id = request.tenant_id.empty() ? "default" : std::move(request.tenant_id);
        slot->state.user_uuid = std::move(request.user_uuid);
        slot->state.persona_id = std::move(request.persona_id);
        slot->state.last_trace_id = trace.trace_id;
        slot->state.status = SessionStatus::Active;
        slot->state.created_at = now;
        slot->state.last_active = now;
        slot->state.emotion_state = EmotionStateTracker(request.emotion_state_config);
        slot->state.personality = std::make_shared<const PersonalityConfig>(request.personality);
        slot->state.prompt_builder = std::make_unique<PromptBuilder>(
            std::move(request.personality),
            std::move(request.emotion_prompt_config),
            request.time_awareness);
        slot->state.max_recent_turns = options_.max_recent_turns;
    }

    SessionSnapshot snapshot;
    {
        std::unique_lock sessions_lock(sessions_mutex_);
        const auto id = slot->state.session_id;
        if (sessions_.find(id) != sessions_.end()) {
            return core::Status::Error(core::ErrorCode::AlreadyExists, "session already exists");
        }
        if (sessions_.size() >= options_.max_active_sessions) {
            logger_.warn("[trace={}] [session] active session limit reached active={} limit={}",
                         core::CurrentTraceId(),
                         sessions_.size(),
                         options_.max_active_sessions);
            return core::Status::Error(core::ErrorCode::ResourceExhausted,
                                       "active session limit reached");
        }
        {
            std::lock_guard slot_lock(slot->mutex);
            snapshot = SnapshotLocked(slot->state);
        }
        sessions_[id] = std::move(slot);
    }

    logger_.info("[trace={}] [session] created session={} user={} persona={}",
                 core::CurrentTraceId(),
                 snapshot.session_id,
                 snapshot.user_uuid,
                 snapshot.persona_id);
    return snapshot;
}

core::Result<SessionSnapshot> SessionManager::GetSessionSnapshot(std::string_view session_id) const {
    auto slot = FindSlot(session_id);
    if (!slot.ok()) {
        return slot.status();
    }
    std::lock_guard lock(slot.value()->mutex);
    return SnapshotLocked(slot.value()->state);
}

core::Status SessionManager::CloseSession(std::string_view session_id, std::string_view trace_id) {
    core::TraceContext trace;
    trace.trace_id = NonEmptyOrGeneratedTrace(std::string(trace_id));
    core::TraceScope trace_scope(trace);

    std::shared_ptr<SessionSlot> removed;
    {
        std::unique_lock lock(sessions_mutex_);
        auto it = sessions_.find(std::string(session_id));
        if (it == sessions_.end()) {
            return core::Status::Error(core::ErrorCode::NotFound, "session not found");
        }
        removed = std::move(it->second);
        {
            std::lock_guard slot_lock(removed->mutex);
            if (removed->state.status == SessionStatus::Closing ||
                removed->state.status == SessionStatus::Closed) {
                return core::Status::Ok();
            }
            removed->state.status = SessionStatus::Closing;
            removed->state.close_reason = "client_close";
            removed->state.last_trace_id = trace.trace_id;
        }
        sessions_.erase(it);
    }

    SessionSnapshot snapshot;
    {
        std::lock_guard lock(removed->mutex);
        removed->state.status = SessionStatus::Closed;
        snapshot = SnapshotLocked(removed->state);
    }
    if (session_closed_callback_) {
        session_closed_callback_(snapshot);
    }

    logger_.info("[trace={}] [session] closed session={} user={} persona={}",
                 core::CurrentTraceId(),
                 session_id,
                 snapshot.user_uuid,
                 snapshot.persona_id);
    return core::Status::Ok();
}

core::Status SessionManager::TouchSession(std::string_view session_id, std::string_view trace_id) {
    auto slot = FindSlot(session_id);
    if (!slot.ok()) {
        return slot.status();
    }
    const auto trace = NonEmptyOrGeneratedTrace(std::string(trace_id));
    {
        std::lock_guard lock(slot.value()->mutex);
        if (slot.value()->state.status != SessionStatus::Active) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "session is not active");
        }
        slot.value()->state.last_active = std::chrono::steady_clock::now();
        slot.value()->state.last_trace_id = trace;
    }
    logger_.debug("[trace={}] [session] touched session={}", trace, session_id);
    return core::Status::Ok();
}

std::vector<SessionSnapshot> SessionManager::CleanupExpired() {
    const auto now = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<SessionSlot>> expired;
    {
        std::unique_lock sessions_lock(sessions_mutex_);
        for (auto it = sessions_.begin(); it != sessions_.end();) {
            bool remove = false;
            {
                std::lock_guard slot_lock(it->second->mutex);
                remove = it->second->state.status == SessionStatus::Active &&
                         now - it->second->state.last_active > options_.idle_timeout;
                if (remove) {
                    it->second->state.status = SessionStatus::Closing;
                    it->second->state.close_reason = "idle_timeout";
                }
            }
            if (remove) {
                expired.push_back(std::move(it->second));
                it = sessions_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::vector<SessionSnapshot> snapshots;
    snapshots.reserve(expired.size());
    for (const auto& slot : expired) {
        SessionSnapshot snapshot;
        {
            std::lock_guard lock(slot->mutex);
            slot->state.status = SessionStatus::Closed;
            snapshot = SnapshotLocked(slot->state);
        }
        snapshots.push_back(snapshot);
        logger_.info("[trace={}] [session] expired session={} user={} persona={}",
                     snapshot.last_trace_id.empty() ? "-" : snapshot.last_trace_id,
                     snapshot.session_id,
                     snapshot.user_uuid,
                     snapshot.persona_id);
        if (session_closed_callback_) {
            session_closed_callback_(snapshot);
        }
    }
    return snapshots;
}

std::size_t SessionManager::SessionCount() const {
    std::shared_lock lock(sessions_mutex_);
    return sessions_.size();
}

core::Status SessionManager::AddTurn(std::string_view session_id,
                                     ConversationTurn turn,
                                     std::string_view trace_id) {
    auto slot = FindSlot(session_id);
    if (!slot.ok()) {
        return slot.status();
    }

    std::lock_guard lock(slot.value()->mutex);
    auto& state = slot.value()->state;
    if (state.status != SessionStatus::Active) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "session is not active");
    }
    state.last_active = std::chrono::steady_clock::now();
    state.last_trace_id = NonEmptyOrGeneratedTrace(std::string(trace_id));
    state.recent_history.push_back(std::move(turn));
    while (state.recent_history.size() > state.max_recent_turns) {
        state.recent_history.pop_front();
    }
    return core::Status::Ok();
}

core::Status SessionManager::RecordRequestMetrics(std::string_view session_id,
                                                  std::chrono::milliseconds latency,
                                                  bool success,
                                                  std::string_view trace_id) {
    auto slot = FindSlot(session_id);
    if (!slot.ok()) {
        return slot.status();
    }
    std::lock_guard lock(slot.value()->mutex);
    auto& metrics = slot.value()->state.metrics;
    ++metrics.request_count;
    if (!success) {
        ++metrics.failed_request_count;
    }
    metrics.last_latency = latency;
    metrics.total_latency += latency;
    if (!trace_id.empty()) {
        slot.value()->state.last_trace_id = std::string(trace_id);
    }
    return core::Status::Ok();
}

core::Status SessionManager::SubmitCompute(DispatchOptions options, SessionTask task) {
    return Submit(compute_pool_, "compute", std::move(options), std::move(task));
}

core::Status SessionManager::SubmitIo(DispatchOptions options, SessionTask task) {
    return Submit(io_pool_, "io", std::move(options), std::move(task));
}

SessionThreadPoolStats SessionManager::PoolStats() const {
    SessionThreadPoolStats stats;
    stats.compute = compute_pool_.Stats();
    stats.io = io_pool_.Stats();
    return stats;
}

void SessionManager::SetSessionClosedCallback(SessionClosedCallback callback) {
    session_closed_callback_ = std::move(callback);
}

core::Status SessionManager::Submit(core::ThreadPool& pool,
                                    std::string_view pool_role,
                                    DispatchOptions options,
                                    SessionTask task) {
    if (!task) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "session task is empty");
    }
    if (options.session_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "session_id is required");
    }

    const std::string task_session_id = options.session_id;
    const std::string task_name = TaskName(pool_role, options);
    auto trace = MakeTrace(options);
    const std::string trace_id = trace.trace_id;
    std::string trusted_tenant_id;
    std::string trusted_user_uuid;
    auto admission_slot = FindSlot(task_session_id);
    if (!admission_slot.ok()) {
        return admission_slot.status();
    }
    {
        std::lock_guard slot_lock(admission_slot.value()->mutex);
        if (admission_slot.value()->state.status != SessionStatus::Active) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "session is not active");
        }
        trusted_tenant_id = admission_slot.value()->state.tenant_id;
        trusted_user_uuid = admission_slot.value()->state.user_uuid;
    }
    const std::string user_uuid = trusted_user_uuid;
    const std::string module = options.module.empty() ? "session" : options.module;
    const std::string operation = options.operation.empty() ? "task" : options.operation;
    core::ThreadPoolTaskMetadata scheduler_metadata;
    scheduler_metadata.concurrency_key = task_session_id;
    scheduler_metadata.tenant_key = trusted_tenant_id;
    if (!trusted_user_uuid.empty()) {
        scheduler_metadata.fairness_key = trusted_tenant_id + ":" + trusted_user_uuid;
    }
    scheduler_metadata.execution_id = options.span_id;

    core::TraceScope trace_scope(trace);
    logger_.info("[trace={}] [session] submit pool={} module={} op={} session={}",
                 core::CurrentTraceId(),
                 pool_role,
                 module,
                 operation,
                 task_session_id);

    auto status = pool.Submit(
        [this,
         task_session_id,
         task = std::move(task),
         pool_role = std::string(pool_role),
         module,
         operation,
         trace_id,
         user_uuid](core::ThreadPoolContext& context) mutable -> core::Status {
            core::TraceContext task_trace;
            task_trace.trace_id = trace_id;
            task_trace.user_uuid = user_uuid;
            core::TraceScope task_trace_scope(task_trace);

            auto slot = FindSlot(task_session_id);
            if (!slot.ok()) {
                logger_.warn("[trace={}] [session] dispatch dropped pool={} module={} op={} session={} reason={}",
                             core::CurrentTraceId(),
                             pool_role,
                             module,
                             operation,
                             task_session_id,
                             slot.status().message());
                return slot.status();
            }

            std::lock_guard lock(slot.value()->mutex);
            if (slot.value()->state.status != SessionStatus::Active) {
                return core::Status::Error(core::ErrorCode::FailedPrecondition, "session is not active");
            }
            slot.value()->state.last_active = std::chrono::steady_clock::now();
            slot.value()->state.last_trace_id = trace_id;

            logger_.debug("[trace={}] [session] dispatch start pool={} module={} op={} session={} worker={}",
                          core::CurrentTraceId(),
                          pool_role,
                          module,
                          operation,
                          task_session_id,
                          context.worker_index());
            auto task_status = task(slot.value()->state, context);
            if (!task_status.ok()) {
                logger_.warn("[trace={}] [session] dispatch failed pool={} module={} op={} session={} code={} reason={}",
                             core::CurrentTraceId(),
                             pool_role,
                             module,
                             operation,
                             task_session_id,
                             static_cast<int>(task_status.code()),
                             task_status.message());
            } else {
                logger_.debug("[trace={}] [session] dispatch done pool={} module={} op={} session={}",
                              core::CurrentTraceId(),
                              pool_role,
                              module,
                              operation,
                              task_session_id);
            }
            return task_status;
        },
        {},
        task_name,
        std::move(scheduler_metadata));

    if (!status.ok()) {
        logger_.warn("[trace={}] [session] submit rejected pool={} module={} op={} session={} reason={}",
                     core::CurrentTraceId(),
                     pool_role,
                     module,
                     operation,
                     task_session_id,
                     status.message());
    }
    return status;
}

core::Result<std::shared_ptr<SessionManager::SessionSlot>> SessionManager::FindSlot(std::string_view session_id) const {
    std::shared_lock lock(sessions_mutex_);
    auto it = sessions_.find(std::string(session_id));
    if (it == sessions_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "session not found");
    }
    return it->second;
}

SessionSnapshot SessionManager::SnapshotLocked(const SessionState& state) const {
    SessionSnapshot snapshot;
    snapshot.session_id = state.session_id;
    snapshot.tenant_id = state.tenant_id;
    snapshot.user_uuid = state.user_uuid;
    snapshot.persona_id = state.persona_id;
    snapshot.last_trace_id = state.last_trace_id;
    snapshot.status = state.status;
    snapshot.close_reason = state.close_reason;
    snapshot.created_at = state.created_at;
    snapshot.last_active = state.last_active;
    snapshot.recent_turn_count = state.recent_history.size();
    snapshot.emotion_state = state.emotion_state.state();
    snapshot.metrics = state.metrics;
    return snapshot;
}

core::TraceContext SessionManager::MakeTrace(DispatchOptions options) const {
    core::TraceContext trace;
    trace.trace_id = NonEmptyOrGeneratedTrace(std::move(options.trace_id));
    trace.span_id = std::move(options.span_id);
    trace.user_uuid = std::move(options.user_uuid);
    return trace;
}

std::string SessionManager::MakeSessionId() {
    static std::atomic<std::uint64_t> counter{1};
    std::ostringstream oss;
    oss << "sess-" << core::GenerateTraceId().substr(0, 16)
        << "-" << counter.fetch_add(1, std::memory_order_relaxed);
    return oss.str();
}

} // namespace agent::service::persona
