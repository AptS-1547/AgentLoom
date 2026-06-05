#pragma once

#include "logger_adapter.h"
#include "persona_algorithm.h"
#include "result.h"
#include "thread_pool.h"
#include "trace_context.h"

#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace agent::service::persona {

enum class SessionStatus {
    Creating,
    Active,
    Disconnected,
    Closing,
    Closed,
};

struct SessionMetrics {
    std::uint64_t turn_count = 0;
    std::uint64_t request_count = 0;
    std::uint64_t failed_request_count = 0;
    std::chrono::milliseconds total_latency{0};
    std::chrono::milliseconds last_latency{0};
};

struct ConversationTurn {
    std::string user_input;
    std::string emotion = "neutral";
    double intensity = 0.0;
    std::string behavior;
    std::string tone;
    std::string response;
    std::string timestamp;
    std::string context_id;
};

struct SessionOptions {
    std::chrono::minutes idle_timeout{15};
    std::size_t max_recent_turns = 20;
};

struct CreateSessionRequest {
    std::string user_uuid;
    std::string persona_id;
    std::string session_id;
    std::string trace_id;
    PersonalityConfig personality;
    std::optional<EmotionPromptConfig> emotion_prompt_config;
    EmotionStateConfig emotion_state_config;
    bool time_awareness = true;
};

struct SessionSnapshot {
    std::string session_id;
    std::string user_uuid;
    std::string persona_id;
    std::string last_trace_id;
    SessionStatus status = SessionStatus::Active;
    std::string close_reason;
    std::chrono::steady_clock::time_point created_at{};
    std::chrono::steady_clock::time_point last_active{};
    std::size_t recent_turn_count = 0;
    EmotionState emotion_state;
    SessionMetrics metrics;
};

struct DispatchOptions {
    std::string session_id;
    std::string trace_id;
    std::string span_id;
    std::string user_uuid;
    std::string module;
    std::string operation;
};

struct SessionThreadPoolStats {
    core::ThreadPoolStats compute;
    core::ThreadPoolStats io;
};

struct SessionState {
    std::string session_id;
    std::string user_uuid;
    std::string persona_id;
    std::string last_trace_id;
    SessionStatus status = SessionStatus::Creating;
    std::string close_reason;
    std::chrono::steady_clock::time_point created_at{};
    std::chrono::steady_clock::time_point last_active{};
    std::deque<ConversationTurn> recent_history;
    EmotionStateTracker emotion_state;
    std::unique_ptr<PromptBuilder> prompt_builder;
    SessionMetrics metrics;
    std::size_t max_recent_turns = 20;
};

class ISessionManager {
public:
    virtual ~ISessionManager() = default;

    virtual core::Result<SessionSnapshot> CreateSession(CreateSessionRequest request) = 0;
    virtual core::Result<SessionSnapshot> GetSessionSnapshot(std::string_view session_id) const = 0;
    virtual core::Status CloseSession(std::string_view session_id, std::string_view trace_id = {}) = 0;
    virtual core::Status TouchSession(std::string_view session_id, std::string_view trace_id = {}) = 0;
    virtual std::vector<SessionSnapshot> CleanupExpired() = 0;
    virtual std::size_t SessionCount() const = 0;
};

class SessionManager final : public ISessionManager {
public:
    using SessionTask = std::function<core::Status(SessionState&, core::ThreadPoolContext&)>;

    SessionManager(core::ThreadPool& compute_pool,
                   core::ThreadPool& io_pool,
                   SessionOptions options = {},
                   core::LoggerAdapter logger = core::LoggerAdapter::ForModule("service"));

    core::Result<SessionSnapshot> CreateSession(CreateSessionRequest request) override;
    core::Result<SessionSnapshot> GetSessionSnapshot(std::string_view session_id) const override;
    core::Status CloseSession(std::string_view session_id, std::string_view trace_id = {}) override;
    core::Status TouchSession(std::string_view session_id, std::string_view trace_id = {}) override;
    std::vector<SessionSnapshot> CleanupExpired() override;
    std::size_t SessionCount() const override;

    core::Status AddTurn(std::string_view session_id,
                         ConversationTurn turn,
                         std::string_view trace_id = {});
    core::Status RecordRequestMetrics(std::string_view session_id,
                                      std::chrono::milliseconds latency,
                                      bool success,
                                      std::string_view trace_id = {});

    core::Status SubmitCompute(DispatchOptions options, SessionTask task);
    core::Status SubmitIo(DispatchOptions options, SessionTask task);
    SessionThreadPoolStats PoolStats() const;

private:
    struct SessionSlot {
        mutable std::mutex mutex;
        SessionState state;
    };

    core::Status Submit(core::ThreadPool& pool,
                        std::string_view pool_role,
                        DispatchOptions options,
                        SessionTask task);
    core::Result<std::shared_ptr<SessionSlot>> FindSlot(std::string_view session_id) const;
    SessionSnapshot SnapshotLocked(const SessionState& state) const;
    core::TraceContext MakeTrace(DispatchOptions options) const;
    std::string MakeSessionId();

    core::ThreadPool& compute_pool_;
    core::ThreadPool& io_pool_;
    SessionOptions options_;
    core::LoggerAdapter logger_;
    mutable std::shared_mutex sessions_mutex_;
    std::unordered_map<std::string, std::shared_ptr<SessionSlot>> sessions_;
};

} // namespace agent::service::persona
