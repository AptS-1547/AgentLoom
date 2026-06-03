#pragma once

#include "logger_adapter.h"
#include "result.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace agent::service::gateway {

enum class ProactiveLevel {
    Off,
    Low,
    Medium,
};

enum class ProactiveState {
    Normal,
    Triggered,
    WaitingResponse,
    Dormant,
};

struct AgentLoopOptions {
    ProactiveLevel proactive_level = ProactiveLevel::Off;
    std::chrono::seconds idle_threshold{300};
    std::chrono::seconds proactive_interval{30};
    std::chrono::milliseconds tick_interval{2000};
    std::size_t max_concurrent_chats = 3;
    bool time_awareness = true;
};

struct ProactiveOptions {
    bool enabled = false;
    double confidence_threshold = 0.6;
    std::size_t recent_turns_limit = 8;
    std::size_t l4_memory_limit = 3;
    std::chrono::duration<double, std::ratio<3600>> idle_trigger_hours{2.0};
    std::chrono::minutes response_wait{30};
    std::chrono::seconds min_interval{30};
};

struct ClassroomSchedulerOptions {
    std::size_t max_concurrent_llm = 3;
    std::chrono::seconds llm_acquire_timeout{30};
    std::chrono::seconds health_check_interval{60};
    std::string default_persona;
};

struct ClassroomPersonaRegistration {
    std::string classroom_id;
    std::string persona_id;
    std::string session_id;
    std::unordered_set<std::string> context_ids;
    std::vector<std::string> context_patterns;
    std::string config_source;
    AgentLoopOptions agent;
    ProactiveOptions proactive;
    bool default_persona = false;
};

struct ClassroomRouteRequest {
    std::string classroom_id;
    std::string context_id;
    std::string persona_hint;
    std::string session_id;
};

struct ClassroomRouteResult {
    std::string classroom_id;
    std::string persona_id;
    std::string session_id;
    std::string context_id;
    bool cache_hit = false;
    bool default_route = false;
};

struct ClassroomPollRequest {
    std::string trace_id;
    std::string classroom_id;
    std::string persona_hint;
    std::string context_id;
    bool system_event = false;
    std::string system_event_content;
};

struct ClassroomPollDecision {
    bool should_speak = false;
    std::string classroom_id;
    std::string persona_id;
    std::string session_id;
    std::string trigger;
    ProactiveState state = ProactiveState::Normal;
};

struct ClassroomHealthPersona {
    std::string persona_id;
    std::string session_id;
    std::vector<std::string> context_ids;
    std::vector<std::string> context_patterns;
    std::string config_source;
    ProactiveState proactive_state = ProactiveState::Normal;
    std::uint64_t uptime_seconds = 0;
};

struct ClassroomHealthReport {
    std::string classroom_id;
    std::vector<ClassroomHealthPersona> personas;
};

struct ProactiveDecisionRequest {
    std::string classroom_id;
    std::string persona_id;
    std::string session_id;
    std::chrono::seconds idle_time{0};
};

struct ProactiveDecision {
    bool should_respond = false;
    double confidence = 0.0;
    std::string reason;
};

class IProactiveDecisionProvider {
public:
    virtual ~IProactiveDecisionProvider() = default;
    virtual core::Result<ProactiveDecision> Decide(const ProactiveDecisionRequest& request) = 0;
};

class IClassroomScheduler {
public:
    virtual ~IClassroomScheduler() = default;
    virtual core::Status RegisterPersona(ClassroomPersonaRegistration request) = 0;
    virtual core::Status UnregisterPersona(std::string_view classroom_id, std::string_view persona_id) = 0;
    virtual core::Status UnregisterSession(std::string_view session_id) = 0;
    virtual core::Result<ClassroomRouteResult> Resolve(const ClassroomRouteRequest& request) = 0;
    virtual core::Status OnUserMessage(std::string_view classroom_id, std::string_view persona_id) = 0;
    virtual core::Status OnProactiveDelivered(std::string_view classroom_id, std::string_view persona_id) = 0;
    virtual core::Result<ClassroomPollDecision> Poll(const ClassroomPollRequest& request) = 0;
    virtual core::Result<ClassroomHealthReport> Health(std::string_view classroom_id) const = 0;
};

class ClassroomScheduler final : public IClassroomScheduler {
public:
    explicit ClassroomScheduler(ClassroomSchedulerOptions options = {},
                                core::LoggerAdapter logger = core::LoggerAdapter::ForModule("classroom"));

    void SetDecisionProvider(std::shared_ptr<IProactiveDecisionProvider> provider);

    core::Status RegisterPersona(ClassroomPersonaRegistration request) override;
    core::Status UnregisterPersona(std::string_view classroom_id, std::string_view persona_id) override;
    core::Status UnregisterSession(std::string_view session_id) override;
    core::Result<ClassroomRouteResult> Resolve(const ClassroomRouteRequest& request) override;
    core::Status OnUserMessage(std::string_view classroom_id, std::string_view persona_id) override;
    core::Status OnProactiveDelivered(std::string_view classroom_id, std::string_view persona_id) override;
    core::Result<ClassroomPollDecision> Poll(const ClassroomPollRequest& request) override;
    core::Result<ClassroomHealthReport> Health(std::string_view classroom_id) const override;

private:
    struct ManagedPersona {
        std::string persona_id;
        std::string session_id;
        std::unordered_set<std::string> context_ids;
        std::vector<std::string> context_patterns;
        std::string config_source;
        AgentLoopOptions agent;
        ProactiveOptions proactive;
        bool default_persona = false;
        ProactiveState proactive_state = ProactiveState::Normal;
        std::chrono::steady_clock::time_point started_at = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point last_user_time = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point last_proactive_time{};
        std::chrono::steady_clock::time_point last_proactive_attempt_time{};
    };

    struct ClassroomState {
        std::unordered_map<std::string, ManagedPersona> personas;
        std::unordered_map<std::string, std::string> context_cache;
        std::string default_persona;
    };

    static bool GlobMatch(std::string_view text, std::string_view pattern);
    static std::string ProactiveTrigger(std::chrono::seconds idle);
    core::Result<ClassroomRouteResult> ResolveLocked(ClassroomState& classroom,
                                                     std::string_view classroom_id,
                                                     const ClassroomRouteRequest& request);
    ManagedPersona* FindPersonaLocked(ClassroomState& classroom,
                                      std::string_view persona_id,
                                      std::string_view session_id);

    ClassroomSchedulerOptions options_;
    core::LoggerAdapter logger_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ClassroomState> classrooms_;
    std::shared_ptr<IProactiveDecisionProvider> decision_provider_;
};

std::string ToString(ProactiveLevel level);
std::string ToString(ProactiveState state);
ProactiveLevel ProactiveLevelFromString(std::string_view value);

} // namespace agent::service::gateway
