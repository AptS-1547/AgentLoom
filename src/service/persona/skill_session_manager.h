#pragma once

#include "logger_adapter.h"
#include "result.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace agent::service::persona {

enum class SkillSessionState {
    Idle,
    Starting,
    Ready,
    Running,
    WaitingInput,
    Closing,
    Closed,
    Failed,
    Expired
};

struct SkillSessionOptions {
    std::chrono::milliseconds startup_timeout{std::chrono::seconds(15)};
    std::chrono::milliseconds max_duration{std::chrono::minutes(2)};
    std::chrono::milliseconds idle_timeout{std::chrono::seconds(60)};
    std::chrono::milliseconds closing_timeout{std::chrono::seconds(10)};
    std::size_t max_recent_observations = 8;
};

struct SkillSessionStartRequest {
    std::string skill_id;
    std::string session_id;
    std::string user_uuid;
    std::string persona_id;
    std::string trace_id;
    std::string source;
    std::string reason;
    std::string arguments_json = "{}";
    std::chrono::milliseconds max_duration{0};
};

struct SkillSessionStopRequest {
    std::string skill_id;
    std::string session_id;
    std::string trace_id;
    std::string source;
    std::string reason;
    bool summarize = true;
    bool write_l3 = false;
};

struct SkillObservation {
    std::string skill_id;
    std::string session_id;
    std::string trace_id;
    std::string summary;
    double confidence = 0.0;
    bool stale = false;
    bool should_inject_prompt = false;
    std::string source;
    std::string metadata_json = "{}";
};

struct SkillSessionSnapshot {
    std::string skill_id;
    std::string session_id;
    std::string user_uuid;
    std::string persona_id;
    std::string trace_id;
    SkillSessionState state = SkillSessionState::Idle;
    std::string status_text;
    std::string last_observation;
    std::string last_error;
    std::string close_reason;
    std::vector<SkillObservation> recent_observations;
    std::chrono::steady_clock::time_point started_at{};
    std::chrono::steady_clock::time_point last_activity_at{};
    std::chrono::steady_clock::time_point closing_started_at{};
    std::chrono::milliseconds max_duration{0};
};

class ISkillSessionManager {
public:
    virtual ~ISkillSessionManager() = default;

    virtual core::Result<SkillSessionSnapshot> Start(const SkillSessionStartRequest& request) = 0;
    virtual core::Result<SkillSessionSnapshot> Stop(const SkillSessionStopRequest& request) = 0;
    virtual core::Result<std::optional<SkillSessionSnapshot>> Get(std::string_view session_id,
                                                                  std::string_view skill_id) const = 0;
    virtual core::Status MarkReady(std::string_view session_id,
                                   std::string_view skill_id,
                                   std::string status_text,
                                   std::string_view trace_id) = 0;
    virtual core::Status MarkFailed(std::string_view session_id,
                                    std::string_view skill_id,
                                    std::string error,
                                    std::string_view trace_id) = 0;
    virtual core::Status RecordObservation(const SkillObservation& observation) = 0;
    virtual std::size_t CleanupExpired(std::stop_token stop_token) = 0;
};

class ISkillObservationSink {
public:
    virtual ~ISkillObservationSink() = default;
    virtual core::Status Publish(SkillObservation observation) = 0;
};

class SkillSessionObservationSink final : public ISkillObservationSink {
public:
    explicit SkillSessionObservationSink(std::shared_ptr<ISkillSessionManager> manager);
    core::Status Publish(SkillObservation observation) override;

private:
    std::shared_ptr<ISkillSessionManager> manager_;
};

class SkillSessionManager final : public ISkillSessionManager,
                                  public std::enable_shared_from_this<SkillSessionManager> {
public:
    explicit SkillSessionManager(SkillSessionOptions options = {},
                                 core::LoggerAdapter logger = core::LoggerAdapter::ForModule("skill"));

    core::Result<SkillSessionSnapshot> Start(const SkillSessionStartRequest& request) override;
    core::Result<SkillSessionSnapshot> Stop(const SkillSessionStopRequest& request) override;
    core::Result<std::optional<SkillSessionSnapshot>> Get(std::string_view session_id,
                                                          std::string_view skill_id) const override;
    core::Status MarkReady(std::string_view session_id,
                           std::string_view skill_id,
                           std::string status_text,
                           std::string_view trace_id) override;
    core::Status MarkFailed(std::string_view session_id,
                            std::string_view skill_id,
                            std::string error,
                            std::string_view trace_id) override;
    core::Status RecordObservation(const SkillObservation& observation) override;
    std::size_t CleanupExpired(std::stop_token stop_token) override;

private:
    static std::string Key(std::string_view session_id, std::string_view skill_id);
    static std::string StateName(SkillSessionState state);
    static bool Terminal(SkillSessionState state) noexcept;

    SkillSessionSnapshot SnapshotLocked(const SkillSessionSnapshot& session) const;
    void ExpireLocked(SkillSessionSnapshot& session, std::string reason, std::chrono::steady_clock::time_point now);

    SkillSessionOptions options_;
    core::LoggerAdapter logger_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, SkillSessionSnapshot> sessions_;
};

} // namespace agent::service::persona
