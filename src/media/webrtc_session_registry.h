#pragma once

#include "logger_adapter.h"
#include "redis_connection_pool.h"
#include "result.h"
#include "webrtc_bin.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <functional>
#include <unordered_map>
#include <vector>

namespace media {

enum class WebRtcSessionState {
    Signaling,
    Connected,
    Reconnecting,
    Closed,
    Failed
};

struct WebRtcSessionCheckpoint {
    std::string session_id;
    std::string trace_id;
    std::string state;
    std::uint64_t connection_id = 0;
    std::uint64_t last_frame_id = 0;
    std::int64_t created_at_ms = 0;
    std::int64_t updated_at_ms = 0;
    std::int64_t last_signaling_at_ms = 0;
    std::int64_t last_ice_at_ms = 0;
    std::int64_t last_frame_at_ms = 0;
    std::string reconnect_token;
    std::string failure_reason;
};

class IWebRtcCheckpointStore {
public:
    virtual ~IWebRtcCheckpointStore() = default;

    virtual core::Status Save(const WebRtcSessionCheckpoint& checkpoint, std::chrono::seconds ttl) = 0;
    virtual core::Result<WebRtcSessionCheckpoint> Load(const std::string& session_id) = 0;
    virtual core::Status Remove(const std::string& session_id) = 0;
};

class RedisWebRtcCheckpointStore final : public IWebRtcCheckpointStore {
public:
    explicit RedisWebRtcCheckpointStore(std::shared_ptr<agent::semantic_cache::RedisConnectionPool> redis,
                                        std::string key_prefix = "rtc:session:");

    core::Status Save(const WebRtcSessionCheckpoint& checkpoint, std::chrono::seconds ttl) override;
    core::Result<WebRtcSessionCheckpoint> Load(const std::string& session_id) override;
    core::Status Remove(const std::string& session_id) override;

private:
    std::string Key(const std::string& session_id) const;

    std::shared_ptr<agent::semantic_cache::RedisConnectionPool> redis_;
    std::string key_prefix_;
};

struct WebRtcSessionRecord {
    std::string session_id;
    std::string trace_id;
    std::uint64_t connection_id = 0;
    WebRtcSessionState state = WebRtcSessionState::Signaling;
    std::shared_ptr<IWebRtcPeer> peer;
    std::chrono::steady_clock::time_point created_at = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_signaling_at = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_ice_at = {};
    std::chrono::steady_clock::time_point last_frame_at = {};
    std::uint64_t last_frame_id = 0;
    std::string reconnect_token;
};

struct WebRtcSessionRegistryOptions {
    std::size_t max_sessions = 256;
    std::chrono::seconds checkpoint_ttl{300};
    std::chrono::seconds signaling_timeout{30};
    std::chrono::seconds reconnect_grace{30};
};

using WebRtcPeerFactory =
    std::function<core::Result<std::shared_ptr<IWebRtcPeer>>(const std::string& session_id, const std::string& name)>;

class WebRtcSessionRegistry {
public:
    WebRtcSessionRegistry(WebRtcSessionRegistryOptions options,
                          WebRtcPeerFactory peer_factory,
                          std::shared_ptr<IWebRtcCheckpointStore> checkpoint_store = nullptr,
                          core::LoggerAdapter logger = core::LoggerAdapter::ForModule("media"));

    core::Result<std::shared_ptr<WebRtcSessionRecord>> GetOrCreate(const std::string& session_id,
                                                                   const std::string& trace_id,
                                                                   std::uint64_t connection_id);
    core::Result<std::shared_ptr<WebRtcSessionRecord>> Find(const std::string& session_id) const;
    core::Status Close(const std::string& session_id);
    void RemoveConnection(std::uint64_t connection_id);
    core::Status RecordFrame(const std::string& session_id, std::uint64_t frame_id);
    core::Status MarkFailed(const std::string& session_id, const std::string& reason);
    core::Result<WebRtcSessionCheckpoint> ValidateResume(const std::string& session_id,
                                                         const std::string& reconnect_token,
                                                         std::uint64_t connection_id,
                                                         const std::string& trace_id);
    std::vector<WebRtcSessionCheckpoint> SnapshotCheckpoints() const;
    std::size_t SessionCount() const;
    std::size_t CleanupExpired(std::stop_token stop_token);
    core::Status SaveCheckpoint(const WebRtcSessionRecord& record);

private:
    static std::string StateName(WebRtcSessionState state);
    static WebRtcSessionState ParseState(std::string_view state);
    static WebRtcSessionCheckpoint ToCheckpoint(const WebRtcSessionRecord& record);

    WebRtcSessionRegistryOptions options_;
    WebRtcPeerFactory peer_factory_;
    std::shared_ptr<IWebRtcCheckpointStore> checkpoint_store_;
    core::LoggerAdapter logger_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<WebRtcSessionRecord>> sessions_;
};

} // namespace media
