#include "webrtc_session_registry.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <random>
#include <sstream>
#include <utility>

namespace media {
namespace {

using Json = nlohmann::json;

std::int64_t NowUnixMs() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

std::string GenerateReconnectToken() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::ostringstream out;
    out << std::hex << rng() << rng();
    return out.str();
}

WebRtcSessionCheckpoint FromJson(const std::string& text) {
    const auto body = Json::parse(text);
    WebRtcSessionCheckpoint checkpoint;
    checkpoint.session_id = body.value("session_id", "");
    checkpoint.trace_id = body.value("trace_id", "");
    checkpoint.state = body.value("state", "");
    checkpoint.connection_id = body.value("connection_id", 0ULL);
    checkpoint.last_frame_id = body.value("last_frame_id", 0ULL);
    checkpoint.created_at_ms = body.value("created_at_ms", 0LL);
    checkpoint.updated_at_ms = body.value("updated_at_ms", 0LL);
    checkpoint.last_signaling_at_ms = body.value("last_signaling_at_ms", 0LL);
    checkpoint.last_ice_at_ms = body.value("last_ice_at_ms", 0LL);
    checkpoint.last_frame_at_ms = body.value("last_frame_at_ms", 0LL);
    checkpoint.reconnect_token = body.value("reconnect_token", "");
    checkpoint.failure_reason = body.value("failure_reason", "");
    return checkpoint;
}

std::string ToJson(const WebRtcSessionCheckpoint& checkpoint) {
    return Json{
        {"session_id", checkpoint.session_id},
        {"trace_id", checkpoint.trace_id},
        {"state", checkpoint.state},
        {"connection_id", checkpoint.connection_id},
        {"last_frame_id", checkpoint.last_frame_id},
        {"created_at_ms", checkpoint.created_at_ms},
        {"updated_at_ms", checkpoint.updated_at_ms},
        {"last_signaling_at_ms", checkpoint.last_signaling_at_ms},
        {"last_ice_at_ms", checkpoint.last_ice_at_ms},
        {"last_frame_at_ms", checkpoint.last_frame_at_ms},
        {"reconnect_token", checkpoint.reconnect_token},
        {"failure_reason", checkpoint.failure_reason},
    }.dump();
}

} // namespace

RedisWebRtcCheckpointStore::RedisWebRtcCheckpointStore(
    std::shared_ptr<agent::semantic_cache::RedisConnectionPool> redis,
    std::string key_prefix)
    : redis_(std::move(redis)), key_prefix_(std::move(key_prefix)) {}

core::Status RedisWebRtcCheckpointStore::Save(const WebRtcSessionCheckpoint& checkpoint, std::chrono::seconds ttl) {
    if (!redis_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "redis checkpoint store is not configured");
    }
    return redis_->Set(Key(checkpoint.session_id), ToJson(checkpoint), ttl);
}

core::Result<WebRtcSessionCheckpoint> RedisWebRtcCheckpointStore::Load(const std::string& session_id) {
    if (!redis_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "redis checkpoint store is not configured");
    }
    auto value = redis_->Get(Key(session_id));
    if (!value.ok()) {
        return value.status();
    }
    try {
        return FromJson(value.value());
    } catch (const std::exception& ex) {
        return core::Status::Error(core::ErrorCode::InternalError, ex.what());
    }
}

core::Status RedisWebRtcCheckpointStore::Remove(const std::string& session_id) {
    if (!redis_) {
        return core::Status::Ok();
    }
    auto removed = redis_->Del({Key(session_id)});
    return removed.ok() ? core::Status::Ok() : removed.status();
}

std::string RedisWebRtcCheckpointStore::Key(const std::string& session_id) const {
    return key_prefix_ + session_id;
}

WebRtcSessionRegistry::WebRtcSessionRegistry(WebRtcSessionRegistryOptions options,
                                             WebRtcPeerFactory peer_factory,
                                             std::shared_ptr<IWebRtcCheckpointStore> checkpoint_store,
                                             core::LoggerAdapter logger)
    : options_(options),
      peer_factory_(std::move(peer_factory)),
      checkpoint_store_(std::move(checkpoint_store)),
      logger_(std::move(logger)) {}

core::Result<std::shared_ptr<WebRtcSessionRecord>> WebRtcSessionRegistry::GetOrCreate(
    const std::string& session_id,
    const std::string& trace_id,
    std::uint64_t connection_id) {
    if (session_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "session_id is empty");
    }

    std::shared_ptr<WebRtcSessionRecord> record;
    {
        std::lock_guard lock(mutex_);
        auto it = sessions_.find(session_id);
        if (it != sessions_.end()) {
            record = it->second;
            record->connection_id = connection_id;
            record->trace_id = trace_id.empty() ? record->trace_id : trace_id;
            record->last_signaling_at = std::chrono::steady_clock::now();
            record->state = WebRtcSessionState::Signaling;
            return record;
        }
        if (sessions_.size() >= options_.max_sessions) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "too many WebRTC signaling sessions");
        }
    }

    auto peer = peer_factory_(session_id, "agent-webrtc-" + session_id);
    if (!peer.ok()) {
        return peer.status();
    }

    std::optional<WebRtcSessionCheckpoint> restored;
    if (checkpoint_store_) {
        auto checkpoint = checkpoint_store_->Load(session_id);
        if (checkpoint.ok() && checkpoint.value().session_id == session_id) {
            restored = std::move(checkpoint).value();
        } else if (!checkpoint.ok() && checkpoint.status().code() != core::ErrorCode::NotFound) {
            logger_.warn("[rtc] checkpoint restore skipped session_id={} reason={}",
                         session_id,
                         checkpoint.status().message());
        }
    }

    record = std::make_shared<WebRtcSessionRecord>();
    record->session_id = session_id;
    record->trace_id = trace_id;
    record->connection_id = connection_id;
    record->peer = std::move(peer).value();
    record->reconnect_token = GenerateReconnectToken();
    if (restored.has_value()) {
        record->trace_id = trace_id.empty() ? restored->trace_id : trace_id;
        record->state = ParseState(restored->state);
        record->last_frame_id = restored->last_frame_id;
        record->reconnect_token = restored->reconnect_token.empty() ? record->reconnect_token : restored->reconnect_token;
        if (restored->last_ice_at_ms > 0) {
            record->last_ice_at = std::chrono::steady_clock::now();
        }
        if (restored->last_frame_at_ms > 0) {
            record->last_frame_at = std::chrono::steady_clock::now();
        }
        if (record->state == WebRtcSessionState::Closed || record->state == WebRtcSessionState::Failed) {
            record->state = WebRtcSessionState::Reconnecting;
        }
    }

    {
        std::lock_guard lock(mutex_);
        sessions_.emplace(session_id, record);
    }
    static_cast<void>(SaveCheckpoint(*record));
    logger_.info("[rtc] session created trace_id={} session_id={} restored={} last_frame_id={}",
                 record->trace_id,
                 record->session_id,
                 restored.has_value(),
                 record->last_frame_id);
    return record;
}

core::Result<std::shared_ptr<WebRtcSessionRecord>> WebRtcSessionRegistry::Find(const std::string& session_id) const {
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "WebRTC signaling session not found");
    }
    return it->second;
}

core::Status WebRtcSessionRegistry::Close(const std::string& session_id) {
    std::shared_ptr<WebRtcSessionRecord> removed;
    {
        std::lock_guard lock(mutex_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return core::Status::Error(core::ErrorCode::NotFound, "WebRTC signaling session not found");
        }
        removed = it->second;
        removed->state = WebRtcSessionState::Closed;
        sessions_.erase(it);
    }
    static_cast<void>(SaveCheckpoint(*removed));
    logger_.info("[rtc] session closed trace_id={} session_id={}", removed->trace_id, removed->session_id);
    return core::Status::Ok();
}

void WebRtcSessionRegistry::RemoveConnection(std::uint64_t connection_id) {
    std::lock_guard lock(mutex_);
    for (auto& [_, record] : sessions_) {
        if (record->connection_id == connection_id) {
            record->state = WebRtcSessionState::Reconnecting;
            record->last_signaling_at = std::chrono::steady_clock::now();
            static_cast<void>(SaveCheckpoint(*record));
        }
    }
}

core::Status WebRtcSessionRegistry::RecordFrame(const std::string& session_id, std::uint64_t frame_id) {
    std::shared_ptr<WebRtcSessionRecord> record;
    {
        std::lock_guard lock(mutex_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return core::Status::Error(core::ErrorCode::NotFound, "WebRTC signaling session not found");
        }
        record = it->second;
        record->state = WebRtcSessionState::Connected;
        record->last_frame_id = frame_id;
        record->last_frame_at = std::chrono::steady_clock::now();
        record->last_signaling_at = record->last_frame_at;
    }
    return SaveCheckpoint(*record);
}

core::Status WebRtcSessionRegistry::MarkFailed(const std::string& session_id, const std::string& reason) {
    std::shared_ptr<WebRtcSessionRecord> record;
    {
        std::lock_guard lock(mutex_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return core::Status::Error(core::ErrorCode::NotFound, "WebRTC signaling session not found");
        }
        record = it->second;
        record->state = WebRtcSessionState::Failed;
        record->last_signaling_at = std::chrono::steady_clock::now();
    }
    auto status = SaveCheckpoint(*record);
    logger_.error("[rtc] session failed trace_id={} session_id={} reason={}",
                  record->trace_id,
                  record->session_id,
                  reason);
    return status;
}

core::Result<WebRtcSessionCheckpoint> WebRtcSessionRegistry::ValidateResume(
    const std::string& session_id,
    const std::string& reconnect_token,
    std::uint64_t connection_id,
    const std::string& trace_id) {
    if (session_id.empty() || reconnect_token.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "session_id or reconnect_token is empty");
    }

    std::shared_ptr<WebRtcSessionRecord> record;
    {
        std::lock_guard lock(mutex_);
        auto it = sessions_.find(session_id);
        if (it != sessions_.end()) {
            record = it->second;
            if (record->reconnect_token != reconnect_token) {
                return core::Status::Error(core::ErrorCode::FailedPrecondition, "reconnect_token mismatch");
            }
            record->connection_id = connection_id;
            record->trace_id = trace_id.empty() ? record->trace_id : trace_id;
            record->state = WebRtcSessionState::Reconnecting;
            record->last_signaling_at = std::chrono::steady_clock::now();
        }
    }
    if (record) {
        auto status = SaveCheckpoint(*record);
        if (!status.ok()) {
            return status;
        }
        return ToCheckpoint(*record);
    }

    if (!checkpoint_store_) {
        return core::Status::Error(core::ErrorCode::NotFound, "WebRTC checkpoint store is not configured");
    }
    auto loaded = checkpoint_store_->Load(session_id);
    if (!loaded.ok()) {
        return loaded.status();
    }
    auto checkpoint = std::move(loaded).value();
    if (checkpoint.reconnect_token != reconnect_token) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "reconnect_token mismatch");
    }
    checkpoint.connection_id = connection_id;
    checkpoint.trace_id = trace_id.empty() ? checkpoint.trace_id : trace_id;
    checkpoint.state = "reconnecting";
    checkpoint.updated_at_ms = NowUnixMs();
    checkpoint.last_signaling_at_ms = checkpoint.updated_at_ms;
    auto saved = checkpoint_store_->Save(checkpoint, options_.checkpoint_ttl);
    if (!saved.ok()) {
        return saved;
    }
    return checkpoint;
}

std::vector<WebRtcSessionCheckpoint> WebRtcSessionRegistry::SnapshotCheckpoints() const {
    std::vector<WebRtcSessionCheckpoint> snapshots;
    std::lock_guard lock(mutex_);
    snapshots.reserve(sessions_.size());
    for (const auto& [_, record] : sessions_) {
        snapshots.push_back(ToCheckpoint(*record));
    }
    return snapshots;
}

std::size_t WebRtcSessionRegistry::SessionCount() const {
    std::lock_guard lock(mutex_);
    return sessions_.size();
}

std::size_t WebRtcSessionRegistry::CleanupExpired(std::stop_token stop_token) {
    const auto now = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<WebRtcSessionRecord>> expired;
    {
        std::lock_guard lock(mutex_);
        for (auto it = sessions_.begin(); it != sessions_.end();) {
            if (stop_token.stop_requested()) {
                break;
            }
            const auto age = now - it->second->last_signaling_at;
            const bool signaling_expired = it->second->state == WebRtcSessionState::Signaling &&
                                           age > options_.signaling_timeout;
            const bool reconnect_expired = it->second->state == WebRtcSessionState::Reconnecting &&
                                           age > options_.reconnect_grace;
            if (signaling_expired || reconnect_expired) {
                it->second->state = WebRtcSessionState::Closed;
                expired.push_back(it->second);
                it = sessions_.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (const auto& record : expired) {
        static_cast<void>(SaveCheckpoint(*record));
        logger_.info("[rtc] session cleanup trace_id={} session_id={} state={}",
                     record->trace_id,
                     record->session_id,
                     StateName(record->state));
    }
    return expired.size();
}

core::Status WebRtcSessionRegistry::SaveCheckpoint(const WebRtcSessionRecord& record) {
    if (!checkpoint_store_) {
        return core::Status::Ok();
    }
    const auto now_ms = NowUnixMs();
    auto checkpoint = ToCheckpoint(record);
    checkpoint.created_at_ms = now_ms;
    checkpoint.updated_at_ms = now_ms;
    checkpoint.last_signaling_at_ms = now_ms;
    checkpoint.last_ice_at_ms = record.last_ice_at == std::chrono::steady_clock::time_point{} ? 0 : now_ms;
    checkpoint.last_frame_at_ms = record.last_frame_at == std::chrono::steady_clock::time_point{} ? 0 : now_ms;
    return checkpoint_store_->Save(checkpoint, options_.checkpoint_ttl);
}

std::string WebRtcSessionRegistry::StateName(WebRtcSessionState state) {
    switch (state) {
    case WebRtcSessionState::Signaling:
        return "signaling";
    case WebRtcSessionState::Connected:
        return "connected";
    case WebRtcSessionState::Reconnecting:
        return "reconnecting";
    case WebRtcSessionState::Closed:
        return "closed";
    case WebRtcSessionState::Failed:
        return "failed";
    }
    return "unknown";
}

WebRtcSessionState WebRtcSessionRegistry::ParseState(std::string_view state) {
    if (state == "connected") {
        return WebRtcSessionState::Connected;
    }
    if (state == "reconnecting") {
        return WebRtcSessionState::Reconnecting;
    }
    if (state == "closed") {
        return WebRtcSessionState::Closed;
    }
    if (state == "failed") {
        return WebRtcSessionState::Failed;
    }
    return WebRtcSessionState::Signaling;
}

WebRtcSessionCheckpoint WebRtcSessionRegistry::ToCheckpoint(const WebRtcSessionRecord& record) {
    const auto now_ms = NowUnixMs();
    WebRtcSessionCheckpoint checkpoint;
    checkpoint.session_id = record.session_id;
    checkpoint.trace_id = record.trace_id;
    checkpoint.state = StateName(record.state);
    checkpoint.connection_id = record.connection_id;
    checkpoint.last_frame_id = record.last_frame_id;
    checkpoint.created_at_ms = now_ms;
    checkpoint.updated_at_ms = now_ms;
    checkpoint.last_signaling_at_ms = now_ms;
    checkpoint.last_ice_at_ms = record.last_ice_at == std::chrono::steady_clock::time_point{} ? 0 : now_ms;
    checkpoint.last_frame_at_ms = record.last_frame_at == std::chrono::steady_clock::time_point{} ? 0 : now_ms;
    checkpoint.reconnect_token = record.reconnect_token;
    return checkpoint;
}

} // namespace media
