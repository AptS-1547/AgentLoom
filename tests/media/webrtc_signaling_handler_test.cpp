#include "webrtc_signaling_handler.h"

#include "memory_pool.h"
#include "third_party/nlohmann/json.hpp"

#include <gtest/gtest.h>

namespace {

using Json = nlohmann::json;

class FakeWebSocketRequest final : public net::IWebSocketStreamRequest {
public:
    FakeWebSocketRequest(std::uint64_t connection_id, Json body)
        : connection_{connection_id, "127.0.0.1"} {
        handshake_.target("/ws/vision/signaling");
        message_.kind = net::WebSocketMessageKind::Text;
        const auto text = body.dump();
        auto buffer = net::SharedBuffer::Copy(pool_, text);
        EXPECT_TRUE(buffer.ok()) << buffer.status().message();
        if (buffer.ok()) {
            message_.total_bytes = buffer.value().size();
            message_.fragments.push_back(std::move(buffer).value());
        }
    }

    const net::BeastHttpRequest& handshake_request() const noexcept override {
        return handshake_;
    }

    const net::ConnectionContext& connection() const noexcept override {
        return connection_;
    }

    net::WebSocketMessage& message() noexcept override {
        return message_;
    }

    const net::WebSocketMessage& message() const noexcept override {
        return message_;
    }

    core::RawMemoryPool& memory_pool() noexcept override {
        return pool_;
    }

    core::ThreadPool* task_pool() const noexcept override {
        return nullptr;
    }

    core::Status Send(net::WebSocketFrame frame) override {
        sent_.push_back(std::string(frame.payload.view()));
        return core::Status::Ok();
    }

    void Close(net::ConnectionCloseInfo close_info) override {
        closed_ = true;
        close_info_ = std::move(close_info);
    }

    Json LastJson() const {
        EXPECT_FALSE(sent_.empty());
        return Json::parse(sent_.back());
    }

    bool closed() const noexcept {
        return closed_;
    }

private:
    core::BucketMemoryPool pool_;
    net::BeastHttpRequest handshake_;
    net::ConnectionContext connection_;
    net::WebSocketMessage message_;
    std::vector<std::string> sent_;
    bool closed_ = false;
    net::ConnectionCloseInfo close_info_;
};

class FakeWebRtcPeer final : public media::IWebRtcPeer {
public:
    core::Status SetLocalDescriptionFromSdp(media::WebRtcSdpType, std::string_view) override {
        return core::Status::Ok();
    }

    core::Status SetRemoteDescriptionFromSdp(media::WebRtcSdpType type, std::string_view sdp) override {
        remote_type = type;
        remote_sdp = std::string(sdp);
        return core::Status::Ok();
    }

    core::Result<std::string> CreateAnswer() override {
        create_answer_calls += 1;
        return std::string("v=0\r\ns=agent-answer\r\n");
    }

    core::Status AddIceCandidate(std::uint32_t mline_index, std::string_view candidate) override {
        last_mline_index = mline_index;
        last_candidate = std::string(candidate);
        return core::Status::Ok();
    }

    void SetLocalIceCandidateHandler(media::IWebRtcPeer::IceCandidateHandler handler) override {
        ice_handler = std::move(handler);
    }

    void ConfigureDecode(media::WebRtcDecodeOptions options) override {
        decode_options = std::move(options);
    }

    media::WebRtcSdpType remote_type = media::WebRtcSdpType::Offer;
    std::string remote_sdp;
    int create_answer_calls = 0;
    std::uint32_t last_mline_index = 0;
    std::string last_candidate;
    media::IWebRtcPeer::IceCandidateHandler ice_handler;
    media::WebRtcDecodeOptions decode_options;
};

class FakeCheckpointStore final : public media::IWebRtcCheckpointStore {
public:
    core::Status Save(const media::WebRtcSessionCheckpoint& checkpoint, std::chrono::seconds ttl) override {
        last_checkpoint = checkpoint;
        last_ttl = ttl;
        ++save_count;
        return core::Status::Ok();
    }

    core::Result<media::WebRtcSessionCheckpoint> Load(const std::string&) override {
        if (seed_checkpoint.has_value()) {
            return *seed_checkpoint;
        }
        if (!last_checkpoint.has_value()) {
            return core::Status::Error(core::ErrorCode::NotFound, "checkpoint not found");
        }
        return *last_checkpoint;
    }

    core::Status Remove(const std::string&) override {
        last_checkpoint.reset();
        return core::Status::Ok();
    }

    std::optional<media::WebRtcSessionCheckpoint> seed_checkpoint;
    std::optional<media::WebRtcSessionCheckpoint> last_checkpoint;
    std::chrono::seconds last_ttl{0};
    int save_count = 0;
};

} // namespace

TEST(WebRtcSignalingHandlerTest, RejectsUnknownMessageType) {
    media::WebRtcSignalingHandler handler;
    auto request = std::make_shared<FakeWebSocketRequest>(
        1,
        Json{{"type", "unknown"}, {"session_id", "vision-session"}, {"payload", Json::object()}});

    handler.Handle(request);

    auto response = request->LastJson();
    EXPECT_EQ(response["type"], "error");
    EXPECT_EQ(response["session_id"], "vision-session");
    EXPECT_EQ(response["payload"]["code"], "INVALID_ARGUMENT");
}

TEST(WebRtcSignalingHandlerTest, CreatesSessionForOfferAndRemovesOnClose) {
    auto peer = std::make_shared<FakeWebRtcPeer>();
    media::WebRtcSignalingHandler handler(
        {},
        [peer](const std::string&, const std::string&) -> core::Result<std::shared_ptr<media::IWebRtcPeer>> {
            return std::static_pointer_cast<media::IWebRtcPeer>(peer);
        });
    const std::string sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 127.0.0.1\r\n"
        "s=-\r\n"
        "t=0 0\r\n";

    auto offer = std::make_shared<FakeWebSocketRequest>(
        7,
        Json{{"type", "offer"},
             {"session_id", "vision-session"},
             {"payload", {{"sdp", sdp}}}});
    handler.Handle(offer);

    auto offer_response = offer->LastJson();
    EXPECT_EQ(offer_response["type"], "answer");
    EXPECT_EQ(offer_response["payload"]["sdp"], "v=0\r\ns=agent-answer\r\n");
    EXPECT_EQ(peer->remote_type, media::WebRtcSdpType::Offer);
    EXPECT_EQ(peer->remote_sdp, sdp);
    EXPECT_EQ(peer->create_answer_calls, 1);
    EXPECT_EQ(handler.session_count(), 1u);

    auto close = std::make_shared<FakeWebSocketRequest>(
        7,
        Json{{"type", "close"}, {"session_id", "vision-session"}});
    handler.Handle(close);

    EXPECT_TRUE(close->closed());
    EXPECT_EQ(handler.session_count(), 0u);
}

TEST(WebRtcSignalingHandlerTest, RejectsIceBeforeOffer) {
    media::WebRtcSignalingHandler handler;
    auto ice = std::make_shared<FakeWebSocketRequest>(
        9,
        Json{{"type", "ice"},
             {"session_id", "missing-session"},
             {"payload", {{"sdp_mline_index", 0}, {"candidate", "candidate:1 1 udp 1 127.0.0.1 9 typ host"}}}});

    handler.Handle(ice);

    auto response = ice->LastJson();
    EXPECT_EQ(response["type"], "error");
    EXPECT_EQ(response["payload"]["code"], "NOT_FOUND");
}

TEST(WebRtcSignalingHandlerTest, ConfigMessageReturnsIceServers) {
    media::WebRtcSignalingOptions options;
    options.ice_servers = {
        media::WebRtcIceServerConfig{.urls = {"stun:stun.example.test:3478"}},
        media::WebRtcIceServerConfig{
            .urls = {"turn:turn.example.test:3478?transport=udp"},
            .username = "user",
            .credential = "secret",
        },
    };
    media::WebRtcSignalingHandler handler(options);
    auto request = std::make_shared<FakeWebSocketRequest>(
        21,
        Json{{"type", "hello"}, {"session_id", "vision-session"}});

    handler.Handle(request);

    auto response = request->LastJson();
    EXPECT_EQ(response["type"], "config");
    EXPECT_EQ(response["session_id"], "vision-session");
    ASSERT_EQ(response["payload"]["ice_servers"].size(), 2u);
    EXPECT_EQ(response["payload"]["ice_servers"][0]["urls"][0], "stun:stun.example.test:3478");
    EXPECT_EQ(response["payload"]["ice_servers"][1]["username"], "user");
    EXPECT_EQ(response["payload"]["ice_servers"][1]["credential"], "secret");
}

TEST(WebRtcSignalingHandlerTest, ResumeRejectsBadToken) {
    auto checkpoint_store = std::make_shared<FakeCheckpointStore>();
    checkpoint_store->seed_checkpoint = media::WebRtcSessionCheckpoint{
        .session_id = "vision-session",
        .trace_id = "trace-old",
        .state = "connected",
        .connection_id = 3,
        .last_frame_id = 41,
        .reconnect_token = "resume-token-001",
    };
    auto registry = std::make_shared<media::WebRtcSessionRegistry>(
        media::WebRtcSessionRegistryOptions{},
        [](const std::string&, const std::string&) -> core::Result<std::shared_ptr<media::IWebRtcPeer>> {
            return std::static_pointer_cast<media::IWebRtcPeer>(std::make_shared<FakeWebRtcPeer>());
        },
        checkpoint_store);
    media::WebRtcSignalingHandler handler({}, registry);
    auto request = std::make_shared<FakeWebSocketRequest>(
        22,
        Json{{"type", "resume"},
             {"session_id", "vision-session"},
             {"payload", {{"reconnect_token", "bad-token"}}}});

    handler.Handle(request);

    auto response = request->LastJson();
    EXPECT_EQ(response["type"], "error");
    EXPECT_EQ(response["payload"]["code"], "FAILED_PRECONDITION");
}

TEST(WebRtcSignalingHandlerTest, ResumeAcceptsCheckpointToken) {
    auto checkpoint_store = std::make_shared<FakeCheckpointStore>();
    checkpoint_store->seed_checkpoint = media::WebRtcSessionCheckpoint{
        .session_id = "vision-session",
        .trace_id = "trace-old",
        .state = "connected",
        .connection_id = 3,
        .last_frame_id = 41,
        .reconnect_token = "resume-token-001",
    };
    auto registry = std::make_shared<media::WebRtcSessionRegistry>(
        media::WebRtcSessionRegistryOptions{.checkpoint_ttl = std::chrono::seconds(60)},
        [](const std::string&, const std::string&) -> core::Result<std::shared_ptr<media::IWebRtcPeer>> {
            return std::static_pointer_cast<media::IWebRtcPeer>(std::make_shared<FakeWebRtcPeer>());
        },
        checkpoint_store);
    media::WebRtcSignalingHandler handler({}, registry);
    auto request = std::make_shared<FakeWebSocketRequest>(
        23,
        Json{{"type", "resume"},
             {"session_id", "vision-session"},
             {"trace_id", "trace-new"},
             {"payload", {{"reconnect_token", "resume-token-001"}}}});

    handler.Handle(request);

    auto response = request->LastJson();
    EXPECT_EQ(response["type"], "resume_ack");
    EXPECT_EQ(response["payload"]["accepted"], true);
    EXPECT_EQ(response["payload"]["state"], "reconnecting");
    EXPECT_EQ(response["payload"]["last_frame_id"], 41u);
    EXPECT_EQ(response["payload"]["reconnect_token"], "resume-token-001");
    ASSERT_TRUE(checkpoint_store->last_checkpoint.has_value());
    EXPECT_EQ(checkpoint_store->last_checkpoint->connection_id, 23u);
    EXPECT_EQ(checkpoint_store->last_checkpoint->trace_id, "trace-new");
    EXPECT_EQ(checkpoint_store->last_checkpoint->state, "reconnecting");
}

TEST(WebRtcSignalingHandlerTest, RecordsFrameCheckpoint) {
    auto peer = std::make_shared<FakeWebRtcPeer>();
    auto checkpoint_store = std::make_shared<FakeCheckpointStore>();
    auto registry = std::make_shared<media::WebRtcSessionRegistry>(
        media::WebRtcSessionRegistryOptions{.checkpoint_ttl = std::chrono::seconds(42)},
        [peer](const std::string&, const std::string&) -> core::Result<std::shared_ptr<media::IWebRtcPeer>> {
            return std::static_pointer_cast<media::IWebRtcPeer>(peer);
        },
        checkpoint_store);

    media::WebRtcSignalingHandler handler({}, registry);
    const std::string sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 127.0.0.1\r\n"
        "s=-\r\n"
        "t=0 0\r\n";
    auto offer = std::make_shared<FakeWebSocketRequest>(
        11,
        Json{{"type", "offer"},
             {"session_id", "vision-session"},
             {"trace_id", "trace-rtc-001"},
             {"payload", {{"sdp", sdp}}}});
    handler.Handle(offer);

    auto status = registry->RecordFrame("vision-session", 17);

    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_TRUE(checkpoint_store->last_checkpoint.has_value());
    EXPECT_EQ(checkpoint_store->last_checkpoint->session_id, "vision-session");
    EXPECT_EQ(checkpoint_store->last_checkpoint->trace_id, "trace-rtc-001");
    EXPECT_EQ(checkpoint_store->last_checkpoint->state, "connected");
    EXPECT_EQ(checkpoint_store->last_checkpoint->last_frame_id, 17u);
    EXPECT_GT(checkpoint_store->last_checkpoint->last_frame_at_ms, 0);
    EXPECT_EQ(checkpoint_store->last_ttl, std::chrono::seconds(42));
}

TEST(WebRtcSignalingHandlerTest, MarkFailedAndSnapshotSessions) {
    auto peer = std::make_shared<FakeWebRtcPeer>();
    auto checkpoint_store = std::make_shared<FakeCheckpointStore>();
    auto registry = std::make_shared<media::WebRtcSessionRegistry>(
        media::WebRtcSessionRegistryOptions{},
        [peer](const std::string&, const std::string&) -> core::Result<std::shared_ptr<media::IWebRtcPeer>> {
            return std::static_pointer_cast<media::IWebRtcPeer>(peer);
        },
        checkpoint_store);

    auto session = registry->GetOrCreate("vision-session", "trace-001", 33);
    ASSERT_TRUE(session.ok()) << session.status().message();
    auto failed = registry->MarkFailed("vision-session", "pipeline failed");

    ASSERT_TRUE(failed.ok()) << failed.message();
    auto snapshots = registry->SnapshotCheckpoints();
    ASSERT_EQ(snapshots.size(), 1u);
    EXPECT_EQ(snapshots[0].session_id, "vision-session");
    EXPECT_EQ(snapshots[0].state, "failed");
    ASSERT_TRUE(checkpoint_store->last_checkpoint.has_value());
    EXPECT_EQ(checkpoint_store->last_checkpoint->state, "failed");
}

TEST(WebRtcSignalingHandlerTest, RestoresSessionCheckpointForReconnect) {
    auto peer = std::make_shared<FakeWebRtcPeer>();
    auto checkpoint_store = std::make_shared<FakeCheckpointStore>();
    checkpoint_store->seed_checkpoint = media::WebRtcSessionCheckpoint{
        .session_id = "vision-session",
        .trace_id = "trace-old",
        .state = "connected",
        .connection_id = 3,
        .last_frame_id = 41,
        .last_frame_at_ms = 123456,
        .reconnect_token = "resume-token-001",
    };
    auto registry = std::make_shared<media::WebRtcSessionRegistry>(
        media::WebRtcSessionRegistryOptions{.checkpoint_ttl = std::chrono::seconds(60)},
        [peer](const std::string&, const std::string&) -> core::Result<std::shared_ptr<media::IWebRtcPeer>> {
            return std::static_pointer_cast<media::IWebRtcPeer>(peer);
        },
        checkpoint_store);

    auto session = registry->GetOrCreate("vision-session", "", 9);

    ASSERT_TRUE(session.ok()) << session.status().message();
    EXPECT_EQ(session.value()->trace_id, "trace-old");
    EXPECT_EQ(session.value()->connection_id, 9u);
    EXPECT_EQ(session.value()->state, media::WebRtcSessionState::Connected);
    EXPECT_EQ(session.value()->last_frame_id, 41u);
    EXPECT_EQ(session.value()->reconnect_token, "resume-token-001");
    ASSERT_TRUE(checkpoint_store->last_checkpoint.has_value());
    EXPECT_EQ(checkpoint_store->last_checkpoint->connection_id, 9u);
    EXPECT_EQ(checkpoint_store->last_checkpoint->last_frame_id, 41u);
}
