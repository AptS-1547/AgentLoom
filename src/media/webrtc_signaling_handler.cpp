#include "webrtc_signaling_handler.h"

#include "shared_buffer.h"
#include "websocket_types.h"
#include "webrtc_media_pipeline.h"

#include "third_party/nlohmann/json.hpp"

#include <cstdint>
#include <limits>
#include <utility>

namespace media {
namespace {

using Json = nlohmann::json;

std::string MessagePayloadToString(const net::WebSocketMessage& message) {
    std::string payload;
    payload.reserve(message.total_bytes);
    for (const auto& fragment : message.fragments) {
        payload.append(fragment.view());
    }
    return payload;
}

const Json* FindPayload(const Json& body) {
    auto it = body.find("payload");
    if (it == body.end() || !it->is_object()) {
        return nullptr;
    }
    return &(*it);
}

core::Result<std::string> RequiredString(const Json& body, const char* field) {
    auto it = body.find(field);
    if (it == body.end() || !it->is_string()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, std::string("missing string field: ") + field);
    }
    return it->get<std::string>();
}

core::Result<std::uint32_t> RequiredUInt32(const Json& body, const char* field) {
    auto it = body.find(field);
    if (it == body.end() || !it->is_number_unsigned()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, std::string("missing unsigned field: ") + field);
    }
    const auto value = it->get<std::uint64_t>();
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, std::string("field is too large: ") + field);
    }
    return static_cast<std::uint32_t>(value);
}

net::WebSocketFrame TextFrame(core::RawMemoryPool& pool, std::string_view text) {
    net::WebSocketFrame frame;
    frame.kind = net::WebSocketMessageKind::Text;
    auto copied = net::SharedBuffer::Copy(pool, text);
    if (copied.ok()) {
        frame.payload = std::move(copied).value();
    }
    return frame;
}

const char* ErrorCodeName(core::ErrorCode code) noexcept {
    switch (code) {
    case core::ErrorCode::Ok:
        return "OK";
    case core::ErrorCode::InvalidArgument:
        return "INVALID_ARGUMENT";
    case core::ErrorCode::NotFound:
        return "NOT_FOUND";
    case core::ErrorCode::ResourceExhausted:
        return "RESOURCE_EXHAUSTED";
    case core::ErrorCode::FailedPrecondition:
        return "FAILED_PRECONDITION";
    case core::ErrorCode::Unavailable:
        return "UNAVAILABLE";
    case core::ErrorCode::InternalError:
        return "INTERNAL_ERROR";
    default:
        return "UNKNOWN";
    }
}

} // namespace

WebRtcSignalingHandler::WebRtcSignalingHandler(WebRtcSignalingOptions options)
    : options_(std::move(options)) {
    if (!options_.frame_executor && options_.compute_pool) {
        options_.frame_executor = std::make_shared<core::KeyedSerialExecutor>(
            options_.compute_pool,
            core::KeyedSerialExecutorOptions{
                .max_keys = options_.max_sessions,
                .queue_capacity_per_key = 1024,
                .task_name_prefix = "vision-frame-stream",
            });
    }
    if (options_.encoded_frame_sink &&
        !std::dynamic_pointer_cast<IOrderedEncodedFrameSink>(options_.encoded_frame_sink)) {
        options_.encoded_frame_sink = std::make_shared<OrderedEncodedFrameSink>(
            options_.encoded_frame_sink,
            OrderedEncodedFrameSinkOptions{
                .max_executions = options_.max_sessions,
                .window_capacity = 1024,
            });
    }
    auto registry_ref = std::make_shared<std::weak_ptr<WebRtcSessionRegistry>>();
    if (!options_.frame_observer) {
        options_.frame_observer = [registry_ref](const VideoFrameView& frame) {
            if (auto registry = registry_ref->lock()) {
                static_cast<void>(registry->RecordFrame(frame.session_id, frame.frame_id));
            }
        };
    }

    WebRtcPeerFactory peer_factory =
        [options = options_, registry_ref](const std::string& session_id,
                                           const std::string& name) -> core::Result<std::shared_ptr<IWebRtcPeer>> {
        auto peer = WebRtcMediaPipeline::Create(WebRtcMediaPipelineOptions{
            .session_id = session_id,
            .name = name,
            .decode = options.decode_options,
            .compute_pool = options.compute_pool,
            .frame_executor = options.frame_executor,
            .io_pool = options.io_pool,
            .frame_sampler = options.frame_sampler,
            .vlm_client = options.vlm_client,
            .frame_encoder = options.frame_encoder,
            .encoded_frame_sink = options.encoded_frame_sink,
            .frame_observer = options.frame_observer,
            .vision_event_sink = options.vision_event_sink,
            .status_observer = [registry_ref](std::string failed_session_id, core::Status status, bool fatal) {
                if (!fatal) {
                    return;
                }
                if (auto registry = registry_ref->lock()) {
                    static_cast<void>(registry->MarkFailed(failed_session_id, status.message()));
                }
            },
        });
        if (!peer.ok()) {
            return peer.status();
        }
        return std::static_pointer_cast<IWebRtcPeer>(std::move(peer).value());
    };
    registry_ = std::make_shared<WebRtcSessionRegistry>(
        WebRtcSessionRegistryOptions{.max_sessions = options_.max_sessions},
        std::move(peer_factory));
    *registry_ref = registry_;
}

WebRtcSignalingHandler::WebRtcSignalingHandler(WebRtcSignalingOptions options, WebRtcPeerFactory peer_factory)
    : WebRtcSignalingHandler(
          options,
          std::make_shared<WebRtcSessionRegistry>(
              WebRtcSessionRegistryOptions{.max_sessions = options.max_sessions},
              std::move(peer_factory))) {}

WebRtcSignalingHandler::WebRtcSignalingHandler(WebRtcSignalingOptions options,
                                               std::shared_ptr<WebRtcSessionRegistry> registry)
    : options_(std::move(options)), registry_(std::move(registry)) {
    if (registry_ && !options_.frame_observer) {
        std::weak_ptr<WebRtcSessionRegistry> weak_registry = registry_;
        options_.frame_observer = [weak_registry](const VideoFrameView& frame) {
            if (auto registry = weak_registry.lock()) {
                static_cast<void>(registry->RecordFrame(frame.session_id, frame.frame_id));
            }
        };
    }
}

void WebRtcSignalingHandler::Handle(std::shared_ptr<net::IWebSocketStreamRequest> request) {
    std::string session_id;

    if (!request->message().ok()) {
        SendError(request, session_id, request->message().status);
        return;
    }
    if (request->message().kind != net::WebSocketMessageKind::Text) {
        SendError(
            request,
            session_id,
            core::Status::Error(core::ErrorCode::InvalidArgument, "WebRTC signaling requires text frames"));
        return;
    }

    Json body;
    try {
        body = Json::parse(MessagePayloadToString(request->message()));
    } catch (const std::exception& ex) {
        SendError(request, session_id, core::Status::Error(core::ErrorCode::InvalidArgument, ex.what()));
        return;
    }

    auto type = RequiredString(body, "type");
    if (!type.ok()) {
        SendError(request, session_id, type.status());
        return;
    }

    auto session = RequiredString(body, "session_id");
    if (!session.ok()) {
        SendError(request, session_id, session.status());
        return;
    }
    session_id = std::move(session).value();
    std::string trace_id;
    if (auto trace = RequiredString(body, "trace_id"); trace.ok()) {
        trace_id = std::move(trace).value();
    } else {
        trace_id = "rtc-" + session_id + "-" + std::to_string(request->connection().connection_id);
    }

    const Json* payload = FindPayload(body);
    if ((type.value() == "offer" || type.value() == "ice" || type.value() == "resume") && !payload) {
        SendError(
            request,
            session_id,
            core::Status::Error(core::ErrorCode::InvalidArgument, "missing object field: payload"));
        return;
    }

    core::Status status;
    if (type.value() == "offer") {
        auto sdp = RequiredString(*payload, "sdp");
        status = sdp.ok()
            ? HandleOffer(request, session_id, trace_id, std::move(sdp).value())
            : sdp.status();
    } else if (type.value() == "ice") {
        auto mline_index = RequiredUInt32(*payload, "sdp_mline_index");
        auto candidate = RequiredString(*payload, "candidate");
        status = mline_index.ok() && candidate.ok()
            ? HandleIce(request, session_id, mline_index.value(), std::move(candidate).value())
            : (!mline_index.ok() ? mline_index.status() : candidate.status());
    } else if (type.value() == "hello" || type.value() == "config") {
        SendConfig(request, session_id);
        status = core::Status::Ok();
    } else if (type.value() == "resume") {
        auto reconnect_token = RequiredString(*payload, "reconnect_token");
        status = reconnect_token.ok()
            ? HandleResume(request, session_id, trace_id, std::move(reconnect_token).value())
            : reconnect_token.status();
    } else if (type.value() == "close") {
        status = HandleClose(session_id);
        if (status.ok()) {
            request->Close(net::ConnectionCloseInfo::Remote("webrtc signaling closed"));
        }
    } else {
        status = core::Status::Error(core::ErrorCode::InvalidArgument, "unknown WebRTC signaling message type");
    }

    if (!status.ok()) {
        SendError(request, session_id, status);
    }
}

void WebRtcSignalingHandler::RemoveConnection(std::uint64_t connection_id) {
    if (registry_) {
        registry_->RemoveConnection(connection_id);
    }
}

std::size_t WebRtcSignalingHandler::session_count() const {
    return registry_ ? registry_->SessionCount() : 0;
}

core::Status WebRtcSignalingHandler::HandleOffer(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                                                 const std::string& session_id,
                                                 const std::string& trace_id,
                                                 const std::string& sdp) {
    if (!registry_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "WebRTC session registry is not configured");
    }
    auto session = registry_->GetOrCreate(session_id, trace_id, request->connection().connection_id);
    if (!session.ok()) {
        return session.status();
    }
    session.value()->peer->SetLocalIceCandidateHandler(
        [this, request, session_id](std::uint32_t mline_index, std::string candidate) {
            SendIce(request, session_id, mline_index, candidate);
        });

    auto status = session.value()->peer->SetRemoteDescriptionFromSdp(WebRtcSdpType::Offer, sdp);
    if (!status.ok()) {
        return status;
    }

    auto answer = session.value()->peer->CreateAnswer();
    if (!answer.ok()) {
        return answer.status();
    }

    SendAnswer(request, session_id, answer.value());
    return core::Status::Ok();
}

core::Status WebRtcSignalingHandler::HandleIce(const std::shared_ptr<net::IWebSocketStreamRequest>&,
                                               const std::string& session_id,
                                               std::uint32_t mline_index,
                                               const std::string& candidate) {
    if (!registry_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "WebRTC session registry is not configured");
    }
    auto session = registry_->Find(session_id);
    if (!session.ok()) {
        return session.status();
    }

    session.value()->last_ice_at = std::chrono::steady_clock::now();
    static_cast<void>(registry_->SaveCheckpoint(*session.value()));
    return session.value()->peer->AddIceCandidate(mline_index, candidate);
}

core::Status WebRtcSignalingHandler::HandleClose(const std::string& session_id) {
    if (!registry_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "WebRTC session registry is not configured");
    }
    return registry_->Close(session_id);
}

core::Status WebRtcSignalingHandler::HandleResume(
    const std::shared_ptr<net::IWebSocketStreamRequest>& request,
    const std::string& session_id,
    const std::string& trace_id,
    const std::string& reconnect_token) {
    if (!registry_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "WebRTC session registry is not configured");
    }
    auto checkpoint = registry_->ValidateResume(
        session_id,
        reconnect_token,
        request->connection().connection_id,
        trace_id);
    if (!checkpoint.ok()) {
        return checkpoint.status();
    }
    SendResumeAck(request, checkpoint.value());
    return core::Status::Ok();
}

void WebRtcSignalingHandler::SendAck(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                                     const std::string& session_id,
                                     std::string type) {
    Json body{
        {"type", std::move(type)},
        {"session_id", session_id},
        {"connection_id", request->connection().connection_id},
        {"payload", Json::object()},
    };
    request->Send(TextFrame(request->memory_pool(), body.dump()));
}

void WebRtcSignalingHandler::SendConfig(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                                        const std::string& session_id) {
    Json ice_servers = Json::array();
    for (const auto& server : options_.ice_servers) {
        Json item{{"urls", server.urls}};
        if (!server.username.empty()) {
            item["username"] = server.username;
        }
        if (!server.credential.empty()) {
            item["credential"] = server.credential;
        }
        ice_servers.push_back(std::move(item));
    }
    Json body{
        {"type", "config"},
        {"session_id", session_id},
        {"connection_id", request->connection().connection_id},
        {"payload", {
            {"ice_servers", std::move(ice_servers)},
        }},
    };
    request->Send(TextFrame(request->memory_pool(), body.dump()));
}

void WebRtcSignalingHandler::SendResumeAck(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                                           const WebRtcSessionCheckpoint& checkpoint) {
    Json body{
        {"type", "resume_ack"},
        {"session_id", checkpoint.session_id},
        {"connection_id", request->connection().connection_id},
        {"payload", {
            {"accepted", true},
            {"state", checkpoint.state},
            {"last_frame_id", checkpoint.last_frame_id},
            {"reconnect_token", checkpoint.reconnect_token},
        }},
    };
    request->Send(TextFrame(request->memory_pool(), body.dump()));
}

void WebRtcSignalingHandler::SendAnswer(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                                        const std::string& session_id,
                                        const std::string& sdp) {
    Json body{
        {"type", "answer"},
        {"session_id", session_id},
        {"connection_id", request->connection().connection_id},
        {"payload", {
            {"sdp", sdp},
        }},
    };
    request->Send(TextFrame(request->memory_pool(), body.dump()));
}

void WebRtcSignalingHandler::SendIce(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                                     const std::string& session_id,
                                     std::uint32_t mline_index,
                                     const std::string& candidate) {
    Json body{
        {"type", "ice"},
        {"session_id", session_id},
        {"connection_id", request->connection().connection_id},
        {"payload", {
            {"sdp_mline_index", mline_index},
            {"candidate", candidate},
        }},
    };
    request->Send(TextFrame(request->memory_pool(), body.dump()));
}

void WebRtcSignalingHandler::SendError(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                                       const std::string& session_id,
                                       const core::Status& status) {
    Json body{
        {"type", "error"},
        {"session_id", session_id},
        {"connection_id", request->connection().connection_id},
        {"payload", {
            {"code", ErrorCodeName(status.code())},
            {"message", status.message()},
        }},
    };
    request->Send(TextFrame(request->memory_pool(), body.dump()));
}

} // namespace media
