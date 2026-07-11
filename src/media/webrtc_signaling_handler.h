#pragma once

#include "frame_encoding.h"
#include "request_interfaces.h"
#include "result.h"
#include "thread_pool.h"
#include "vision_runtime_interfaces.h"
#include "webrtc_bin.h"
#include "webrtc_session_registry.h"

#include <memory>
#include <string>
#include <functional>
#include <vector>

namespace media {

struct WebRtcIceServerConfig {
    std::vector<std::string> urls;
    std::string username;
    std::string credential;
};

struct WebRtcSignalingOptions {
    std::size_t max_sessions = 256;
    WebRtcDecodeOptions decode_options;
    std::shared_ptr<core::ThreadPool> compute_pool;
    std::shared_ptr<core::ThreadPool> io_pool;
    std::shared_ptr<IFrameSampler> frame_sampler;
    std::shared_ptr<IVlmVisionClient> vlm_client;
    std::shared_ptr<IVideoFrameEncoder> frame_encoder;
    std::shared_ptr<IEncodedVideoFrameSink> encoded_frame_sink;
    FrameObserver frame_observer;
    std::shared_ptr<IVisionEventSink> vision_event_sink;
    std::vector<WebRtcIceServerConfig> ice_servers;
};

class WebRtcSignalingHandler {
public:
    explicit WebRtcSignalingHandler(WebRtcSignalingOptions options = {});
    WebRtcSignalingHandler(WebRtcSignalingOptions options, WebRtcPeerFactory peer_factory);
    WebRtcSignalingHandler(WebRtcSignalingOptions options, std::shared_ptr<WebRtcSessionRegistry> registry);

    void Handle(std::shared_ptr<net::IWebSocketStreamRequest> request);
    void RemoveConnection(std::uint64_t connection_id);
    std::size_t session_count() const;

private:
    core::Status HandleOffer(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                             const std::string& session_id,
                             const std::string& trace_id,
                             const std::string& sdp);
    core::Status HandleIce(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                           const std::string& session_id,
                           std::uint32_t mline_index,
                           const std::string& candidate);
    core::Status HandleClose(const std::string& session_id);
    core::Status HandleResume(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                              const std::string& session_id,
                              const std::string& trace_id,
                              const std::string& reconnect_token);

    void SendAck(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                 const std::string& session_id,
                 std::string type);
    void SendConfig(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                    const std::string& session_id);
    void SendResumeAck(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                       const WebRtcSessionCheckpoint& checkpoint);
    void SendAnswer(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                    const std::string& session_id,
                    const std::string& sdp);
    void SendIce(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                 const std::string& session_id,
                 std::uint32_t mline_index,
                 const std::string& candidate);
    void SendError(const std::shared_ptr<net::IWebSocketStreamRequest>& request,
                   const std::string& session_id,
                   const core::Status& status);

    WebRtcSignalingOptions options_;
    std::shared_ptr<WebRtcSessionRegistry> registry_;
};

} // namespace media
