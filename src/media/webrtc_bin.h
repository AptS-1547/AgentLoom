#pragma once

#include "result.h"

#include <gst/gst.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace media {

enum class WebRtcSdpType {
    Offer,
    Answer,
    PrAnswer,
    Rollback
};

struct WebRtcBinOptions {
    std::string name = "agent-webrtcbin";
    bool bundle_policy_max_bundle = true;
};

enum class VideoDecoderPreference {
    Auto,
    Software,
    Nvidia,
    D3D11,
    Vaapi
};

struct WebRtcDecodeOptions {
    VideoDecoderPreference video_decoder = VideoDecoderPreference::Auto;
    bool enable_appsink = true;
    std::optional<std::string> appsink_name;
    bool force_system_memory = true;
};

class IWebRtcPeer {
public:
    using IceCandidateHandler = std::function<void(std::uint32_t mline_index, std::string candidate)>;

    virtual ~IWebRtcPeer() = default;

    virtual core::Status SetLocalDescriptionFromSdp(WebRtcSdpType type, std::string_view sdp) = 0;
    virtual core::Status SetRemoteDescriptionFromSdp(WebRtcSdpType type, std::string_view sdp) = 0;
    virtual core::Result<std::string> CreateAnswer() = 0;
    virtual core::Status AddIceCandidate(std::uint32_t mline_index, std::string_view candidate) = 0;
    virtual void SetLocalIceCandidateHandler(IceCandidateHandler handler) = 0;
    virtual void ConfigureDecode(WebRtcDecodeOptions options) = 0;
};

class GstInitializer {
public:
    static core::Status EnsureInitialized();
};

class WebRtcBin final : public IWebRtcPeer {
public:
    static core::Result<std::shared_ptr<WebRtcBin>> Create(WebRtcBinOptions options = {});

    ~WebRtcBin();

    WebRtcBin(const WebRtcBin&) = delete;
    WebRtcBin& operator=(const WebRtcBin&) = delete;
    WebRtcBin(WebRtcBin&&) = delete;
    WebRtcBin& operator=(WebRtcBin&&) = delete;

    GstElement* element() const noexcept {
        return webrtcbin_;
    }

    const std::string& name() const noexcept {
        return name_;
    }

    core::Status SetLocalDescriptionFromSdp(WebRtcSdpType type, std::string_view sdp) override;
    core::Status SetRemoteDescriptionFromSdp(WebRtcSdpType type, std::string_view sdp) override;
    core::Status SetLocalDescriptionFromSdp(std::string_view sdp);
    core::Status SetRemoteDescriptionFromSdp(std::string_view sdp);
    core::Result<std::string> CreateAnswer() override;
    core::Status AddIceCandidate(std::uint32_t mline_index, std::string_view candidate) override;
    void SetLocalIceCandidateHandler(IceCandidateHandler handler) override;

    void ConfigureDecode(WebRtcDecodeOptions options) override;
    const WebRtcDecodeOptions& decode_options() const noexcept {
        return decode_options_;
    }

private:
    explicit WebRtcBin(GstElement* webrtcbin, std::string name) noexcept;

    core::Status SetDescriptionFromSdp(WebRtcSdpType type, std::string_view sdp, bool local);
    static void OnIceCandidate(GstElement* element, guint mline_index, gchar* candidate, gpointer user_data);

    GstElement* webrtcbin_ = nullptr;
    std::string name_;
    WebRtcDecodeOptions decode_options_;
    IceCandidateHandler ice_candidate_handler_;
};

} // namespace media
