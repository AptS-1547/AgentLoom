#include "webrtc_bin.h"

#include <gst/sdp/gstsdpmessage.h>
#include <gst/webrtc/webrtc.h>

#include <mutex>
#include <utility>

namespace media {
namespace {

std::string GstParseErrorMessage(GError* error) {
    if (!error) {
        return "unknown GStreamer error";
    }
    std::string message = error->message ? error->message : "unknown GStreamer error";
    g_error_free(error);
    return message;
}

struct GstSdpMessageDeleter {
    void operator()(GstSDPMessage* message) const noexcept {
        if (message) {
            gst_sdp_message_free(message);
        }
    }
};

struct GstWebRtcSessionDescriptionDeleter {
    void operator()(GstWebRTCSessionDescription* description) const noexcept {
        if (description) {
            gst_webrtc_session_description_free(description);
        }
    }
};

using UniqueSdpMessage = std::unique_ptr<GstSDPMessage, GstSdpMessageDeleter>;
using UniqueSessionDescription =
    std::unique_ptr<GstWebRTCSessionDescription, GstWebRtcSessionDescriptionDeleter>;

struct GstPromiseDeleter {
    void operator()(GstPromise* promise) const noexcept {
        if (promise) {
            gst_promise_unref(promise);
        }
    }
};

using UniquePromise = std::unique_ptr<GstPromise, GstPromiseDeleter>;

core::Status CheckWebRtcPlugin() {
    GstElementFactory* factory = gst_element_factory_find("webrtcbin");
    if (!factory) {
        return core::Status::Error(core::ErrorCode::Unavailable, "GStreamer webrtcbin factory is not available");
    }
    gst_object_unref(factory);
    return core::Status::Ok();
}

core::Result<GstWebRTCSDPType> ToGstSdpType(WebRtcSdpType type) {
    switch (type) {
    case WebRtcSdpType::Offer:
        return GST_WEBRTC_SDP_TYPE_OFFER;
    case WebRtcSdpType::Answer:
        return GST_WEBRTC_SDP_TYPE_ANSWER;
    case WebRtcSdpType::PrAnswer:
        return GST_WEBRTC_SDP_TYPE_PRANSWER;
    case WebRtcSdpType::Rollback:
        return GST_WEBRTC_SDP_TYPE_ROLLBACK;
    }

    return core::Status::Error(core::ErrorCode::InvalidArgument, "unsupported WebRTC SDP type");
}

} // namespace

core::Status GstInitializer::EnsureInitialized() {
    static std::once_flag init_once;
    static core::Status init_status;

    std::call_once(init_once, [] {
        GError* error = nullptr;
        if (!gst_init_check(nullptr, nullptr, &error)) {
            init_status = core::Status::Error(core::ErrorCode::Unavailable, GstParseErrorMessage(error));
            return;
        }
        init_status = CheckWebRtcPlugin();
    });

    return init_status;
}

core::Result<std::shared_ptr<WebRtcBin>> WebRtcBin::Create(WebRtcBinOptions options) {
    auto init_status = GstInitializer::EnsureInitialized();
    if (!init_status.ok()) {
        return init_status;
    }

    GstElement* element = gst_element_factory_make("webrtcbin", options.name.empty() ? nullptr : options.name.c_str());
    if (!element) {
        return core::Status::Error(core::ErrorCode::Unavailable, "failed to create GStreamer webrtcbin element");
    }

    if (options.bundle_policy_max_bundle) {
        g_object_set(G_OBJECT(element), "bundle-policy", GST_WEBRTC_BUNDLE_POLICY_MAX_BUNDLE, nullptr);
    }

    auto webrtc = std::shared_ptr<WebRtcBin>(new WebRtcBin(element, std::move(options.name)));
    g_signal_connect(element, "on-ice-candidate", G_CALLBACK(&WebRtcBin::OnIceCandidate), webrtc.get());
    return webrtc;
}

WebRtcBin::WebRtcBin(GstElement* webrtcbin, std::string name) noexcept
    : webrtcbin_(webrtcbin), name_(std::move(name)) {}

WebRtcBin::~WebRtcBin() {
    if (webrtcbin_) {
        gst_element_set_state(webrtcbin_, GST_STATE_NULL);
        gst_object_unref(webrtcbin_);
        webrtcbin_ = nullptr;
    }
}

core::Status WebRtcBin::SetLocalDescriptionFromSdp(std::string_view sdp) {
    return SetLocalDescriptionFromSdp(WebRtcSdpType::Answer, sdp);
}

core::Status WebRtcBin::SetRemoteDescriptionFromSdp(std::string_view sdp) {
    return SetRemoteDescriptionFromSdp(WebRtcSdpType::Offer, sdp);
}

core::Status WebRtcBin::SetLocalDescriptionFromSdp(WebRtcSdpType type, std::string_view sdp) {
    return SetDescriptionFromSdp(type, sdp, true);
}

core::Status WebRtcBin::SetRemoteDescriptionFromSdp(WebRtcSdpType type, std::string_view sdp) {
    return SetDescriptionFromSdp(type, sdp, false);
}

core::Status WebRtcBin::AddIceCandidate(std::uint32_t mline_index, std::string_view candidate) {
    if (!webrtcbin_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "webrtcbin is not initialized");
    }
    if (candidate.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "ICE candidate is empty");
    }

    const std::string candidate_text(candidate);
    g_signal_emit_by_name(webrtcbin_, "add-ice-candidate", mline_index, candidate_text.c_str());
    return core::Status::Ok();
}

core::Result<std::string> WebRtcBin::CreateAnswer() {
    if (!webrtcbin_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "webrtcbin is not initialized");
    }

    UniquePromise promise(gst_promise_new());
    if (!promise) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to create GStreamer promise");
    }

    g_signal_emit_by_name(webrtcbin_, "create-answer", nullptr, promise.get());
    const auto wait_result = gst_promise_wait(promise.get());
    if (wait_result != GST_PROMISE_RESULT_REPLIED) {
        return core::Status::Error(core::ErrorCode::InternalError, "GStreamer create-answer did not return a reply");
    }

    const GstStructure* reply = gst_promise_get_reply(promise.get());
    if (!reply) {
        return core::Status::Error(core::ErrorCode::InternalError, "GStreamer create-answer reply is empty");
    }

    GstWebRTCSessionDescription* raw_answer = nullptr;
    if (!gst_structure_get(reply, "answer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &raw_answer, nullptr) || !raw_answer) {
        return core::Status::Error(core::ErrorCode::InternalError, "GStreamer create-answer reply has no answer");
    }

    UniqueSessionDescription answer(raw_answer);
    gchar* raw_sdp_text = gst_sdp_message_as_text(answer->sdp);
    if (!raw_sdp_text) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to serialize WebRTC answer SDP");
    }

    std::string sdp_text(raw_sdp_text);
    g_free(raw_sdp_text);

    UniquePromise set_local_promise(gst_promise_new());
    if (!set_local_promise) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to create GStreamer promise");
    }

    g_signal_emit_by_name(webrtcbin_, "set-local-description", answer.get(), set_local_promise.get());
    const auto set_local_result = gst_promise_wait(set_local_promise.get());
    if (set_local_result != GST_PROMISE_RESULT_REPLIED) {
        return core::Status::Error(core::ErrorCode::InternalError, "GStreamer set-local-description failed");
    }
    return sdp_text;
}

void WebRtcBin::ConfigureDecode(WebRtcDecodeOptions options) {
    decode_options_ = std::move(options);
}

void WebRtcBin::SetLocalIceCandidateHandler(IceCandidateHandler handler) {
    ice_candidate_handler_ = std::move(handler);
}

void WebRtcBin::OnIceCandidate(GstElement*, guint mline_index, gchar* candidate, gpointer user_data) {
    auto* self = static_cast<WebRtcBin*>(user_data);
    if (!self || !self->ice_candidate_handler_ || !candidate) {
        return;
    }
    self->ice_candidate_handler_(static_cast<std::uint32_t>(mline_index), std::string(candidate));
}

core::Status WebRtcBin::SetDescriptionFromSdp(WebRtcSdpType type, std::string_view sdp, bool local) {
    if (!webrtcbin_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "webrtcbin is not initialized");
    }
    if (sdp.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "SDP text is empty");
    }

    GstSDPMessage* raw_sdp = nullptr;
    const std::string sdp_text(sdp);
    GstSDPResult parse_result = gst_sdp_message_new_from_text(sdp_text.c_str(), &raw_sdp);
    if (parse_result != GST_SDP_OK || !raw_sdp) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "failed to parse SDP text");
    }

    UniqueSdpMessage parsed_sdp(raw_sdp);
    auto gst_type = ToGstSdpType(type);
    if (!gst_type.ok()) {
        return gst_type.status();
    }

    UniqueSessionDescription description(
        gst_webrtc_session_description_new(gst_type.value(), parsed_sdp.release()));
    if (!description) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to create WebRTC session description");
    }

    UniquePromise promise(gst_promise_new());
    if (!promise) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to create GStreamer promise");
    }

    g_signal_emit_by_name(
        webrtcbin_,
        local ? "set-local-description" : "set-remote-description",
        description.get(),
        promise.get());
    const auto result = gst_promise_wait(promise.get());
    if (result != GST_PROMISE_RESULT_REPLIED) {
        return core::Status::Error(
            core::ErrorCode::InternalError,
            local ? "GStreamer set-local-description failed" : "GStreamer set-remote-description failed");
    }

    return core::Status::Ok();
}

} // namespace media
