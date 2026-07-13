#pragma once

#include "frame_encoding.h"
#include "keyed_serial_executor.h"
#include "ordered_encoded_frame_sink.h"
#include "thread_pool.h"
#include "vision_runtime_interfaces.h"
#include "webrtc_bin.h"

#include <gst/gst.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace media {

struct WebRtcMediaPipelineOptions {
    std::string execution_id;
    std::string session_id;
    std::string name = "agent-webrtc-media-pipeline";
    WebRtcDecodeOptions decode;
    std::shared_ptr<core::ThreadPool> compute_pool;
    std::shared_ptr<core::IKeyedSerialExecutor> frame_executor;
    std::shared_ptr<core::ThreadPool> io_pool;
    std::shared_ptr<IFrameSampler> frame_sampler;
    std::shared_ptr<IVlmVisionClient> vlm_client;
    std::shared_ptr<IVideoFrameEncoder> frame_encoder;
    std::shared_ptr<IEncodedVideoFrameSink> encoded_frame_sink;
    FrameObserver frame_observer;
    std::shared_ptr<IVisionEventSink> vision_event_sink;
    std::function<void(std::string session_id, core::Status status, bool fatal)> status_observer;
};

class WebRtcMediaPipeline final : public IWebRtcPeer,
                                  public std::enable_shared_from_this<WebRtcMediaPipeline> {
public:
    static core::Result<std::shared_ptr<WebRtcMediaPipeline>> Create(WebRtcMediaPipelineOptions options);

    ~WebRtcMediaPipeline();

    WebRtcMediaPipeline(const WebRtcMediaPipeline&) = delete;
    WebRtcMediaPipeline& operator=(const WebRtcMediaPipeline&) = delete;
    WebRtcMediaPipeline(WebRtcMediaPipeline&&) = delete;
    WebRtcMediaPipeline& operator=(WebRtcMediaPipeline&&) = delete;

    core::Status SetLocalDescriptionFromSdp(WebRtcSdpType type, std::string_view sdp) override;
    core::Status SetRemoteDescriptionFromSdp(WebRtcSdpType type, std::string_view sdp) override;
    core::Result<std::string> CreateAnswer() override;
    core::Status AddIceCandidate(std::uint32_t mline_index, std::string_view candidate) override;
    void SetLocalIceCandidateHandler(IceCandidateHandler handler) override;
    void ConfigureDecode(WebRtcDecodeOptions options) override;

    GstElement* pipeline() const noexcept {
        return pipeline_;
    }

    GstElement* webrtcbin() const noexcept;

private:
    explicit WebRtcMediaPipeline(WebRtcMediaPipelineOptions options) noexcept;

    core::Status Build();
    core::Status Start();
    core::Status ValidateDecoderPreference() const;
    core::Status LinkIncomingPad(GstPad* pad);
    core::Status LinkDecodebinPad(GstElement* decodebin, GstPad* pad, GstElement* videoconvert);
    core::Status HandleSample(GstSample* sample);
    void SubmitFrame(VideoFrameView frame);
    std::string FrameStreamKey() const;

    static void OnPadAdded(GstElement* element, GstPad* pad, gpointer user_data);
    static gint OnAutoplugSelect(GstElement* decodebin, GstPad* pad, GstCaps* caps, GstElementFactory* factory, gpointer user_data);
    static void OnDecodebinPadAdded(GstElement* decodebin, GstPad* pad, gpointer user_data);
    static GstFlowReturn OnNewSample(GstElement* appsink, gpointer user_data);
    static void OnBusMessage(GstBus* bus, GstMessage* message, gpointer user_data);
    void NotifyPipelineStatus(core::Status status, bool fatal);

    WebRtcMediaPipelineOptions options_;
    std::shared_ptr<WebRtcBin> webrtc_;
    GstElement* pipeline_ = nullptr;
    GstBus* bus_ = nullptr;
    gulong bus_handler_id_ = 0;
    std::mutex pipeline_mutex_;
    std::atomic<std::uint64_t> frame_id_{0};
    std::shared_ptr<std::atomic<std::uint64_t>> selected_sequence_ =
        std::make_shared<std::atomic<std::uint64_t>>(0);
    std::shared_ptr<core::TaskGroup> publish_tasks_ = std::make_shared<core::TaskGroup>();
};

} // namespace media
