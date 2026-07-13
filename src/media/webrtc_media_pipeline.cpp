#include "webrtc_media_pipeline.h"

#include <gst/app/gstappsink.h>
#include <gst/video/video-info.h>

#include <cstring>
#include <string_view>
#include <utility>

namespace media {
namespace {

struct GstCapsDeleter {
    void operator()(GstCaps* caps) const noexcept {
        if (caps) {
            gst_caps_unref(caps);
        }
    }
};

using UniqueCaps = std::unique_ptr<GstCaps, GstCapsDeleter>;

struct GErrorDeleter {
    void operator()(GError* error) const noexcept {
        if (error) {
            g_error_free(error);
        }
    }
};

struct GCharDeleter {
    void operator()(gchar* text) const noexcept {
        if (text) {
            g_free(text);
        }
    }
};

using UniqueGError = std::unique_ptr<GError, GErrorDeleter>;
using UniqueGChar = std::unique_ptr<gchar, GCharDeleter>;

constexpr gint kAutoplugTry = 0;
constexpr gint kAutoplugSkip = 2;

bool Contains(std::string_view value, std::string_view needle) noexcept {
    return value.find(needle) != std::string_view::npos;
}

bool FactoryNameStartsWith(GstElementFactory* factory, std::string_view prefix) {
    const char* name = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
    return name && std::string_view(name).starts_with(prefix);
}

bool IsDecoderFactory(GstElementFactory* factory) {
    const char* klass = gst_element_factory_get_klass(factory);
    return klass && Contains(klass, "Decoder");
}

bool IsHardwareDecoderFactory(GstElementFactory* factory) {
    const char* klass = gst_element_factory_get_klass(factory);
    if (klass && Contains(klass, "Hardware")) {
        return true;
    }

    const char* name = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
    if (!name) {
        return false;
    }
    const std::string_view factory_name(name);
    return factory_name.starts_with("nv") ||
           factory_name.starts_with("vaapi") ||
           factory_name.starts_with("d3d") ||
           factory_name.starts_with("msdk") ||
           factory_name.starts_with("qsv");
}

bool IsNvidiaDecoderFactory(GstElementFactory* factory) {
    return FactoryNameStartsWith(factory, "nv");
}

bool IsVaapiDecoderFactory(GstElementFactory* factory) {
    return FactoryNameStartsWith(factory, "vaapi");
}

bool HasAnyFactory(std::initializer_list<const char*> names) {
    for (const char* name : names) {
        GstElementFactory* factory = gst_element_factory_find(name);
        if (factory) {
            gst_object_unref(factory);
            return true;
        }
    }
    return false;
}

bool HasSoftwareDecoderFactory() {
    return HasAnyFactory({"avdec_h264", "avdec_vp8", "avdec_vp9", "openh264dec"});
}

bool HasNvidiaDecoderFactory() {
    return HasAnyFactory({"nvh264dec", "nvh265dec", "nvvp8dec", "nvvp9dec", "nvav1dec"});
}

bool HasVaapiDecoderFactory() {
    return HasAnyFactory({"vaapih264dec", "vaapih265dec", "vaapivp8dec", "vaapivp9dec", "vaapiav1dec"});
}

core::Status AddMany(GstElement* pipeline, std::initializer_list<GstElement*> elements) {
    for (auto* element : elements) {
        if (!element) {
            return core::Status::Error(core::ErrorCode::InternalError, "GStreamer element is null");
        }
        gst_bin_add(GST_BIN(pipeline), element);
    }
    return core::Status::Ok();
}

core::Status LinkMany(std::initializer_list<GstElement*> elements) {
    GstElement* previous = nullptr;
    for (auto* element : elements) {
        if (!element) {
            return core::Status::Error(core::ErrorCode::InternalError, "GStreamer element is null");
        }
        if (previous && !gst_element_link(previous, element)) {
            return core::Status::Error(core::ErrorCode::InternalError, "failed to link GStreamer elements");
        }
        previous = element;
    }
    return core::Status::Ok();
}

VideoPixelFormat ParsePixelFormat(const GstStructure* structure) {
    const char* format = gst_structure_get_string(structure, "format");
    if (!format) {
        return VideoPixelFormat::Unknown;
    }
    if (std::strcmp(format, "RGB") == 0) {
        return VideoPixelFormat::Rgb;
    }
    if (std::strcmp(format, "BGR") == 0) {
        return VideoPixelFormat::Bgr;
    }
    if (std::strcmp(format, "NV12") == 0) {
        return VideoPixelFormat::Nv12;
    }
    if (std::strcmp(format, "I420") == 0) {
        return VideoPixelFormat::I420;
    }
    return VideoPixelFormat::Unknown;
}

} // namespace

core::Result<std::shared_ptr<WebRtcMediaPipeline>> WebRtcMediaPipeline::Create(WebRtcMediaPipelineOptions options) {
    if (static_cast<bool>(options.frame_encoder) != static_cast<bool>(options.encoded_frame_sink)) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "frame_encoder and encoded_frame_sink must be configured together");
    }
    if (!options.frame_executor && options.compute_pool) {
        options.frame_executor = std::make_shared<core::KeyedSerialExecutor>(
            options.compute_pool,
            core::KeyedSerialExecutorOptions{
                .max_keys = 1,
                .queue_capacity_per_key = 1024,
                .task_name_prefix = "vision-frame-stream",
            });
    }
    if (options.encoded_frame_sink &&
        !std::dynamic_pointer_cast<IOrderedEncodedFrameSink>(options.encoded_frame_sink)) {
        options.encoded_frame_sink = std::make_shared<OrderedEncodedFrameSink>(
            options.encoded_frame_sink,
            OrderedEncodedFrameSinkOptions{
                .max_executions = 1,
                .window_capacity = 1024,
            });
    }
    auto init_status = GstInitializer::EnsureInitialized();
    if (!init_status.ok()) {
        return init_status;
    }

    auto pipeline = std::shared_ptr<WebRtcMediaPipeline>(new WebRtcMediaPipeline(std::move(options)));
    auto status = pipeline->Build();
    if (!status.ok()) {
        return status;
    }
    status = pipeline->ValidateDecoderPreference();
    if (!status.ok()) {
        return status;
    }
    status = pipeline->Start();
    if (!status.ok()) {
        return status;
    }
    return pipeline;
}

WebRtcMediaPipeline::WebRtcMediaPipeline(WebRtcMediaPipelineOptions options) noexcept
    : options_(std::move(options)) {}

WebRtcMediaPipeline::~WebRtcMediaPipeline() {
    {
        std::lock_guard lock(pipeline_mutex_);
        if (bus_) {
            if (bus_handler_id_ != 0) {
                g_signal_handler_disconnect(bus_, bus_handler_id_);
                bus_handler_id_ = 0;
            }
            gst_bus_remove_signal_watch(bus_);
            gst_object_unref(bus_);
            bus_ = nullptr;
        }
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }
    if (options_.frame_executor) {
        const auto drain_status = options_.frame_executor->WaitIdle(FrameStreamKey(), std::chrono::seconds(30));
        if (!drain_status.ok()) {
            NotifyPipelineStatus(drain_status, false);
        }
    }
    static_cast<void>(publish_tasks_->Seal());
    auto published = publish_tasks_->WaitFor(std::chrono::seconds(30));
    if (!published.ok()) {
        NotifyPipelineStatus(published.status(), false);
    } else if (!published.value().first_failure.ok()) {
        NotifyPipelineStatus(published.value().first_failure, false);
    }
    if (auto ordered_sink = std::dynamic_pointer_cast<IOrderedEncodedFrameSink>(options_.encoded_frame_sink)) {
        auto sealed = ordered_sink->SealExecution(
            options_.session_id,
            FrameStreamKey(),
            selected_sequence_->load(std::memory_order_acquire));
        if (!sealed.ok()) {
            NotifyPipelineStatus(sealed.status(), false);
        }
    }
}

core::Status WebRtcMediaPipeline::SetLocalDescriptionFromSdp(WebRtcSdpType type, std::string_view sdp) {
    return webrtc_->SetLocalDescriptionFromSdp(type, sdp);
}

core::Status WebRtcMediaPipeline::SetRemoteDescriptionFromSdp(WebRtcSdpType type, std::string_view sdp) {
    return webrtc_->SetRemoteDescriptionFromSdp(type, sdp);
}

core::Result<std::string> WebRtcMediaPipeline::CreateAnswer() {
    return webrtc_->CreateAnswer();
}

core::Status WebRtcMediaPipeline::AddIceCandidate(std::uint32_t mline_index, std::string_view candidate) {
    return webrtc_->AddIceCandidate(mline_index, candidate);
}

void WebRtcMediaPipeline::SetLocalIceCandidateHandler(IceCandidateHandler handler) {
    if (webrtc_) {
        webrtc_->SetLocalIceCandidateHandler(std::move(handler));
    }
}

void WebRtcMediaPipeline::ConfigureDecode(WebRtcDecodeOptions options) {
    options_.decode = std::move(options);
    if (webrtc_) {
        webrtc_->ConfigureDecode(options_.decode);
    }
}

GstElement* WebRtcMediaPipeline::webrtcbin() const noexcept {
    return webrtc_ ? webrtc_->element() : nullptr;
}

core::Status WebRtcMediaPipeline::Build() {
    std::lock_guard lock(pipeline_mutex_);
    pipeline_ = gst_pipeline_new(options_.name.empty() ? nullptr : options_.name.c_str());
    if (!pipeline_) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to create GStreamer pipeline");
    }

    auto webrtc = WebRtcBin::Create(WebRtcBinOptions{.name = options_.name + "-webrtcbin"});
    if (!webrtc.ok()) {
        return webrtc.status();
    }
    webrtc_ = std::move(webrtc).value();
    webrtc_->ConfigureDecode(options_.decode);

    gst_bin_add(GST_BIN(pipeline_), webrtc_->element());
    gst_object_ref(webrtc_->element());
    g_object_set_data(G_OBJECT(pipeline_), "agent-webrtc-pipeline", this);
    g_signal_connect(webrtc_->element(), "pad-added", G_CALLBACK(&WebRtcMediaPipeline::OnPadAdded), this);
    bus_ = gst_element_get_bus(pipeline_);
    if (!bus_) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to get GStreamer bus");
    }
    gst_bus_add_signal_watch(bus_);
    bus_handler_id_ = g_signal_connect(bus_, "message", G_CALLBACK(&WebRtcMediaPipeline::OnBusMessage), this);
    return core::Status::Ok();
}

core::Status WebRtcMediaPipeline::Start() {
    std::lock_guard lock(pipeline_mutex_);
    if (!pipeline_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "GStreamer pipeline is not initialized");
    }
    const auto result = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (result == GST_STATE_CHANGE_FAILURE) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to start GStreamer pipeline");
    }
    return core::Status::Ok();
}

core::Status WebRtcMediaPipeline::ValidateDecoderPreference() const {
    switch (options_.decode.video_decoder) {
    case VideoDecoderPreference::Auto:
        return core::Status::Ok();
    case VideoDecoderPreference::Software:
        return HasSoftwareDecoderFactory()
            ? core::Status::Ok()
            : core::Status::Error(core::ErrorCode::Unavailable, "no software video decoder factory is available");
    case VideoDecoderPreference::Nvidia:
        return HasNvidiaDecoderFactory()
            ? core::Status::Ok()
            : core::Status::Error(core::ErrorCode::Unavailable, "no NVIDIA video decoder factory is available");
    case VideoDecoderPreference::Vaapi:
        return HasVaapiDecoderFactory()
            ? core::Status::Ok()
            : core::Status::Error(core::ErrorCode::Unavailable, "no VAAPI video decoder factory is available");
    case VideoDecoderPreference::D3D11:
        return core::Status::Error(core::ErrorCode::Unimplemented, "D3D11 video decoder preference is not implemented");
    }

    return core::Status::Error(core::ErrorCode::InvalidArgument, "unknown video decoder preference");
}

core::Status WebRtcMediaPipeline::LinkIncomingPad(GstPad* pad) {
    if (!pipeline_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "GStreamer pipeline is not initialized");
    }

    GstElement* queue = gst_element_factory_make("queue", nullptr);
    GstElement* decodebin = gst_element_factory_make("decodebin", nullptr);
    GstElement* videoconvert = gst_element_factory_make("videoconvert", nullptr);
    GstElement* appsink = gst_element_factory_make("appsink", options_.decode.appsink_name.value_or("").c_str());
    if (!queue || !decodebin || !videoconvert || !appsink) {
        return core::Status::Error(core::ErrorCode::Unavailable, "failed to create GStreamer decode elements");
    }

    UniqueCaps caps(gst_caps_new_simple(
        "video/x-raw",
        "format",
        G_TYPE_STRING,
        "RGB",
        nullptr));
    g_object_set(G_OBJECT(appsink), "emit-signals", TRUE, "sync", FALSE, "max-buffers", 2, "drop", TRUE, nullptr);
    gst_app_sink_set_caps(GST_APP_SINK(appsink), caps.get());

    auto status = AddMany(pipeline_, {queue, decodebin, videoconvert, appsink});
    if (!status.ok()) {
        return status;
    }
    status = LinkMany({queue, decodebin});
    if (!status.ok()) {
        return status;
    }
    status = LinkMany({videoconvert, appsink});
    if (!status.ok()) {
        return status;
    }

    GstPad* queue_sink = gst_element_get_static_pad(queue, "sink");
    if (!queue_sink) {
        return core::Status::Error(core::ErrorCode::InternalError, "queue sink pad is unavailable");
    }
    const auto link_result = gst_pad_link(pad, queue_sink);
    gst_object_unref(queue_sink);
    if (link_result != GST_PAD_LINK_OK) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to link WebRTC incoming pad");
    }

    g_signal_connect(decodebin, "autoplug-select", G_CALLBACK(&WebRtcMediaPipeline::OnAutoplugSelect), this);
    g_signal_connect(decodebin, "pad-added", G_CALLBACK(&WebRtcMediaPipeline::OnDecodebinPadAdded), videoconvert);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(&WebRtcMediaPipeline::OnNewSample), this);
    gst_element_sync_state_with_parent(queue);
    gst_element_sync_state_with_parent(decodebin);
    gst_element_sync_state_with_parent(videoconvert);
    gst_element_sync_state_with_parent(appsink);
    return core::Status::Ok();
}

core::Status WebRtcMediaPipeline::LinkDecodebinPad(GstElement*, GstPad* pad, GstElement* videoconvert) {
    GstPad* sink_pad = gst_element_get_static_pad(videoconvert, "sink");
    if (!sink_pad) {
        return core::Status::Error(core::ErrorCode::InternalError, "videoconvert sink pad is unavailable");
    }
    if (gst_pad_is_linked(sink_pad)) {
        gst_object_unref(sink_pad);
        return core::Status::Ok();
    }

    const auto result = gst_pad_link(pad, sink_pad);
    gst_object_unref(sink_pad);
    if (result != GST_PAD_LINK_OK) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to link decodebin video pad");
    }
    return core::Status::Ok();
}

core::Status WebRtcMediaPipeline::HandleSample(GstSample* sample) {
    if (!sample) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "GStreamer sample is null");
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    if (!buffer || !caps || gst_caps_is_empty(caps)) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "GStreamer sample has no buffer or caps");
    }

    const GstStructure* structure = gst_caps_get_structure(caps, 0);
    gint width = 0;
    gint height = 0;
    gst_structure_get_int(structure, "width", &width);
    gst_structure_get_int(structure, "height", &height);

    auto frame_buffer = GstMappedFrameBuffer::Create(buffer);
    if (!frame_buffer.ok()) {
        return frame_buffer.status();
    }

    VideoFrameView frame;
    frame.session_id = options_.session_id;
    frame.frame_id = ++frame_id_;
    if (GST_BUFFER_PTS_IS_VALID(buffer)) {
        frame.timestamp_us = static_cast<std::int64_t>(GST_BUFFER_PTS(buffer) / GST_USECOND);
    }
    frame.width = static_cast<std::uint32_t>(width > 0 ? width : 0);
    frame.height = static_cast<std::uint32_t>(height > 0 ? height : 0);
    GstVideoInfo video_info;
    gst_video_info_init(&video_info);
    if (gst_video_info_from_caps(&video_info, caps)) {
        const auto stride = GST_VIDEO_INFO_PLANE_STRIDE(&video_info, 0);
        frame.row_stride_bytes = stride > 0 ? static_cast<std::size_t>(stride) : 0;
    }
    frame.format = ParsePixelFormat(structure);
    frame.buffer = std::move(frame_buffer).value();
    frame.bytes = frame.buffer->bytes();
    if (options_.frame_observer) {
        options_.frame_observer(frame);
    }
    SubmitFrame(std::move(frame));
    return core::Status::Ok();
}

void WebRtcMediaPipeline::SubmitFrame(VideoFrameView frame) {
    if (!options_.frame_sampler || !options_.frame_executor) {
        return;
    }

    auto sampler = options_.frame_sampler;
    auto event_sink = options_.vision_event_sink;
    auto encoder = options_.frame_encoder;
    auto encoded_sink = options_.encoded_frame_sink;
    auto io_pool = options_.io_pool;
    auto execution_id = FrameStreamKey();
    auto selected_sequence = selected_sequence_;
    auto publish_tasks = publish_tasks_;
    const auto stream_key = FrameStreamKey();
    auto status = options_.frame_executor->Submit(
        stream_key,
        [sampler,
         event_sink,
         encoder,
         encoded_sink,
         io_pool,
         execution_id = std::move(execution_id),
         selected_sequence = std::move(selected_sequence),
         publish_tasks,
         frame = std::move(frame)]() mutable -> core::Status {
            if (frame.buffer) {
                frame.bytes = frame.buffer->bytes();
            }
            auto decision = sampler->Evaluate(frame);
            if (!decision.ok()) {
                return decision.status();
            }
            if (decision.value().submit_to_vlm && event_sink) {
                VisionEvent event;
                event.event_id = frame.session_id + "-frame-" + std::to_string(frame.frame_id);
                event.session_id = frame.session_id;
                event.peak_frame_id = frame.frame_id;
                event.representative_frame_id = frame.frame_id;
                event.timestamp_seconds = static_cast<double>(frame.frame_id);
                event.captured_at = frame.captured_at;
                event.peak_score = decision.value().saliency_score;
                event.metrics["saliency_score"] = decision.value().saliency_score;
                event.metrics["width"] = static_cast<double>(frame.width);
                event.metrics["height"] = static_cast<double>(frame.height);
                const auto event_status = event_sink->Publish(std::move(event));
                if (!event_status.ok()) {
                    return event_status;
                }
            }
            if (decision.value().submit_to_vlm && encoder && encoded_sink) {
                const auto sequence = selected_sequence->fetch_add(1, std::memory_order_relaxed) + 1;
                auto encoded = encoder->Encode(frame, decision.value().saliency_score);
                if (!encoded.ok()) {
                    if (auto ordered_sink = std::dynamic_pointer_cast<IOrderedEncodedFrameSink>(encoded_sink)) {
                        static_cast<void>(ordered_sink->MarkSkipped(
                            frame.session_id,
                            execution_id,
                            sequence,
                            encoded.status()));
                    }
                    return encoded.status();
                }
                encoded.value().metadata().execution_id = execution_id;
                encoded.value().metadata().selected_sequence = sequence;
                if (!io_pool) {
                    return encoded_sink->Publish(std::move(encoded).value());
                }
                auto submit_status = io_pool->Submit(
                    *publish_tasks,
                    [encoded_sink, encoded_frame = std::move(encoded).value()]() mutable -> core::Status {
                        return encoded_sink->Publish(std::move(encoded_frame));
                    },
                    {},
                    "vision-frame-ipc-publish");
                if (!submit_status.ok()) {
                    if (auto ordered_sink = std::dynamic_pointer_cast<IOrderedEncodedFrameSink>(encoded_sink)) {
                        static_cast<void>(ordered_sink->MarkSkipped(
                            frame.session_id,
                            execution_id,
                            sequence,
                            submit_status));
                    }
                }
                return submit_status;
            }
            return core::Status::Ok();
        },
        "vision-frame-sampler");
    if (!status.ok()) {
        NotifyPipelineStatus(status, false);
    }
}

std::string WebRtcMediaPipeline::FrameStreamKey() const {
    return options_.execution_id.empty() ? options_.session_id : options_.execution_id;
}

void WebRtcMediaPipeline::OnPadAdded(GstElement*, GstPad* pad, gpointer user_data) {
    auto* self = static_cast<WebRtcMediaPipeline*>(user_data);
    if (self) {
        (void)self->LinkIncomingPad(pad);
    }
}

gint WebRtcMediaPipeline::OnAutoplugSelect(
    GstElement*,
    GstPad*,
    GstCaps*,
    GstElementFactory* factory,
    gpointer user_data) {
    auto* self = static_cast<WebRtcMediaPipeline*>(user_data);
    if (!self || !factory || !IsDecoderFactory(factory)) {
        return kAutoplugTry;
    }

    switch (self->options_.decode.video_decoder) {
    case VideoDecoderPreference::Auto:
        return kAutoplugTry;
    case VideoDecoderPreference::Software:
        return IsHardwareDecoderFactory(factory) ? kAutoplugSkip : kAutoplugTry;
    case VideoDecoderPreference::Nvidia:
        return IsNvidiaDecoderFactory(factory) ? kAutoplugTry : kAutoplugSkip;
    case VideoDecoderPreference::Vaapi:
        return IsVaapiDecoderFactory(factory) ? kAutoplugTry : kAutoplugSkip;
    case VideoDecoderPreference::D3D11:
        return kAutoplugSkip;
    }
    return kAutoplugTry;
}

void WebRtcMediaPipeline::OnDecodebinPadAdded(GstElement* decodebin, GstPad* pad, gpointer user_data) {
    auto* videoconvert = static_cast<GstElement*>(user_data);
    GstObject* parent = gst_element_get_parent(decodebin);
    if (!parent) {
        return;
    }
    auto* self = static_cast<WebRtcMediaPipeline*>(g_object_get_data(G_OBJECT(parent), "agent-webrtc-pipeline"));
    gst_object_unref(parent);
    if (self) {
        (void)self->LinkDecodebinPad(decodebin, pad, videoconvert);
    }
}

GstFlowReturn WebRtcMediaPipeline::OnNewSample(GstElement* appsink, gpointer user_data) {
    auto* self = static_cast<WebRtcMediaPipeline*>(user_data);
    if (!self) {
        return GST_FLOW_ERROR;
    }

    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
    if (!sample) {
        return GST_FLOW_ERROR;
    }

    auto status = self->HandleSample(sample);
    gst_sample_unref(sample);
    return status.ok() ? GST_FLOW_OK : GST_FLOW_ERROR;
}

void WebRtcMediaPipeline::OnBusMessage(GstBus*, GstMessage* message, gpointer user_data) {
    auto* self = static_cast<WebRtcMediaPipeline*>(user_data);
    if (!self || !message) {
        return;
    }

    switch (GST_MESSAGE_TYPE(message)) {
    case GST_MESSAGE_ERROR: {
        GError* raw_error = nullptr;
        gchar* raw_debug = nullptr;
        gst_message_parse_error(message, &raw_error, &raw_debug);
        UniqueGError error(raw_error);
        UniqueGChar debug(raw_debug);
        std::string detail = error && error->message ? error->message : "GStreamer pipeline error";
        if (debug && std::strlen(debug.get()) > 0) {
            detail += " debug=";
            detail += debug.get();
        }
        self->NotifyPipelineStatus(core::Status::Error(core::ErrorCode::InternalError, detail), true);
        break;
    }
    case GST_MESSAGE_WARNING: {
        GError* raw_error = nullptr;
        gchar* raw_debug = nullptr;
        gst_message_parse_warning(message, &raw_error, &raw_debug);
        UniqueGError error(raw_error);
        UniqueGChar debug(raw_debug);
        std::string detail = error && error->message ? error->message : "GStreamer pipeline warning";
        if (debug && std::strlen(debug.get()) > 0) {
            detail += " debug=";
            detail += debug.get();
        }
        self->NotifyPipelineStatus(core::Status::Error(core::ErrorCode::Unavailable, detail), false);
        break;
    }
    case GST_MESSAGE_EOS:
        self->NotifyPipelineStatus(
            core::Status::Error(core::ErrorCode::Unavailable, "GStreamer pipeline reached EOS"),
            true);
        break;
    default:
        break;
    }
}

void WebRtcMediaPipeline::NotifyPipelineStatus(core::Status status, bool fatal) {
    if (options_.status_observer) {
        options_.status_observer(options_.session_id, std::move(status), fatal);
    }
}

} // namespace media
