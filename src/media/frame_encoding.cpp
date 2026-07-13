#include "frame_encoding.h"

#include "webrtc_bin.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video-info.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace media {
namespace {

struct GstObjectDeleter {
    void operator()(GstObject* object) const noexcept {
        if (object) {
            gst_object_unref(object);
        }
    }
};

struct GstCapsDeleter {
    void operator()(GstCaps* caps) const noexcept {
        if (caps) {
            gst_caps_unref(caps);
        }
    }
};

struct GstSampleDeleter {
    void operator()(GstSample* sample) const noexcept {
        if (sample) {
            gst_sample_unref(sample);
        }
    }
};

struct GstElementDeleter {
    void operator()(GstElement* element) const noexcept {
        if (element) {
            gst_object_unref(element);
        }
    }
};

struct GstBufferDeleter {
    void operator()(GstBuffer* buffer) const noexcept {
        if (buffer) {
            gst_buffer_unref(buffer);
        }
    }
};

using UniqueCaps = std::unique_ptr<GstCaps, GstCapsDeleter>;
using UniqueSample = std::unique_ptr<GstSample, GstSampleDeleter>;
using UniqueElement = std::unique_ptr<GstElement, GstElementDeleter>;
using UniqueBuffer = std::unique_ptr<GstBuffer, GstBufferDeleter>;

struct EncoderCandidate {
    VideoImageEncoderBackend backend;
    const char* factory_name;
    bool hardware;
};

bool HasFactory(const char* name) {
    GstElementFactory* factory = gst_element_factory_find(name);
    if (!factory) {
        return false;
    }
    gst_object_unref(factory);
    return true;
}

std::vector<EncoderCandidate> Candidates(const VideoFrameEncodingOptions& options) {
    const auto software = options.format == EncodedVideoFrameFormat::Jpeg
        ? std::vector<EncoderCandidate>{{VideoImageEncoderBackend::Software, "jpegenc", false},
                                        {VideoImageEncoderBackend::Software, "avenc_mjpeg", false}}
        : std::vector<EncoderCandidate>{{VideoImageEncoderBackend::Software, "pngenc", false},
                                        {VideoImageEncoderBackend::Software, "rspngenc", false},
                                        {VideoImageEncoderBackend::Software, "avenc_png", false}};

    if (options.format == EncodedVideoFrameFormat::Png) {
        return options.preference == VideoImageEncoderPreference::Auto ||
                       options.preference == VideoImageEncoderPreference::Software
            ? software
            : std::vector<EncoderCandidate>{};
    }

    switch (options.preference) {
    case VideoImageEncoderPreference::Auto:
        return {
            {VideoImageEncoderBackend::Nvidia, "nvjpegenc", true},
            {VideoImageEncoderBackend::Qsv, "qsvjpegenc", true},
            {VideoImageEncoderBackend::Vaapi, "vaapijpegenc", true},
            {VideoImageEncoderBackend::Vaapi, "vajpegenc", true},
            {VideoImageEncoderBackend::D3D11, "d3d11jpegenc", true},
            software[0],
            software[1],
        };
    case VideoImageEncoderPreference::Software:
        return software;
    case VideoImageEncoderPreference::Nvidia:
        return {{VideoImageEncoderBackend::Nvidia, "nvjpegenc", true}};
    case VideoImageEncoderPreference::D3D11:
        return {{VideoImageEncoderBackend::D3D11, "d3d11jpegenc", true}};
    case VideoImageEncoderPreference::Vaapi:
        return {{VideoImageEncoderBackend::Vaapi, "vaapijpegenc", true},
                {VideoImageEncoderBackend::Vaapi, "vajpegenc", true}};
    case VideoImageEncoderPreference::Qsv:
        return {{VideoImageEncoderBackend::Qsv, "qsvjpegenc", true}};
    }
    return {};
}

core::Status ValidateOptions(const VideoFrameEncodingOptions& options) {
    if (options.jpeg_quality < 1 || options.jpeg_quality > 100) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "jpeg_quality must be in [1, 100]");
    }
    if (options.png_compression < 0 || options.png_compression > 9) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "png_compression must be in [0, 9]");
    }
    if (options.max_encoded_bytes == 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "max_encoded_bytes must be positive");
    }
    if (options.timeout <= std::chrono::milliseconds::zero()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "encoder timeout must be positive");
    }
    return core::Status::Ok();
}

core::Status ValidateFrame(const VideoFrameView& frame) {
    if (frame.session_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "video frame session_id is required");
    }
    if (frame.format != VideoPixelFormat::Rgb && frame.format != VideoPixelFormat::Bgr) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "image encoder expects RGB or BGR input");
    }
    if (frame.width == 0 || frame.height == 0 || frame.bytes.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "video frame is empty");
    }
    if (frame.width > std::numeric_limits<std::size_t>::max() / frame.height / 3) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "video frame dimensions overflow");
    }
    const auto packed_stride = static_cast<std::size_t>(frame.width) * 3;
    const auto row_stride = frame.row_stride_bytes == 0 ? packed_stride : frame.row_stride_bytes;
    if (row_stride < packed_stride) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "video frame stride is smaller than width*3");
    }
    const auto expected = row_stride * frame.height;
    if (frame.bytes.size() < expected) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "video frame payload is smaller than width*height*3");
    }
    return core::Status::Ok();
}

void SetIntegerPropertyIfPresent(GstElement* element, const char* name, int value) {
    if (g_object_class_find_property(G_OBJECT_GET_CLASS(element), name)) {
        g_object_set(G_OBJECT(element), name, value, nullptr);
    }
}

std::int64_t TimestampMicros(std::chrono::steady_clock::time_point value) {
    return std::chrono::duration_cast<std::chrono::microseconds>(value.time_since_epoch()).count();
}

} // namespace

EncodedVideoFrame::EncodedVideoFrame(
    EncodedVideoFrameMetadata metadata,
    core::SharedMemoryBlock payload,
    std::size_t payload_size) noexcept
    : metadata_(std::move(metadata)),
      payload_(std::move(payload)),
      payload_size_(payload_size) {}

const EncodedVideoFrameMetadata& EncodedVideoFrame::metadata() const noexcept {
    return metadata_;
}

EncodedVideoFrameMetadata& EncodedVideoFrame::metadata() noexcept {
    return metadata_;
}

std::span<const std::byte> EncodedVideoFrame::bytes() const noexcept {
    return {payload_.data(), std::min(payload_.size(), payload_size_)};
}

bool EncodedVideoFrame::valid() const noexcept {
    return !metadata_.session_id.empty() && payload_.valid() && payload_size_ > 0 && payload_size_ <= payload_.size();
}

class GStreamerVideoFrameEncoder::Impl {
public:
    static core::Result<std::unique_ptr<Impl>> Create(
        core::RawMemoryPool& memory_pool,
        VideoFrameEncodingOptions options,
        core::LoggerAdapter logger) {
        const auto validation = ValidateOptions(options);
        if (!validation.ok()) {
            return validation;
        }
        const auto init_status = GstInitializer::EnsureInitialized();
        if (!init_status.ok()) {
            return init_status;
        }

        auto effective_logger = logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("frame-encoder");
        for (const auto& candidate : Candidates(options)) {
            if (!HasFactory(candidate.factory_name)) {
                continue;
            }
            auto impl = std::unique_ptr<Impl>(new Impl(memory_pool, options, effective_logger));
            const auto status = impl->Build(candidate);
            if (status.ok()) {
                const auto probe = impl->Probe();
                if (!probe.ok()) {
                    effective_logger.warn(
                        "[frame-encoder] factory probe failed factory={} code={} message={}",
                        candidate.factory_name,
                        static_cast<int>(probe.code()),
                        probe.message());
                    continue;
                }
                effective_logger.info(
                    "[frame-encoder] selected factory={} hardware={}",
                    candidate.factory_name,
                    candidate.hardware);
                return impl;
            }
            effective_logger.warn(
                "[frame-encoder] factory initialization failed factory={} code={} message={}",
                candidate.factory_name,
                static_cast<int>(status.code()),
                status.message());
        }

        return core::Status::Error(
            core::ErrorCode::Unavailable,
            "no compatible image encoder factory is available");
    }

    ~Impl() {
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
            appsrc_ = nullptr;
            appsink_ = nullptr;
        }
    }

    core::Result<EncodedVideoFrame> Encode(
        const VideoFrameView& frame,
        double saliency,
        std::string trace_id) {
        const auto validation = ValidateFrame(frame);
        if (!validation.ok()) {
            return LogFailure(validation, frame);
        }

        std::lock_guard lock(mutex_);
        const char* input_format = frame.format == VideoPixelFormat::Rgb ? "RGB" : "BGR";
        UniqueCaps caps(gst_caps_new_simple(
            "video/x-raw",
            "format", G_TYPE_STRING, input_format,
            "width", G_TYPE_INT, static_cast<int>(frame.width),
            "height", G_TYPE_INT, static_cast<int>(frame.height),
            "framerate", GST_TYPE_FRACTION, 0, 1,
            nullptr));
        if (!caps) {
            return LogFailure(
                core::Status::Error(core::ErrorCode::OutOfMemory, "failed to allocate encoder input caps"),
                frame);
        }
        gst_app_src_set_caps(GST_APP_SRC(appsrc_), caps.get());

        GstVideoInfo input_info;
        gst_video_info_init(&input_info);
        if (!gst_video_info_from_caps(&input_info, caps.get())) {
            return LogFailure(
                core::Status::Error(core::ErrorCode::InvalidArgument, "failed to derive encoder input layout"),
                frame);
        }
        const auto target_stride_value = GST_VIDEO_INFO_PLANE_STRIDE(&input_info, 0);
        if (target_stride_value <= 0 || GST_VIDEO_INFO_SIZE(&input_info) == 0) {
            return LogFailure(
                core::Status::Error(core::ErrorCode::InvalidArgument, "encoder input layout is invalid"),
                frame);
        }
        const auto target_stride = static_cast<std::size_t>(target_stride_value);
        UniqueBuffer input(gst_buffer_new_allocate(nullptr, GST_VIDEO_INFO_SIZE(&input_info), nullptr));
        if (!input) {
            return LogFailure(
                core::Status::Error(core::ErrorCode::OutOfMemory, "failed to allocate encoder input buffer"),
                frame);
        }
        GstMapInfo input_map{};
        if (!gst_buffer_map(input.get(), &input_map, GST_MAP_WRITE)) {
            return LogFailure(
                core::Status::Error(core::ErrorCode::InternalError, "failed to map encoder input buffer"),
                frame);
        }
        const auto packed_stride = static_cast<std::size_t>(frame.width) * 3;
        const auto source_stride = frame.row_stride_bytes == 0 ? packed_stride : frame.row_stride_bytes;
        for (std::size_t row = 0; row < frame.height; ++row) {
            std::memcpy(
                input_map.data + row * target_stride,
                frame.bytes.data() + row * source_stride,
                packed_stride);
            if (target_stride > packed_stride) {
                std::memset(
                    input_map.data + row * target_stride + packed_stride,
                    0,
                    target_stride - packed_stride);
            }
        }
        gst_buffer_unmap(input.get(), &input_map);
        const auto timestamp_us = frame.timestamp_us.value_or(TimestampMicros(frame.captured_at));
        GST_BUFFER_PTS(input.get()) = static_cast<GstClockTime>(
            std::max<std::int64_t>(0, timestamp_us)) * GST_USECOND;
        GST_BUFFER_DURATION(input.get()) = GST_CLOCK_TIME_NONE;

        const auto flow = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), input.release());
        if (flow != GST_FLOW_OK) {
            return LogFailure(
                core::Status::Error(core::ErrorCode::Unavailable, "image encoder rejected input frame"),
                frame);
        }

        UniqueSample sample(gst_app_sink_try_pull_sample(
            GST_APP_SINK(appsink_),
            static_cast<GstClockTime>(options_.timeout.count()) * GST_MSECOND));
        if (!sample) {
            return LogFailure(
                core::Status::Error(core::ErrorCode::Timeout, "image encoder output timed out"),
                frame);
        }

        GstBuffer* output = gst_sample_get_buffer(sample.get());
        GstMapInfo map{};
        if (!output || !gst_buffer_map(output, &map, GST_MAP_READ)) {
            return LogFailure(
                core::Status::Error(core::ErrorCode::InternalError, "failed to map encoded image buffer"),
                frame);
        }
        const auto unmap = [&] { gst_buffer_unmap(output, &map); };
        if (map.size == 0 || map.size > options_.max_encoded_bytes) {
            unmap();
            return LogFailure(
                core::Status::Error(core::ErrorCode::ResourceExhausted, "encoded image exceeds configured capacity"),
                frame);
        }

        auto block = memory_pool_.allocate(map.size, alignof(std::max_align_t));
        if (!block.ok()) {
            unmap();
            return LogFailure(block.status(), frame);
        }
        std::memcpy(block.value().data(), map.data, map.size);
        const auto payload_size = map.size;
        unmap();

        auto shared = core::SharedMemoryBlock::adopt(std::move(block).value());
        if (!shared.ok()) {
            return LogFailure(shared.status(), frame);
        }
        EncodedVideoFrameMetadata metadata;
        metadata.session_id = frame.session_id;
        metadata.trace_id = std::move(trace_id);
        metadata.frame_id = frame.frame_id;
        metadata.timestamp_us = timestamp_us;
        metadata.width = frame.width;
        metadata.height = frame.height;
        metadata.format = options_.format;
        metadata.saliency = saliency;
        return EncodedVideoFrame(std::move(metadata), std::move(shared).value(), payload_size);
    }

    const VideoFrameEncoderSelection& Selection() const noexcept {
        return selection_;
    }

private:
    Impl(
        core::RawMemoryPool& memory_pool,
        VideoFrameEncodingOptions options,
        core::LoggerAdapter logger)
        : memory_pool_(memory_pool),
          options_(std::move(options)),
          logger_(std::move(logger)) {}

    core::Status Build(const EncoderCandidate& candidate) {
        UniqueElement pipeline(gst_pipeline_new(nullptr));
        UniqueElement appsrc(gst_element_factory_make("appsrc", nullptr));
        UniqueElement convert(gst_element_factory_make("videoconvert", nullptr));
        UniqueElement encoder(gst_element_factory_make(candidate.factory_name, nullptr));
        UniqueElement appsink(gst_element_factory_make("appsink", nullptr));
        if (!pipeline || !appsrc || !convert || !encoder || !appsink) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to create image encoder elements");
        }

        g_object_set(
            G_OBJECT(appsrc.get()),
            "format", GST_FORMAT_TIME,
            "is-live", FALSE,
            "block", TRUE,
            nullptr);
        g_object_set(
            G_OBJECT(appsink.get()),
            "sync", FALSE,
            "max-buffers", 1,
            "drop", FALSE,
            nullptr);
        if (options_.format == EncodedVideoFrameFormat::Jpeg) {
            SetIntegerPropertyIfPresent(encoder.get(), "quality", options_.jpeg_quality);
        } else {
            SetIntegerPropertyIfPresent(encoder.get(), "compression-level", options_.png_compression);
            SetIntegerPropertyIfPresent(encoder.get(), "compression", options_.png_compression);
        }

        if (!gst_bin_add(GST_BIN(pipeline.get()), appsrc.get())) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to add appsrc to image encoder pipeline");
        }
        appsrc_ = appsrc.release();
        if (!gst_bin_add(GST_BIN(pipeline.get()), convert.get())) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to add videoconvert to image encoder pipeline");
        }
        GstElement* convert_element = convert.release();
        if (!gst_bin_add(GST_BIN(pipeline.get()), encoder.get())) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to add encoder to image encoder pipeline");
        }
        GstElement* encoder_element = encoder.release();
        if (!gst_bin_add(GST_BIN(pipeline.get()), appsink.get())) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to add appsink to image encoder pipeline");
        }
        appsink_ = appsink.release();
        if (!gst_element_link_many(appsrc_, convert_element, encoder_element, appsink_, nullptr)) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to link image encoder pipeline");
        }
        if (gst_element_set_state(pipeline.get(), GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to start image encoder pipeline");
        }

        pipeline_ = pipeline.release();
        selection_ = {
            .backend = candidate.backend,
            .factory_name = candidate.factory_name,
            .hardware = candidate.hardware,
        };
        return core::Status::Ok();
    }

    core::Status Probe() {
        constexpr std::uint32_t width = 16;
        constexpr std::uint32_t height = 16;
        std::vector<std::byte> payload(static_cast<std::size_t>(width) * height * 3, std::byte{0x60});
        VideoFrameView frame;
        frame.session_id = "encoder-probe";
        frame.frame_id = 1;
        frame.width = width;
        frame.height = height;
        frame.row_stride_bytes = width * 3;
        frame.format = VideoPixelFormat::Rgb;
        frame.bytes = {
            reinterpret_cast<const char*>(payload.data()),
            payload.size(),
        };
        auto encoded = Encode(frame, 0.0, {});
        return encoded.ok() ? core::Status::Ok() : encoded.status();
    }

    core::Status LogFailure(core::Status status, const VideoFrameView& frame) {
        logger_.warn(
            "[frame-encoder] encode failed session={} frame={} factory={} code={} message={}",
            frame.session_id,
            frame.frame_id,
            selection_.factory_name,
            static_cast<int>(status.code()),
            status.message());
        return status;
    }

    core::RawMemoryPool& memory_pool_;
    VideoFrameEncodingOptions options_;
    core::LoggerAdapter logger_;
    VideoFrameEncoderSelection selection_;
    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GstElement* appsink_ = nullptr;
    std::mutex mutex_;
};

core::Result<std::shared_ptr<GStreamerVideoFrameEncoder>> GStreamerVideoFrameEncoder::Create(
    core::RawMemoryPool& memory_pool,
    VideoFrameEncodingOptions options,
    core::LoggerAdapter logger) {
    auto impl = Impl::Create(memory_pool, std::move(options), std::move(logger));
    if (!impl.ok()) {
        return impl.status();
    }
    return std::shared_ptr<GStreamerVideoFrameEncoder>(
        new GStreamerVideoFrameEncoder(std::move(impl).value()));
}

GStreamerVideoFrameEncoder::GStreamerVideoFrameEncoder(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

GStreamerVideoFrameEncoder::~GStreamerVideoFrameEncoder() = default;

core::Result<EncodedVideoFrame> GStreamerVideoFrameEncoder::Encode(
    const VideoFrameView& frame,
    double saliency,
    std::string trace_id) {
    return impl_->Encode(frame, saliency, std::move(trace_id));
}

const VideoFrameEncoderSelection& GStreamerVideoFrameEncoder::Selection() const noexcept {
    return impl_->Selection();
}

} // namespace media
