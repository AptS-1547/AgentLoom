#pragma once

#include "logger_adapter.h"
#include "shared_memory_block.h"
#include "vision_runtime_interfaces.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace media {

enum class EncodedVideoFrameFormat {
    Jpeg,
    Png
};

enum class VideoImageEncoderPreference {
    Auto,
    Software,
    Nvidia,
    D3D11,
    Vaapi,
    Qsv
};

enum class VideoImageEncoderBackend {
    Software,
    Nvidia,
    D3D11,
    Vaapi,
    Qsv
};

struct VideoFrameEncodingOptions {
    EncodedVideoFrameFormat format = EncodedVideoFrameFormat::Jpeg;
    VideoImageEncoderPreference preference = VideoImageEncoderPreference::Auto;
    int jpeg_quality = 85;
    int png_compression = 3;
    std::size_t max_encoded_bytes = 4 * 1024 * 1024;
    std::chrono::milliseconds timeout{2000};
};

struct VideoFrameEncoderSelection {
    VideoImageEncoderBackend backend = VideoImageEncoderBackend::Software;
    std::string factory_name;
    bool hardware = false;
};

struct EncodedVideoFrameMetadata {
    std::string session_id;
    std::string trace_id;
    std::uint64_t frame_id = 0;
    std::int64_t timestamp_us = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    EncodedVideoFrameFormat format = EncodedVideoFrameFormat::Jpeg;
    double saliency = 0.0;
};

class EncodedVideoFrame {
public:
    EncodedVideoFrame() = default;
    EncodedVideoFrame(
        EncodedVideoFrameMetadata metadata,
        core::SharedMemoryBlock payload,
        std::size_t payload_size) noexcept;

    const EncodedVideoFrameMetadata& metadata() const noexcept;
    std::span<const std::byte> bytes() const noexcept;
    bool valid() const noexcept;

private:
    EncodedVideoFrameMetadata metadata_;
    core::SharedMemoryBlock payload_;
    std::size_t payload_size_ = 0;
};

class IVideoFrameEncoder {
public:
    virtual ~IVideoFrameEncoder() = default;

    virtual core::Result<EncodedVideoFrame> Encode(
        const VideoFrameView& frame,
        double saliency,
        std::string trace_id = {}) = 0;
    virtual const VideoFrameEncoderSelection& Selection() const noexcept = 0;
};

class IEncodedVideoFrameSink {
public:
    virtual ~IEncodedVideoFrameSink() = default;
    virtual core::Status Publish(EncodedVideoFrame frame) = 0;
};

class GStreamerVideoFrameEncoder final : public IVideoFrameEncoder {
public:
    static core::Result<std::shared_ptr<GStreamerVideoFrameEncoder>> Create(
        core::RawMemoryPool& memory_pool,
        VideoFrameEncodingOptions options = {},
        core::LoggerAdapter logger = {});

    ~GStreamerVideoFrameEncoder() override;

    GStreamerVideoFrameEncoder(const GStreamerVideoFrameEncoder&) = delete;
    GStreamerVideoFrameEncoder& operator=(const GStreamerVideoFrameEncoder&) = delete;

    core::Result<EncodedVideoFrame> Encode(
        const VideoFrameView& frame,
        double saliency,
        std::string trace_id = {}) override;
    const VideoFrameEncoderSelection& Selection() const noexcept override;

private:
    class Impl;
    explicit GStreamerVideoFrameEncoder(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace media
