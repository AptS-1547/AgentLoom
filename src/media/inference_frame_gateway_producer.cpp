#include "inference_frame_gateway_producer.h"

#include <chrono>
#include <utility>

namespace media {
namespace {

ipc::media::SharedFrameFormat ToSharedFormat(EncodedVideoFrameFormat format) {
    switch (format) {
    case EncodedVideoFrameFormat::Jpeg:
        return ipc::media::SharedFrameFormat::Jpeg;
    case EncodedVideoFrameFormat::Png:
        return ipc::media::SharedFrameFormat::Png;
    }
    return ipc::media::SharedFrameFormat::Unknown;
}

} // namespace

InferenceFrameGatewayProducer::InferenceFrameGatewayProducer(
    ipc::media::IInferenceFrameIpcSink& sink,
    core::LoggerAdapter logger)
    : sink_(sink),
      logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("frame-ipc-producer")) {}

core::Status InferenceFrameGatewayProducer::Publish(EncodedVideoFrame frame) {
    return PublishBorrowed(frame);
}

core::Status InferenceFrameGatewayProducer::PublishBorrowed(const EncodedVideoFrame& frame) {
    if (!frame.valid()) {
        rejected_frames_.fetch_add(1, std::memory_order_relaxed);
        const auto status = core::Status::Error(core::ErrorCode::InvalidArgument, "encoded video frame is invalid");
        logger_.warn(
            "[frame-ipc-producer] frame rejected code={} message={}",
            static_cast<int>(status.code()),
            status.message());
        return status;
    }

    const auto& metadata = frame.metadata();
    const auto payload = frame.bytes();
    const auto published_at_unix_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const auto status = sink_.Publish({
        .execution_id = metadata.execution_id,
        .session_id = metadata.session_id,
        .trace_id = metadata.trace_id,
        .selected_sequence = metadata.selected_sequence,
        .transport_sequence = metadata.transport_sequence,
        .frame_id = metadata.frame_id,
        .timestamp_us = metadata.timestamp_us,
        .published_at_unix_us = published_at_unix_us,
        .width = metadata.width,
        .height = metadata.height,
        .format = static_cast<std::uint32_t>(ToSharedFormat(metadata.format)),
        .saliency = metadata.saliency,
        .payload = payload,
    });
    if (!status.ok()) {
        rejected_frames_.fetch_add(1, std::memory_order_relaxed);
        if (status.code() == core::ErrorCode::ResourceExhausted) {
            logger_.debug(
                "[frame-ipc-producer] backpressure session={} frame={} bytes={}",
                metadata.session_id,
                metadata.frame_id,
                payload.size());
        } else {
            logger_.warn(
                "[frame-ipc-producer] publish failed session={} trace={} frame={} code={} message={}",
                metadata.session_id,
                metadata.trace_id,
                metadata.frame_id,
                static_cast<int>(status.code()),
                status.message());
        }
        return status;
    }

    published_frames_.fetch_add(1, std::memory_order_relaxed);
    published_bytes_.fetch_add(payload.size(), std::memory_order_relaxed);
    return core::Status::Ok();
}

InferenceFrameGatewayProducerSnapshot InferenceFrameGatewayProducer::Snapshot() const {
    return {
        .published_frames = published_frames_.load(std::memory_order_relaxed),
        .rejected_frames = rejected_frames_.load(std::memory_order_relaxed),
        .published_bytes = published_bytes_.load(std::memory_order_relaxed),
    };
}

} // namespace media
