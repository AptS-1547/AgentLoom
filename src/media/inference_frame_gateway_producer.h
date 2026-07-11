#pragma once

#include "frame_encoding.h"
#include "inference_frame_shared_memory.h"

#include <atomic>
#include <cstddef>
#include <memory>

namespace media {

struct InferenceFrameGatewayProducerSnapshot {
    std::size_t published_frames = 0;
    std::size_t rejected_frames = 0;
    std::size_t published_bytes = 0;
};

class IInferenceFrameGatewayProducer : public IEncodedVideoFrameSink {
public:
    ~IInferenceFrameGatewayProducer() override = default;
    virtual InferenceFrameGatewayProducerSnapshot Snapshot() const = 0;
};

class InferenceFrameGatewayProducer final : public IInferenceFrameGatewayProducer {
public:
    explicit InferenceFrameGatewayProducer(
        ipc::media::IInferenceFrameIpcSink& sink,
        core::LoggerAdapter logger = {});

    core::Status Publish(EncodedVideoFrame frame) override;
    InferenceFrameGatewayProducerSnapshot Snapshot() const override;

private:
    ipc::media::IInferenceFrameIpcSink& sink_;
    core::LoggerAdapter logger_;
    std::atomic<std::size_t> published_frames_{0};
    std::atomic<std::size_t> rejected_frames_{0};
    std::atomic<std::size_t> published_bytes_{0};
};

} // namespace media
