#pragma once

#include "inference_frame_backlog.h"
#include "inference_frame_shared_memory.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string_view>

namespace media::inference {

struct InferenceFrameIpcReceiverSnapshot {
    std::size_t received_frames = 0;
    std::size_t copied_frames = 0;
    std::size_t submitted_frames = 0;
    std::size_t rejected_frames = 0;
    bool shutdown = false;
};

using InferenceFrameIpcReceiverObserver = std::function<void(
    const ipc::media::SharedFrameMetadata& metadata,
    const core::Status& status)>;

class IInferenceFrameIpcReceiver {
public:
    virtual ~IInferenceFrameIpcReceiver() = default;

    virtual core::Status PollOnce() = 0;
    virtual core::Status CloseSession(std::string_view session_id) = 0;
    virtual void Shutdown() = 0;
    virtual InferenceFrameIpcReceiverSnapshot Snapshot() const = 0;
};

class InferenceFrameIpcReceiver final : public IInferenceFrameIpcReceiver {
public:
    InferenceFrameIpcReceiver(
        ipc::media::IInferenceFrameIpcSource& source,
        core::RawMemoryPool& memory_pool,
        IInferenceFrameBacklog& backlog,
        core::LoggerAdapter logger = {},
        InferenceFrameIpcReceiverObserver observer = {});
    ~InferenceFrameIpcReceiver() override;

    InferenceFrameIpcReceiver(const InferenceFrameIpcReceiver&) = delete;
    InferenceFrameIpcReceiver& operator=(const InferenceFrameIpcReceiver&) = delete;

    core::Status PollOnce() override;
    core::Status CloseSession(std::string_view session_id) override;
    void Shutdown() override;
    InferenceFrameIpcReceiverSnapshot Snapshot() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace media::inference
