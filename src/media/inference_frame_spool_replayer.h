#pragma once

#include "inference_frame_backlog.h"
#include "inference_frame_spool.h"

#include <cstddef>
#include <memory>

namespace media::inference {

struct InferenceFrameSpoolReplayOptions {
    std::size_t max_records_per_pump = 8;
};

struct InferenceFrameSpoolReplaySnapshot {
    std::size_t replayed_to_backlog = 0;
    std::size_t backpressure_events = 0;
    bool pending_frame = false;
    bool spool_exhausted = false;
    bool complete = false;
};

class IInferenceFrameSpoolReplayer {
public:
    virtual ~IInferenceFrameSpoolReplayer() = default;

    virtual core::Result<InferenceFrameSpoolReplaySnapshot> Pump() = 0;
    virtual InferenceFrameSpoolReplaySnapshot Snapshot() const = 0;
};

class InferenceFrameSpoolReplayer final : public IInferenceFrameSpoolReplayer {
public:
    InferenceFrameSpoolReplayer(
        IInferenceFrameSpool& spool,
        core::RawMemoryPool& memory_pool,
        IInferenceFrameBacklog& backlog,
        InferenceFrameSpoolReplayOptions options = {},
        core::LoggerAdapter logger = {});
    ~InferenceFrameSpoolReplayer() override;

    InferenceFrameSpoolReplayer(const InferenceFrameSpoolReplayer&) = delete;
    InferenceFrameSpoolReplayer& operator=(const InferenceFrameSpoolReplayer&) = delete;

    core::Result<InferenceFrameSpoolReplaySnapshot> Pump() override;
    InferenceFrameSpoolReplaySnapshot Snapshot() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace media::inference
