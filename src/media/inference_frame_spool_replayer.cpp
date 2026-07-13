#include "inference_frame_spool_replayer.h"

#include <mutex>
#include <optional>
#include <utility>

namespace media::inference {

class InferenceFrameSpoolReplayer::Impl {
public:
    Impl(IInferenceFrameSpool& spool,
         core::RawMemoryPool& memory_pool,
         IInferenceFrameBacklog& backlog,
         InferenceFrameSpoolReplayOptions options,
         core::LoggerAdapter logger)
        : spool_(spool),
          memory_pool_(memory_pool),
          backlog_(backlog),
          options_(options),
          logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("media-spool-replay")) {}

    core::Result<InferenceFrameSpoolReplaySnapshot> Pump() {
        std::lock_guard lock(mutex_);
        if (options_.max_records_per_pump == 0) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "spool replay max_records_per_pump must be positive");
        }
        if (CompleteLocked()) {
            return SnapshotLocked();
        }

        std::size_t admitted_this_pump = 0;
        while (admitted_this_pump < options_.max_records_per_pump) {
            if (!pending_frame_) {
                auto replay = spool_.ReplayNext();
                if (!replay.ok()) {
                    logger_.warn(
                        "[media-spool-replay] read failed code={} message={}",
                        static_cast<int>(replay.status().code()),
                        replay.status().message());
                    return replay.status();
                }
                if (!replay.value().has_value()) {
                    spool_exhausted_ = true;
                    break;
                }

                const auto& lease = replay.value().value();
                auto copied = CopyInferenceFrame(
                    memory_pool_,
                    lease.metadata().frame,
                    lease.bytes());
                if (!copied.ok()) {
                    logger_.warn(
                        "[media-spool-replay] copy failed execution={} sequence={} code={} message={}",
                        lease.metadata().execution_id,
                        lease.metadata().selected_sequence,
                        static_cast<int>(copied.status().code()),
                        copied.status().message());
                    return copied.status();
                }
                pending_frame_.emplace(std::move(copied).value());
            }

            auto admission = backlog_.TrySubmit(std::move(pending_frame_).value());
            pending_frame_.reset();
            if (admission.accepted()) {
                ++replayed_to_backlog_;
                ++admitted_this_pump;
                continue;
            }
            if (admission.rejected_frame.has_value()) {
                pending_frame_.emplace(std::move(admission.rejected_frame).value());
            }
            if (admission.status.code() == core::ErrorCode::ResourceExhausted && pending_frame_) {
                ++backpressure_events_;
                return SnapshotLocked();
            }
            logger_.warn(
                "[media-spool-replay] backlog admission failed code={} message={}",
                static_cast<int>(admission.status.code()),
                admission.status.message());
            return admission.status;
        }
        return SnapshotLocked();
    }

    InferenceFrameSpoolReplaySnapshot Snapshot() const {
        std::lock_guard lock(mutex_);
        return SnapshotLocked();
    }

private:
    bool CompleteLocked() const noexcept {
        return spool_exhausted_ && !pending_frame_;
    }

    InferenceFrameSpoolReplaySnapshot SnapshotLocked() const noexcept {
        return {
            .replayed_to_backlog = replayed_to_backlog_,
            .backpressure_events = backpressure_events_,
            .pending_frame = pending_frame_.has_value(),
            .spool_exhausted = spool_exhausted_,
            .complete = CompleteLocked(),
        };
    }

    IInferenceFrameSpool& spool_;
    core::RawMemoryPool& memory_pool_;
    IInferenceFrameBacklog& backlog_;
    InferenceFrameSpoolReplayOptions options_;
    core::LoggerAdapter logger_;
    mutable std::mutex mutex_;
    std::optional<OwnedInferenceFrame> pending_frame_;
    std::size_t replayed_to_backlog_ = 0;
    std::size_t backpressure_events_ = 0;
    bool spool_exhausted_ = false;
};

InferenceFrameSpoolReplayer::InferenceFrameSpoolReplayer(
    IInferenceFrameSpool& spool,
    core::RawMemoryPool& memory_pool,
    IInferenceFrameBacklog& backlog,
    InferenceFrameSpoolReplayOptions options,
    core::LoggerAdapter logger)
    : impl_(std::make_unique<Impl>(
          spool,
          memory_pool,
          backlog,
          options,
          std::move(logger))) {}

InferenceFrameSpoolReplayer::~InferenceFrameSpoolReplayer() = default;

core::Result<InferenceFrameSpoolReplaySnapshot> InferenceFrameSpoolReplayer::Pump() {
    return impl_->Pump();
}

InferenceFrameSpoolReplaySnapshot InferenceFrameSpoolReplayer::Snapshot() const {
    return impl_->Snapshot();
}

} // namespace media::inference
