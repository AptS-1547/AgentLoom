#include "inference_frame_ipc_receiver.h"

#include <atomic>
#include <exception>
#include <utility>

namespace media::inference {
namespace {

core::Result<InferenceFrameFormat> ParseFormat(std::uint32_t value) {
    switch (static_cast<ipc::media::SharedFrameFormat>(value)) {
    case ipc::media::SharedFrameFormat::Jpeg:
        return InferenceFrameFormat::Jpeg;
    case ipc::media::SharedFrameFormat::Png:
        return InferenceFrameFormat::Png;
    case ipc::media::SharedFrameFormat::Rgb:
        return InferenceFrameFormat::Rgb;
    case ipc::media::SharedFrameFormat::Bgr:
        return InferenceFrameFormat::Bgr;
    case ipc::media::SharedFrameFormat::Nv12:
        return InferenceFrameFormat::Nv12;
    case ipc::media::SharedFrameFormat::I420:
        return InferenceFrameFormat::I420;
    case ipc::media::SharedFrameFormat::Unknown:
        break;
    }
    return core::Status::Error(core::ErrorCode::InvalidArgument, "shared frame format is invalid");
}

} // namespace

class InferenceFrameIpcReceiver::Impl {
public:
    Impl(
        ipc::media::IInferenceFrameIpcSource& source,
        core::RawMemoryPool& memory_pool,
        IInferenceFrameBacklog& backlog,
        core::LoggerAdapter logger,
        InferenceFrameIpcReceiverObserver observer)
        : source_(source),
          memory_pool_(memory_pool),
          backlog_(backlog),
          logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("media-inference")),
          observer_(std::move(observer)) {}

    core::Status PollOnce() {
        if (shutdown_.load(std::memory_order_acquire)) {
            return core::Status::Error(core::ErrorCode::Cancelled, "inference frame IPC receiver is shut down");
        }

        try {
            auto claimed = source_.TryClaim();
            if (!claimed.ok()) {
                return claimed.status();
            }
            received_frames_.fetch_add(1, std::memory_order_relaxed);

            auto format = ParseFormat(claimed.value().metadata().format);
            if (!format.ok()) {
                claimed.value().Acknowledge();
                return Reject(format.status(), claimed.value().metadata());
            }

            InferenceFrameMetadata metadata;
            metadata.session_id = claimed.value().metadata().session_id;
            metadata.trace_id = claimed.value().metadata().trace_id;
            metadata.frame_id = claimed.value().metadata().frame_id;
            metadata.timestamp_us = claimed.value().metadata().timestamp_us;
            metadata.width = claimed.value().metadata().width;
            metadata.height = claimed.value().metadata().height;
            metadata.format = format.value();
            metadata.saliency = claimed.value().metadata().saliency;

            auto copied = CopyInferenceFrame(
                memory_pool_,
                std::move(metadata),
                claimed.value().payload());
            const auto acknowledge_status = claimed.value().Acknowledge();
            if (!acknowledge_status.ok()) {
                return Reject(acknowledge_status, claimed.value().metadata());
            }
            if (!copied.ok()) {
                return Reject(copied.status(), claimed.value().metadata());
            }
            copied_frames_.fetch_add(1, std::memory_order_relaxed);

            const auto submit_status = backlog_.Submit(std::move(copied).value());
            if (!submit_status.ok()) {
                return Reject(submit_status, claimed.value().metadata());
            }
            submitted_frames_.fetch_add(1, std::memory_order_relaxed);
            Notify(claimed.value().metadata(), core::Status::Ok());
            return core::Status::Ok();
        } catch (const std::exception&) {
            return Reject(
                core::Status::Error(core::ErrorCode::InternalError, "inference frame IPC receiver failed unexpectedly"),
                {});
        } catch (...) {
            return Reject(
                core::Status::Error(core::ErrorCode::Unknown, "inference frame IPC receiver failed unexpectedly"),
                {});
        }
    }

    core::Status CloseSession(std::string_view session_id) {
        const auto status = backlog_.CloseSession(session_id);
        if (!status.ok()) {
            logger_.warn(
                "[frame-ipc-receiver] close session failed session={} code={} message={}",
                session_id,
                static_cast<int>(status.code()),
                status.message());
        }
        return status;
    }

    void Shutdown() {
        if (!shutdown_.exchange(true, std::memory_order_acq_rel)) {
            source_.Shutdown();
        }
    }

    InferenceFrameIpcReceiverSnapshot Snapshot() const {
        return {
            .received_frames = received_frames_.load(std::memory_order_relaxed),
            .copied_frames = copied_frames_.load(std::memory_order_relaxed),
            .submitted_frames = submitted_frames_.load(std::memory_order_relaxed),
            .rejected_frames = rejected_frames_.load(std::memory_order_relaxed),
            .shutdown = shutdown_.load(std::memory_order_acquire),
        };
    }

private:
    core::Status Reject(
        core::Status status,
        const ipc::media::SharedFrameMetadata& metadata) {
        rejected_frames_.fetch_add(1, std::memory_order_relaxed);
        logger_.warn(
            "[frame-ipc-receiver] frame rejected session={} trace={} frame={} code={} message={}",
            metadata.session_id,
            metadata.trace_id,
            metadata.frame_id,
            static_cast<int>(status.code()),
            status.message());
        Notify(metadata, status);
        return status;
    }

    void Notify(
        const ipc::media::SharedFrameMetadata& metadata,
        const core::Status& status) const noexcept {
        if (!observer_) {
            return;
        }
        try {
            observer_(metadata, status);
        } catch (const std::exception&) {
            logger_.warn("[frame-ipc-receiver] outcome observer failed unexpectedly");
        } catch (...) {
            logger_.warn("[frame-ipc-receiver] outcome observer failed unexpectedly");
        }
    }

    ipc::media::IInferenceFrameIpcSource& source_;
    core::RawMemoryPool& memory_pool_;
    IInferenceFrameBacklog& backlog_;
    core::LoggerAdapter logger_;
    InferenceFrameIpcReceiverObserver observer_;
    std::atomic<std::size_t> received_frames_{0};
    std::atomic<std::size_t> copied_frames_{0};
    std::atomic<std::size_t> submitted_frames_{0};
    std::atomic<std::size_t> rejected_frames_{0};
    std::atomic<bool> shutdown_{false};
};

InferenceFrameIpcReceiver::InferenceFrameIpcReceiver(
    ipc::media::IInferenceFrameIpcSource& source,
    core::RawMemoryPool& memory_pool,
    IInferenceFrameBacklog& backlog,
    core::LoggerAdapter logger,
    InferenceFrameIpcReceiverObserver observer)
    : impl_(std::make_unique<Impl>(
          source,
          memory_pool,
          backlog,
          std::move(logger),
          std::move(observer))) {}

InferenceFrameIpcReceiver::~InferenceFrameIpcReceiver() = default;

core::Status InferenceFrameIpcReceiver::PollOnce() {
    return impl_->PollOnce();
}

core::Status InferenceFrameIpcReceiver::CloseSession(std::string_view session_id) {
    return impl_->CloseSession(session_id);
}

void InferenceFrameIpcReceiver::Shutdown() {
    impl_->Shutdown();
}

InferenceFrameIpcReceiverSnapshot InferenceFrameIpcReceiver::Snapshot() const {
    return impl_->Snapshot();
}

} // namespace media::inference
