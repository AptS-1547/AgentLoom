#include "inference_frame_ipc_control.h"

#include <utility>

namespace ipc::media {

InferenceFrameIpcGrantReceiver::InferenceFrameIpcGrantReceiver(core::LoggerAdapter logger)
    : logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("frame-ipc-control")) {}

core::Status InferenceFrameIpcGrantReceiver::ApplyGrant(const InferenceFrameIpcGrant& grant) {
    std::unique_lock lock(mutex_);
    if (snapshot_.state == InferenceFrameIpcControlState::Shutdown) {
        return core::Status::Error(core::ErrorCode::Cancelled, "shared frame grant receiver is shut down");
    }
    if (grant.channel_name.empty() || grant.epoch == 0 ||
        grant.slot_count < 2 || grant.payload_capacity == 0) {
        const auto status = core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "shared frame IPC grant is invalid");
        snapshot_.state = InferenceFrameIpcControlState::Failed;
        snapshot_.last_status = status;
        ++snapshot_.failures;
        return status;
    }

    core::Status status;
    if (source_) {
        status = source_->ApplyGrant(grant);
    } else {
        auto opened = ReconnectableInferenceFrameIpcSource::Open({
            .name = grant.channel_name,
            .slot_count = grant.slot_count,
            .payload_capacity = grant.payload_capacity,
            .expected_epoch = grant.epoch,
        }, logger_);
        if (opened.ok()) {
            source_ = std::move(opened).value();
            status = core::Status::Ok();
        } else {
            status = opened.status();
        }
    }

    if (!status.ok()) {
        snapshot_.state = InferenceFrameIpcControlState::Failed;
        snapshot_.last_status = status;
        ++snapshot_.failures;
        logger_.warn(
            "[frame-ipc-control] grant apply failed name={} epoch={} code={} message={}",
            grant.channel_name,
            grant.epoch,
            static_cast<int>(status.code()),
            status.message());
        return status;
    }

    snapshot_.state = InferenceFrameIpcControlState::Granted;
    snapshot_.grant = grant;
    snapshot_.last_status = core::Status::Ok();
    ++snapshot_.grants_applied;
    logger_.info(
        "[frame-ipc-control] grant applied name={} epoch={}",
        grant.channel_name,
        grant.epoch);
    return core::Status::Ok();
}

core::Status InferenceFrameIpcGrantReceiver::Revoke(
    std::uint64_t epoch,
    std::string_view reason) {
    std::unique_lock lock(mutex_);
    if (snapshot_.state == InferenceFrameIpcControlState::Shutdown) {
        return core::Status::Error(core::ErrorCode::Cancelled, "shared frame grant receiver is shut down");
    }
    if (snapshot_.grant.epoch != 0 && snapshot_.grant.epoch != epoch) {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "shared frame revoke epoch does not match active grant");
    }
    if (source_) {
        source_->Shutdown();
        source_.reset();
    }
    snapshot_.state = InferenceFrameIpcControlState::Fenced;
    snapshot_.last_status = core::Status::Error(core::ErrorCode::Cancelled, std::string(reason));
    ++snapshot_.revocations;
    logger_.warn(
        "[frame-ipc-control] grant revoked epoch={} reason={}",
        epoch,
        reason);
    return core::Status::Ok();
}

core::Status InferenceFrameIpcGrantReceiver::Probe(std::uint64_t expected_epoch) {
    std::shared_lock lock(mutex_);
    if (!source_ || snapshot_.state != InferenceFrameIpcControlState::Granted) {
        return core::Status::Error(core::ErrorCode::Unavailable, "shared frame inference peer is not granted");
    }
    if (snapshot_.grant.epoch != expected_epoch) {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "shared frame inference peer epoch does not match");
    }
    return core::Status::Ok();
}

core::Result<ClaimedSharedFrame> InferenceFrameIpcGrantReceiver::TryClaim() {
    std::shared_lock lock(mutex_);
    if (!source_ || snapshot_.state != InferenceFrameIpcControlState::Granted) {
        return core::Status::Error(core::ErrorCode::Cancelled, "shared frame grant is not active");
    }
    return source_->TryClaim();
}

void InferenceFrameIpcGrantReceiver::Shutdown() {
    std::unique_lock lock(mutex_);
    if (source_) {
        source_->Shutdown();
        source_.reset();
    }
    snapshot_.state = InferenceFrameIpcControlState::Shutdown;
    snapshot_.last_status = core::Status::Ok();
}

InferenceFrameSharedMemorySnapshot InferenceFrameIpcGrantReceiver::Snapshot() const {
    std::shared_lock lock(mutex_);
    return source_ ? source_->Snapshot() : InferenceFrameSharedMemorySnapshot{
        .fenced = snapshot_.state == InferenceFrameIpcControlState::Fenced,
        .shutdown = snapshot_.state == InferenceFrameIpcControlState::Shutdown,
    };
}

InferenceFrameIpcControlSnapshot InferenceFrameIpcGrantReceiver::ControlSnapshot() const {
    std::shared_lock lock(mutex_);
    return snapshot_;
}

InferenceFrameIpcLeaseCoordinator::InferenceFrameIpcLeaseCoordinator(
    std::shared_ptr<IRecoverableInferenceFrameIpcSink> sink,
    std::shared_ptr<IInferenceFrameIpcControlSignal> signal,
    core::LoggerAdapter logger)
    : sink_(std::move(sink)),
      signal_(std::move(signal)),
      logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("frame-ipc-control")) {}

core::Status InferenceFrameIpcLeaseCoordinator::Start() {
    std::lock_guard lock(mutex_);
    if (!sink_ || !signal_) {
        return FailLocked(
            core::Status::Error(core::ErrorCode::FailedPrecondition, "shared frame IPC control dependencies are missing"),
            "start");
    }
    if (snapshot_.state == InferenceFrameIpcControlState::Shutdown) {
        return core::Status::Error(core::ErrorCode::Cancelled, "shared frame IPC coordinator is shut down");
    }
    if (snapshot_.state == InferenceFrameIpcControlState::Granted) {
        return core::Status::Ok();
    }

    const auto grant = sink_->CurrentGrant();
    const auto status = signal_->ApplyGrant(grant);
    if (!status.ok()) {
        static_cast<void>(sink_->Fence());
        snapshot_.grant = grant;
        snapshot_.state = InferenceFrameIpcControlState::Fenced;
        return FailLocked(status, "publish initial grant");
    }
    snapshot_.state = InferenceFrameIpcControlState::Granted;
    snapshot_.grant = grant;
    snapshot_.last_status = core::Status::Ok();
    ++snapshot_.grants_applied;
    return core::Status::Ok();
}

core::Status InferenceFrameIpcLeaseCoordinator::Revoke(std::string reason) {
    std::lock_guard lock(mutex_);
    if (snapshot_.state == InferenceFrameIpcControlState::Shutdown) {
        return core::Status::Error(core::ErrorCode::Cancelled, "shared frame IPC coordinator is shut down");
    }
    if (snapshot_.state == InferenceFrameIpcControlState::Fenced) {
        return core::Status::Ok();
    }
    if (!sink_) {
        return FailLocked(
            core::Status::Error(core::ErrorCode::FailedPrecondition, "shared frame IPC sink is missing"),
            "revoke");
    }

    const auto grant = sink_->CurrentGrant();
    const auto fence_status = sink_->Fence();
    if (!fence_status.ok()) {
        return FailLocked(fence_status, "fence producer");
    }
    if (signal_) {
        const auto revoke_status = signal_->Revoke(grant.epoch, reason);
        if (!revoke_status.ok()) {
            logger_.warn(
                "[frame-ipc-control] remote revoke notification failed epoch={} code={} message={}",
                grant.epoch,
                static_cast<int>(revoke_status.code()),
                revoke_status.message());
        }
    }
    snapshot_.state = InferenceFrameIpcControlState::Fenced;
    snapshot_.grant = grant;
    snapshot_.last_status = core::Status::Error(core::ErrorCode::Cancelled, std::move(reason));
    ++snapshot_.revocations;
    return core::Status::Ok();
}

core::Status InferenceFrameIpcLeaseCoordinator::CheckPeer() {
    std::lock_guard lock(mutex_);
    if (snapshot_.state == InferenceFrameIpcControlState::Shutdown) {
        return core::Status::Error(core::ErrorCode::Cancelled, "shared frame IPC coordinator is shut down");
    }
    if (snapshot_.state != InferenceFrameIpcControlState::Granted || !sink_ || !signal_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "shared frame IPC peer is not granted");
    }

    const auto status = signal_->Probe(snapshot_.grant.epoch);
    if (status.ok()) {
        return status;
    }
    const auto fence_status = sink_->Fence();
    if (!fence_status.ok()) {
        return FailLocked(fence_status, "fence unavailable peer");
    }
    snapshot_.state = InferenceFrameIpcControlState::Fenced;
    snapshot_.last_status = status;
    ++snapshot_.revocations;
    ++snapshot_.failures;
    logger_.error(
        "[frame-ipc-control] peer probe failed; producer fenced epoch={} code={} message={}",
        snapshot_.grant.epoch,
        static_cast<int>(status.code()),
        status.message());
    return status;
}

core::Status InferenceFrameIpcLeaseCoordinator::Recover() {
    std::lock_guard lock(mutex_);
    if (snapshot_.state == InferenceFrameIpcControlState::Shutdown) {
        return core::Status::Error(core::ErrorCode::Cancelled, "shared frame IPC coordinator is shut down");
    }
    if (!sink_ || !signal_) {
        return FailLocked(
            core::Status::Error(core::ErrorCode::FailedPrecondition, "shared frame IPC control dependencies are missing"),
            "recover");
    }

    snapshot_.state = InferenceFrameIpcControlState::Recovering;
    auto status = sink_->Recreate();
    if (!status.ok()) {
        return FailLocked(status, "recreate producer");
    }
    const auto grant = sink_->CurrentGrant();
    status = signal_->ApplyGrant(grant);
    if (!status.ok()) {
        static_cast<void>(sink_->Fence());
        snapshot_.grant = grant;
        snapshot_.state = InferenceFrameIpcControlState::Fenced;
        return FailLocked(status, "publish recovery grant");
    }

    snapshot_.state = InferenceFrameIpcControlState::Granted;
    snapshot_.grant = grant;
    snapshot_.last_status = core::Status::Ok();
    ++snapshot_.grants_applied;
    ++snapshot_.recoveries;
    return core::Status::Ok();
}

void InferenceFrameIpcLeaseCoordinator::Shutdown() {
    std::lock_guard lock(mutex_);
    if (snapshot_.state == InferenceFrameIpcControlState::Shutdown) {
        return;
    }
    if (sink_) {
        const auto grant = sink_->CurrentGrant();
        static_cast<void>(sink_->Fence());
        if (signal_) {
            static_cast<void>(signal_->Revoke(grant.epoch, "IPC coordinator shutdown"));
        }
        sink_->Shutdown();
    }
    snapshot_.state = InferenceFrameIpcControlState::Shutdown;
    snapshot_.last_status = core::Status::Ok();
}

InferenceFrameIpcControlSnapshot InferenceFrameIpcLeaseCoordinator::Snapshot() const {
    std::lock_guard lock(mutex_);
    return snapshot_;
}

core::Status InferenceFrameIpcLeaseCoordinator::FailLocked(
    core::Status status,
    std::string_view operation) {
    snapshot_.state = InferenceFrameIpcControlState::Failed;
    snapshot_.last_status = status;
    ++snapshot_.failures;
    logger_.error(
        "[frame-ipc-control] {} failed code={} message={}",
        operation,
        static_cast<int>(status.code()),
        status.message());
    return status;
}

} // namespace ipc::media
