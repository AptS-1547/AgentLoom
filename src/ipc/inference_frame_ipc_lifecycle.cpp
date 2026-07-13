#include "inference_frame_ipc_lifecycle.h"

#include <mutex>
#include <utility>

namespace ipc::media {

core::Result<std::unique_ptr<RecoverableInferenceFrameIpcSink>>
RecoverableInferenceFrameIpcSink::Create(
    InferenceFrameSharedMemoryOptions options,
    core::LoggerAdapter logger) {
    auto effective_logger = logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("frame-ipc-lifecycle");
    options.remove_existing = true;
    auto channel = SharedMemoryInferenceFrameChannel::Create(options, effective_logger);
    if (!channel.ok()) {
        return channel.status();
    }
    return std::unique_ptr<RecoverableInferenceFrameIpcSink>(
        new RecoverableInferenceFrameIpcSink(
            std::move(options),
            std::move(channel).value(),
            effective_logger));
}

RecoverableInferenceFrameIpcSink::RecoverableInferenceFrameIpcSink(
    InferenceFrameSharedMemoryOptions options,
    std::unique_ptr<SharedMemoryInferenceFrameChannel> channel,
    core::LoggerAdapter logger) noexcept
    : options_(std::move(options)),
      channel_(std::move(channel)),
      logger_(std::move(logger)) {}

core::Status RecoverableInferenceFrameIpcSink::Publish(const SharedFramePublishRequest& request) {
    std::shared_lock lock(mutex_);
    if (shutdown_ || !channel_) {
        return core::Status::Error(core::ErrorCode::Unavailable, "shared frame producer is disconnected");
    }
    return channel_->Publish(request);
}

core::Status RecoverableInferenceFrameIpcSink::Fence() {
    std::unique_lock lock(mutex_);
    if (shutdown_ || !channel_) {
        return core::Status::Error(core::ErrorCode::Unavailable, "shared frame producer is disconnected");
    }
    const auto status = channel_->Fence();
    if (!status.ok()) {
        logger_.error(
            "[frame-ipc-lifecycle] producer fence failed name={} code={} message={}",
            options_.name,
            static_cast<int>(status.code()),
            status.message());
    }
    return status;
}

core::Status RecoverableInferenceFrameIpcSink::Recreate() {
    std::unique_lock lock(mutex_);
    if (shutdown_) {
        return core::Status::Error(core::ErrorCode::Cancelled, "shared frame producer is shut down");
    }

    if (channel_) {
        const auto fence_status = channel_->Fence();
        if (!fence_status.ok()) {
            ++recovery_failures_;
            logger_.error(
                "[frame-ipc-lifecycle] producer recreate fence failed name={} code={} message={}",
                options_.name,
                static_cast<int>(fence_status.code()),
                fence_status.message());
            return fence_status;
        }
        channel_->Shutdown();
        channel_.reset();
    }
    SharedMemoryInferenceFrameChannel::Remove(options_.name);
    options_.remove_existing = true;
    auto replacement = SharedMemoryInferenceFrameChannel::Create(options_, logger_);
    if (!replacement.ok()) {
        ++recovery_failures_;
        logger_.error(
            "[frame-ipc-lifecycle] producer recreate failed name={} code={} message={}",
            options_.name,
            static_cast<int>(replacement.status().code()),
            replacement.status().message());
        return replacement.status();
    }
    channel_ = std::move(replacement).value();
    ++recoveries_;
    logger_.info(
        "[frame-ipc-lifecycle] producer recreated name={} epoch={}",
        options_.name,
        channel_->Snapshot().epoch);
    return core::Status::Ok();
}

void RecoverableInferenceFrameIpcSink::Shutdown() {
    std::unique_lock lock(mutex_);
    shutdown_ = true;
    if (channel_) {
        channel_->Shutdown();
    }
}

InferenceFrameSharedMemorySnapshot RecoverableInferenceFrameIpcSink::Snapshot() const {
    std::shared_lock lock(mutex_);
    return channel_ ? channel_->Snapshot() : InferenceFrameSharedMemorySnapshot{.shutdown = shutdown_};
}

InferenceFrameIpcGrant RecoverableInferenceFrameIpcSink::CurrentGrant() const {
    std::shared_lock lock(mutex_);
    if (!channel_) {
        return {};
    }
    const auto snapshot = channel_->Snapshot();
    return {
        .channel_name = options_.name,
        .epoch = snapshot.epoch,
        .slot_count = snapshot.slot_count,
        .payload_capacity = snapshot.payload_capacity,
    };
}

InferenceFrameIpcLifecycleSnapshot RecoverableInferenceFrameIpcSink::LifecycleSnapshot() const {
    std::shared_lock lock(mutex_);
    return {
        .channel = channel_ ? channel_->Snapshot() : InferenceFrameSharedMemorySnapshot{.shutdown = shutdown_},
        .recoveries = recoveries_,
        .recovery_failures = recovery_failures_,
        .connected = static_cast<bool>(channel_) && !shutdown_,
    };
}

core::Result<std::unique_ptr<ReconnectableInferenceFrameIpcSource>>
ReconnectableInferenceFrameIpcSource::Open(
    InferenceFrameSharedMemoryOptions options,
    core::LoggerAdapter logger) {
    auto effective_logger = logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("frame-ipc-lifecycle");
    auto channel = SharedMemoryInferenceFrameChannel::Open(options, effective_logger);
    if (!channel.ok()) {
        return channel.status();
    }
    return std::unique_ptr<ReconnectableInferenceFrameIpcSource>(
        new ReconnectableInferenceFrameIpcSource(
            std::move(options),
            std::move(channel).value(),
            effective_logger));
}

ReconnectableInferenceFrameIpcSource::ReconnectableInferenceFrameIpcSource(
    InferenceFrameSharedMemoryOptions options,
    std::unique_ptr<SharedMemoryInferenceFrameChannel> channel,
    core::LoggerAdapter logger) noexcept
    : options_(std::move(options)),
      channel_(std::move(channel)),
      logger_(std::move(logger)) {}

core::Result<ClaimedSharedFrame> ReconnectableInferenceFrameIpcSource::TryClaim() {
    std::shared_lock lock(mutex_);
    if (shutdown_ || !channel_) {
        return core::Status::Error(core::ErrorCode::Unavailable, "shared frame consumer is disconnected");
    }
    return channel_->TryClaim();
}

core::Status ReconnectableInferenceFrameIpcSource::Reconnect() {
    std::unique_lock lock(mutex_);
    if (shutdown_) {
        return core::Status::Error(core::ErrorCode::Cancelled, "shared frame consumer is shut down");
    }

    if (channel_) {
        channel_->Shutdown();
        channel_.reset();
    }
    auto replacement = SharedMemoryInferenceFrameChannel::Open(options_, logger_);
    if (!replacement.ok()) {
        ++recovery_failures_;
        logger_.warn(
            "[frame-ipc-lifecycle] consumer reconnect failed name={} code={} message={}",
            options_.name,
            static_cast<int>(replacement.status().code()),
            replacement.status().message());
        return replacement.status();
    }
    channel_ = std::move(replacement).value();
    ++recoveries_;
    logger_.info(
        "[frame-ipc-lifecycle] consumer reconnected name={} epoch={}",
        options_.name,
        channel_->Snapshot().epoch);
    return core::Status::Ok();
}

core::Status ReconnectableInferenceFrameIpcSource::ApplyGrant(InferenceFrameIpcGrant grant) {
    if (grant.channel_name.empty() || grant.epoch == 0 ||
        grant.slot_count < 2 || grant.payload_capacity == 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "shared frame IPC grant is invalid");
    }

    std::unique_lock lock(mutex_);
    if (shutdown_) {
        return core::Status::Error(core::ErrorCode::Cancelled, "shared frame consumer is shut down");
    }

    InferenceFrameSharedMemoryOptions replacement_options{
        .name = std::move(grant.channel_name),
        .slot_count = grant.slot_count,
        .payload_capacity = grant.payload_capacity,
        .expected_epoch = grant.epoch,
    };
    auto replacement = SharedMemoryInferenceFrameChannel::Open(replacement_options, logger_);
    if (!replacement.ok()) {
        ++recovery_failures_;
        logger_.warn(
            "[frame-ipc-lifecycle] consumer grant rejected name={} epoch={} code={} message={}",
            replacement_options.name,
            grant.epoch,
            static_cast<int>(replacement.status().code()),
            replacement.status().message());
        return replacement.status();
    }
    const auto replacement_snapshot = replacement.value()->Snapshot();
    if (replacement_snapshot.slot_count != grant.slot_count ||
        replacement_snapshot.payload_capacity != grant.payload_capacity) {
        ++recovery_failures_;
        logger_.warn(
            "[frame-ipc-lifecycle] consumer grant layout rejected name={} epoch={} slots={} payload_capacity={}",
            replacement_options.name,
            grant.epoch,
            grant.slot_count,
            grant.payload_capacity);
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "shared frame layout does not match control-plane grant");
    }

    if (channel_) {
        channel_->Shutdown();
    }
    channel_ = std::move(replacement).value();
    options_ = std::move(replacement_options);
    ++recoveries_;
    logger_.info(
        "[frame-ipc-lifecycle] consumer applied grant name={} epoch={}",
        options_.name,
        grant.epoch);
    return core::Status::Ok();
}

void ReconnectableInferenceFrameIpcSource::Shutdown() {
    std::unique_lock lock(mutex_);
    shutdown_ = true;
    if (channel_) {
        channel_->Shutdown();
    }
}

InferenceFrameSharedMemorySnapshot ReconnectableInferenceFrameIpcSource::Snapshot() const {
    std::shared_lock lock(mutex_);
    return channel_ ? channel_->Snapshot() : InferenceFrameSharedMemorySnapshot{.shutdown = shutdown_};
}

InferenceFrameIpcLifecycleSnapshot ReconnectableInferenceFrameIpcSource::LifecycleSnapshot() const {
    std::shared_lock lock(mutex_);
    return {
        .channel = channel_ ? channel_->Snapshot() : InferenceFrameSharedMemorySnapshot{.shutdown = shutdown_},
        .recoveries = recoveries_,
        .recovery_failures = recovery_failures_,
        .connected = static_cast<bool>(channel_) && !shutdown_,
    };
}

} // namespace ipc::media
