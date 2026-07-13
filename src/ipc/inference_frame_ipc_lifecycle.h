#pragma once

#include "inference_frame_shared_memory.h"

#include <cstddef>
#include <memory>
#include <shared_mutex>

namespace ipc::media {

struct InferenceFrameIpcGrant {
    std::string channel_name;
    std::uint64_t epoch = 0;
    std::size_t slot_count = 0;
    std::size_t payload_capacity = 0;
};

struct InferenceFrameIpcLifecycleSnapshot {
    InferenceFrameSharedMemorySnapshot channel;
    std::size_t recoveries = 0;
    std::size_t recovery_failures = 0;
    bool connected = false;
};

class IRecoverableInferenceFrameIpcSink : public IInferenceFrameIpcSink {
public:
    ~IRecoverableInferenceFrameIpcSink() override = default;
    virtual core::Status Fence() = 0;
    virtual core::Status Recreate() = 0;
    virtual InferenceFrameIpcGrant CurrentGrant() const = 0;
    virtual InferenceFrameIpcLifecycleSnapshot LifecycleSnapshot() const = 0;
};

class IReconnectableInferenceFrameIpcSource : public IInferenceFrameIpcSource {
public:
    ~IReconnectableInferenceFrameIpcSource() override = default;
    virtual core::Status Reconnect() = 0;
    virtual core::Status ApplyGrant(InferenceFrameIpcGrant grant) = 0;
    virtual InferenceFrameIpcLifecycleSnapshot LifecycleSnapshot() const = 0;
};

class RecoverableInferenceFrameIpcSink final : public IRecoverableInferenceFrameIpcSink {
public:
    static core::Result<std::unique_ptr<RecoverableInferenceFrameIpcSink>> Create(
        InferenceFrameSharedMemoryOptions options,
        core::LoggerAdapter logger = {});

    core::Status Publish(const SharedFramePublishRequest& request) override;
    core::Status Fence() override;
    core::Status Recreate() override;
    void Shutdown() override;
    InferenceFrameSharedMemorySnapshot Snapshot() const override;
    InferenceFrameIpcGrant CurrentGrant() const override;
    InferenceFrameIpcLifecycleSnapshot LifecycleSnapshot() const override;

private:
    RecoverableInferenceFrameIpcSink(
        InferenceFrameSharedMemoryOptions options,
        std::unique_ptr<SharedMemoryInferenceFrameChannel> channel,
        core::LoggerAdapter logger) noexcept;

    InferenceFrameSharedMemoryOptions options_;
    mutable std::shared_mutex mutex_;
    std::unique_ptr<SharedMemoryInferenceFrameChannel> channel_;
    core::LoggerAdapter logger_;
    std::size_t recoveries_ = 0;
    std::size_t recovery_failures_ = 0;
    bool shutdown_ = false;
};

class ReconnectableInferenceFrameIpcSource final : public IReconnectableInferenceFrameIpcSource {
public:
    static core::Result<std::unique_ptr<ReconnectableInferenceFrameIpcSource>> Open(
        InferenceFrameSharedMemoryOptions options,
        core::LoggerAdapter logger = {});

    core::Result<ClaimedSharedFrame> TryClaim() override;
    core::Status Reconnect() override;
    core::Status ApplyGrant(InferenceFrameIpcGrant grant) override;
    void Shutdown() override;
    InferenceFrameSharedMemorySnapshot Snapshot() const override;
    InferenceFrameIpcLifecycleSnapshot LifecycleSnapshot() const override;

private:
    ReconnectableInferenceFrameIpcSource(
        InferenceFrameSharedMemoryOptions options,
        std::unique_ptr<SharedMemoryInferenceFrameChannel> channel,
        core::LoggerAdapter logger) noexcept;

    InferenceFrameSharedMemoryOptions options_;
    mutable std::shared_mutex mutex_;
    std::unique_ptr<SharedMemoryInferenceFrameChannel> channel_;
    core::LoggerAdapter logger_;
    std::size_t recoveries_ = 0;
    std::size_t recovery_failures_ = 0;
    bool shutdown_ = false;
};

} // namespace ipc::media
