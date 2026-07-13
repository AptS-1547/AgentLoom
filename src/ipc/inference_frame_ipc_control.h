#pragma once

#include "inference_frame_ipc_lifecycle.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>

namespace ipc::media {

enum class InferenceFrameIpcControlState {
    Idle,
    Granted,
    Fenced,
    Recovering,
    Failed,
    Shutdown,
};

struct InferenceFrameIpcControlSnapshot {
    InferenceFrameIpcControlState state = InferenceFrameIpcControlState::Idle;
    InferenceFrameIpcGrant grant;
    std::size_t grants_applied = 0;
    std::size_t revocations = 0;
    std::size_t recoveries = 0;
    std::size_t failures = 0;
    core::Status last_status;
};

class IInferenceFrameIpcControlSignal {
public:
    virtual ~IInferenceFrameIpcControlSignal() = default;
    virtual core::Status ApplyGrant(const InferenceFrameIpcGrant& grant) = 0;
    virtual core::Status Revoke(std::uint64_t epoch, std::string_view reason) = 0;
    virtual core::Status Probe(std::uint64_t expected_epoch) = 0;
};

class IInferenceFrameIpcGrantReceiver
    : public IInferenceFrameIpcSource,
      public IInferenceFrameIpcControlSignal {
public:
    ~IInferenceFrameIpcGrantReceiver() override = default;
    virtual InferenceFrameIpcControlSnapshot ControlSnapshot() const = 0;
};

class InferenceFrameIpcGrantReceiver final : public IInferenceFrameIpcGrantReceiver {
public:
    explicit InferenceFrameIpcGrantReceiver(core::LoggerAdapter logger = {});

    core::Status ApplyGrant(const InferenceFrameIpcGrant& grant) override;
    core::Status Revoke(std::uint64_t epoch, std::string_view reason) override;
    core::Status Probe(std::uint64_t expected_epoch) override;
    core::Result<ClaimedSharedFrame> TryClaim() override;
    void Shutdown() override;
    InferenceFrameSharedMemorySnapshot Snapshot() const override;
    InferenceFrameIpcControlSnapshot ControlSnapshot() const override;

private:
    mutable std::shared_mutex mutex_;
    std::unique_ptr<ReconnectableInferenceFrameIpcSource> source_;
    core::LoggerAdapter logger_;
    InferenceFrameIpcControlSnapshot snapshot_;
};

class IInferenceFrameIpcLeaseCoordinator {
public:
    virtual ~IInferenceFrameIpcLeaseCoordinator() = default;
    virtual core::Status Start() = 0;
    virtual core::Status Revoke(std::string reason) = 0;
    virtual core::Status CheckPeer() = 0;
    virtual core::Status Recover() = 0;
    virtual void Shutdown() = 0;
    virtual InferenceFrameIpcControlSnapshot Snapshot() const = 0;
};

class InferenceFrameIpcLeaseCoordinator final : public IInferenceFrameIpcLeaseCoordinator {
public:
    InferenceFrameIpcLeaseCoordinator(
        std::shared_ptr<IRecoverableInferenceFrameIpcSink> sink,
        std::shared_ptr<IInferenceFrameIpcControlSignal> signal,
        core::LoggerAdapter logger = {});

    core::Status Start() override;
    core::Status Revoke(std::string reason) override;
    core::Status CheckPeer() override;
    core::Status Recover() override;
    void Shutdown() override;
    InferenceFrameIpcControlSnapshot Snapshot() const override;

private:
    core::Status FailLocked(core::Status status, std::string_view operation);

    std::shared_ptr<IRecoverableInferenceFrameIpcSink> sink_;
    std::shared_ptr<IInferenceFrameIpcControlSignal> signal_;
    core::LoggerAdapter logger_;
    mutable std::mutex mutex_;
    InferenceFrameIpcControlSnapshot snapshot_;
};

} // namespace ipc::media
