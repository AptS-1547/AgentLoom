#pragma once

#include "inference_frame_backlog.h"
#include "thread_pool.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>

namespace media::inference {

struct InferenceFrameCoordinatorOptions {
    std::size_t worker_count = 2;
    std::chrono::milliseconds wait_timeout{100};
    bool shutdown_backlog = true;
};

struct InferenceFrameCoordinatorSnapshot {
    std::size_t processed_frames = 0;
    std::size_t successful_frames = 0;
    std::size_t failed_frames = 0;
    std::size_t result_publish_failures = 0;
    std::uint64_t total_latency_us = 0;
    std::uint64_t max_latency_us = 0;
    bool running = false;
};

class IInferenceFrameCoordinator {
public:
    virtual ~IInferenceFrameCoordinator() = default;

    virtual core::Status Start() = 0;
    virtual void Shutdown() = 0;
    virtual InferenceFrameCoordinatorSnapshot Snapshot() const = 0;
};

class InferenceFrameCoordinator final : public IInferenceFrameCoordinator {
public:
    InferenceFrameCoordinator(
        IInferenceFrameBacklog& backlog,
        IVlmVisionClient& vlm_client,
        IInferenceFrameResultTable& result_table,
        InferenceFrameCoordinatorOptions options = {},
        core::LoggerAdapter logger = {});
    ~InferenceFrameCoordinator() override;

    InferenceFrameCoordinator(const InferenceFrameCoordinator&) = delete;
    InferenceFrameCoordinator& operator=(const InferenceFrameCoordinator&) = delete;

    core::Status Start() override;
    void Shutdown() override;
    InferenceFrameCoordinatorSnapshot Snapshot() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace media::inference
