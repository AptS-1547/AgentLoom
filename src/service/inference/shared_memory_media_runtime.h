#pragma once

#include "inference_frame_backlog.h"
#include "inference_frame_ipc_control.h"
#include "logger_adapter.h"
#include "multimodal_service.h"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace service {

struct SharedMediaExecutionOpenRequest {
    std::string execution_id;
    std::string session_id;
    std::string skill_id = "vision.observe";
    std::string trace_id;
};

struct SharedMediaExecutionSealRequest {
    std::string execution_id;
    std::string session_id;
    std::size_t expected_selected_frames = 0;
    std::uint64_t final_transport_sequence = 0;
    std::string reason;
};

struct SharedMediaExecutionSnapshot {
    std::string execution_id;
    std::string session_id;
    std::string state;
    std::size_t expected_selected_frames = 0;
    std::size_t selected_frames = 0;
    std::size_t committed_frames = 0;
    std::size_t hot_frames = 0;
    std::size_t spooled_frames = 0;
    std::size_t terminal_frames = 0;
    std::size_t failed_inference_frames = 0;
    bool seal_requested = false;
    bool replay_complete = false;
    bool complete = false;
    std::int64_t input_sealed_at_unix_us = 0;
    core::Status status = core::Status::Ok();
    std::vector<media::inference::InferenceFrameResultRecord> results;
};

struct SharedMemoryMediaRuntimeOptions {
    std::filesystem::path spool_root = "build/media-inference-spool";
    std::size_t max_executions = 256;
    std::size_t receiver_workers = 2;
    std::size_t vlm_workers = 1;
    std::size_t backlog_segments_per_session = 2;
    std::size_t backlog_slots_per_segment = 8;
    std::size_t max_results_per_execution = 4096;
    std::size_t spool_segment_bytes = 64 * 1024 * 1024;
    std::size_t max_spool_bytes_per_execution = 2ull * 1024 * 1024 * 1024;
    std::chrono::milliseconds receiver_idle_delay{1};
    std::chrono::milliseconds seal_wait_timeout{30000};
    int max_tokens = 128;
    int context_size = 4096;
    float temperature = 0.1F;
    float top_p = 0.9F;
    int top_k = 40;
    bool allow_cache = false;
};

class ISharedMemoryMediaRuntime {
public:
    virtual ~ISharedMemoryMediaRuntime() = default;

    virtual core::Result<SharedMediaExecutionSnapshot> Open(
        const SharedMediaExecutionOpenRequest& request) = 0;
    virtual core::Result<SharedMediaExecutionSnapshot> Seal(
        const SharedMediaExecutionSealRequest& request) = 0;
    virtual core::Result<SharedMediaExecutionSnapshot> Get(
        std::string_view session_id,
        std::string_view execution_id,
        bool include_results) const = 0;
    virtual void Shutdown() = 0;
};

class SharedMemoryMediaRuntime final : public ISharedMemoryMediaRuntime {
public:
    static core::Result<std::unique_ptr<SharedMemoryMediaRuntime>> Create(
        IMultimodalService& inference_service,
        ipc::media::IInferenceFrameIpcGrantReceiver& ipc_source,
        SharedMemoryMediaRuntimeOptions options = {},
        core::LoggerAdapter logger = {});

    ~SharedMemoryMediaRuntime() override;

    SharedMemoryMediaRuntime(const SharedMemoryMediaRuntime&) = delete;
    SharedMemoryMediaRuntime& operator=(const SharedMemoryMediaRuntime&) = delete;

    core::Result<SharedMediaExecutionSnapshot> Open(
        const SharedMediaExecutionOpenRequest& request) override;
    core::Result<SharedMediaExecutionSnapshot> Seal(
        const SharedMediaExecutionSealRequest& request) override;
    core::Result<SharedMediaExecutionSnapshot> Get(
        std::string_view session_id,
        std::string_view execution_id,
        bool include_results) const override;
    void Shutdown() override;

private:
    class Impl;
    explicit SharedMemoryMediaRuntime(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace service
