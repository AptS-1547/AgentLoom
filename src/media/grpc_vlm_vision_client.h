#pragma once

#include "logger_adapter.h"
#include "vision_inference_interfaces.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

namespace media {

struct GrpcVlmVisionClientOptions {
    std::string target = "127.0.0.1:50051";
    std::chrono::milliseconds timeout{120000};
    std::size_t max_receive_message_bytes = 16 * 1024 * 1024;
    std::size_t max_send_message_bytes = 32 * 1024 * 1024;
    int max_tokens = 128;
    int context_size = 4096;
    float temperature = 0.1F;
    float top_p = 0.9F;
    int top_k = 40;
    std::string task_type = "controlled-vision-observation";
    std::string prompt;
    std::string auth_metadata_key = "x-agent-auth";
    std::string auth_token;
    bool allow_cache = false;
    bool force_refresh = true;
    bool allow_stale_cache = false;
};

struct GrpcVlmVisionClientSnapshot {
    std::size_t requests = 0;
    std::size_t successful_requests = 0;
    std::size_t failed_requests = 0;
    std::size_t cache_hits = 0;
    std::uint64_t total_latency_us = 0;
    std::uint64_t max_latency_us = 0;
};

class GrpcVlmVisionClient final : public IVlmVisionClient {
public:
    static core::Result<std::unique_ptr<GrpcVlmVisionClient>> Create(
        GrpcVlmVisionClientOptions options,
        core::LoggerAdapter logger = {});

    ~GrpcVlmVisionClient() override;

    GrpcVlmVisionClient(const GrpcVlmVisionClient&) = delete;
    GrpcVlmVisionClient& operator=(const GrpcVlmVisionClient&) = delete;

    core::Result<VisionInferenceResult> Analyze(
        const VisionInferenceRequest& request) override;

    GrpcVlmVisionClientSnapshot Snapshot() const;

private:
    class Impl;
    explicit GrpcVlmVisionClient(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace media
