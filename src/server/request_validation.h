#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <grpcpp/server_context.h>

#include "multimodal_inference.pb.h"

namespace request_validation {

struct AuthOptions {
    std::string token;
    std::string metadata_key = "x-agent-auth";
};

struct RequestLimits {
    size_t max_image_bytes = 20 * 1024 * 1024;
    size_t max_image_pixels = 16 * 1024 * 1024;
    uint32_t max_image_width = 8192;
    uint32_t max_image_height = 8192;
    size_t max_prompt_bytes = 8192;
    int32_t min_context_size = 128;
    int32_t max_context_size = 8192;
    int32_t max_vlm_tokens = 2048;
    float max_temperature = 2.0f;
    int32_t max_top_k = 1000;
    size_t max_vlm_session_id_bytes = 128;
    size_t max_vlm_request_id_bytes = 128;
    size_t max_vlm_task_type_bytes = 64;
    int32_t max_sequence_length = 512;
    int32_t max_batch_size = 64;
    int64_t max_token_id = 10000000;
    float max_abs_personality = 100.0f;
};

struct ValidationResult {
    bool ok = true;
    std::string error;
};

ValidationResult Ok();
ValidationResult Fail(std::string error);

ValidationResult ValidateAuth(
    const grpc::ServerContext& context,
    const AuthOptions& options);

ValidationResult ValidateEmotionRequest(
    const multimodal_inference::EmotionRequest& request,
    const RequestLimits& limits);

ValidationResult ValidateEmotionBatchRequest(
    const multimodal_inference::EmotionBatchRequest& request,
    const RequestLimits& limits);

ValidationResult ValidateVLMRequest(
    const multimodal_inference::VLMRequest& request,
    const RequestLimits& limits);

ValidationResult ValidateSaliencyRequest(
    const multimodal_inference::SaliencyRequest& request,
    const RequestLimits& limits);

}
