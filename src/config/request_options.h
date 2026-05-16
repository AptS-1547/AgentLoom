#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace request_validation {

struct AuthOptions {
    std::string token;
    std::string metadata_key = "x-agent-auth";
};

struct RequestLimits {
    std::size_t max_image_bytes = 20 * 1024 * 1024;
    std::size_t max_image_pixels = 16 * 1024 * 1024;
    std::uint32_t max_image_width = 8192;
    std::uint32_t max_image_height = 8192;
    std::size_t max_prompt_bytes = 8192;
    std::int32_t min_context_size = 128;
    std::int32_t max_context_size = 8192;
    std::int32_t max_vlm_tokens = 2048;
    float max_temperature = 2.0f;
    std::int32_t max_top_k = 1000;
    std::size_t max_vlm_session_id_bytes = 128;
    std::size_t max_vlm_request_id_bytes = 128;
    std::size_t max_vlm_task_type_bytes = 64;
    std::int32_t max_sequence_length = 512;
    std::int32_t max_batch_size = 64;
    std::int64_t max_token_id = 10000000;
    float max_abs_personality = 100.0f;
};

} // namespace request_validation
