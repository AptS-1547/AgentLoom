#pragma once

#include "result.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace media {

struct VisionInferenceRequest {
    std::string session_id;
    std::uint64_t frame_id = 0;
    std::span<const std::byte> encoded_image;
    std::string mime_type = "image/jpeg";
    std::optional<std::string> prompt_hint;
};

struct VisionInferenceResult {
    std::string scene_hint;
    std::string action_hint;
    std::string object_hint;
    std::string agent_hint;
    std::string memory_candidate;
    std::vector<std::string> facts;
    std::vector<std::string> weak_interpretations;
    std::string raw_text;
    std::vector<float> image_embedding;
    double confidence = 0.0;
    double image_encode_ms = 0.0;
    double prompt_eval_ms = 0.0;
    double eval_ms = 0.0;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    bool cache_hit = false;
    bool cache_stale = false;
    bool prompt_kv_cache_hit = false;
    std::string result_source;
    bool prompt_kv_near_candidate = false;
    bool prompt_kv_near_accepted = false;
    bool prompt_kv_near_same_session = false;
    float prompt_kv_global_cosine = 0.0f;
    float prompt_kv_mean_token_cosine = 0.0f;
    float prompt_kv_p05_token_cosine = 0.0f;
    float prompt_kv_min_token_cosine = 0.0f;
    float prompt_kv_relative_l2 = 0.0f;
    float prompt_kv_max_abs_error = 0.0f;
};

class IVlmVisionClient {
public:
    virtual ~IVlmVisionClient() = default;

    /// 同步分析一张已编码关键帧。
    /// @param request session/frame 标识、MIME 类型和图像字节；encoded_image 只在调用期间有效。
    /// @return 受控视觉线索；单帧结果不得直接作为身份、情绪或长期事实。
    virtual core::Result<VisionInferenceResult> Analyze(
        const VisionInferenceRequest& request) = 0;
};

} // namespace media
