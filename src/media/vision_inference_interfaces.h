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
