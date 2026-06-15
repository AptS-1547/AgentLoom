#pragma once

#include "server_options.h"

#include "result.h"

#include "multimodal_inference.pb.h"

#include <memory>

namespace service {

class EmotionInferenceService {
public:
    explicit EmotionInferenceService(const MultimodalServerOptions& options);
    ~EmotionInferenceService();

    EmotionInferenceService(const EmotionInferenceService&) = delete;
    EmotionInferenceService& operator=(const EmotionInferenceService&) = delete;

    core::Status PredictEmotion(const multimodal_inference::EmotionRequest& request,
                                multimodal_inference::EmotionResponse& response);

    core::Status PredictEmotionBatch(const multimodal_inference::EmotionBatchRequest& request,
                                     multimodal_inference::EmotionBatchResponse& response);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace service
