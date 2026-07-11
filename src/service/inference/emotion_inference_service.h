#pragma once

#include "server_options.h"

#include "result.h"

#include "multimodal_inference.pb.h"

#include <memory>

namespace service {

class IEmotionInferenceService {
public:
    virtual ~IEmotionInferenceService() = default;

    virtual core::Status PredictEmotion(
        const multimodal_inference::EmotionRequest& request,
        multimodal_inference::EmotionResponse& response) = 0;

    virtual core::Status PredictEmotionBatch(
        const multimodal_inference::EmotionBatchRequest& request,
        multimodal_inference::EmotionBatchResponse& response) = 0;
};

class EmotionInferenceService final : public IEmotionInferenceService {
public:
    explicit EmotionInferenceService(const MultimodalServerOptions& options);
    ~EmotionInferenceService();

    EmotionInferenceService(const EmotionInferenceService&) = delete;
    EmotionInferenceService& operator=(const EmotionInferenceService&) = delete;

    core::Status PredictEmotion(const multimodal_inference::EmotionRequest& request,
                                multimodal_inference::EmotionResponse& response) override;

    core::Status PredictEmotionBatch(const multimodal_inference::EmotionBatchRequest& request,
                                     multimodal_inference::EmotionBatchResponse& response) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace service
