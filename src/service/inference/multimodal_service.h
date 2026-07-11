#pragma once

#include "server_options.h"

#include "result.h"

#include "multimodal_inference.pb.h"

#include <functional>
#include <memory>

namespace service {

using VlmTokenEmitter = std::function<void(multimodal_inference::VLMToken)>;

class IMultimodalService {
public:
    virtual ~IMultimodalService() = default;

    virtual core::Status PredictEmotion(
        const multimodal_inference::EmotionRequest& request,
        multimodal_inference::EmotionResponse& response) = 0;

    virtual core::Status PredictEmotionBatch(
        const multimodal_inference::EmotionBatchRequest& request,
        multimodal_inference::EmotionBatchResponse& response) = 0;

    virtual core::Status DetectSaliency(
        const multimodal_inference::SaliencyRequest& request,
        multimodal_inference::SaliencyResponse& response) = 0;

    virtual core::Status GenerateVLM(
        const multimodal_inference::VLMRequest& request,
        VlmTokenEmitter emit) = 0;

    virtual core::Status GenerateVLMSync(
        const multimodal_inference::VLMRequest& request,
        multimodal_inference::VLMResponse& response) = 0;
};

class MultimodalService final : public IMultimodalService {
public:
    explicit MultimodalService(const MultimodalServerOptions& options);
    ~MultimodalService();

    MultimodalService(const MultimodalService&) = delete;
    MultimodalService& operator=(const MultimodalService&) = delete;

    core::Status PredictEmotion(const multimodal_inference::EmotionRequest& request,
                                multimodal_inference::EmotionResponse& response) override;

    core::Status PredictEmotionBatch(const multimodal_inference::EmotionBatchRequest& request,
                                     multimodal_inference::EmotionBatchResponse& response) override;

    core::Status DetectSaliency(const multimodal_inference::SaliencyRequest& request,
                                multimodal_inference::SaliencyResponse& response) override;

    core::Status GenerateVLM(const multimodal_inference::VLMRequest& request,
                             VlmTokenEmitter emit) override;

    core::Status GenerateVLMSync(const multimodal_inference::VLMRequest& request,
                                 multimodal_inference::VLMResponse& response) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace service
