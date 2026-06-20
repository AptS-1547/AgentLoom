#include "emotion_inference_service.h"

#include "onnx_model.h"
#include "onnx_session_utils.h"
#include "../../core/logger_adapter.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace service {

namespace {

core::LoggerAdapter& Logger() {
    static core::LoggerAdapter logger = core::LoggerAdapter::ForModule("service");
    return logger;
}

core::Status Error(core::ErrorCode code, std::string message) {
    return core::Status::Error(code, std::move(message));
}

bert::ModelRuntimeOptions ToModelRuntimeOptions(const BertRuntimeConfigOptions& config) {
    bert::ModelRuntimeOptions options;
    options.execution_provider = config.execution_provider;
    options.allow_cpu_fallback = config.allow_cpu_fallback;
    options.cuda_device_id = config.cuda_device_id;
    options.intra_op_num_threads = config.intra_op_num_threads;
    options.inter_op_num_threads = config.inter_op_num_threads;
    options.enable_cpu_mem_arena = config.enable_cpu_mem_arena;
    options.enable_mem_pattern = config.enable_mem_pattern;
    return options;
}

} // namespace

class EmotionInferenceService::Impl {
public:
    explicit Impl(const MultimodalServerOptions& options) {
        if (options.bert_model.empty()) {
            Logger().warn("[emotion] BERT model path is empty; service will reject inference requests");
            return;
        }

        Logger().info("[emotion] loading BERT model: {}", options.bert_model);
        if (!bert_model_.LoadModel(options.bert_model, ToModelRuntimeOptions(options.bert_runtime))) {
            Logger().error("[emotion] failed to load BERT model path={} provider={} reason={}",
                           options.bert_model,
                           options.bert_runtime.execution_provider,
                           bert_model_.LastError().empty() ? "unknown error" : bert_model_.LastError());
        } else {
            Logger().info("[emotion] BERT model loaded: {}", bert_model_.GetInfo());
        }
    }

    core::Status PredictEmotion(const multimodal_inference::EmotionRequest& request,
                                multimodal_inference::EmotionResponse& response) {
        if (!bert_model_.IsLoaded()) {
            response.set_error("BERT model not loaded");
            return Error(core::ErrorCode::FailedPrecondition, "BERT model not loaded");
        }

        std::vector<std::int64_t> input_ids(request.input_ids().begin(), request.input_ids().end());
        std::vector<std::int64_t> attention_mask(request.attention_mask().begin(), request.attention_mask().end());
        std::vector<float> personality(request.personality().begin(), request.personality().end());

        auto result = bert_model_.Predict(input_ids, attention_mask, personality);
        if (!result.success) {
            response.set_error(result.error_message);
            return Error(core::ErrorCode::InternalError, result.error_message);
        }

        response.mutable_emotion_logits()->Reserve(static_cast<int>(result.emotion_logits.size()));
        for (float value : result.emotion_logits) {
            response.add_emotion_logits(value);
        }
        response.mutable_behavior_logits()->Reserve(static_cast<int>(result.behavior_logits.size()));
        for (float value : result.behavior_logits) {
            response.add_behavior_logits(value);
        }
        response.mutable_tone_logits()->Reserve(static_cast<int>(result.tone_logits.size()));
        for (float value : result.tone_logits) {
            response.add_tone_logits(value);
        }
        response.set_intensity(result.intensity);
        response.mutable_response_length_logits()->Reserve(static_cast<int>(result.response_length_logits.size()));
        for (float value : result.response_length_logits) {
            response.add_response_length_logits(value);
        }
        return core::Status::Ok();
    }

    core::Status PredictEmotionBatch(const multimodal_inference::EmotionBatchRequest& request,
                                     multimodal_inference::EmotionBatchResponse& response) {
        const auto batch_size = static_cast<std::size_t>(request.batch_size());
        const auto seq_len = static_cast<std::size_t>(request.seq_length());

        if (!bert_model_.IsLoaded()) {
            response.set_error("BERT model not loaded");
            return Error(core::ErrorCode::FailedPrecondition, "BERT model not loaded");
        }

        std::vector<std::int64_t> input_ids(request.input_ids().begin(), request.input_ids().end());
        std::vector<std::int64_t> attention_mask(request.attention_mask().begin(), request.attention_mask().end());
        std::vector<float> personality(request.personality().begin(), request.personality().end());

        auto results = bert_model_.PredictBatch(input_ids, attention_mask, personality, batch_size, seq_len);
        if (results.empty()) {
            response.set_error("Batch inference failed");
            return Error(core::ErrorCode::InternalError, "Batch inference failed");
        }

        response.mutable_emotion_logits()->Reserve(static_cast<int>(batch_size * results.front().emotion_logits.size()));
        response.mutable_behavior_logits()->Reserve(static_cast<int>(batch_size * results.front().behavior_logits.size()));
        response.mutable_tone_logits()->Reserve(static_cast<int>(batch_size * results.front().tone_logits.size()));
        response.mutable_intensity()->Reserve(static_cast<int>(batch_size));
        response.mutable_response_length_logits()->Reserve(static_cast<int>(batch_size * results.front().response_length_logits.size()));

        for (const auto& result : results) {
            for (float value : result.emotion_logits) {
                response.add_emotion_logits(value);
            }
            for (float value : result.behavior_logits) {
                response.add_behavior_logits(value);
            }
            for (float value : result.tone_logits) {
                response.add_tone_logits(value);
            }
            response.add_intensity(result.intensity);
            for (float value : result.response_length_logits) {
                response.add_response_length_logits(value);
            }
        }

        return core::Status::Ok();
    }

private:
    bert::OnnxBERTModel bert_model_;
};

EmotionInferenceService::EmotionInferenceService(const MultimodalServerOptions& options)
    : impl_(std::make_unique<Impl>(options)) {}

EmotionInferenceService::~EmotionInferenceService() = default;

core::Status EmotionInferenceService::PredictEmotion(
    const multimodal_inference::EmotionRequest& request,
    multimodal_inference::EmotionResponse& response) {
    return impl_->PredictEmotion(request, response);
}

core::Status EmotionInferenceService::PredictEmotionBatch(
    const multimodal_inference::EmotionBatchRequest& request,
    multimodal_inference::EmotionBatchResponse& response) {
    return impl_->PredictEmotionBatch(request, response);
}

} // namespace service
