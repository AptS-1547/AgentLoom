#include "grpc_emotion_analyzer.h"

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>

namespace agent::service::persona {

namespace {

constexpr std::string_view kEmotionLabels[] = {
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "neutral",
    "excitement",
    "tenderness",
    "curiosity",
};

constexpr std::string_view kBehaviorLabels[] = {
    "respond_positive",
    "respond_negative",
    "ask_question",
    "share_experience",
    "give_advice",
    "express_empathy",
    "make_joke",
    "change_topic",
    "seek_clarification",
    "agree",
    "disagree",
    "neutral_acknowledge",
};

constexpr std::string_view kToneLabels[] = {
    "enthusiastic",
    "calm",
    "playful",
    "serious",
    "warm",
    "cold",
    "sarcastic",
    "supportive",
};

core::ErrorCode FromGrpcCode(grpc::StatusCode code) noexcept {
    switch (code) {
    case grpc::StatusCode::OK: return core::ErrorCode::Ok;
    case grpc::StatusCode::INVALID_ARGUMENT: return core::ErrorCode::InvalidArgument;
    case grpc::StatusCode::NOT_FOUND: return core::ErrorCode::NotFound;
    case grpc::StatusCode::DEADLINE_EXCEEDED: return core::ErrorCode::Timeout;
    case grpc::StatusCode::CANCELLED: return core::ErrorCode::Cancelled;
    case grpc::StatusCode::ALREADY_EXISTS: return core::ErrorCode::AlreadyExists;
    case grpc::StatusCode::PERMISSION_DENIED:
    case grpc::StatusCode::UNAUTHENTICATED:
        return core::ErrorCode::PermissionDenied;
    case grpc::StatusCode::FAILED_PRECONDITION: return core::ErrorCode::FailedPrecondition;
    case grpc::StatusCode::UNIMPLEMENTED: return core::ErrorCode::Unimplemented;
    case grpc::StatusCode::RESOURCE_EXHAUSTED: return core::ErrorCode::ResourceExhausted;
    case grpc::StatusCode::UNAVAILABLE: return core::ErrorCode::Unavailable;
    default: return core::ErrorCode::InternalError;
    }
}

std::vector<double> Softmax(const google::protobuf::RepeatedField<float>& logits) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(logits.size()));
    if (logits.empty()) {
        return out;
    }

    float max_value = -std::numeric_limits<float>::infinity();
    for (float value : logits) {
        max_value = std::max(max_value, value);
    }

    double sum = 0.0;
    for (float value : logits) {
        const double prob = std::exp(static_cast<double>(value - max_value));
        out.push_back(prob);
        sum += prob;
    }
    if (sum <= 0.0) {
        return out;
    }
    for (double& value : out) {
        value /= sum;
    }
    return out;
}

std::size_t ArgMax(const std::vector<double>& values) {
    return static_cast<std::size_t>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

} // namespace

GrpcEmotionAnalyzer::GrpcEmotionAnalyzer(GrpcEmotionAnalyzerOptions options,
                                         std::shared_ptr<::vector::HfTokenizer> tokenizer)
    : GrpcEmotionAnalyzer(
          options,
          std::move(tokenizer),
          grpc::CreateChannel(options.target, grpc::InsecureChannelCredentials())) {}

GrpcEmotionAnalyzer::GrpcEmotionAnalyzer(GrpcEmotionAnalyzerOptions options,
                                         std::shared_ptr<::vector::HfTokenizer> tokenizer,
                                         std::shared_ptr<grpc::Channel> channel)
    : options_(std::move(options)),
      tokenizer_(std::move(tokenizer)),
      stub_(multimodal_inference::MultimodalInference::NewStub(std::move(channel))) {}

core::Result<EmotionAnalysis> GrpcEmotionAnalyzer::Analyze(
    std::string_view text,
    std::string_view trace_id,
    std::shared_ptr<const PersonalityConfig> personality) {
    if (!stub_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "emotion grpc stub is not initialized");
    }
    if (!tokenizer_ || !tokenizer_->valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "emotion tokenizer is not initialized");
    }
    if (text.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "emotion text is required");
    }

    std::unique_lock lock(tokenizer_mutex_);
    auto tokenized = tokenizer_->Encode(text, options_.tokenizer_options);
    lock.unlock();
    if (!tokenized.ok()) {
        return tokenized.status();
    }

    multimodal_inference::EmotionRequest request;
    request.mutable_input_ids()->Add(tokenized.value().input_ids.begin(), tokenized.value().input_ids.end());
    request.mutable_attention_mask()->Add(tokenized.value().attention_mask.begin(), tokenized.value().attention_mask.end());
    const auto personality_vector = BuildPersonalityVector(std::move(personality));
    request.mutable_personality()->Add(personality_vector.begin(), personality_vector.end());

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + options_.deadline);
    if (!options_.auth_token.empty()) {
        context.AddMetadata(options_.auth_metadata_key, options_.auth_token);
    }
    if (!trace_id.empty()) {
        context.AddMetadata("x-trace-id", std::string(trace_id));
    }

    multimodal_inference::EmotionResponse response;
    const auto status = stub_->PredictEmotion(&context, request, &response);
    if (!status.ok()) {
        return FromGrpcStatus(status);
    }
    if (!response.error().empty()) {
        return core::Status::Error(core::ErrorCode::InternalError, response.error());
    }
    return BuildAnalysis(response);
}

core::Status GrpcEmotionAnalyzer::AnalyzeAsync(
    std::string text,
    std::string trace_id,
    std::shared_ptr<const PersonalityConfig> personality,
    AnalyzeCompletion completion) {
    if (!completion) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "emotion completion is required");
    }
    if (!stub_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                   "emotion grpc stub is not initialized");
    }
    if (!tokenizer_ || !tokenizer_->valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                   "emotion tokenizer is not initialized");
    }
    if (text.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "emotion text is required");
    }

    std::unique_lock lock(tokenizer_mutex_);
    auto tokenized = tokenizer_->Encode(text, options_.tokenizer_options);
    lock.unlock();
    if (!tokenized.ok()) {
        return tokenized.status();
    }

    struct AsyncCall final {
        grpc::ClientContext context;
        multimodal_inference::EmotionRequest request;
        multimodal_inference::EmotionResponse response;
        AnalyzeCompletion completion;
    };
    auto call = std::make_shared<AsyncCall>();
    call->request.mutable_input_ids()->Add(
        tokenized.value().input_ids.begin(), tokenized.value().input_ids.end());
    call->request.mutable_attention_mask()->Add(
        tokenized.value().attention_mask.begin(), tokenized.value().attention_mask.end());
    const auto personality_vector = BuildPersonalityVector(std::move(personality));
    call->request.mutable_personality()->Add(
        personality_vector.begin(), personality_vector.end());
    call->context.set_deadline(std::chrono::system_clock::now() + options_.deadline);
    if (!options_.auth_token.empty()) {
        call->context.AddMetadata(options_.auth_metadata_key, options_.auth_token);
    }
    if (!trace_id.empty()) {
        call->context.AddMetadata("x-trace-id", std::move(trace_id));
    }
    call->completion = std::move(completion);

    stub_->async()->PredictEmotion(
        &call->context,
        &call->request,
        &call->response,
        [call](grpc::Status status) mutable {
            auto completion = std::move(call->completion);
            if (!status.ok()) {
                completion(FromGrpcStatus(status));
                return;
            }
            if (!call->response.error().empty()) {
                completion(core::Status::Error(
                    core::ErrorCode::InternalError, call->response.error()));
                return;
            }
            completion(BuildAnalysis(call->response));
        });
    return core::Status::Ok();
}

core::Result<std::vector<EmotionAnalysis>> GrpcEmotionAnalyzer::AnalyzeBatch(
    std::span<const std::string_view> texts,
    std::string_view trace_id,
    std::shared_ptr<const PersonalityConfig> personality) {
    if (!stub_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "emotion grpc stub is not initialized");
    }
    if (!tokenizer_ || !tokenizer_->valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "emotion tokenizer is not initialized");
    }
    if (texts.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "emotion batch is empty");
    }

    std::unique_lock lock(tokenizer_mutex_);
    auto tokenized = tokenizer_->EncodeBatch(texts, options_.tokenizer_options);
    lock.unlock();
    if (!tokenized.ok()) {
        return tokenized.status();
    }

    multimodal_inference::EmotionBatchRequest request;
    request.mutable_input_ids()->Add(tokenized.value().input_ids.begin(), tokenized.value().input_ids.end());
    request.mutable_attention_mask()->Add(tokenized.value().attention_mask.begin(), tokenized.value().attention_mask.end());
    request.set_batch_size(static_cast<int>(tokenized.value().batch_size));
    request.set_seq_length(static_cast<int>(tokenized.value().sequence_length));

    const auto personality_vector = BuildPersonalityVector(std::move(personality));
    request.mutable_personality()->Reserve(static_cast<int>(personality_vector.size() * texts.size()));
    for (std::size_t index = 0; index < texts.size(); ++index) {
        request.mutable_personality()->Add(personality_vector.begin(), personality_vector.end());
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + options_.deadline);
    if (!options_.auth_token.empty()) {
        context.AddMetadata(options_.auth_metadata_key, options_.auth_token);
    }
    if (!trace_id.empty()) {
        context.AddMetadata("x-trace-id", std::string(trace_id));
    }

    multimodal_inference::EmotionBatchResponse response;
    const auto status = stub_->PredictEmotionBatch(&context, request, &response);
    if (!status.ok()) {
        return FromGrpcStatus(status);
    }
    if (!response.error().empty()) {
        return core::Status::Error(core::ErrorCode::InternalError, response.error());
    }

    constexpr int kEmotionCount = static_cast<int>(std::size(kEmotionLabels));
    constexpr int kBehaviorCount = static_cast<int>(std::size(kBehaviorLabels));
    constexpr int kToneCount = static_cast<int>(std::size(kToneLabels));
    const int batch_size = request.batch_size();
    if (response.emotion_logits_size() != batch_size * kEmotionCount ||
        response.behavior_logits_size() != batch_size * kBehaviorCount ||
        response.tone_logits_size() != batch_size * kToneCount ||
        response.intensity_size() != batch_size) {
        return core::Status::Error(core::ErrorCode::InternalError, "emotion grpc batch response has unexpected shape");
    }

    std::vector<EmotionAnalysis> analyses;
    analyses.reserve(static_cast<std::size_t>(batch_size));
    for (int row = 0; row < batch_size; ++row) {
        multimodal_inference::EmotionResponse item;
        for (int col = 0; col < kEmotionCount; ++col) {
            item.add_emotion_logits(response.emotion_logits(row * kEmotionCount + col));
        }
        for (int col = 0; col < kBehaviorCount; ++col) {
            item.add_behavior_logits(response.behavior_logits(row * kBehaviorCount + col));
        }
        for (int col = 0; col < kToneCount; ++col) {
            item.add_tone_logits(response.tone_logits(row * kToneCount + col));
        }
        item.set_intensity(response.intensity(row));
        auto analysis = BuildAnalysis(item);
        if (!analysis.ok()) {
            return analysis.status();
        }
        analyses.push_back(std::move(analysis).value());
    }
    return analyses;
}

core::Status GrpcEmotionAnalyzer::FromGrpcStatus(const grpc::Status& status) {
    if (status.ok()) {
        return core::Status::Ok();
    }
    return core::Status::Error(FromGrpcCode(status.error_code()), status.error_message());
}

std::vector<float> GrpcEmotionAnalyzer::BuildPersonalityVector(std::shared_ptr<const PersonalityConfig> personality) {
    std::vector<float> values(11, 0.5f);
    if (!personality) {
        return values;
    }

    values[0] = static_cast<float>(personality->openness);
    values[2] = static_cast<float>(personality->extraversion);
    values[6] = static_cast<float>(personality->humor_tendency);
    values[7] = static_cast<float>(personality->empathy_level);
    values[8] = static_cast<float>(personality->curiosity_level);
    values[9] = static_cast<float>(personality->formality);
    return values;
}

core::Result<EmotionAnalysis> GrpcEmotionAnalyzer::BuildAnalysis(
    const multimodal_inference::EmotionResponse& response) {
    if (response.emotion_logits_size() != static_cast<int>(std::size(kEmotionLabels))) {
        return core::Status::Error(core::ErrorCode::InternalError, "emotion grpc response has unexpected emotion logits size");
    }
    if (response.behavior_logits_size() != static_cast<int>(std::size(kBehaviorLabels))) {
        return core::Status::Error(core::ErrorCode::InternalError, "emotion grpc response has unexpected behavior logits size");
    }
    if (response.tone_logits_size() != static_cast<int>(std::size(kToneLabels))) {
        return core::Status::Error(core::ErrorCode::InternalError, "emotion grpc response has unexpected tone logits size");
    }

    const auto emotion_probs = Softmax(response.emotion_logits());
    const auto behavior_probs = Softmax(response.behavior_logits());
    const auto tone_probs = Softmax(response.tone_logits());
    const auto emotion_index = ArgMax(emotion_probs);
    const auto behavior_index = ArgMax(behavior_probs);
    const auto tone_index = ArgMax(tone_probs);

    EmotionAnalysis analysis;
    analysis.emotion.primary = std::string(kEmotionLabels[emotion_index]);
    analysis.emotion.primary_prob = emotion_probs[emotion_index];
    analysis.emotion.intensity = std::clamp(static_cast<double>(response.intensity()), 0.0, 1.0);
    for (std::size_t index = 0; index < emotion_probs.size(); ++index) {
        analysis.emotion.probabilities.emplace(std::string(kEmotionLabels[index]), emotion_probs[index]);
    }
    analysis.behavior = std::string(kBehaviorLabels[behavior_index]);
    analysis.tone = std::string(kToneLabels[tone_index]);
    return analysis;
}

} // namespace agent::service::persona
