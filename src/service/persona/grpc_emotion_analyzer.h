#pragma once

#include "persona_runtime.h"
#include "hf_tokenizer.h"

#include "multimodal_inference.grpc.pb.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace agent::service::persona {

struct GrpcEmotionAnalyzerOptions {
    std::string target = "127.0.0.1:50051";
    std::chrono::milliseconds deadline{3000};
    std::string auth_token;
    std::string auth_metadata_key = "authorization";
    ::vector::EncodeOptions tokenizer_options{128, true, true, false, true};
};

class GrpcEmotionAnalyzer final : public IEmotionAnalyzer,
                                  public IAsyncEmotionAnalyzer {
public:
    GrpcEmotionAnalyzer(GrpcEmotionAnalyzerOptions options,
                        std::shared_ptr<::vector::HfTokenizer> tokenizer);
    GrpcEmotionAnalyzer(GrpcEmotionAnalyzerOptions options,
                        std::shared_ptr<::vector::HfTokenizer> tokenizer,
                        std::shared_ptr<grpc::Channel> channel);

    core::Result<EmotionAnalysis> Analyze(
        std::string_view text,
        std::string_view trace_id,
        std::shared_ptr<const PersonalityConfig> personality = nullptr) override;
    core::Status AnalyzeAsync(
        std::string text,
        std::string trace_id,
        std::shared_ptr<const PersonalityConfig> personality,
        AnalyzeCompletion completion) override;

    core::Result<std::vector<EmotionAnalysis>> AnalyzeBatch(
        std::span<const std::string_view> texts,
        std::string_view trace_id,
        std::shared_ptr<const PersonalityConfig> personality = nullptr);

private:
    static core::Status FromGrpcStatus(const grpc::Status& status);
    static std::vector<float> BuildPersonalityVector(std::shared_ptr<const PersonalityConfig> personality);
    static core::Result<EmotionAnalysis> BuildAnalysis(const multimodal_inference::EmotionResponse& response);

    GrpcEmotionAnalyzerOptions options_;
    std::shared_ptr<::vector::HfTokenizer> tokenizer_;
    std::unique_ptr<multimodal_inference::MultimodalInference::Stub> stub_;
    std::mutex tokenizer_mutex_;
};

} // namespace agent::service::persona
