#pragma once

#include "persona_runtime.h"

#include <chrono>
#include <map>
#include <memory>
#include <regex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace agent::llm {
class ILlmClient;
}

namespace vector {
class EmbeddingPipeline;
}

namespace agent::service::persona {

struct EmotionEvidence {
    std::string label;
    double score = 0.0;
    std::string source;
    std::string matched;
};

struct EmotionKeywordRule {
    std::string label;
    std::string pattern;
    double score = 1.0;
};

class IEmotionEvidenceProvider {
public:
    virtual ~IEmotionEvidenceProvider() = default;
    virtual core::Result<std::vector<EmotionEvidence>> Collect(std::string_view text,
                                                               std::string_view trace_id) const = 0;
    virtual core::Result<std::vector<std::vector<EmotionEvidence>>> CollectBatch(
        std::span<const std::string_view> texts,
        std::string_view trace_id) const;
};

class KeywordEmotionEvidenceProvider final : public IEmotionEvidenceProvider {
public:
    explicit KeywordEmotionEvidenceProvider(std::vector<EmotionKeywordRule> rules);

    core::Result<std::vector<EmotionEvidence>> Collect(std::string_view text,
                                                       std::string_view trace_id) const override;

private:
    struct CompiledRule {
        std::string label;
        std::string pattern;
        double score = 1.0;
        std::regex regex;
    };

    std::vector<CompiledRule> rules_;
};

struct EmotionVectorPrototype {
    std::string label;
    std::string text;
    double score = 1.0;
};

struct EmotionVectorEvidenceOptions {
    double similarity_threshold = 0.72;
    std::size_t max_evidence = 3;
};

class VectorEmotionEvidenceProvider final : public IEmotionEvidenceProvider {
public:
    static core::Result<std::shared_ptr<VectorEmotionEvidenceProvider>> Create(
        std::shared_ptr<::vector::EmbeddingPipeline> embedding,
        std::vector<EmotionVectorPrototype> prototypes,
        EmotionVectorEvidenceOptions options = {});

    core::Result<std::vector<EmotionEvidence>> Collect(std::string_view text,
                                                       std::string_view trace_id) const override;
    core::Result<std::vector<std::vector<EmotionEvidence>>> CollectBatch(
        std::span<const std::string_view> texts,
        std::string_view trace_id) const override;

private:
    struct EmbeddedPrototype {
        EmotionVectorPrototype prototype;
        std::vector<float> embedding;
    };

    VectorEmotionEvidenceProvider(std::shared_ptr<::vector::EmbeddingPipeline> embedding,
                                  std::vector<EmbeddedPrototype> prototypes,
                                  EmotionVectorEvidenceOptions options);

    static double CosineSimilarity(const std::vector<float>& lhs, const std::vector<float>& rhs);
    static double CosineSimilarity(std::span<const float> lhs, const std::vector<float>& rhs);
    std::vector<EmotionEvidence> CollectFromEmbedding(std::span<const float> query) const;

    std::shared_ptr<::vector::EmbeddingPipeline> embedding_;
    std::vector<EmbeddedPrototype> prototypes_;
    EmotionVectorEvidenceOptions options_;
};

struct LlmEmotionFallbackOptions {
    std::string model;
    double default_confidence = 0.65;
    std::chrono::milliseconds timeout{3000};
};

class LlmEmotionFallbackAnalyzer final : public IEmotionAnalyzer {
public:
    LlmEmotionFallbackAnalyzer(std::shared_ptr<agent::llm::ILlmClient> llm,
                               LlmEmotionFallbackOptions options = {});

    core::Result<EmotionAnalysis> Analyze(std::string_view text,
                                          std::string_view trace_id,
                                          std::shared_ptr<const PersonalityConfig> personality = nullptr) override;

private:
    std::shared_ptr<agent::llm::ILlmClient> llm_;
    LlmEmotionFallbackOptions options_;
};

struct EmotionFusionAnalyzerOptions {
    bool enabled = true;
    double bert_weight = 1.0;
    double default_reliability = 0.7;
    double accept_confidence = 0.55;
    double ambiguity_margin = 0.12;
    double head_bias = 0.0;
    double bert_signal_weight = 2.0;
    double keyword_signal_weight = 1.2;
    double vector_signal_weight = 1.0;
    double llm_signal_weight = 0.8;
    double margin_signal_weight = 0.5;
    double llm_gate_confidence = 0.70;
    double llm_gate_min_delta = -0.05;
    std::map<std::string, double> source_weights{{"keyword", 0.75}, {"vector", 0.85}, {"llm", 0.9}};
    std::map<std::string, double> label_reliability{
        {"joy", 0.78},
        {"sadness", 0.72},
        {"anger", 0.69},
        {"fear", 0.64},
        {"surprise", 0.58},
        {"disgust", 0.52},
        {"neutral", 0.82},
        {"excitement", 0.74},
        {"tenderness", 0.70},
        {"curiosity", 0.76},
    };
};

struct EmotionFusionFeature {
    std::string label;
    double bert_prob = 0.0;
    double reliability = 0.0;
    double keyword_score = 0.0;
    double vector_score = 0.0;
    double llm_score = 0.0;
    double margin_bonus = 0.0;
};

class FusedEmotionAnalyzer final : public IEmotionAnalyzer {
public:
    FusedEmotionAnalyzer(std::shared_ptr<IEmotionAnalyzer> primary,
                         EmotionFusionAnalyzerOptions options,
                         std::vector<std::shared_ptr<IEmotionEvidenceProvider>> providers = {},
                         std::shared_ptr<IEmotionAnalyzer> fallback = nullptr);

    core::Result<EmotionAnalysis> Analyze(std::string_view text,
                                          std::string_view trace_id,
                                          std::shared_ptr<const PersonalityConfig> personality = nullptr) override;

    core::Result<EmotionAnalysis> FuseForTesting(const EmotionAnalysis& primary,
                                                 const std::vector<EmotionEvidence>& evidence) const;

    std::vector<EmotionFusionFeature> BuildFeaturesForCalibration(
        const EmotionAnalysis& primary,
        const std::vector<EmotionEvidence>& evidence) const;
    std::map<std::string, double> BuildLogitsForCalibration(
        const std::vector<EmotionFusionFeature>& features,
        const EmotionFusionAnalyzerOptions& options) const;
    std::map<std::string, double> SoftmaxForCalibration(
        const std::map<std::string, double>& logits) const;

private:
    std::map<std::string, double> BuildLogits(const EmotionAnalysis& primary,
                                              const std::vector<EmotionEvidence>& evidence) const;
    std::map<std::string, double> Softmax(const std::map<std::string, double>& logits) const;
    EmotionAnalysis BuildResult(const EmotionAnalysis& primary,
                                const std::map<std::string, double>& probabilities) const;
    EmotionAnalysis ApplyFallbackGate(const EmotionAnalysis& fused,
                                      const EmotionAnalysis& fallback) const;
    bool ShouldUseFallback(const EmotionAnalysis& fused) const;
    double Reliability(std::string_view label) const;
    double SourceWeight(std::string_view source) const;

    std::shared_ptr<IEmotionAnalyzer> primary_;
    EmotionFusionAnalyzerOptions options_;
    std::vector<std::shared_ptr<IEmotionEvidenceProvider>> providers_;
    std::shared_ptr<IEmotionAnalyzer> fallback_;
};

std::vector<EmotionKeywordRule> DefaultEmotionKeywordRules();
std::vector<EmotionVectorPrototype> DefaultEmotionVectorPrototypes();

} // namespace agent::service::persona
