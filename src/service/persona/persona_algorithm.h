#pragma once

#include "result.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace agent::service::persona {

struct EmotionVa {
    double valence = 0.0;
    double arousal = 0.1;
};

struct EmotionStateConfig {
    double alpha = 0.75;
    double beta = 0.25;
    double gamma = 0.25;
    double delta = 0.15;
    double baseline_valence = 0.15;
    double baseline_arousal = 0.28;
    double kappa = 0.05;
    double negativity_bias = 1.3;
    double noise_sigma = 0.05;
    double injection_threshold = 0.12;
    int save_interval_turns = 5;
    bool persist_to_l4 = true;
};

struct EmotionState {
    double valence = 0.0;
    double arousal = 0.1;
    int turn_count = 0;
    std::string last_emotion = "neutral";
    double prev_valence = 0.0;
    std::string sustained_label = "neutral";
    int sustained_turns = 0;
};

struct GenerationParams {
    double temperature = 0.7;
    int max_tokens = 1024;
    double top_p = 0.9;
};

struct EmotionGenerationOptions {
    GenerationParams default_generation{0.7, 1024, 0.9};
    int min_tokens = 100;
    double max_token_ratio = 1.25;
    double default_token_weight = 1.0;
    double high_intensity_threshold = 0.7;
    double high_intensity_multiplier = 1.1;
    std::map<std::string, double> token_weights{
        {"neutral", 1.0},
        {"joy", 1.0},
        {"excitement", 1.05},
        {"sadness", 1.1},
        {"fear", 1.1},
        {"anger", 1.05},
        {"disgust", 1.0},
        {"surprise", 1.05},
        {"tenderness", 1.05},
        {"curiosity", 1.15},
    };
};

struct EmotionStateSnapshot {
    EmotionState state;
    EmotionStateConfig config;
};

class IEmotionStateTracker {
public:
    virtual ~IEmotionStateTracker() = default;

    virtual const EmotionState& state() const noexcept = 0;
    virtual const EmotionStateConfig& config() const noexcept = 0;
    virtual core::Result<EmotionState> Update(std::string_view user_emotion,
                                              double user_intensity,
                                              std::string_view ai_emotion,
                                              double ai_intensity) = 0;
    virtual core::Result<EmotionState> ApplyStimulus(double valence_delta,
                                                     double arousal_delta) = 0;
    virtual GenerationParams GetParamAdjustments(const GenerationParams& base) const = 0;
    virtual std::optional<std::string> GetPromptHint() const = 0;
    virtual EmotionStateSnapshot Snapshot() const = 0;
};

class EmotionStateTracker final : public IEmotionStateTracker {
public:
    explicit EmotionStateTracker(EmotionStateConfig config = {},
                                 std::optional<EmotionState> initial_state = std::nullopt,
                                 std::uint32_t random_seed = std::random_device{}());

    const EmotionState& state() const noexcept override;
    const EmotionStateConfig& config() const noexcept override;

    core::Result<EmotionState> Update(std::string_view user_emotion,
                                      double user_intensity,
                                      std::string_view ai_emotion,
                                      double ai_intensity) override;
    core::Result<EmotionState> ApplyStimulus(double valence_delta,
                                             double arousal_delta) override;
    GenerationParams GetParamAdjustments(const GenerationParams& base) const override;
    std::optional<std::string> GetPromptHint() const override;
    EmotionStateSnapshot Snapshot() const override;

    static const std::unordered_map<std::string, EmotionVa>& EmotionMap();
    static const std::unordered_map<std::string, std::string>& StateHintMap();
    static std::string StateToLabel(double valence, double arousal);

private:
    core::Result<EmotionVa> Project(std::string_view emotion, double intensity) const;
    void UpdateLabelTrace();

    EmotionStateConfig config_;
    EmotionState state_;
    std::mt19937 rng_;
};

struct PersonalityConfig {
    std::string name;
    std::string description;
    std::vector<std::string> traits;
    double openness = 0.5;
    double extraversion = 0.5;
    double humor_tendency = 0.5;
    double empathy_level = 0.5;
    double curiosity_level = 0.5;
    double formality = 0.5;
};

struct EmotionInfo {
    std::string primary = "neutral";
    double intensity = 0.0;
    double primary_prob = 1.0;
    std::map<std::string, double> probabilities;
};

struct EmotionAnalysis {
    EmotionInfo emotion;
    std::string behavior;
    std::string tone;
};

struct EmotionPromptConfig {
    std::map<std::string, std::string> emotion_map;
    std::map<std::string, double> emotion_reliability;
    std::map<std::string, double> confidence_thresholds{{"strong", 0.5}, {"weak", 0.3}};
    std::map<std::string, double> intensity_levels{{"high_min", 0.7}};
};

struct PromptBlock {
    std::string type = "text";
    std::string text;
    bool cache_ephemeral = false;
};

class IPromptBuilder {
public:
    virtual ~IPromptBuilder() = default;

    virtual core::Result<std::string> BuildSystemPrompt(
        std::string_view recalled_context,
        const std::optional<EmotionAnalysis>& emotion_analysis,
        const std::optional<std::string>& emotion_state_hint) const = 0;

    virtual core::Result<std::vector<PromptBlock>> BuildSystemPromptBlocks(
        std::string_view recalled_context,
        const std::optional<EmotionAnalysis>& emotion_analysis,
        const std::optional<std::string>& emotion_state_hint) const = 0;
};

class PromptBuilder final : public IPromptBuilder {
public:
    PromptBuilder(PersonalityConfig personality,
                  std::optional<EmotionPromptConfig> emotion_prompt_config = std::nullopt,
                  bool time_awareness = true);

    core::Result<std::string> BuildSystemPrompt(
        std::string_view recalled_context,
        const std::optional<EmotionAnalysis>& emotion_analysis,
        const std::optional<std::string>& emotion_state_hint) const override;

    core::Result<std::vector<PromptBlock>> BuildSystemPromptBlocks(
        std::string_view recalled_context,
        const std::optional<EmotionAnalysis>& emotion_analysis,
        const std::optional<std::string>& emotion_state_hint) const override;

    core::Result<std::string> BuildEmotionDirectives(const EmotionAnalysis& emotion_analysis) const;
    std::string PersonalitySection() const;

private:
    std::string CurrentTimeText() const;

    PersonalityConfig personality_;
    std::optional<EmotionPromptConfig> emotion_prompt_config_;
    bool time_awareness_ = true;
};

struct EmotionFusionConfig {
    double w_bert = 0.7;
    double w_llm = 0.3;
    double bias = 0.0;
    double llm_confidence_default = 0.6;
};

struct LlmEmotionResult {
    std::string emotion;
    double confidence = 0.0;
};

class IEmotionFusion {
public:
    virtual ~IEmotionFusion() = default;
    virtual core::Result<EmotionAnalysis> Fuse(
        const EmotionAnalysis& bert_result,
        const std::optional<LlmEmotionResult>& llm_result) const = 0;
};

class EmotionNeuronFusion final : public IEmotionFusion {
public:
    EmotionNeuronFusion(EmotionFusionConfig config,
                        std::map<std::string, double> emotion_reliability);

    core::Result<EmotionAnalysis> Fuse(
        const EmotionAnalysis& bert_result,
        const std::optional<LlmEmotionResult>& llm_result) const override;

private:
    std::map<std::string, double> Softmax(const std::map<std::string, double>& scores) const;

    EmotionFusionConfig config_;
    std::map<std::string, double> emotion_reliability_;
    std::vector<std::string> emotion_labels_;
};

} // namespace agent::service::persona
