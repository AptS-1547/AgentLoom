#include "persona_algorithm.h"

#include <gtest/gtest.h>

#include <numeric>

namespace {

using agent::service::persona::EmotionAnalysis;
using agent::service::persona::EmotionInfo;
using agent::service::persona::EmotionNeuronFusion;
using agent::service::persona::EmotionPromptConfig;
using agent::service::persona::EmotionStateConfig;
using agent::service::persona::EmotionStateTracker;
using agent::service::persona::GenerationParams;
using agent::service::persona::LlmEmotionResult;
using agent::service::persona::PersonalityConfig;
using agent::service::persona::PromptBuilder;

EmotionPromptConfig MakePromptConfig() {
    EmotionPromptConfig cfg;
    cfg.emotion_map = {
        {"sadness", "用户有些难过，回应时多陪伴"},
        {"joy", "用户很开心，可以轻松一点"},
    };
    cfg.emotion_reliability = {
        {"sadness", 0.8},
        {"joy", 0.9},
        {"neutral", 0.7},
    };
    cfg.confidence_thresholds = {{"strong", 0.5}, {"weak", 0.3}};
    cfg.intensity_levels = {{"high_min", 0.7}};
    return cfg;
}

TEST(EmotionStateTrackerTest, UpdatesOuStateWithoutNoise) {
    EmotionStateConfig cfg;
    cfg.noise_sigma = 0.0;
    EmotionStateTracker tracker(cfg, std::nullopt, 7);

    auto updated = tracker.Update("sadness", 1.0, "neutral", 0.5);
    ASSERT_TRUE(updated.ok()) << updated.status().message();

    const auto state = std::move(updated).value();
    EXPECT_EQ(state.turn_count, 1);
    EXPECT_LT(state.valence, 0.0);
    EXPECT_GT(state.arousal, 0.0);
    EXPECT_FALSE(state.last_emotion.empty());
}

TEST(EmotionStateTrackerTest, AppliesStimulusAndBuildsHint) {
    EmotionStateConfig cfg;
    cfg.noise_sigma = 0.0;
    cfg.injection_threshold = 0.01;
    EmotionStateTracker tracker(cfg, std::nullopt, 11);

    auto state = tracker.ApplyStimulus(0.8, 0.4);
    ASSERT_TRUE(state.ok()) << state.status().message();
    EXPECT_GT(state.value().valence, cfg.baseline_valence);

    auto hint = tracker.GetPromptHint();
    ASSERT_TRUE(hint.has_value());
    EXPECT_FALSE(hint->empty());
}

TEST(EmotionStateTrackerTest, AdjustsGenerationParams) {
    EmotionStateConfig cfg;
    cfg.noise_sigma = 0.0;
    EmotionStateTracker tracker(cfg, std::nullopt, 13);
    ASSERT_TRUE(tracker.Update("fear", 1.0, "sadness", 0.8).ok());

    auto adjusted = tracker.GetParamAdjustments(GenerationParams{0.7, 1000, 0.9});
    EXPECT_GE(adjusted.temperature, 0.525);
    EXPECT_LE(adjusted.temperature, 0.945);
    EXPECT_GE(adjusted.max_tokens, 1000);
    EXPECT_LE(adjusted.max_tokens, 1250);
    EXPECT_GE(adjusted.top_p, 0.8);
    EXPECT_LE(adjusted.top_p, 0.99);
}

TEST(PromptBuilderTest, BuildsSystemPromptWithStrongEmotionDirective) {
    PersonalityConfig personality;
    personality.name = "小橘";
    personality.description = "温柔但直接的教育陪伴者";

    PromptBuilder builder(personality, MakePromptConfig(), false);

    EmotionAnalysis analysis;
    analysis.emotion.primary = "sadness";
    analysis.emotion.intensity = 0.9;
    analysis.emotion.primary_prob = 0.9;

    auto prompt = builder.BuildSystemPrompt("用户: 今天考试没考好", analysis, std::string("保持耐心"));
    ASSERT_TRUE(prompt.ok()) << prompt.status().message();
    EXPECT_NE(prompt.value().find("你是小橘。"), std::string::npos);
    EXPECT_NE(prompt.value().find("（强烈）用户有些难过"), std::string::npos);
    EXPECT_NE(prompt.value().find("<feeling>保持耐心</feeling>"), std::string::npos);
}

TEST(PromptBuilderTest, BuildsPromptBlocksWithCacheFlags) {
    PersonalityConfig personality;
    personality.name = "小橘";
    personality.traits = {"耐心", "好奇"};
    PromptBuilder builder(personality, MakePromptConfig(), false);

    EmotionAnalysis analysis;
    analysis.emotion.primary = "joy";
    analysis.emotion.intensity = 0.4;
    analysis.emotion.primary_prob = 0.7;

    auto blocks = builder.BuildSystemPromptBlocks("长期记忆", analysis, std::nullopt);
    ASSERT_TRUE(blocks.ok()) << blocks.status().message();
    ASSERT_EQ(blocks.value().size(), 3u);
    EXPECT_TRUE(blocks.value()[0].cache_ephemeral);
    EXPECT_TRUE(blocks.value()[1].cache_ephemeral);
    EXPECT_FALSE(blocks.value()[2].cache_ephemeral);
}

TEST(EmotionNeuronFusionTest, FusesBertAndLlmScores) {
    EmotionAnalysis bert;
    bert.emotion.primary = "sadness";
    bert.emotion.intensity = 0.6;
    bert.emotion.primary_prob = 0.6;
    bert.emotion.probabilities = {
        {"sadness", 0.6},
        {"joy", 0.1},
        {"neutral", 0.3},
    };

    EmotionNeuronFusion fusion({0.4, 0.9, 0.0, 0.6},
                               {{"sadness", 0.8}, {"joy", 0.9}, {"neutral", 0.7}});
    auto fused = fusion.Fuse(bert, LlmEmotionResult{"joy", 0.95});
    ASSERT_TRUE(fused.ok()) << fused.status().message();
    EXPECT_EQ(fused.value().emotion.primary, "joy");

    double sum = 0.0;
    for (const auto& [_, p] : fused.value().emotion.probabilities) {
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

} // namespace
