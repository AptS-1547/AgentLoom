#include "emotion_fusion_analyzer.h"

#include <gtest/gtest.h>

#include <numeric>

namespace {

using agent::service::persona::EmotionAnalysis;
using agent::service::persona::EmotionEvidence;
using agent::service::persona::EmotionFusionAnalyzerOptions;
using agent::service::persona::FusedEmotionAnalyzer;

class StaticEmotionAnalyzer final : public agent::service::persona::IEmotionAnalyzer {
public:
    explicit StaticEmotionAnalyzer(EmotionAnalysis analysis) : analysis_(std::move(analysis)) {}

    core::Result<EmotionAnalysis> Analyze(
        std::string_view,
        std::string_view,
        std::shared_ptr<const agent::service::persona::PersonalityConfig> = nullptr) override {
        return analysis_;
    }

private:
    EmotionAnalysis analysis_;
};

class RecordingAsyncEmotionAnalyzer final
    : public agent::service::persona::IEmotionAnalyzer,
      public agent::service::persona::IAsyncEmotionAnalyzer {
public:
    explicit RecordingAsyncEmotionAnalyzer(EmotionAnalysis analysis)
        : analysis_(std::move(analysis)) {}

    core::Result<EmotionAnalysis> Analyze(
        std::string_view,
        std::string_view,
        std::shared_ptr<const agent::service::persona::PersonalityConfig> = nullptr) override {
        return core::Status::Error(core::ErrorCode::InternalError,
                                   "unexpected synchronous analysis");
    }

    core::Status AnalyzeAsync(
        std::string text,
        std::string trace_id,
        std::shared_ptr<const agent::service::persona::PersonalityConfig>,
        AnalyzeCompletion completion) override {
        observed_text = std::move(text);
        observed_trace_id = std::move(trace_id);
        completion(analysis_);
        return core::Status::Ok();
    }

    std::string observed_text;
    std::string observed_trace_id;

private:
    EmotionAnalysis analysis_;
};

EmotionAnalysis MakeEmotion(std::string primary,
                            double primary_prob,
                            std::map<std::string, double> probabilities) {
    EmotionAnalysis analysis;
    analysis.emotion.primary = std::move(primary);
    analysis.emotion.primary_prob = primary_prob;
    analysis.emotion.intensity = primary_prob;
    analysis.emotion.probabilities = std::move(probabilities);
    analysis.behavior = "neutral_acknowledge";
    analysis.tone = "calm";
    return analysis;
}

double ProbSum(const EmotionAnalysis& analysis) {
    double sum = 0.0;
    for (const auto& [_, probability] : analysis.emotion.probabilities) {
        sum += probability;
    }
    return sum;
}

} // namespace

TEST(FusedEmotionAnalyzerTest, KeywordEvidenceOverridesLowConfidenceBert) {
    EmotionFusionAnalyzerOptions options;
    options.label_reliability = {{"neutral", 0.82}, {"sadness", 0.72}, {"joy", 0.78}};
    options.source_weights = {{"keyword", 0.9}};
    FusedEmotionAnalyzer fusion(nullptr, options);

    auto bert = MakeEmotion("neutral", 0.42, {{"neutral", 0.42}, {"sadness", 0.38}, {"joy", 0.20}});
    auto fused = fusion.FuseForTesting(bert, {EmotionEvidence{"sadness", 0.95, "keyword", "考砸"}});

    ASSERT_TRUE(fused.ok()) << fused.status().message();
    EXPECT_EQ(fused.value().emotion.primary, "sadness");
    EXPECT_NEAR(ProbSum(fused.value()), 1.0, 1e-9);
}

TEST(FusedEmotionAnalyzerTest, HighConfidenceBertSurvivesWeakKeywordEvidence) {
    EmotionFusionAnalyzerOptions options;
    options.label_reliability = {{"neutral", 0.82}, {"sadness", 0.72}, {"joy", 0.78}};
    options.source_weights = {{"keyword", 0.25}};
    FusedEmotionAnalyzer fusion(nullptr, options);

    auto bert = MakeEmotion("joy", 0.88, {{"joy", 0.88}, {"neutral", 0.08}, {"sadness", 0.04}});
    auto fused = fusion.FuseForTesting(bert, {EmotionEvidence{"sadness", 0.40, "keyword", "不想"}});

    ASSERT_TRUE(fused.ok()) << fused.status().message();
    EXPECT_EQ(fused.value().emotion.primary, "joy");
    EXPECT_NEAR(ProbSum(fused.value()), 1.0, 1e-9);
}

TEST(FusedEmotionAnalyzerTest, SoftmaxHeadKeepsStrongEvidenceAsDominantSignal) {
    EmotionFusionAnalyzerOptions options;
    options.label_reliability = {{"neutral", 0.82}, {"anger", 0.69}, {"joy", 0.78}};
    options.source_weights = {{"keyword", 1.0}};
    options.bert_signal_weight = 1.5;
    options.keyword_signal_weight = 3.0;
    FusedEmotionAnalyzer fusion(nullptr, options);

    auto bert = MakeEmotion("neutral", 0.50, {{"neutral", 0.50}, {"joy", 0.30}, {"anger", 0.20}});
    auto fused = fusion.FuseForTesting(bert, {EmotionEvidence{"anger", 1.0, "keyword", "烦死了"}});

    ASSERT_TRUE(fused.ok()) << fused.status().message();
    EXPECT_EQ(fused.value().emotion.primary, "anger");
    EXPECT_NEAR(ProbSum(fused.value()), 1.0, 1e-9);
    EXPECT_GT(fused.value().emotion.primary_prob, 0.70);
}

TEST(FusedEmotionAnalyzerTest, LlmFallbackGateCanInterceptWithoutSoftmaxEvidence) {
    EmotionFusionAnalyzerOptions options;
    options.accept_confidence = 0.99;
    options.ambiguity_margin = 0.99;
    options.llm_gate_confidence = 0.70;
    options.llm_gate_min_delta = -0.10;
    options.label_reliability = {{"neutral", 0.82}, {"anger", 0.69}, {"joy", 0.78}};
    auto bert = std::make_shared<StaticEmotionAnalyzer>(
        MakeEmotion("neutral", 0.46, {{"neutral", 0.46}, {"joy", 0.32}, {"anger", 0.22}}));
    auto llm = std::make_shared<StaticEmotionAnalyzer>(
        MakeEmotion("anger", 0.86, {{"anger", 0.86}, {"neutral", 0.10}, {"joy", 0.04}}));
    FusedEmotionAnalyzer fusion(bert, options, {}, llm);

    auto fused = fusion.Analyze("真服了，这一步讲得太乱了", "trace-test", nullptr);

    ASSERT_TRUE(fused.ok()) << fused.status().message();
    EXPECT_EQ(fused.value().emotion.primary, "anger");
    EXPECT_NEAR(ProbSum(fused.value()), 1.0, 1e-9);
}

TEST(FusedEmotionAnalyzerTest, AsyncPathPreservesOwnedTextAndTrace) {
    EmotionFusionAnalyzerOptions options;
    options.enabled = false;
    auto primary = std::make_shared<RecordingAsyncEmotionAnalyzer>(
        MakeEmotion("curiosity", 0.9, {{"curiosity", 0.9}, {"neutral", 0.1}}));
    FusedEmotionAnalyzer fusion(primary, options);
    std::optional<core::Result<EmotionAnalysis>> completed;

    auto status = fusion.AnalyzeAsync(
        "为什么天空是蓝色的？",
        "trace-async-fusion",
        nullptr,
        [&completed](core::Result<EmotionAnalysis> result) mutable {
            completed.emplace(std::move(result));
        });

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(primary->observed_text, "为什么天空是蓝色的？");
    EXPECT_EQ(primary->observed_trace_id, "trace-async-fusion");
    ASSERT_TRUE(completed.has_value());
    ASSERT_TRUE(completed->ok()) << completed->status().message();
    EXPECT_EQ(completed->value().emotion.primary, "curiosity");
}

TEST(KeywordEmotionEvidenceProviderTest, MatchesDefaultChineseRules) {
    auto provider = agent::service::persona::KeywordEmotionEvidenceProvider(
        agent::service::persona::DefaultEmotionKeywordRules());

    auto evidence = provider.Collect("这次考试考砸了，我有点难过", "trace-test");

    ASSERT_TRUE(evidence.ok()) << evidence.status().message();
    ASSERT_FALSE(evidence.value().empty());
    EXPECT_NE(std::find_if(evidence.value().begin(),
                           evidence.value().end(),
                           [](const auto& item) { return item.label == "sadness"; }),
              evidence.value().end());
}
