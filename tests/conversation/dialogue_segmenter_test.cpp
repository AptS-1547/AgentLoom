#include "dialogue_segmenter.h"

#include <gtest/gtest.h>

#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace {

class FakeEmbeddingProvider final : public agent::conversation::ITextEmbeddingProvider {
public:
    core::Result<std::vector<float>> EmbedText(std::string_view text) override {
        const auto found = embeddings.find(std::string(text));
        if (found == embeddings.end()) {
            return core::Status::Error(core::ErrorCode::NotFound, "test embedding is missing");
        }
        return found->second;
    }

    std::unordered_map<std::string, std::vector<float>> embeddings;
};

class BatchOnlyEmbeddingProvider final : public agent::conversation::ITextEmbeddingProvider {
public:
    core::Result<std::vector<float>> EmbedText(std::string_view) override {
        return core::Status::Error(
            core::ErrorCode::InternalError,
            "single embedding path must not be called");
    }

    core::Result<Batch> EmbedBatch(
        std::span<const std::string_view> texts) override {
        ++batch_calls;
        Batch batch;
        batch.batch_size = texts.size();
        batch.dimension = 2;
        for (const auto text : texts) {
            if (text.starts_with("a")) {
                batch.embeddings.insert(batch.embeddings.end(), {1.0f, 0.0f});
            } else {
                batch.embeddings.insert(batch.embeddings.end(), {0.0f, 1.0f});
            }
        }
        return batch;
    }

    bool Normalized() const noexcept override { return true; }

    std::size_t batch_calls = 0;
};

agent::conversation::DialogueTurn Turn(std::string id, std::string text, std::int64_t timestamp_us = 0) {
    return {
        .turn_id = std::move(id),
        .speaker = "speaker",
        .text = std::move(text),
        .timestamp_us = timestamp_us,
    };
}

} // namespace

TEST(BoundedDpDialogueSegmenterTest, SplitsDistinctTopicsAndKeepsOwnedSpansContiguous) {
    auto embeddings = std::make_shared<FakeEmbeddingProvider>();
    embeddings->embeddings = {
        {"price question", {1.0f, 0.0f}},
        {"price answer", {1.0f, 0.0f}},
        {"delivery question", {0.0f, 1.0f}},
        {"delivery answer", {0.0f, 1.0f}},
    };
    agent::conversation::BoundedDpDialogueSegmenter segmenter(
        embeddings,
        {
            .min_block_turns = 2,
            .max_block_turns = 4,
            .boundary_penalty = 0.2,
            .short_block_penalty = 1.0,
        });

    auto result = segmenter.Segment(
        "session-1",
        {
            Turn("t1", "price question"),
            Turn("t2", "price answer"),
            Turn("t3", "delivery question"),
            Turn("t4", "delivery answer"),
        });

    ASSERT_TRUE(result.ok()) << result.status().message();
    ASSERT_EQ(result.value().blocks.size(), 2u);
    EXPECT_EQ(result.value().blocks[0].owned_begin, 0u);
    EXPECT_EQ(result.value().blocks[0].owned_end, 2u);
    EXPECT_EQ(result.value().blocks[0].turn_ids, (std::vector<std::string>{"t1", "t2"}));
    EXPECT_EQ(result.value().blocks[1].owned_begin, 2u);
    EXPECT_EQ(result.value().blocks[1].owned_end, 4u);
    EXPECT_EQ(result.value().blocks[1].turn_ids, (std::vector<std::string>{"t3", "t4"}));
}

TEST(BoundedDpDialogueSegmenterTest, EnforcesMaximumBlockLength) {
    auto embeddings = std::make_shared<FakeEmbeddingProvider>();
    for (int index = 0; index < 5; ++index) {
        embeddings->embeddings.emplace("same-" + std::to_string(index), std::vector<float>{1.0f, 0.0f});
    }
    agent::conversation::BoundedDpDialogueSegmenter segmenter(
        embeddings,
        {
            .max_block_turns = 2,
            .boundary_penalty = 10.0,
        });
    std::vector<agent::conversation::DialogueTurn> turns;
    for (int index = 0; index < 5; ++index) {
        turns.push_back(Turn("t" + std::to_string(index), "same-" + std::to_string(index)));
    }

    auto result = segmenter.Segment("bounded", turns);

    ASSERT_TRUE(result.ok()) << result.status().message();
    ASSERT_EQ(result.value().blocks.size(), 3u);
    for (const auto& block : result.value().blocks) {
        EXPECT_LE(block.owned_end - block.owned_begin, 2u);
    }
    EXPECT_EQ(result.value().blocks.front().owned_begin, 0u);
    EXPECT_EQ(result.value().blocks.back().owned_end, 5u);
}

TEST(BoundedDpDialogueSegmenterTest, LargeClockGapCanSplitSemanticallySimilarTurns) {
    auto embeddings = std::make_shared<FakeEmbeddingProvider>();
    embeddings->embeddings = {
        {"same-a", {1.0f, 0.0f}},
        {"same-b", {1.0f, 0.0f}},
    };
    agent::conversation::BoundedDpDialogueSegmenter segmenter(
        embeddings,
        {
            .max_block_turns = 2,
            .boundary_penalty = 0.25,
            .clock_half_life_seconds = 10.0,
        });

    auto result = segmenter.Segment(
        "clock-gap",
        {
            Turn("t1", "same-a", 1'000'000),
            Turn("t2", "same-b", 101'000'000),
        });

    ASSERT_TRUE(result.ok()) << result.status().message();
    ASSERT_EQ(result.value().blocks.size(), 2u);
    EXPECT_EQ(result.value().blocks[0].owned_end, 1u);
    EXPECT_EQ(result.value().blocks[1].owned_begin, 1u);
}

TEST(BoundedDpDialogueSegmenterTest, NearbyContinuityBonusCanKeepAdjacentTurnsTogether) {
    auto embeddings = std::make_shared<FakeEmbeddingProvider>();
    embeddings->embeddings = {
        {"question", {1.0f, 0.0f}},
        {"short answer", {0.0f, 1.0f}},
    };
    const std::vector<agent::conversation::DialogueTurn> turns{
        Turn("t1", "question"),
        Turn("t2", "short answer"),
    };
    agent::conversation::BoundedDpDialogueSegmenter without_bonus(
        embeddings,
        {
            .max_block_turns = 2,
            .boundary_penalty = 1.0,
            .time_continuity_bonus = 0.0,
            .turn_half_life = 8.0,
        });
    agent::conversation::BoundedDpDialogueSegmenter with_bonus(
        embeddings,
        {
            .max_block_turns = 2,
            .boundary_penalty = 1.0,
            .time_continuity_bonus = 0.2,
            .turn_half_life = 8.0,
        });

    auto split = without_bonus.Segment("without-bonus", turns);
    auto merged = with_bonus.Segment("with-bonus", turns);

    ASSERT_TRUE(split.ok()) << split.status().message();
    ASSERT_TRUE(merged.ok()) << merged.status().message();
    EXPECT_EQ(split.value().blocks.size(), 2u);
    EXPECT_EQ(merged.value().blocks.size(), 1u);
}

TEST(LinearBoundaryLogitModelTest, EigenAutoDiffAndAdamLearnSeparableExamples) {
    auto created_model = agent::conversation::LinearBoundaryLogitModel::Create();
    ASSERT_TRUE(created_model.ok()) << created_model.status().message();
    auto model = std::move(created_model).value();
    core::optimization::AdamOptions optimizer_options;
    optimizer_options.learning_rate = 0.05;
    optimizer_options.max_gradient_norm = 5.0;
    auto created_optimizer = core::optimization::AdamOptimizer::Create(optimizer_options);
    ASSERT_TRUE(created_optimizer.ok()) << created_optimizer.status().message();
    auto optimizer = std::move(created_optimizer).value();

    auto example = [](double cosine, double boundary) {
        agent::conversation::BoundaryTrainingExample value;
        value.features.values[static_cast<std::size_t>(
            agent::conversation::BoundaryFeature::SemanticSimilarity)] = cosine;
        value.features.values[static_cast<std::size_t>(
            agent::conversation::BoundaryFeature::TimeContinuity)] = 1.0;
        value.boundary_label = boundary;
        return value;
    };
    const std::array examples{
        example(0.95, 0.0),
        example(0.80, 0.0),
        example(-0.20, 1.0),
        example(-0.60, 1.0),
    };
    double final_loss = 0.0;
    for (int epoch = 0; epoch < 250; ++epoch) {
        auto trained = model->TrainBatch(examples, *optimizer);
        ASSERT_TRUE(trained.ok()) << trained.status().message();
        final_loss = trained.value();
    }

    EXPECT_LT(final_loss, 0.05);
    EXPECT_LT(model->BoundaryProbability(examples[0].features), 0.1);
    EXPECT_GT(model->BoundaryProbability(examples[3].features), 0.9);
    EXPECT_GT(model->PenaltyAdjustment(examples[0].features), 0.0);
    EXPECT_LT(model->PenaltyAdjustment(examples[3].features), 0.0);

    auto restored_result = agent::conversation::LinearBoundaryLogitModel::Create();
    ASSERT_TRUE(restored_result.ok()) << restored_result.status().message();
    auto restored = std::move(restored_result).value();
    ASSERT_TRUE(restored->LoadParameters(model->Parameters()).ok());
    EXPECT_NEAR(
        restored->BoundaryProbability(examples[3].features),
        model->BoundaryProbability(examples[3].features),
        1e-12);
}

TEST(BoundedDpDialogueSegmenterTest, RejectsInconsistentEmbeddingDimensions) {
    auto embeddings = std::make_shared<FakeEmbeddingProvider>();
    embeddings->embeddings = {
        {"one", {1.0f, 0.0f}},
        {"two", {1.0f, 0.0f, 0.0f}},
    };
    agent::conversation::BoundedDpDialogueSegmenter segmenter(embeddings);

    auto result = segmenter.Segment("bad", {Turn("t1", "one"), Turn("t2", "two")});

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::InvalidArgument);
}

TEST(BoundedDpDialogueSegmenterTest, UsesConfiguredEmbeddingBatches) {
    auto embeddings = std::make_shared<BatchOnlyEmbeddingProvider>();
    agent::conversation::BoundedDpDialogueSegmenter segmenter(
        embeddings,
        {
            .max_block_turns = 4,
            .embedding_batch_size = 2,
            .boundary_penalty = 0.2,
        });

    auto result = segmenter.Segment(
        "batched",
        {
            Turn("t1", "a-1"),
            Turn("t2", "a-2"),
            Turn("t3", "b-1"),
            Turn("t4", "b-2"),
            Turn("t5", "b-3"),
        });

    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_EQ(embeddings->batch_calls, 3u);
}

TEST(BoundedDpDialogueSegmenterTest, RejectsZeroEmbeddingBatchSize) {
    auto embeddings = std::make_shared<FakeEmbeddingProvider>();
    embeddings->embeddings = {{"one", {1.0f, 0.0f}}};
    agent::conversation::BoundedDpDialogueSegmenter segmenter(
        embeddings,
        {.embedding_batch_size = 0});

    auto result = segmenter.Segment("bad-batch", {Turn("t1", "one")});

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::InvalidArgument);
}
