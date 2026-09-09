#include "async_embedding.h"

#include <gtest/gtest.h>

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class FakeEmbeddingProvider final : public vector::IEmbeddingBatchProvider {
public:
    core::Result<vector::EmbeddingBatch> EncodeBatch(
        std::span<const std::string_view> texts) const override {
        vector::EmbeddingBatch result;
        result.batch_size = texts.size();
        result.dimension = 1;
        result.embeddings.reserve(texts.size());
        for (const auto text : texts) {
            result.embeddings.push_back(static_cast<float>(text.size()));
        }
        return result;
    }
};

// SegmentAsync 分块 parity 测试：与同步 Segment 使用同一套「单位向量」映射，
// 用同步路径作 oracle，断言协程路径的分块结果逐块一致。
class UnitEmbeddingBatchProvider final : public vector::IEmbeddingBatchProvider {
public:
    core::Result<vector::EmbeddingBatch> EncodeBatch(
        std::span<const std::string_view> texts) const override {
        vector::EmbeddingBatch result;
        result.batch_size = texts.size();
        result.dimension = 2;
        result.embeddings.reserve(texts.size() * 2);
        for (const auto text : texts) {
            if (text.starts_with("a")) {
                result.embeddings.insert(result.embeddings.end(), {1.0f, 0.0f});
            } else {
                result.embeddings.insert(result.embeddings.end(), {0.0f, 1.0f});
            }
        }
        return result;
    }
};

// 同步 oracle：与 UnitEmbeddingBatchProvider 相同的单位向量映射，声明已归一化。
class UnitEmbeddingSyncProvider final : public agent::conversation::ITextEmbeddingProvider {
public:
    core::Result<std::vector<float>> EmbedText(std::string_view) override {
        return core::Status::Error(
            core::ErrorCode::InternalError,
            "single embedding path must not be called");
    }

    core::Result<Batch> EmbedBatch(
        std::span<const std::string_view> texts) override {
        Batch batch;
        batch.batch_size = texts.size();
        batch.dimension = 2;
        batch.embeddings.reserve(texts.size() * 2);
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
};

agent::conversation::DialogueTurn MakeTurn(
    std::string id, std::string text, std::int64_t timestamp_us = 0) {
    return {
        .turn_id = std::move(id),
        .speaker = "speaker",
        .text = std::move(text),
        .timestamp_us = timestamp_us,
    };
}

class AsyncEmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(compute_pool_.Start().ok());
        provider_ = std::make_shared<FakeEmbeddingProvider>();
        coordinator_ = std::make_unique<vector::EmbeddingBatchCoordinator>(
            compute_pool_, provider_,
            vector::EmbeddingBatchCoordinatorOptions{.max_batch_size = 8});
        ASSERT_TRUE(coordinator_->Start().ok());
    }
    void TearDown() override {
        coordinator_->Shutdown();
        compute_pool_.Shutdown(true);
    }

    core::ThreadPool compute_pool_{{.worker_count = 2,
                                    .queue_capacity = 32,
                                    .name = "async-embedding-test"}};
    std::shared_ptr<FakeEmbeddingProvider> provider_;
    std::unique_ptr<vector::EmbeddingBatchCoordinator> coordinator_;
};

TEST_F(AsyncEmbeddingTest, EmbedBatchAsyncAssemblesOrderedBatch) {
    std::vector<std::string> texts = {"hello", "world", "foo"};
    std::vector<std::string_view> views(texts.begin(), texts.end());

    auto task = agent::conversation::EmbedBatchAsync(*coordinator_, views);
    auto result = task.get();
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_EQ(result.value().batch_size, 3u);
    EXPECT_EQ(result.value().dimension, 1u);
    ASSERT_EQ(result.value().embeddings.size(), 3u);
    EXPECT_FLOAT_EQ(result.value().embeddings[0], 5.0f);  // "hello"
    EXPECT_FLOAT_EQ(result.value().embeddings[1], 5.0f);  // "world"
    EXPECT_FLOAT_EQ(result.value().embeddings[2], 3.0f);  // "foo"
}

TEST_F(AsyncEmbeddingTest, EmptyBatchReturnsImmediately) {
    std::vector<std::string_view> views;
    auto task = agent::conversation::EmbedBatchAsync(*coordinator_, views);
    auto result = task.get();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().batch_size, 0u);
}

TEST_F(AsyncEmbeddingTest, SegmentAsyncMatchesSyncSegmentation) {
    // 用同步 Segment 作 oracle，断言协程 SegmentAsync 分块结果逐块一致。
    auto coordinator = std::make_shared<vector::EmbeddingBatchCoordinator>(
        compute_pool_,
        std::make_shared<UnitEmbeddingBatchProvider>(),
        vector::EmbeddingBatchCoordinatorOptions{.max_batch_size = 8});
    ASSERT_TRUE(coordinator->Start().ok());

    const agent::conversation::DialogueSegmenterOptions options{
        .min_block_turns = 2,
        .max_block_turns = 4,
        .boundary_penalty = 0.2,
        .short_block_penalty = 1.0,
    };
    agent::conversation::BoundedDpDialogueSegmenter async_segmenter(coordinator, options);
    agent::conversation::BoundedDpDialogueSegmenter sync_segmenter(
        std::make_shared<UnitEmbeddingSyncProvider>(), options);

    const std::vector<agent::conversation::DialogueTurn> turns{
        MakeTurn("t1", "a-1"),
        MakeTurn("t2", "a-2"),
        MakeTurn("t3", "b-1"),
        MakeTurn("t4", "b-2"),
    };

    auto sync_result = sync_segmenter.Segment("parity", turns);
    ASSERT_TRUE(sync_result.ok()) << sync_result.status().message();
    auto async_result = async_segmenter.SegmentAsync("parity", turns).get();
    ASSERT_TRUE(async_result.ok()) << async_result.status().message();

    const auto& expected = sync_result.value().blocks;
    const auto& actual = async_result.value().blocks;
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual[i].owned_begin, expected[i].owned_begin);
        EXPECT_EQ(actual[i].owned_end, expected[i].owned_end);
        EXPECT_EQ(actual[i].turn_ids, expected[i].turn_ids);
    }

    coordinator->Shutdown();
}

TEST_F(AsyncEmbeddingTest, SegmentAsyncRejectsNonNormalizedEmbeddings) {
    // 协程路径：非单位范数（FakeEmbeddingProvider 返回 {text.size()}，范数≠1）应被探测并拒绝。
    auto coordinator = std::make_shared<vector::EmbeddingBatchCoordinator>(
        compute_pool_,
        std::make_shared<FakeEmbeddingProvider>(),
        vector::EmbeddingBatchCoordinatorOptions{.max_batch_size = 8});
    ASSERT_TRUE(coordinator->Start().ok());

    agent::conversation::BoundedDpDialogueSegmenter segmenter(coordinator);
    auto result = segmenter.SegmentAsync(
        "reject",
        {MakeTurn("t1", "hello"), MakeTurn("t2", "world")}).get();

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::InvalidArgument);

    coordinator->Shutdown();
}

} // namespace
