#include "embedding_pipeline.h"
#include "hf_tokenizer.h"
#include "onnx_text_embedding_model.h"
#include "text_embedding_model.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace {

using vector::EmbeddingBatch;
using vector::EmbeddingModelOptions;
using vector::EmbeddingPipeline;
using vector::EmbeddingPipelineOptions;
using vector::HfTokenizer;
using vector::OnnxTextEmbeddingModel;
using vector::PoolingStrategy;
using vector::TokenizedBatch;

const char* kTokenizerFixtureEnv = "HF_TOKENIZER_FIXTURE_JSON";
const char* kOnnxFixtureEnv = "HF_TEXT_EMBEDDING_ONNX";

std::filesystem::path TokenizerFixturePath() {
    const char* env = std::getenv(kTokenizerFixtureEnv);
    if (env == nullptr || *env == '\0') {
        return {};
    }
    return std::filesystem::path(env);
}

std::filesystem::path OnnxFixturePath() {
    const char* env = std::getenv(kOnnxFixtureEnv);
    if (env == nullptr || *env == '\0') {
        return {};
    }
    return std::filesystem::path(env);
}

bool TokenizerAvailable() {
    auto p = TokenizerFixturePath();
    if (p.empty()) return false;
    std::error_code ec;
    return std::filesystem::exists(p, ec) && !ec;
}

bool OnnxAvailable() {
    auto p = OnnxFixturePath();
    if (p.empty()) return false;
    std::error_code ec;
    return std::filesystem::exists(p, ec) && !ec;
}

#define REQUIRE_TOKENIZER()                                                   \
    do {                                                                      \
        if (!TokenizerAvailable()) {                                          \
            GTEST_SKIP() << "Set " << kTokenizerFixtureEnv                    \
                         << " to run tokenizer-dependent tests.";             \
        }                                                                     \
    } while (0)

#define REQUIRE_ONNX()                                                        \
    do {                                                                      \
        if (!OnnxAvailable()) {                                               \
            GTEST_SKIP() << "Set " << kOnnxFixtureEnv                         \
                         << " to an ONNX embedding model to run E2E tests."; \
        }                                                                     \
    } while (0)

HfTokenizer LoadTokenizer() {
    auto result = HfTokenizer::LoadFromFile(TokenizerFixturePath());
    EXPECT_TRUE(result.ok()) << result.status().message();
    return std::move(result).value();
}

} // namespace

TEST(PoolingUnitTest, MeanPoolSingleTokenAllAttended) {
    constexpr std::size_t B = 1, S = 1, H = 4;
    std::array<float, B * S * H> hidden{1.0f, 2.0f, 3.0f, 4.0f};
    std::array<std::int64_t, B * S> mask{1};
    std::array<float, B * H> out{};

    auto status = vector::pooling::MeanPool(hidden.data(), mask.data(), B, S, H, out.data());
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);
}

TEST(PoolingUnitTest, MeanPoolTwoTokensPartialMask) {
    constexpr std::size_t B = 1, S = 2, H = 2;
    std::array<float, B * S * H> hidden{
        2.0f, 4.0f,
        6.0f, 8.0f,
    };
    std::array<std::int64_t, B * S> mask{1, 0};
    std::array<float, B * H> out{};

    auto status = vector::pooling::MeanPool(hidden.data(), mask.data(), B, S, H, out.data());
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[1], 4.0f);
}

TEST(PoolingUnitTest, MeanPoolBatchOfTwo) {
    constexpr std::size_t B = 2, S = 2, H = 2;
    std::array<float, B * S * H> hidden{
        1.0f, 2.0f,
        3.0f, 4.0f,
        10.0f, 20.0f,
        30.0f, 40.0f,
    };
    std::array<std::int64_t, B * S> mask{
        1, 1,
        1, 0,
    };
    std::array<float, B * H> out{};

    auto status = vector::pooling::MeanPool(hidden.data(), mask.data(), B, S, H, out.data());
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[1], 3.0f);
    EXPECT_FLOAT_EQ(out[2], 10.0f);
    EXPECT_FLOAT_EQ(out[3], 20.0f);
}

TEST(PoolingUnitTest, ClsPoolExtractsFirstToken) {
    constexpr std::size_t B = 2, S = 3, H = 2;
    std::array<float, B * S * H> hidden{
        1.0f, 2.0f,
        99.0f, 99.0f,
        99.0f, 99.0f,
        10.0f, 20.0f,
        99.0f, 99.0f,
        99.0f, 99.0f,
    };
    std::array<float, B * H> out{};

    auto status = vector::pooling::ClsPool(hidden.data(), B, S, H, out.data());
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 10.0f);
    EXPECT_FLOAT_EQ(out[3], 20.0f);
}

TEST(PoolingUnitTest, L2NormalizeZeroVectorStaysZero) {
    std::array<float, 3> data{0.0f, 0.0f, 0.0f};
    vector::pooling::L2NormalizeRows(data.data(), 1, 3);
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 0.0f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
}

TEST(PoolingUnitTest, L2NormalizeUnitVectorStaysUnit) {
    std::array<float, 3> data{1.0f, 0.0f, 0.0f};
    vector::pooling::L2NormalizeRows(data.data(), 1, 3);
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 0.0f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
}

TEST(PoolingUnitTest, L2NormalizeMakesNormOne) {
    std::array<float, 3> data{3.0f, 4.0f, 0.0f};
    vector::pooling::L2NormalizeRows(data.data(), 1, 3);
    const float norm = std::sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
    EXPECT_NEAR(norm, 1.0f, 1e-6f);
}

TEST(OnnxTextEmbeddingE2ETest, LoadAndEmbedSingleText) {
    REQUIRE_TOKENIZER();
    REQUIRE_ONNX();

    auto tk = LoadTokenizer();
    ASSERT_TRUE(tk.valid());

    EmbeddingModelOptions opts;
    opts.model_path = OnnxFixturePath();
    opts.pooling = PoolingStrategy::Mean;
    opts.normalize = true;

    auto model_result = OnnxTextEmbeddingModel::Load(std::move(opts));
    ASSERT_TRUE(model_result.ok()) << model_result.status().message();
    auto model = std::move(model_result).value();

    auto tokenized = tk.Encode("hello embedding world");
    ASSERT_TRUE(tokenized.ok()) << tokenized.status().message();

    auto embedded = model->Embed(tokenized.value());
    ASSERT_TRUE(embedded.ok()) << embedded.status().message();

    const auto& batch = embedded.value();
    EXPECT_EQ(batch.batch_size, 1u);
    EXPECT_GT(batch.dimension, 0u);
    EXPECT_EQ(batch.embeddings.size(), batch.batch_size * batch.dimension);

    const float norm = std::sqrt(
        std::inner_product(batch.embeddings.begin(), batch.embeddings.end(),
                           batch.embeddings.begin(), 0.0f));
    EXPECT_NEAR(norm, 1.0f, 1e-5f);
}

TEST(OnnxTextEmbeddingE2ETest, EmbedBatchProducesSameShapeForAll) {
    REQUIRE_TOKENIZER();
    REQUIRE_ONNX();

    auto tk = LoadTokenizer();
    ASSERT_TRUE(tk.valid());

    EmbeddingModelOptions opts;
    opts.model_path = OnnxFixturePath();
    opts.pooling = PoolingStrategy::Mean;
    opts.normalize = true;

    auto model_result = OnnxTextEmbeddingModel::Load(std::move(opts));
    ASSERT_TRUE(model_result.ok()) << model_result.status().message();
    auto model = std::move(model_result).value();

    std::array<std::string_view, 3> texts{
        "short",
        "a longer sentence with more tokens",
        "中文测试句子",
    };

    auto tokenized = tk.EncodeBatch(std::span<const std::string_view>(texts));
    ASSERT_TRUE(tokenized.ok()) << tokenized.status().message();

    auto embedded = model->Embed(tokenized.value());
    ASSERT_TRUE(embedded.ok()) << embedded.status().message();

    EXPECT_EQ(embedded.value().batch_size, texts.size());
    EXPECT_GT(embedded.value().dimension, 0u);
    EXPECT_EQ(embedded.value().embeddings.size(),
              embedded.value().batch_size * embedded.value().dimension);
}

TEST(EmbeddingPipelineE2ETest, EncodeReturnsNormalizedVector) {
    REQUIRE_TOKENIZER();
    REQUIRE_ONNX();

    auto tk = LoadTokenizer();
    ASSERT_TRUE(tk.valid());

    EmbeddingModelOptions opts;
    opts.model_path = OnnxFixturePath();
    opts.pooling = PoolingStrategy::Mean;
    opts.normalize = true;

    auto model_result = OnnxTextEmbeddingModel::Load(std::move(opts));
    ASSERT_TRUE(model_result.ok()) << model_result.status().message();

    auto pipeline = EmbeddingPipeline(
        std::make_shared<HfTokenizer>(std::move(tk)),
        std::move(model_result).value(),
        EmbeddingPipelineOptions{}
    );

    auto vec = pipeline.Encode("pipeline test sentence");
    ASSERT_TRUE(vec.ok()) << vec.status().message();
    EXPECT_GT(vec.value().size(), 0u);

    const float norm = std::sqrt(
        std::inner_product(vec.value().begin(), vec.value().end(),
                           vec.value().begin(), 0.0f));
    EXPECT_NEAR(norm, 1.0f, 1e-5f);
}

TEST(EmbeddingPipelineE2ETest, SharedModelSupportsConcurrentPipelines) {
    REQUIRE_TOKENIZER();
    REQUIRE_ONNX();

    auto first_tokenizer = LoadTokenizer();
    auto second_tokenizer = LoadTokenizer();
    ASSERT_TRUE(first_tokenizer.valid());
    ASSERT_TRUE(second_tokenizer.valid());

    EmbeddingModelOptions opts;
    opts.model_path = OnnxFixturePath();
    opts.execution_provider = "cpu";
    opts.allow_cpu_fallback = false;
    opts.pooling = PoolingStrategy::Mean;
    opts.normalize = true;

    auto loaded = OnnxTextEmbeddingModel::Load(std::move(opts));
    ASSERT_TRUE(loaded.ok()) << loaded.status().message();
    std::shared_ptr<vector::IEmbeddingModel> shared_model(std::move(loaded).value());

    EmbeddingPipeline first(
        std::make_shared<HfTokenizer>(std::move(first_tokenizer)),
        shared_model);
    EmbeddingPipeline second(
        std::make_shared<HfTokenizer>(std::move(second_tokenizer)),
        shared_model);
    EXPECT_GE(shared_model.use_count(), 3);

    std::array<std::string_view, 3> first_texts{
        "shared model first pipeline",
        "parallel embedding request",
        "第一组并发文本",
    };
    std::array<std::string_view, 2> second_texts{
        "shared model second pipeline",
        "第二组并发文本",
    };

    auto first_result = std::async(std::launch::async, [&] {
        return first.EncodeBatch(first_texts);
    });
    auto second_result = std::async(std::launch::async, [&] {
        return second.EncodeBatch(second_texts);
    });

    auto first_batch = first_result.get();
    auto second_batch = second_result.get();
    ASSERT_TRUE(first_batch.ok()) << first_batch.status().message();
    ASSERT_TRUE(second_batch.ok()) << second_batch.status().message();
    EXPECT_EQ(first_batch.value().batch_size, first_texts.size());
    EXPECT_EQ(second_batch.value().batch_size, second_texts.size());
    EXPECT_EQ(first_batch.value().dimension, second_batch.value().dimension);
    EXPECT_EQ(shared_model->Dimension(), first_batch.value().dimension);
}
