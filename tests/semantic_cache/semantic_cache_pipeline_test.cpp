#include "../../src/semantic_cache/semantic_cache_pipeline.h"
#include "../../src/core/result.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <cmath>

using namespace agent::semantic_cache;
using namespace storage;

class MockVectorRepository : public IVectorRepository {
public:
    std::vector<CacheRecord> mock_data_;

    core::Result<std::vector<CacheRecord>> Search(
        const std::vector<float>& embedding,
        std::size_t top_k) override
    {
        if (mock_data_.empty()) {
            return std::vector<CacheRecord>{};
        }

        std::vector<std::pair<float, CacheRecord>> scored;
        scored.reserve(mock_data_.size());

        for (const auto& record : mock_data_) {
            float score = DotProduct(embedding, record.embedding);
            scored.emplace_back(score, record);
        }

        std::partial_sort(
            scored.begin(),
            scored.begin() + std::min(top_k, scored.size()),
            scored.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );

        std::vector<CacheRecord> results;
        results.reserve(std::min(top_k, scored.size()));
        for (std::size_t i = 0; i < std::min(top_k, scored.size()); ++i) {
            results.push_back(scored[i].second);
        }

        return std::move(results);
    }

    core::Status Store(const std::vector<CacheRecord>& records) override {
        mock_data_.insert(mock_data_.end(), records.begin(), records.end());
        return core::Status::Ok();
    }

private:
    static float DotProduct(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) return 0.0f;
        float sum = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
};

std::vector<float> GenerateRandomEmbedding(std::size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> embedding(dim);
    float norm = 0.0f;

    for (auto& val : embedding) {
        val = dist(rng);
        norm += val * val;
    }

    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (auto& val : embedding) {
            val /= norm;
        }
    }

    return embedding;
}

TEST(SemanticCacheCodecTest, SerializeDeserialize) {
    std::mt19937 rng(123);
    CacheRecord original;
    original.embedding = GenerateRandomEmbedding(384, rng);
    original.input = "测试输入";
    original.response = "测试响应";

    std::string serialized = SerializeCacheRecord(original);
    ASSERT_FALSE(serialized.empty());

    auto result = DeserializeCacheRecord(serialized);
    ASSERT_TRUE(result.ok());

    const auto& deserialized = result.value();
    ASSERT_EQ(deserialized.embedding.size(), original.embedding.size());
    for (std::size_t i = 0; i < original.embedding.size(); ++i) {
        EXPECT_FLOAT_EQ(deserialized.embedding[i], original.embedding[i]);
    }
    EXPECT_EQ(deserialized.input, original.input);
    EXPECT_EQ(deserialized.response, original.response);
}

TEST(SemanticCacheCodecTest, Serialize384Dim) {
    CacheRecord record;
    record.embedding.resize(384);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& val : record.embedding) {
        val = dist(rng);
    }

    record.input = "这是一个384维度的测试";
    record.response = "响应内容";

    std::string serialized = SerializeCacheRecord(record);

    std::size_t expected_size =
        sizeof(std::uint32_t) +                    // dim
        384 * sizeof(float) +                      // embedding
        sizeof(std::uint32_t) + record.input.size() +
        sizeof(std::uint32_t) + record.response.size();

    EXPECT_EQ(serialized.size(), expected_size);

    auto result = DeserializeCacheRecord(serialized);
    ASSERT_TRUE(result.ok());

    const auto& deserialized = result.value();
    ASSERT_EQ(deserialized.embedding.size(), 384u);
    EXPECT_EQ(deserialized.input, record.input);
    EXPECT_EQ(deserialized.response, record.response);
}

TEST(SemanticCacheCodecTest, InvalidData) {
    auto result1 = DeserializeCacheRecord("");
    EXPECT_FALSE(result1.ok());

    std::string truncated(10, '\0');
    auto result2 = DeserializeCacheRecord(truncated);
    EXPECT_FALSE(result2.ok());

    std::string invalid_dim;
    std::uint32_t bad_dim = 999999;
    invalid_dim.append(reinterpret_cast<const char*>(&bad_dim), sizeof(bad_dim));
    auto result3 = DeserializeCacheRecord(invalid_dim);
    EXPECT_FALSE(result3.ok());
}

TEST(MockVectorRepositoryTest, EmptySearch) {
    MockVectorRepository repo;
    std::vector<float> query(384, 0.1f);

    auto result = repo.Search(query, 5);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result.value().empty());
}

TEST(MockVectorRepositoryTest, StoreAndSearch) {
    MockVectorRepository repo;
    std::mt19937 rng(42);

    std::vector<CacheRecord> records;
    for (int i = 0; i < 10; ++i) {
        CacheRecord record;
        record.embedding = GenerateRandomEmbedding(384, rng);
        record.input = "输入" + std::to_string(i);
        record.response = "响应" + std::to_string(i);
        records.push_back(record);
    }

    auto store_status = repo.Store(records);
    ASSERT_TRUE(store_status.ok());

    std::vector<float> query = GenerateRandomEmbedding(384, rng);
    auto search_result = repo.Search(query, 3);
    ASSERT_TRUE(search_result.ok());

    const auto& hits = search_result.value();
    EXPECT_EQ(hits.size(), 3u);
}

TEST(MockVectorRepositoryTest, TopKOrdering) {
    MockVectorRepository repo;
    std::mt19937 rng(42);

    std::vector<float> base_embedding = GenerateRandomEmbedding(384, rng);

    for (int i = 0; i < 5; ++i) {
        CacheRecord record;
        record.embedding = base_embedding;

        float noise = i * 0.01f;
        for (auto& val : record.embedding) {
            val += noise;
        }

        float norm = 0.0f;
        for (auto val : record.embedding) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        for (auto& val : record.embedding) {
            val /= norm;
        }

        record.input = "输入" + std::to_string(i);
        record.response = "响应" + std::to_string(i);
        repo.mock_data_.push_back(record);
    }

    auto search_result = repo.Search(base_embedding, 3);
    ASSERT_TRUE(search_result.ok());

    const auto& hits = search_result.value();
    EXPECT_EQ(hits.size(), 3u);

    EXPECT_EQ(hits[0].input, "输入0");
}

TEST(DotProductTest, Unrolled384) {
    std::mt19937 rng(42);
    std::vector<float> a = GenerateRandomEmbedding(384, rng);
    std::vector<float> b = GenerateRandomEmbedding(384, rng);

    float simd_result = dot_product_unrolled<384>(a.data(), b.data());

    float naive_result = 0.0f;
    for (std::size_t i = 0; i < 384; ++i) {
        naive_result += a[i] * b[i];
    }

    EXPECT_NEAR(simd_result, naive_result, 1e-4f);
}

TEST(DotProductTest, SmallDimensions) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};

    float result = dot_product_small<3>(a, b);
    EXPECT_FLOAT_EQ(result, 32.0f);
}

TEST(DotProductTest, NormalizedVectors) {
    std::mt19937 rng(42);
    std::vector<float> a = GenerateRandomEmbedding(384, rng);
    std::vector<float> b = GenerateRandomEmbedding(384, rng);

    float dot = dot_product_unrolled<384>(a.data(), b.data());

    EXPECT_GE(dot, -1.0f);
    EXPECT_LE(dot, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
