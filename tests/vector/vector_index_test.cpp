#include "exact_vector_index.h"
#include "faiss_vector_index.h"
#include "vector_index.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

namespace {

using vector::ExactVectorIndex;
using vector::FaissVectorIndex;
using vector::IVectorIndex;
using vector::VectorIndexOptions;
using vector::VectorSearchResult;

std::filesystem::path TempFile(const std::string& tag) {
    auto p = std::filesystem::temp_directory_path() /
             (std::string("agent_vector_index_test_") + tag + ".bin");
    std::error_code ec;
    std::filesystem::remove(p, ec);
    return p;
}

void NormalizeRow(float* row, std::size_t dim) {
    double sq = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        sq += static_cast<double>(row[i]) * row[i];
    }
    const float n = static_cast<float>(std::sqrt(sq));
    if (n <= 1e-12f) return;
    for (std::size_t i = 0; i < dim; ++i) {
        row[i] /= n;
    }
}

std::vector<float> RandomNormalizedCorpus(std::size_t count, std::size_t dim, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            data[i * dim + d] = dist(rng);
        }
        NormalizeRow(data.data() + i * dim, dim);
    }
    return data;
}

template <typename IndexT>
core::Result<std::unique_ptr<IndexT>> CreateIndex(std::size_t dim) {
    VectorIndexOptions opts;
    opts.dimension = dim;
    return IndexT::Create(opts);
}

} // namespace

// --- Common contract tests, parameterised over both backends. ---

class VectorIndexCommonTest
    : public ::testing::TestWithParam<std::string> {
protected:
    std::unique_ptr<IVectorIndex> MakeIndex(std::size_t dim) {
        const auto& backend = GetParam();
        if (backend == "exact") {
            auto r = CreateIndex<ExactVectorIndex>(dim);
            EXPECT_TRUE(r.ok()) << r.status().message();
            return std::move(r).value();
        }
        if (backend == "faiss") {
            auto r = CreateIndex<FaissVectorIndex>(dim);
            EXPECT_TRUE(r.ok()) << r.status().message();
            return std::move(r).value();
        }
        ADD_FAILURE() << "unknown backend: " << backend;
        return nullptr;
    }
};

TEST_P(VectorIndexCommonTest, CreateZeroDimensionFails) {
    const auto& backend = GetParam();
    if (backend == "exact") {
        auto r = ExactVectorIndex::Create({0});
        EXPECT_FALSE(r.ok());
    } else {
        auto r = FaissVectorIndex::Create({0});
        EXPECT_FALSE(r.ok());
    }
}

TEST_P(VectorIndexCommonTest, EmptyIndexSearchReturnsEmpty) {
    auto idx = MakeIndex(4);
    ASSERT_NE(idx, nullptr);
    EXPECT_EQ(idx->Size(), 0u);

    std::array<float, 4> query{1, 0, 0, 0};
    auto r = idx->Search(query, 5);
    ASSERT_TRUE(r.ok()) << r.status().message();
    EXPECT_TRUE(r.value().empty());
}

TEST_P(VectorIndexCommonTest, AddDimensionMismatchFails) {
    auto idx = MakeIndex(4);
    ASSERT_NE(idx, nullptr);

    std::array<float, 6> wrong_vec{0, 0, 0, 0, 0, 0};
    std::array<std::int64_t, 1> ids{1};
    auto status = idx->Add(wrong_vec, ids);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), core::ErrorCode::InvalidArgument);
}

TEST_P(VectorIndexCommonTest, SearchDimensionMismatchFails) {
    auto idx = MakeIndex(4);
    ASSERT_NE(idx, nullptr);

    std::array<float, 5> wrong_query{1, 0, 0, 0, 0};
    auto r = idx->Search(wrong_query, 1);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), core::ErrorCode::InvalidArgument);
}

TEST_P(VectorIndexCommonTest, AddAndSelfRetrieval) {
    auto idx = MakeIndex(4);
    ASSERT_NE(idx, nullptr);

    std::array<float, 12> vectors{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
    };
    std::array<std::int64_t, 3> ids{10, 20, 30};
    ASSERT_TRUE(idx->Add(vectors, ids).ok());
    EXPECT_EQ(idx->Size(), 3u);

    // Query equal to row 1 should return id=20 first.
    std::array<float, 4> q{0.0f, 1.0f, 0.0f, 0.0f};
    auto r = idx->Search(q, 1);
    ASSERT_TRUE(r.ok()) << r.status().message();
    ASSERT_EQ(r.value().size(), 1u);
    EXPECT_EQ(r.value()[0].id, 20);
    EXPECT_NEAR(r.value()[0].score, 1.0f, 1e-5f);
}

TEST_P(VectorIndexCommonTest, TopKOrdering) {
    auto idx = MakeIndex(2);
    ASSERT_NE(idx, nullptr);

    // 3 unit vectors at 0, 30, 60 degrees.
    std::array<float, 6> vectors{
        1.0f, 0.0f,
        0.8660254f, 0.5f,
        0.5f, 0.8660254f,
    };
    std::array<std::int64_t, 3> ids{100, 200, 300};
    ASSERT_TRUE(idx->Add(vectors, ids).ok());

    std::array<float, 2> q{1.0f, 0.0f};
    auto r = idx->Search(q, 3);
    ASSERT_TRUE(r.ok());
    ASSERT_EQ(r.value().size(), 3u);
    EXPECT_EQ(r.value()[0].id, 100);
    EXPECT_EQ(r.value()[1].id, 200);
    EXPECT_EQ(r.value()[2].id, 300);
    EXPECT_GT(r.value()[0].score, r.value()[1].score);
    EXPECT_GT(r.value()[1].score, r.value()[2].score);
}

TEST_P(VectorIndexCommonTest, TopKGreaterThanSizeReturnsAll) {
    auto idx = MakeIndex(2);
    std::array<float, 4> v{1, 0, 0, 1};
    std::array<std::int64_t, 2> ids{1, 2};
    ASSERT_TRUE(idx->Add(v, ids).ok());

    std::array<float, 2> q{1, 0};
    auto r = idx->Search(q, 100);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value().size(), 2u);
}

TEST_P(VectorIndexCommonTest, RemoveReducesSize) {
    auto idx = MakeIndex(2);
    std::array<float, 6> v{1, 0, 0, 1, 1, 1};
    std::array<std::int64_t, 3> ids{1, 2, 3};
    ASSERT_TRUE(idx->Add(v, ids).ok());
    EXPECT_EQ(idx->Size(), 3u);

    std::array<std::int64_t, 1> to_remove{2};
    ASSERT_TRUE(idx->Remove(to_remove).ok());
    EXPECT_EQ(idx->Size(), 2u);

    // Search should no longer return id=2.
    std::array<float, 2> q{0, 1};
    auto r = idx->Search(q, 5);
    ASSERT_TRUE(r.ok());
    for (const auto& hit : r.value()) {
        EXPECT_NE(hit.id, 2);
    }
}

TEST_P(VectorIndexCommonTest, ResetClearsIndex) {
    auto idx = MakeIndex(2);
    std::array<float, 4> v{1, 0, 0, 1};
    std::array<std::int64_t, 2> ids{1, 2};
    ASSERT_TRUE(idx->Add(v, ids).ok());
    ASSERT_TRUE(idx->Reset().ok());
    EXPECT_EQ(idx->Size(), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    Backends,
    VectorIndexCommonTest,
    ::testing::Values("exact", "faiss"));

// --- Save / Load round-trip tests (backend-specific format) ---

TEST(ExactVectorIndexTest, SaveLoadRoundTrip) {
    auto idx_r = ExactVectorIndex::Create({4});
    ASSERT_TRUE(idx_r.ok());
    auto idx = std::move(idx_r).value();

    std::array<float, 8> v{1, 0, 0, 0, 0, 1, 0, 0};
    std::array<std::int64_t, 2> ids{1, 2};
    ASSERT_TRUE(idx->Add(v, ids).ok());

    auto path = TempFile("exact_roundtrip");
    ASSERT_TRUE(idx->Save(path).ok());

    auto loaded_r = ExactVectorIndex::LoadFromFile(path);
    ASSERT_TRUE(loaded_r.ok()) << loaded_r.status().message();
    auto loaded = std::move(loaded_r).value();

    EXPECT_EQ(loaded->Dimension(), 4u);
    EXPECT_EQ(loaded->Size(), 2u);

    std::array<float, 4> q{1, 0, 0, 0};
    auto r = loaded->Search(q, 1);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value()[0].id, 1);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(ExactVectorIndexTest, LoadBadMagicFails) {
    auto path = TempFile("exact_bad_magic");
    {
        std::ofstream os(path, std::ios::binary);
        os.write("NOTAVECT", 8);
    }
    auto r = ExactVectorIndex::LoadFromFile(path);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), core::ErrorCode::FailedPrecondition);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(FaissVectorIndexTest, SaveLoadRoundTrip) {
    auto idx_r = FaissVectorIndex::Create({4});
    ASSERT_TRUE(idx_r.ok());
    auto idx = std::move(idx_r).value();

    std::array<float, 8> v{1, 0, 0, 0, 0, 1, 0, 0};
    std::array<std::int64_t, 2> ids{42, 99};
    ASSERT_TRUE(idx->Add(v, ids).ok());

    auto path = TempFile("faiss_roundtrip");
    ASSERT_TRUE(idx->Save(path).ok());

    auto loaded_r = FaissVectorIndex::LoadFromFile(path);
    ASSERT_TRUE(loaded_r.ok()) << loaded_r.status().message();
    auto loaded = std::move(loaded_r).value();

    EXPECT_EQ(loaded->Dimension(), 4u);
    EXPECT_EQ(loaded->Size(), 2u);

    std::array<float, 4> q{0, 1, 0, 0};
    auto r = loaded->Search(q, 1);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value()[0].id, 99);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(FaissVectorIndexTest, LoadDimensionExpectationCheck) {
    auto idx_r = FaissVectorIndex::Create({4});
    ASSERT_TRUE(idx_r.ok());
    auto idx = std::move(idx_r).value();

    std::array<float, 4> v{1, 0, 0, 0};
    std::array<std::int64_t, 1> ids{1};
    ASSERT_TRUE(idx->Add(v, ids).ok());

    auto path = TempFile("faiss_dim_check");
    ASSERT_TRUE(idx->Save(path).ok());

    auto wrong = FaissVectorIndex::LoadFromFile(path, 8);
    EXPECT_FALSE(wrong.ok());
    EXPECT_EQ(wrong.status().code(), core::ErrorCode::FailedPrecondition);

    auto right = FaissVectorIndex::LoadFromFile(path, 4);
    EXPECT_TRUE(right.ok());

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// --- Parity: ExactVectorIndex and FaissVectorIndex must agree top-k on random corpus. ---

TEST(VectorIndexParityTest, ExactAndFaissAgreeOnRandomCorpus) {
    constexpr std::size_t kDim = 32;
    constexpr std::size_t kN = 128;
    constexpr std::size_t kQ = 16;
    constexpr std::size_t kTopK = 5;

    auto corpus = RandomNormalizedCorpus(kN, kDim, /*seed=*/1234);
    std::vector<std::int64_t> ids(kN);
    std::iota(ids.begin(), ids.end(), 1);

    auto exact_r = ExactVectorIndex::Create({kDim});
    auto faiss_r = FaissVectorIndex::Create({kDim});
    ASSERT_TRUE(exact_r.ok());
    ASSERT_TRUE(faiss_r.ok());
    auto exact_idx = std::move(exact_r).value();
    auto faiss_idx = std::move(faiss_r).value();

    ASSERT_TRUE(exact_idx->Add(corpus, ids).ok());
    ASSERT_TRUE(faiss_idx->Add(corpus, ids).ok());

    auto queries = RandomNormalizedCorpus(kQ, kDim, /*seed=*/5678);
    for (std::size_t qi = 0; qi < kQ; ++qi) {
        std::span<const float> q(queries.data() + qi * kDim, kDim);

        auto er = exact_idx->Search(q, kTopK);
        auto fr = faiss_idx->Search(q, kTopK);
        ASSERT_TRUE(er.ok()) << er.status().message();
        ASSERT_TRUE(fr.ok()) << fr.status().message();
        ASSERT_EQ(er.value().size(), kTopK);
        ASSERT_EQ(fr.value().size(), kTopK);

        for (std::size_t i = 0; i < kTopK; ++i) {
            EXPECT_EQ(er.value()[i].id, fr.value()[i].id)
                << "query " << qi << ", rank " << i;
            EXPECT_NEAR(er.value()[i].score, fr.value()[i].score, 1e-4f)
                << "query " << qi << ", rank " << i;
        }
    }
}

TEST(VectorIndexParityTest, ExactAndFaissAgreeAfterRemove) {
    constexpr std::size_t kDim = 16;
    constexpr std::size_t kN = 64;

    auto corpus = RandomNormalizedCorpus(kN, kDim, /*seed=*/777);
    std::vector<std::int64_t> ids(kN);
    std::iota(ids.begin(), ids.end(), 1);

    auto exact_r = ExactVectorIndex::Create({kDim});
    auto faiss_r = FaissVectorIndex::Create({kDim});
    auto exact_idx = std::move(exact_r).value();
    auto faiss_idx = std::move(faiss_r).value();
    ASSERT_TRUE(exact_idx->Add(corpus, ids).ok());
    ASSERT_TRUE(faiss_idx->Add(corpus, ids).ok());

    std::vector<std::int64_t> to_remove{5, 17, 33, 50};
    ASSERT_TRUE(exact_idx->Remove(to_remove).ok());
    ASSERT_TRUE(faiss_idx->Remove(to_remove).ok());

    EXPECT_EQ(exact_idx->Size(), faiss_idx->Size());

    auto queries = RandomNormalizedCorpus(4, kDim, /*seed=*/999);
    for (std::size_t qi = 0; qi < 4; ++qi) {
        std::span<const float> q(queries.data() + qi * kDim, kDim);
        auto er = exact_idx->Search(q, 3);
        auto fr = faiss_idx->Search(q, 3);
        ASSERT_TRUE(er.ok());
        ASSERT_TRUE(fr.ok());
        for (const auto& hit : er.value()) {
            for (auto rid : to_remove) {
                EXPECT_NE(hit.id, rid);
            }
        }
        for (std::size_t i = 0; i < er.value().size(); ++i) {
            EXPECT_EQ(er.value()[i].id, fr.value()[i].id);
        }
    }
}

TEST(VectorIndexBackendNameTest, BackendNamesAreDistinct) {
    auto e = ExactVectorIndex::Create({4});
    auto f = FaissVectorIndex::Create({4});
    ASSERT_TRUE(e.ok());
    ASSERT_TRUE(f.ok());
    EXPECT_EQ(e.value()->BackendName(), "exact_ip");
    EXPECT_EQ(f.value()->BackendName(), "faiss_flat_ip");
}
