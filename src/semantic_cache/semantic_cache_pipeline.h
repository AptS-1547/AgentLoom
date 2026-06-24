#pragma once

#include "context_risk_detector.h"
#include "isemantic_cache.h"
#include "semantic_cache_policy.h"
#include "semantic_cache_types.h"
#include "result.h"
#include "../vector/hf_tokenizer.h"
#include "../storage/sqlite/sqlite_connection.h"
#include "../vector/onnx_text_embedding_model.h"
#include <cstdint>
#include <deque>
#include <memory>
#include <array>
#include <string>
#include <string_view>
#include <immintrin.h>
#include <mutex>
#include <chrono>
#include <algorithm>


#if defined(_MSC_VER)
#define AGENT_SEMANTIC_CACHE_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define AGENT_SEMANTIC_CACHE_RESTRICT __restrict__
#else
#define AGENT_SEMANTIC_CACHE_RESTRICT
#endif

// This file implements a simple semantic cache pipeline that stores CacheRecords in Redis in batches.
template<size_t Dim, typename T = float>
inline T dot_product_unrolled(const T* AGENT_SEMANTIC_CACHE_RESTRICT a,
                              const T* AGENT_SEMANTIC_CACHE_RESTRICT b) noexcept
{
    static_assert(Dim > 0, "Dimension must be > 0");
    static_assert(std::is_same_v<T, float>, "Only float is supported in this AVX2 version");

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    constexpr size_t vec_chunk = 32; 
    constexpr size_t full_blocks = Dim / vec_chunk;

    for (size_t i = 0; i < full_blocks * vec_chunk; i += vec_chunk) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);

        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);

        __m256 a2 = _mm256_loadu_ps(a + i + 16);
        __m256 b2 = _mm256_loadu_ps(b + i + 16);
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);

        __m256 a3 = _mm256_loadu_ps(a + i + 24);
        __m256 b3 = _mm256_loadu_ps(b + i + 24);
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);
    }

    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

  
    constexpr size_t remainder_start = full_blocks * vec_chunk;


    if constexpr (Dim % vec_chunk >= 8) {
        __m256 a_rem = _mm256_loadu_ps(a + remainder_start);
        __m256 b_rem = _mm256_loadu_ps(b + remainder_start);
        sum0 = _mm256_fmadd_ps(a_rem, b_rem, sum0);
    }
    if constexpr (Dim % vec_chunk >= 16) {
        __m256 a_rem = _mm256_loadu_ps(a + remainder_start + 8);
        __m256 b_rem = _mm256_loadu_ps(b + remainder_start + 8);
        sum1 = _mm256_fmadd_ps(a_rem, b_rem, sum1);
        sum0 = _mm256_add_ps(sum0, sum1);
    }
    if constexpr (Dim % vec_chunk >= 24) {
        __m256 a_rem = _mm256_loadu_ps(a + remainder_start + 16);
        __m256 b_rem = _mm256_loadu_ps(b + remainder_start + 16);
        sum2 = _mm256_fmadd_ps(a_rem, b_rem, sum2);
        sum0 = _mm256_add_ps(sum0, sum2);
    }

    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    float result = _mm_cvtss_f32(sum128);


    constexpr size_t leftover8 = Dim - remainder_start;
    if constexpr (leftover8 > 0 && leftover8 < 8) {
    
        for (size_t i = remainder_start; i < Dim; ++i) {
            result += a[i] * b[i];
        }
    }

    return result;
}

template<size_t Dim>
inline float dot_product_small(const float* AGENT_SEMANTIC_CACHE_RESTRICT a,
                               const float* AGENT_SEMANTIC_CACHE_RESTRICT b) noexcept
{
    static_assert(Dim <= 8, "Use only for small dims");
    if constexpr (Dim == 0) return 0.0f;
    else if constexpr (Dim == 1) return a[0] * b[0];
    else if constexpr (Dim == 2) return a[0]*b[0] + a[1]*b[1];
    else if constexpr (Dim == 3) return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    else if constexpr (Dim == 4) return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    else if constexpr (Dim == 5) return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4];
    else if constexpr (Dim == 6) return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4] + a[5]*b[5];
    else if constexpr (Dim == 7) return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4] + a[5]*b[5] + a[6]*b[6];
    else {
        float sum = 0;
        for (size_t i = 0; i < Dim; ++i) sum += a[i] * b[i];
        return sum;
    }
}
namespace storage {

    struct CacheRecord {
        std::vector<float> embedding;
        std::string input;
        std::string response;
        agent::semantic_cache::CacheEntryMetadata metadata;
        std::string payload_type;
        std::string prompt_version;
        std::string model_version;
        std::string corpus_version;
        std::string policy_version;
        std::unordered_map<std::string, std::string> extra_metadata;
    };

    class IVectorRepository{
        public:
            virtual ~IVectorRepository() = default;
            virtual core::Result<std::vector<storage::CacheRecord>> Search(const std::vector<float>& embedding, std::size_t top_k) = 0;
            virtual core::Status Store(const std::vector<storage::CacheRecord>& records) = 0;
    };
}





namespace agent::semantic_cache {

class RedisConnectionPool;

inline constexpr std::uint32_t kExpectedEmbeddingDim = 384;

std::string SerializeCacheRecord(const storage::CacheRecord& r);
core::Result<storage::CacheRecord> DeserializeCacheRecord(std::string_view buf);

/// Dependencies injected into the pipeline.  Forward-declared opaque types
/// keep the public header free of Faiss / Ort / tokenizer details.
///
/// TODO(orange): pick the concrete handle types when wiring this up — likely
///   - tokenizer:        agent::vector::HfTokenizer (or pool)
///   - embedding:        agent::vector::OnnxTextEmbeddingModel
///   - index manager:    agent::vector::VectorIndexManager
///   - repository:       agent::storage::IVectorRepository
///
/// They can be stored as raw references / shared_ptr / unique_ptr — pick
/// whatever matches the lifetime story you want.  This header just owns the
/// composition; concrete types live in the .cpp.
namespace cache_vector {
    
    enum class CacheMode{
        CONTEXT_WINDOW,
        SEMANTIC_SIMILARITY
    };

    struct SearchWithContextResult {
        storage::CacheRecord hit;
        float score = 0.0f;
        std::vector<storage::CacheRecord> context;
    };

    struct ScoredCacheRecord {
        storage::CacheRecord record;
        float score = 0.0f;
    };

    class VectorIndexManager : storage::IVectorRepository {
        public:
            VectorIndexManager(std::string user_uuid,
                               std::shared_ptr<RedisConnectionPool> redis_pool,
                               storage::sqlite::SqliteConnection sqlite_conn,
                               std::size_t max_cached_records = 1000);
            VectorIndexManager(const VectorIndexManager&) = delete;
            VectorIndexManager& operator=(const VectorIndexManager&) = delete;

            core::Status AddRecord(const storage::CacheRecord& record);
            core::Result<std::vector<storage::CacheRecord>> Search(const std::vector<float>& embedding, std::size_t top_k) override;
            core::Result<std::vector<ScoredCacheRecord>> SearchAllBatches(
                const std::vector<float>& embedding,
                std::size_t top_k);
            core::Result<std::vector<SearchWithContextResult>> SearchWithContext(
                const std::vector<float>& embedding,
                std::size_t top_k,
                std::size_t neighbors_per_hit);
            core::Status Store(const std::vector<storage::CacheRecord>& records) override;

            core::Status StoreTimestampIndex();
            core::Status LoadTimestampIndex();
            std::size_t CurrentSize() const;
            const std::string& user_uuid() const noexcept { return user_uuid_; }
            const std::vector<std::int64_t>& backpack_timestamps() const noexcept { return backpack_timestamps_; }
            ~VectorIndexManager() = default;

        private:
            core::Status ReloadNextBatch();
            core::Status RebuildTimestampIndexFromRedis();
            core::Status LoadActiveBatch();
            core::Status SaveActiveBatch();
            core::Status PromoteBatchToIndex();
            std::string BuildBatchKey(std::int64_t timestamp) const;
            std::size_t max_cached_records_;
            std::string user_uuid_;
            std::shared_ptr<RedisConnectionPool> redis_pool_;
            storage::sqlite::SqliteConnection sqlite_conn_;
            std::deque<std::int64_t> timestamp_index_;
            std::vector<std::int64_t> backpack_timestamps_;
            std::int64_t active_timestamp_;
            std::size_t active_count_;
            std::size_t read_cursor_ = 0;
            std::vector<storage::CacheRecord> search_records_;
            std::mutex mutex_;
    };
}

struct SemanticCachePipelineDeps {
    std::shared_ptr<IContextRiskDetector> risk_detector;
    std::shared_ptr<IPolicyMatcher> policy_matcher;
    std::shared_ptr<vector::HfTokenizer> tokenizer;
    std::shared_ptr<vector::OnnxTextEmbeddingModel> embedding_model;
    std::shared_ptr<cache_vector::VectorIndexManager> index_manager;
    std::shared_ptr<storage::IVectorRepository> repository;
};


struct SemanticCachePipelineOptions {
    std::size_t top_k = 8;
    float similarity_floor = 0.85f;
    bool enable_global_scope = true;
    vector::EncodeOptions tokenizer_options;
};

/// Pipeline implementation of ISemanticCache.
///
/// Encapsulates the lookup chain described in NEXT_RUNTIME_ROADMAP §5:
///   precheck → tokenize → embed → vector search → metadata filter →
///   policy filter → payload load.
///
/// Construction is intentionally factory-style so wiring failures surface
/// as core::Status rather than exceptions in the ctor.
class SemanticCachePipeline : public ISemanticCache {
public:
    static core::Result<std::unique_ptr<SemanticCachePipeline>> Create(
        SemanticCachePipelineOptions options,
        SemanticCachePipelineDeps deps);

    ~SemanticCachePipeline() override;

    core::Result<CacheLookupResult> Lookup(const CacheLookupRequest& req) override;
    core::Status Store(const CacheStoreRequest& req) override;

private:
    SemanticCachePipeline(SemanticCachePipelineOptions options,
                           SemanticCachePipelineDeps deps);

    SemanticCachePipelineOptions options_;
    SemanticCachePipelineDeps deps_;
};

}  // namespace agent::semantic_cache
