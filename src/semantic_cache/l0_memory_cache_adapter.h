#pragma once

#include "isemantic_cache.h"
#include "semantic_cache_pipeline.h"

#include "embedding_pipeline.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace agent::semantic_cache {

class RedisConnectionPool;

struct L0MemoryCacheAdapterOptions {
    std::size_t top_k = 5;
    std::size_t candidate_multiplier = 4;
    std::size_t neighbors_per_hit = 1;
    float similarity_floor = 0.78f;
    std::int64_t warm_window_seconds = 3600;
    std::int64_t half_life_seconds = 172800;
    std::int64_t max_age_seconds = 604800;
};

/// L0 short-context memory adapter.
///
/// This class intentionally implements ISemanticCache only as an integration
/// adapter for SemanticMemoryContextProvider. It is not the answer semantic
/// cache: Store() admits conversation turns into L0, while Lookup() retrieves
/// related short-term dialogue context through cache_vector::VectorIndexManager.
class L0MemoryCacheAdapter final : public ISemanticCache {
public:
    L0MemoryCacheAdapter(std::shared_ptr<::vector::EmbeddingPipeline> embedding,
                         std::shared_ptr<cache_vector::VectorIndexManager> index,
                         L0MemoryCacheAdapterOptions options = {});
    L0MemoryCacheAdapter(std::shared_ptr<::vector::EmbeddingPipeline> embedding,
                         std::shared_ptr<RedisConnectionPool> redis_pool,
                         std::string sqlite_path,
                         std::size_t max_cached_records,
                         L0MemoryCacheAdapterOptions options = {});

    core::Result<CacheLookupResult> Lookup(const CacheLookupRequest& req) override;
    core::Status Store(const CacheStoreRequest& req) override;
    void ReleaseSession(std::string_view session_id);

private:
    struct SessionIndexEntry {
        explicit SessionIndexEntry(std::shared_ptr<cache_vector::VectorIndexManager> value)
            : index(std::move(value)) {}

        std::shared_ptr<cache_vector::VectorIndexManager> index;
        // VectorIndexManager 仍包含批次游标等可变状态；同一 Session 暂时严格保序。
        std::mutex operation_mutex;
    };

    core::Result<std::shared_ptr<SessionIndexEntry>> ResolveIndex(
        const CacheLookupRequest& req);

    std::shared_ptr<::vector::EmbeddingPipeline> embedding_;
    std::shared_ptr<cache_vector::VectorIndexManager> index_;
    std::shared_ptr<RedisConnectionPool> redis_pool_;
    std::shared_ptr<storage::sqlite::SqliteConnectionPool> sqlite_pool_;
    std::size_t max_cached_records_ = 1000;
    std::shared_ptr<SessionIndexEntry> fixed_index_entry_;
    std::unordered_map<std::string, std::shared_ptr<SessionIndexEntry>> per_session_indices_;
    L0MemoryCacheAdapterOptions options_;
    // 仅保护 Session 到索引的映射，严禁在持锁期间执行 Embedding、Redis 或 SQLite I/O。
    mutable std::shared_mutex indices_mutex_;
};

} // namespace agent::semantic_cache
