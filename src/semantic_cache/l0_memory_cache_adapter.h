#pragma once

#include "isemantic_cache.h"
#include "semantic_cache_pipeline.h"

#include "embedding_pipeline.h"

#include <cstddef>
#include <memory>
#include <mutex>

namespace agent::semantic_cache {

struct L0MemoryCacheAdapterOptions {
    std::size_t top_k = 5;
    std::size_t neighbors_per_hit = 1;
    float similarity_floor = 0.78f;
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

    core::Result<CacheLookupResult> Lookup(const CacheLookupRequest& req) override;
    core::Status Store(const CacheStoreRequest& req) override;

private:
    std::shared_ptr<::vector::EmbeddingPipeline> embedding_;
    std::shared_ptr<cache_vector::VectorIndexManager> index_;
    L0MemoryCacheAdapterOptions options_;
    mutable std::mutex mutex_;
};

} // namespace agent::semantic_cache
