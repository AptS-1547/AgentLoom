#pragma once

#include "document_types.h"
#include "redis_connection_pool.h"

#include <chrono>
#include <memory>
#include <string>

namespace agent::document {

struct RedisDocumentLlmChunkCacheOptions {
    std::string key_prefix = "agent:document:llm_chunk";
    std::chrono::seconds ttl{7 * 24 * 60 * 60};
};

class RedisDocumentLlmChunkCache final : public IDocumentLlmChunkCache {
public:
    RedisDocumentLlmChunkCache(std::shared_ptr<semantic_cache::RedisConnectionPool> redis,
                               RedisDocumentLlmChunkCacheOptions options = {});

    core::Result<ChunkTrunk> Lookup(const DocumentLlmChunkCacheKey& key) override;
    core::Status Store(const DocumentLlmChunkCacheKey& key, const ChunkTrunk& chunk) override;

private:
    std::string RedisKey(const DocumentLlmChunkCacheKey& key) const;

    std::shared_ptr<semantic_cache::RedisConnectionPool> redis_;
    RedisDocumentLlmChunkCacheOptions options_;
};

} // namespace agent::document
