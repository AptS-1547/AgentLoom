#include "l0_memory_cache_adapter.h"

#include <chrono>
#include <sstream>
#include <utility>

namespace agent::semantic_cache {

L0MemoryCacheAdapter::L0MemoryCacheAdapter(
    std::shared_ptr<::vector::EmbeddingPipeline> embedding,
    std::shared_ptr<cache_vector::VectorIndexManager> index,
    L0MemoryCacheAdapterOptions options)
    : embedding_(std::move(embedding)),
      index_(std::move(index)),
      options_(options) {}

core::Result<CacheLookupResult> L0MemoryCacheAdapter::Lookup(const CacheLookupRequest& req) {
    if (!embedding_ || !index_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "L0 memory adapter is not initialized");
    }
    if (req.text.empty()) {
        return CacheLookupResult{};
    }

    std::lock_guard lock(mutex_);
    auto query_embedding = embedding_->Encode(req.text);
    if (!query_embedding.ok()) {
        return query_embedding.status();
    }

    auto hits = index_->SearchWithContext(query_embedding.value(), options_.top_k, options_.neighbors_per_hit);
    if (!hits.ok()) {
        if (hits.status().code() == core::ErrorCode::NotFound) {
            return CacheLookupResult{};
        }
        return hits.status();
    }

    std::ostringstream payload;
    float best_score = 0.0f;
    std::size_t accepted = 0;
    for (const auto& hit : hits.value()) {
        if (hit.score < options_.similarity_floor) {
            continue;
        }
        if (accepted == 0) {
            best_score = hit.score;
        }
        payload << "Score: " << hit.score << "\n"
                << "Q: " << hit.hit.input << "\n"
                << "A: " << hit.hit.response << "\n";
        for (const auto& neighbor : hit.context) {
            if (!neighbor.input.empty() || !neighbor.response.empty()) {
                payload << "Context Q: " << neighbor.input << "\n"
                        << "Context A: " << neighbor.response << "\n";
            }
        }
        ++accepted;
    }

    CacheLookupResult result;
    result.hit = accepted > 0;
    result.similarity_score = best_score;
    result.payload = payload.str();
    result.retrieved_at = std::chrono::system_clock::now();
    return result;
}

core::Status L0MemoryCacheAdapter::Store(const CacheStoreRequest& req) {
    if (!embedding_ || !index_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "L0 memory adapter is not initialized");
    }
    if (req.origin.text.empty() || req.response_payload.empty()) {
        return core::Status::Ok();
    }

    std::lock_guard lock(mutex_);
    auto text_embedding = embedding_->Encode(req.origin.text);
    if (!text_embedding.ok()) {
        return text_embedding.status();
    }

    storage::CacheRecord record;
    record.embedding = std::move(text_embedding).value();
    record.input = req.origin.text;
    record.response = req.response_payload;
    return index_->AddRecord(record);
}

} // namespace agent::semantic_cache
