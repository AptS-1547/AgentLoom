#include "document_llm_chunk_cache.h"

#include <nlohmann/json.hpp>

namespace agent::document {
namespace {

nlohmann::json SliceToJson(const ChunkSlice& slice) {
    return {
        {"title", slice.title},
        {"summary", slice.summary},
        {"text", slice.text},
        {"kind", slice.kind},
        {"confidence", slice.confidence},
    };
}

ChunkSlice SliceFromJson(const nlohmann::json& json) {
    ChunkSlice slice;
    if (!json.is_object()) {
        return slice;
    }
    slice.title = json.value("title", std::string{});
    slice.summary = json.value("summary", std::string{});
    slice.text = json.value("text", std::string{});
    slice.kind = json.value("kind", std::string{"paragraph"});
    slice.confidence = json.value("confidence", 0.5);
    return slice;
}

nlohmann::json ChunkToCacheJson(const ChunkTrunk& chunk) {
    nlohmann::json slices = nlohmann::json::array();
    for (const auto& slice : chunk.slices) {
        slices.push_back(SliceToJson(slice));
    }
    return {
        {"title", chunk.title},
        {"summary", chunk.summary},
        {"slices", slices},
        {"confidence", chunk.confidence},
        {"metadata", chunk.metadata},
    };
}

ChunkTrunk ChunkFromCacheJson(const nlohmann::json& json) {
    ChunkTrunk chunk;
    chunk.title = json.value("title", std::string{});
    chunk.summary = json.value("summary", std::string{});
    chunk.confidence = json.value("confidence", 0.72);
    chunk.source = "llm";
    if (const auto it = json.find("metadata"); it != json.end() && it->is_object()) {
        chunk.metadata = it->get<std::map<std::string, std::string>>();
    }
    if (const auto it = json.find("slices"); it != json.end() && it->is_array()) {
        for (const auto& item : *it) {
            chunk.slices.push_back(SliceFromJson(item));
        }
    }
    return chunk;
}

} // namespace

RedisDocumentLlmChunkCache::RedisDocumentLlmChunkCache(
    std::shared_ptr<semantic_cache::RedisConnectionPool> redis,
    RedisDocumentLlmChunkCacheOptions options)
    : redis_(std::move(redis)),
      options_(std::move(options)) {}

std::string RedisDocumentLlmChunkCache::RedisKey(const DocumentLlmChunkCacheKey& key) const {
    return options_.key_prefix + ":" + key.prompt_version + ":" + key.model + ":" + key.text_hash;
}

core::Result<ChunkTrunk> RedisDocumentLlmChunkCache::Lookup(const DocumentLlmChunkCacheKey& key) {
    if (!redis_ || !redis_->running()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "document llm cache redis is not running");
    }
    auto value = redis_->Get(RedisKey(key));
    if (!value.ok()) {
        return value.status();
    }
    try {
        return ChunkFromCacheJson(nlohmann::json::parse(value.value()));
    } catch (const std::exception& e) {
        return core::Status::Error(core::ErrorCode::InternalError, std::string("document llm cache parse failed: ") + e.what());
    }
}

core::Status RedisDocumentLlmChunkCache::Store(const DocumentLlmChunkCacheKey& key, const ChunkTrunk& chunk) {
    if (!redis_ || !redis_->running()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "document llm cache redis is not running");
    }
    return redis_->Set(
        RedisKey(key),
        ChunkToCacheJson(chunk).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace),
        options_.ttl);
}

} // namespace agent::document
