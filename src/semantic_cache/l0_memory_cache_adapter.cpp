#include "l0_memory_cache_adapter.h"

#include "redis_connection_pool.h"
#include "sqlite/sqlite_connection_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <utility>
#include <vector>

namespace agent::semantic_cache {
namespace {

std::int64_t NowUnixMs() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

bool MatchesOwner(const CacheLookupRequest& req, const storage::CacheRecord& record) {
    if (!req.tenant_id.empty() && record.metadata.tenant_id != req.tenant_id) {
        return false;
    }
    if (!req.user_id.empty() && record.metadata.user_id != req.user_id) {
        return false;
    }
    return true;
}

bool IsExpired(const storage::CacheRecord& record, std::int64_t now_ms) {
    return record.metadata.expires_at_ms > 0 && record.metadata.expires_at_ms <= now_ms;
}

std::int64_t SecondsToMs(std::int64_t seconds) {
    if (seconds <= 0) {
        return 0;
    }
    constexpr std::int64_t kMaxSafeSeconds = std::numeric_limits<std::int64_t>::max() / 1000;
    if (seconds > kMaxSafeSeconds) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return seconds * 1000;
}

float TimeDecay(std::int64_t created_at_ms,
                std::int64_t now_ms,
                const L0MemoryCacheAdapterOptions& options) {
    if (created_at_ms <= 0 || now_ms <= created_at_ms) {
        return 1.0f;
    }

    const auto age_ms = now_ms - created_at_ms;
    const auto warm_ms = SecondsToMs(options.warm_window_seconds);
    const auto half_life_ms = SecondsToMs(options.half_life_seconds);
    const auto max_age_ms = SecondsToMs(options.max_age_seconds);

    if (max_age_ms > 0 && age_ms > max_age_ms) {
        return 0.0f;
    }
    if (age_ms <= warm_ms) {
        return 1.0f;
    }
    if (half_life_ms <= 0) {
        return 0.0f;
    }

    const double periods = static_cast<double>(age_ms - warm_ms) /
                           static_cast<double>(half_life_ms);
    return static_cast<float>(std::pow(0.5, periods));
}

struct RankedL0Hit {
    cache_vector::SearchWithContextResult hit;
    float decayed_score = 0.0f;
    float decay = 1.0f;
};

std::shared_ptr<storage::sqlite::SqliteConnectionPool> MakeL0SqlitePool(
    std::string sqlite_path) {
    if (sqlite_path.empty()) {
        return nullptr;
    }
    storage::sqlite::SqliteConnectionPoolOptions options;
    options.path = std::move(sqlite_path);
    options.read_connection_count = 4;
    options.write_connection_count = 1;
    options.busy_timeout_ms = 5000;
    options.enable_wal = true;
    return std::make_shared<storage::sqlite::SqliteConnectionPool>(std::move(options));
}

} // namespace

L0MemoryCacheAdapter::L0MemoryCacheAdapter(
    std::shared_ptr<::vector::EmbeddingPipeline> embedding,
    std::shared_ptr<cache_vector::VectorIndexManager> index,
    L0MemoryCacheAdapterOptions options)
    : embedding_(std::move(embedding)),
      index_(std::move(index)),
      options_(options) {
    if (index_) {
        fixed_index_entry_ = std::make_shared<SessionIndexEntry>(index_);
    }
}

L0MemoryCacheAdapter::L0MemoryCacheAdapter(
    std::shared_ptr<::vector::EmbeddingPipeline> embedding,
    std::shared_ptr<RedisConnectionPool> redis_pool,
    std::string sqlite_path,
    std::size_t max_cached_records,
    L0MemoryCacheAdapterOptions options)
    : embedding_(std::move(embedding)),
      redis_pool_(std::move(redis_pool)),
      sqlite_pool_(MakeL0SqlitePool(std::move(sqlite_path))),
      max_cached_records_(max_cached_records),
      options_(options) {}

core::Result<std::shared_ptr<L0MemoryCacheAdapter::SessionIndexEntry>> L0MemoryCacheAdapter::ResolveIndex(
    const CacheLookupRequest& req) {
    if (fixed_index_entry_) {
        return fixed_index_entry_;
    }
    if (req.session_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "L0 memory session_id is required");
    }
    if (req.user_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "L0 memory user_id is required");
    }
    if (!redis_pool_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "L0 memory redis pool is not initialized");
    }
    if (!sqlite_pool_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "L0 memory sqlite path is not configured");
    }

    const std::string key(req.session_id);
    {
        std::shared_lock lock(indices_mutex_);
        auto found = per_session_indices_.find(key);
        if (found != per_session_indices_.end()) {
            return found->second;
        }
    }

    if (auto status = sqlite_pool_->Start(); !status.ok()) {
        return status;
    }
    // 索引初始化可能访问 SQLite，放在映射锁之外，避免一个新 Session 阻塞全部热路径。
    auto index = std::make_shared<cache_vector::VectorIndexManager>(
        key,
        redis_pool_,
        sqlite_pool_,
        max_cached_records_);
    auto candidate = std::make_shared<SessionIndexEntry>(std::move(index));

    // 同 Session 并发首次访问允许构造候选对象，但最终只发布一个共享实例。
    std::unique_lock lock(indices_mutex_);
    auto found = per_session_indices_.try_emplace(key, candidate).first;
    return found->second;
}

void L0MemoryCacheAdapter::ReleaseSession(std::string_view session_id) {
    if (session_id.empty()) {
        return;
    }
    std::unique_lock lock(indices_mutex_);
    per_session_indices_.erase(std::string(session_id));
}

core::Result<CacheLookupResult> L0MemoryCacheAdapter::Lookup(const CacheLookupRequest& req) {
    if (!embedding_ || (!index_ && !redis_pool_)) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "L0 memory adapter is not initialized");
    }
    if (req.text.empty()) {
        return CacheLookupResult{};
    }

    auto query_embedding = embedding_->Encode(req.text);
    if (!query_embedding.ok()) {
        return query_embedding.status();
    }
    auto entry = ResolveIndex(req);
    if (!entry.ok()) {
        return entry.status();
    }

    const auto candidate_multiplier = std::max<std::size_t>(1, options_.candidate_multiplier);
    const auto max_top_k = std::numeric_limits<std::size_t>::max() / candidate_multiplier;
    const auto candidate_k = options_.top_k > max_top_k
        ? std::numeric_limits<std::size_t>::max()
        : options_.top_k * candidate_multiplier;
    core::Result<std::vector<cache_vector::SearchWithContextResult>> hits = [&] {
        std::lock_guard lock(entry.value()->operation_mutex);
        return entry.value()->index->SearchWithContext(
            query_embedding.value(), candidate_k, options_.neighbors_per_hit);
    }();
    if (!hits.ok()) {
        if (hits.status().code() == core::ErrorCode::NotFound) {
            return CacheLookupResult{};
        }
        return hits.status();
    }

    const auto now_ms = NowUnixMs();
    std::vector<RankedL0Hit> ranked;
    ranked.reserve(hits.value().size());
    for (auto& hit : hits.value()) {
        if (!MatchesOwner(req, hit.hit) || IsExpired(hit.hit, now_ms)) {
            continue;
        }
        const float decay = TimeDecay(hit.hit.metadata.created_at_ms, now_ms, options_);
        const float decayed_score = hit.score * decay;
        if (decayed_score < options_.similarity_floor) {
            continue;
        }
        std::erase_if(hit.context, [&](const storage::CacheRecord& neighbor) {
            return !MatchesOwner(req, neighbor) || IsExpired(neighbor, now_ms);
        });
        ranked.push_back(RankedL0Hit{std::move(hit), decayed_score, decay});
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.decayed_score > rhs.decayed_score;
    });
    if (ranked.size() > options_.top_k) {
        ranked.resize(options_.top_k);
    }

    std::ostringstream payload;
    float best_score = 0.0f;
    std::size_t accepted = 0;
    for (const auto& ranked_hit : ranked) {
        const auto& hit = ranked_hit.hit;
        if (accepted == 0) {
            best_score = ranked_hit.decayed_score;
        }
        payload << "Score: " << ranked_hit.decayed_score << "\n"
                << "RawScore: " << hit.score << "\n"
                << "TimeDecay: " << ranked_hit.decay << "\n"
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
    if (!embedding_ || (!index_ && !redis_pool_)) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "L0 memory adapter is not initialized");
    }
    if (req.origin.text.empty() || req.response_payload.empty()) {
        return core::Status::Ok();
    }

    auto text_embedding = embedding_->Encode(req.origin.text);
    if (!text_embedding.ok()) {
        return text_embedding.status();
    }
    auto entry = ResolveIndex(req.origin);
    if (!entry.ok()) {
        return entry.status();
    }

    storage::CacheRecord record;
    record.embedding = std::move(text_embedding).value();
    record.input = req.origin.text;
    record.response = req.response_payload;
    record.metadata.scope = req.origin.scope;
    record.metadata.answer_type = req.answer_type;
    record.metadata.tenant_id = req.origin.tenant_id;
    record.metadata.user_id = req.origin.user_id;
    record.metadata.session_id = req.origin.session_id;
    record.metadata.subject = req.origin.subject;
    record.metadata.grade = req.origin.grade;
    record.metadata.topic = req.origin.topic;
    record.metadata.persona_id = req.origin.persona_id;
    record.metadata.quality_score = req.quality_score;
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    record.metadata.created_at_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    if (req.ttl) {
        record.metadata.expires_at_ms = record.metadata.created_at_ms +
            std::chrono::duration_cast<std::chrono::milliseconds>(*req.ttl).count();
    }
    record.extra_metadata = req.origin.extra;
    std::lock_guard lock(entry.value()->operation_mutex);
    return entry.value()->index->AddRecord(record);
}

} // namespace agent::semantic_cache
