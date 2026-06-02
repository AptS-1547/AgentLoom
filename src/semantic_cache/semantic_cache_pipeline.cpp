#include "semantic_cache_pipeline.h"
#include "redis_connection_pool.h"
#include "../storage/sqlite/sqlite_statement.h"
#include "../storage/sqlite/sqlite_transaction.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>

namespace agent::semantic_cache {

namespace {
// Per-thread RNG for batch sampling. Seeded once per thread on first use.
// Uniform sampling matches the L0 "unbiased subset of full history" assumption
// better than strict LRU rotation for context-selection workloads.
thread_local std::mt19937 g_batch_rng{std::random_device{}()};
}  // namespace

// ──────────────────────────────────────────────────────────────────────────
// CacheRecord binary serialization
// ──────────────────────────────────────────────────────────────────────────

std::string SerializeCacheRecord(const storage::CacheRecord& r) {
    std::uint32_t dim = static_cast<std::uint32_t>(r.embedding.size());
    std::uint32_t input_len = static_cast<std::uint32_t>(r.input.size());
    std::uint32_t response_len = static_cast<std::uint32_t>(r.response.size());

    std::string buf;
    buf.resize(4 + dim * 4 + 4 + input_len + 4 + response_len);

    char* p = buf.data();
    std::memcpy(p, &dim, 4);              p += 4;
    std::memcpy(p, r.embedding.data(), dim * 4); p += dim * 4;
    std::memcpy(p, &input_len, 4);        p += 4;
    std::memcpy(p, r.input.data(), input_len);   p += input_len;
    std::memcpy(p, &response_len, 4);     p += 4;
    std::memcpy(p, r.response.data(), response_len);

    return buf;
}

core::Result<storage::CacheRecord> DeserializeCacheRecord(std::string_view buf) {
    const char* p = buf.data();
    const char* end = p + buf.size();

    if (buf.size() < 12) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "buffer too small");
    }

    std::uint32_t dim;
    std::memcpy(&dim, p, 4); p += 4;

    if (dim != kExpectedEmbeddingDim) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
            "dimension mismatch: expected " + std::to_string(kExpectedEmbeddingDim)
            + " got " + std::to_string(dim));
    }
    if (p + dim * 4 > end) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "truncated embedding");
    }

    storage::CacheRecord r;
    r.embedding.resize(dim);
    std::memcpy(r.embedding.data(), p, dim * 4); p += dim * 4;

    if (p + 4 > end) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "truncated input_len");
    }
    std::uint32_t input_len;
    std::memcpy(&input_len, p, 4); p += 4;
    if (p + input_len > end) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "truncated input");
    }
    r.input.assign(p, input_len); p += input_len;

    if (p + 4 > end) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "truncated response_len");
    }
    std::uint32_t response_len;
    std::memcpy(&response_len, p, 4); p += 4;
    if (p + response_len > end) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "truncated response");
    }
    r.response.assign(p, response_len);

    return r;
}

// ──────────────────────────────────────────────────────────────────────────
// SemanticCachePipeline (unchanged skeleton)
// ──────────────────────────────────────────────────────────────────────────

core::Result<std::unique_ptr<SemanticCachePipeline>> SemanticCachePipeline::Create(
    SemanticCachePipelineOptions options,
    SemanticCachePipelineDeps deps) {
    // TODO(orange): validate deps (all required handles non-null) and
    // return InvalidArgument if anything mandatory is missing.
    auto self = std::unique_ptr<SemanticCachePipeline>(
        new SemanticCachePipeline(std::move(options), std::move(deps)));
    return self;
}

SemanticCachePipeline::SemanticCachePipeline(SemanticCachePipelineOptions options,
                                              SemanticCachePipelineDeps deps)
    : options_(std::move(options)), deps_(std::move(deps)) {}

SemanticCachePipeline::~SemanticCachePipeline() = default;

core::Result<CacheLookupResult> SemanticCachePipeline::Lookup(const CacheLookupRequest& req) {
    // TODO(orange): implement.
    return CacheLookupResult{};  // miss
}

core::Status SemanticCachePipeline::Store(const CacheStoreRequest& req) {
    // TODO(orange): implement.
    return core::Status::Ok();
}

// ──────────────────────────────────────────────────────────────────────────
// VectorIndexManager implementation
// ──────────────────────────────────────────────────────────────────────────

cache_vector::VectorIndexManager::VectorIndexManager(
    std::string user_uuid,
    std::shared_ptr<RedisConnectionPool> redis_pool,
    storage::sqlite::SqliteConnection sqlite_conn,
    std::size_t max_cached_records)
    : user_uuid_(std::move(user_uuid)),
      redis_pool_(std::move(redis_pool)),
      sqlite_conn_(std::move(sqlite_conn)),
      active_timestamp_(0),
      active_count_(0),
      max_cached_records_(max_cached_records) {
    LoadActiveBatch();
    LoadTimestampIndex();
}

std::size_t cache_vector::VectorIndexManager::CurrentSize() const {
    return active_count_ + timestamp_index_.size() * max_cached_records_;
}

std::string cache_vector::VectorIndexManager::BuildBatchKey(std::int64_t timestamp) const {
    return "cache:batch:" + user_uuid_ + ":" + std::to_string(timestamp);
}

core::Status cache_vector::VectorIndexManager::AddRecord(const storage::CacheRecord& record) {
    if (active_timestamp_ == 0) {
        active_timestamp_ = std::chrono::system_clock::now().time_since_epoch().count();
    }

    std::string key = BuildBatchKey(active_timestamp_);
    std::string serialized = SerializeCacheRecord(record);
    std::string field = std::to_string(active_count_);

    auto result = redis_pool_->HSet(key, field, serialized);
    if (!result.ok()) {
        return result.status();
    }

    active_count_ += 1;
    auto status = SaveActiveBatch();
    if (!status.ok()) {
        return status;
    }

    if (active_count_ >= max_cached_records_) {
        return PromoteBatchToIndex();
    }

    return core::Status::Ok();
}

core::Status cache_vector::VectorIndexManager::Store(const std::vector<storage::CacheRecord>& records) {
    if (records.empty()) {
        return core::Status::Ok();
    }

    if (active_timestamp_ == 0) {
        active_timestamp_ = std::chrono::system_clock::now().time_since_epoch().count();
    }

    std::string key = BuildBatchKey(active_timestamp_);
    std::unordered_map<std::string, std::string> field_values;

    for (size_t i = 0; i < records.size(); ++i) {
        std::string serialized = SerializeCacheRecord(records[i]);
        field_values[std::to_string(active_count_ + i)] = serialized;
    }

    auto result = redis_pool_->HMSet(key, field_values);
    if (!result.ok()) {
        return result.status();
    }

    active_count_ += records.size();
    auto status = SaveActiveBatch();
    if (!status.ok()) {
        return status;
    }

    if (active_count_ >= max_cached_records_) {
        return PromoteBatchToIndex();
    }

    return core::Status::Ok();
}

core::Status cache_vector::VectorIndexManager::ReloadNextBatch() {

    if (!redis_pool_ || !redis_pool_->running()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "redis pool not available");
    }

    std::int64_t timestamp;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (timestamp_index_.empty()) {
            // Fallback: rebuild from Redis SCAN
            auto rebuild_status = RebuildTimestampIndexFromRedis();
            if (!rebuild_status.ok() || timestamp_index_.empty()) {
                return core::Status::Error(core::ErrorCode::NotFound, "no batches to reload");
            }
            read_cursor_ = 0;
        }

        // All batches have been read this round — reset cursor for next round
        if (read_cursor_ >= timestamp_index_.size()) {
            read_cursor_ = 0;
        }

        // Pick a random unread batch from [read_cursor_, end), swap to cursor position
        std::uniform_int_distribution<std::size_t> dist(read_cursor_, timestamp_index_.size() - 1);
        auto idx = dist(g_batch_rng);
        std::swap(timestamp_index_[read_cursor_], timestamp_index_[idx]);
        timestamp = timestamp_index_[read_cursor_];
        ++read_cursor_;
    }

    std::string key = BuildBatchKey(timestamp);

    auto result = redis_pool_->HGetAll(key);
    if (!result.ok()) {
        return result.status();
    }

    auto& field_values = result.value();
    search_records_.clear();
    search_records_.reserve(field_values.size());

    for (const auto& [field, serialized] : field_values) {
        auto record_result = DeserializeCacheRecord(serialized);
        if (!record_result.ok()) {
            return record_result.status();
        }
        search_records_.push_back(std::move(record_result.value()));
    }

    return core::Status::Ok();
}

core::Status cache_vector::VectorIndexManager::RebuildTimestampIndexFromRedis() {
    if (!redis_pool_ || !redis_pool_->running()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "redis pool not available");
    }

    std::string pattern = "cache:batch:" + user_uuid_ + ":*";
    std::deque<std::int64_t> rebuilt;

    auto result = redis_pool_->Scan(pattern);
    if (!result.ok()) {
        return result.status();
    }

    for (const auto& key : result.value()) {
        auto last_colon = key.rfind(':');
        if (last_colon != std::string::npos && last_colon + 1 < key.size()) {
            try {
                auto ts = std::stoll(key.substr(last_colon + 1));
                if (ts != active_timestamp_) {
                    rebuilt.push_back(ts);
                }
            } catch (...) {
            }
        }
    }

    if (!rebuilt.empty()) {
        timestamp_index_ = std::move(rebuilt);
        read_cursor_ = 0;
        StoreTimestampIndex();
    }

    return core::Status::Ok();
}

core::Result<std::vector<storage::CacheRecord>> cache_vector::VectorIndexManager::Search(
    const std::vector<float>& embedding,
    std::size_t top_k) {

    if (embedding.size() != kExpectedEmbeddingDim) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
            "embedding dimension mismatch: expected " + std::to_string(kExpectedEmbeddingDim)
            + " got " + std::to_string(embedding.size()));
    }

    if (top_k == 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "top_k must be > 0");
    }

    auto status = LoadActiveBatch();
    if (!status.ok()) {
        return status;
    }

    status = LoadTimestampIndex();
    if (!status.ok()) {
        return status;
    }

    // If both active batch and index are empty, try rebuilding from Redis
    if (active_timestamp_ == 0 && timestamp_index_.empty()) {
        status = RebuildTimestampIndexFromRedis();
        if (!status.ok()) {
            return status;
        }
    }

    std::size_t total = active_count_ + timestamp_index_.size() * max_cached_records_;
    if (total == 0) {
        return core::Status::Error(core::ErrorCode::NotFound, "no records in index");
    }

    search_records_.clear();

    if (active_timestamp_ != 0) {
        std::string key = BuildBatchKey(active_timestamp_);
        auto result = redis_pool_->HGetAll(key);
        if (result.ok()) {
            const auto& field_values = result.value();
            search_records_.reserve(field_values.size());
            for (const auto& [field, serialized] : field_values) {
                auto record_result = DeserializeCacheRecord(serialized);
                if (record_result.ok()) {
                    search_records_.push_back(std::move(record_result.value()));
                }
            }
        }
    }

    if (total < 2 * max_cached_records_) {
        for (auto ts : timestamp_index_) {
            std::string key = BuildBatchKey(ts);
            auto result = redis_pool_->HGetAll(key);
            if (result.ok()) {
                const auto& field_values = result.value();
                for (const auto& [field, serialized] : field_values) {
                    auto record_result = DeserializeCacheRecord(serialized);
                    if (record_result.ok()) {
                        search_records_.push_back(std::move(record_result.value()));
                    }
                }
            }
        }
    } else {
        ReloadNextBatch();
    }

    if (search_records_.empty()) {
        return core::Status::Error(core::ErrorCode::NotFound, "no records loaded");
    }

    std::size_t n = std::min(top_k, search_records_.size());

    std::partial_sort(
        search_records_.begin(),
        search_records_.begin() + n,
        search_records_.end(),
        [&embedding](const storage::CacheRecord& a, const storage::CacheRecord& b) {
            float sim_a = dot_product_unrolled<kExpectedEmbeddingDim>(embedding.data(), a.embedding.data());
            float sim_b = dot_product_unrolled<kExpectedEmbeddingDim>(embedding.data(), b.embedding.data());
            return sim_a > sim_b;
        });

    std::vector<storage::CacheRecord> result;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        result.push_back(search_records_[i]);
    }

    return result;
}

core::Result<std::vector<cache_vector::SearchWithContextResult>> cache_vector::VectorIndexManager::SearchWithContext(
    const std::vector<float>& embedding,
    std::size_t top_k,
    std::size_t neighbors_per_hit) {

    if (embedding.size() != kExpectedEmbeddingDim) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
            "embedding dimension mismatch: expected " + std::to_string(kExpectedEmbeddingDim)
            + " got " + std::to_string(embedding.size()));
    }

    if (top_k == 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "top_k must be > 0");
    }

    auto status = LoadActiveBatch();
    if (!status.ok()) {
        return status;
    }

    status = LoadTimestampIndex();
    if (!status.ok()) {
        return status;
    }

    std::size_t total = active_count_ + timestamp_index_.size() * max_cached_records_;
    if (total == 0) {
        return core::Status::Error(core::ErrorCode::NotFound, "no records in index");
    }

    search_records_.clear();

    if (active_timestamp_ != 0) {
        std::string key = BuildBatchKey(active_timestamp_);
        auto result = redis_pool_->HGetAll(key);
        if (result.ok()) {
            const auto& field_values = result.value();
            search_records_.reserve(field_values.size());
            for (const auto& [field, serialized] : field_values) {
                auto record_result = DeserializeCacheRecord(serialized);
                if (record_result.ok()) {
                    search_records_.push_back(std::move(record_result.value()));
                }
            }
        }
    }

    const std::size_t active_end = search_records_.size();

    if (total < 2 * max_cached_records_) {
        for (auto ts : timestamp_index_) {
            std::string key = BuildBatchKey(ts);
            auto result = redis_pool_->HGetAll(key);
            if (result.ok()) {
                const auto& field_values = result.value();
                for (const auto& [field, serialized] : field_values) {
                    auto record_result = DeserializeCacheRecord(serialized);
                    if (record_result.ok()) {
                        search_records_.push_back(std::move(record_result.value()));
                    }
                }
            }
        }
    } else {
        ReloadNextBatch();
    }

    if (search_records_.empty()) {
        return core::Status::Error(core::ErrorCode::NotFound, "no records loaded");
    }

    struct IndexedRecord {
        std::size_t index;
        float score;
    };

    std::vector<IndexedRecord> indexed;
    indexed.reserve(search_records_.size());
    for (std::size_t i = 0; i < search_records_.size(); ++i) {
        float score = dot_product_unrolled<kExpectedEmbeddingDim>(
            embedding.data(), search_records_[i].embedding.data());
        indexed.push_back({i, score});
    }

    std::size_t n = std::min(top_k, indexed.size());

    std::partial_sort(
        indexed.begin(),
        indexed.begin() + n,
        indexed.end(),
        [](const IndexedRecord& a, const IndexedRecord& b) {
            return a.score > b.score;
        });

    std::vector<SearchWithContextResult> results;
    results.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        SearchWithContextResult res;
        res.hit = search_records_[indexed[i].index];
        res.score = indexed[i].score;

        std::size_t idx = indexed[i].index;
        std::size_t seg_start, seg_end;
        if (idx < active_end) {
            seg_start = 0;
            seg_end = active_end;
        } else {
            seg_start = active_end;
            seg_end = search_records_.size();
        }

        std::size_t half = neighbors_per_hit / 2;
        std::size_t start = (idx >= seg_start + half) ? (idx - half) : seg_start;
        std::size_t end = std::min(idx + half + 1, seg_end);

        res.context.reserve(end - start);
        for (std::size_t j = start; j < end; ++j) {
            if (j != idx) {
                res.context.push_back(search_records_[j]);
            }
        }

        results.push_back(std::move(res));
    }

    return results;
}

core::Status cache_vector::VectorIndexManager::LoadActiveBatch() {
    if (!sqlite_conn_.valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "sqlite connection not available");
    }

    auto status = sqlite_conn_.Execute(
        "CREATE TABLE IF NOT EXISTS active_batch ("
        "user_uuid TEXT PRIMARY KEY, "
        "timestamp INTEGER NOT NULL, "
        "count INTEGER NOT NULL)");
    if (!status.ok()) {
        return status;
    }

    auto stmt_result = sqlite_conn_.Prepare(
        "SELECT timestamp, count FROM active_batch WHERE user_uuid = ?");
    if (!stmt_result.ok()) {
        return stmt_result.status();
    }
    auto stmt = std::move(stmt_result.value());

    status = stmt.BindText(1, user_uuid_);
    if (!status.ok()) {
        return status;
    }

    auto step_result = stmt.Step();
    if (!step_result.ok()) {
        return step_result.status();
    }

    if (step_result.value() == storage::sqlite::SqliteStepResult::Row) {
        active_timestamp_ = stmt.ColumnInt64(0);
        active_count_ = static_cast<std::size_t>(stmt.ColumnInt64(1));
    } else {
        active_timestamp_ = 0;
        active_count_ = 0;
    }

    return core::Status::Ok();
}

core::Status cache_vector::VectorIndexManager::SaveActiveBatch() {
    if (!sqlite_conn_.valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "sqlite connection not available");
    }

    auto stmt_result = sqlite_conn_.Prepare(
        "INSERT OR REPLACE INTO active_batch (user_uuid, timestamp, count) VALUES (?, ?, ?)");
    if (!stmt_result.ok()) {
        return stmt_result.status();
    }
    auto stmt = std::move(stmt_result.value());

    auto status = stmt.BindText(1, user_uuid_);
    if (!status.ok()) {
        return status;
    }

    status = stmt.BindInt64(2, active_timestamp_);
    if (!status.ok()) {
        return status;
    }

    status = stmt.BindInt64(3, static_cast<std::int64_t>(active_count_));
    if (!status.ok()) {
        return status;
    }

    auto step_result = stmt.Step();
    if (!step_result.ok()) {
        return step_result.status();
    }

    return core::Status::Ok();
}

core::Status cache_vector::VectorIndexManager::PromoteBatchToIndex() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        timestamp_index_.push_back(active_timestamp_);
        backpack_timestamps_.push_back(active_timestamp_);
    }

    auto status = StoreTimestampIndex();
    if (!status.ok()) {
        return status;
    }

    auto stmt_result = sqlite_conn_.Prepare("DELETE FROM active_batch WHERE user_uuid = ?");
    if (!stmt_result.ok()) {
        return stmt_result.status();
    }
    auto stmt = std::move(stmt_result.value());

    status = stmt.BindText(1, user_uuid_);
    if (!status.ok()) {
        return status;
    }

    auto step_result = stmt.Step();
    if (!step_result.ok()) {
        return step_result.status();
    }

    active_timestamp_ = 0;
    active_count_ = 0;

    return core::Status::Ok();
}

core::Status cache_vector::VectorIndexManager::StoreTimestampIndex() {
    if (!sqlite_conn_.valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "sqlite connection not available");
    }

    storage::sqlite::SqliteTransaction txn(sqlite_conn_);
    auto status = txn.Begin();
    if (!status.ok()) return status;

    status = sqlite_conn_.Execute(
        "CREATE TABLE IF NOT EXISTS cache_timestamp_index ("
        "user_uuid TEXT NOT NULL, "
        "timestamp INTEGER NOT NULL, "
        "PRIMARY KEY (user_uuid, timestamp))");
    if (!status.ok()) {
        txn.Rollback();
        return status;
    }

    status = sqlite_conn_.Execute("DELETE FROM cache_timestamp_index WHERE user_uuid = ?");
    if (!status.ok()) {
        // Table might not have user_uuid column yet (old schema), recreate
        status = sqlite_conn_.Execute("DROP TABLE IF EXISTS cache_timestamp_index");
        if (!status.ok()) {
            txn.Rollback();
            return status;
        }
        status = sqlite_conn_.Execute(
            "CREATE TABLE cache_timestamp_index ("
            "user_uuid TEXT NOT NULL, "
            "timestamp INTEGER NOT NULL, "
            "PRIMARY KEY (user_uuid, timestamp))");
        if (!status.ok()) {
            txn.Rollback();
            return status;
        }
    }

    auto stmt_result = sqlite_conn_.Prepare("INSERT OR IGNORE INTO cache_timestamp_index (user_uuid, timestamp) VALUES (?, ?)");
    if (!stmt_result.ok()) {
        txn.Rollback();
        return stmt_result.status();
    }
    auto stmt = std::move(stmt_result.value());

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto ts : timestamp_index_) {
        status = stmt.BindText(1, user_uuid_);
        if (!status.ok()) {
            txn.Rollback();
            return status;
        }
        status = stmt.BindInt64(2, ts);
        if (!status.ok()) {
            txn.Rollback();
            return status;
        }
        auto step_result = stmt.Step();
        if (!step_result.ok()) {
            txn.Rollback();
            return step_result.status();
        }
        status = stmt.Reset();
        if (!status.ok()) {
            txn.Rollback();
            return status;
        }
    }

    return txn.Commit();
}

core::Status cache_vector::VectorIndexManager::LoadTimestampIndex() {
    if (!sqlite_conn_.valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "sqlite connection not available");
    }

    auto status = sqlite_conn_.Execute(
        "CREATE TABLE IF NOT EXISTS cache_timestamp_index ("
        "user_uuid TEXT NOT NULL, "
        "timestamp INTEGER NOT NULL, "
        "PRIMARY KEY (user_uuid, timestamp))");
    if (!status.ok()) return status;

    auto stmt_result = sqlite_conn_.Prepare(
        "SELECT timestamp FROM cache_timestamp_index WHERE user_uuid = ? ORDER BY timestamp");
    if (!stmt_result.ok()) {
        return stmt_result.status();
    }
    auto stmt = std::move(stmt_result.value());

    status = stmt.BindText(1, user_uuid_);
    if (!status.ok()) {
        return status;
    }

    std::deque<std::int64_t> loaded;
    while (true) {
        auto step_result = stmt.Step();
        if (!step_result.ok()) {
            return step_result.status();
        }
        if (step_result.value() == storage::sqlite::SqliteStepResult::Done) {
            break;
        }
        loaded.push_back(stmt.ColumnInt64(0));
    }

    std::lock_guard<std::mutex> lock(mutex_);
    timestamp_index_ = std::move(loaded);

    return core::Status::Ok();
}

}  // namespace agent::semantic_cache
