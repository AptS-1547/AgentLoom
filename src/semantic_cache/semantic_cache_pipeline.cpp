#include "semantic_cache_pipeline.h"
#include "redis_connection_pool.h"
#include "../storage/sqlite/sqlite_statement.h"
#include "../storage/sqlite/sqlite_transaction.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>
#include <sstream>
#include <unordered_map>

namespace agent::semantic_cache {

namespace {
// Per-thread RNG for batch sampling. Seeded once per thread on first use.
// Uniform sampling matches the L0 "unbiased subset of full history" assumption
// better than strict LRU rotation for context-selection workloads.
thread_local std::mt19937 g_batch_rng{std::random_device{}()};
constexpr std::uint32_t kCacheRecordExtensionMagic = 0x43524D45; // CRME
constexpr std::uint32_t kCacheRecordExtensionVersion = 1;

core::Result<storage::sqlite::SqliteConnectionLease> AcquireIndexConnection(
    const std::shared_ptr<storage::sqlite::SqliteConnectionPool>& pool,
    bool write) {
    if (!pool) {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "semantic cache SQLite pool is not configured");
    }
    if (auto status = pool->Start(); !status.ok()) {
        return status;
    }
    return write ? pool->WaitAcquireWrite() : pool->WaitAcquireRead();
}

void AppendPod(std::string& buf, const auto& value) {
    const auto* raw = reinterpret_cast<const char*>(&value);
    buf.append(raw, sizeof(value));
}

core::Status ReadPod(const char*& p, const char* end, auto& value, std::string_view name) {
    if (p + sizeof(value) > end) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "truncated " + std::string(name));
    }
    std::memcpy(&value, p, sizeof(value));
    p += sizeof(value);
    return core::Status::Ok();
}

void AppendString(std::string& buf, const std::string& value) {
    const auto len = static_cast<std::uint32_t>(value.size());
    AppendPod(buf, len);
    buf.append(value);
}

core::Status ReadString(const char*& p, const char* end, std::string& value, std::string_view name) {
    std::uint32_t len = 0;
    auto status = ReadPod(p, end, len, name);
    if (!status.ok()) {
        return status;
    }
    if (p + len > end) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "truncated " + std::string(name));
    }
    value.assign(p, len);
    p += len;
    return core::Status::Ok();
}

void AppendStringMap(std::string& buf, const std::unordered_map<std::string, std::string>& values) {
    const auto count = static_cast<std::uint32_t>(values.size());
    AppendPod(buf, count);
    for (const auto& [key, value] : values) {
        AppendString(buf, key);
        AppendString(buf, value);
    }
}

core::Status ReadStringMap(const char*& p,
                           const char* end,
                           std::unordered_map<std::string, std::string>& values) {
    std::uint32_t count = 0;
    auto status = ReadPod(p, end, count, "metadata map count");
    if (!status.ok()) {
        return status;
    }
    values.clear();
    values.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::string key;
        std::string value;
        status = ReadString(p, end, key, "metadata map key");
        if (!status.ok()) {
            return status;
        }
        status = ReadString(p, end, value, "metadata map value");
        if (!status.ok()) {
            return status;
        }
        values.emplace(std::move(key), std::move(value));
    }
    return core::Status::Ok();
}

core::Result<std::vector<float>> EncodeText(const SemanticCachePipelineOptions& options,
                                            const SemanticCachePipelineDeps& deps,
                                            std::string_view text) {
    if (!deps.tokenizer || !deps.embedding_model) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                   "semantic cache tokenizer and embedding model are required");
    }
    auto tokenized = deps.tokenizer->Encode(text, options.tokenizer_options);
    if (!tokenized.ok()) {
        return tokenized.status();
    }
    auto embedded = deps.embedding_model->Embed(tokenized.value());
    if (!embedded.ok()) {
        return embedded.status();
    }
    auto batch = std::move(embedded).value();
    if (batch.batch_size != 1 || batch.dimension != kExpectedEmbeddingDim ||
        batch.embeddings.size() != kExpectedEmbeddingDim) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "semantic cache embedding shape mismatch");
    }
    return std::move(batch.embeddings);
}

std::string BuildLookupPayload(const storage::CacheRecord& record) {
    std::ostringstream payload;
    payload << record.response;
    return payload.str();
}

std::int64_t NowUnixMs() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

CacheEntryMetadata MetadataFromStoreRequest(const CacheStoreRequest& req) {
    CacheEntryMetadata metadata;
    metadata.scope = req.origin.scope;
    metadata.answer_type = req.answer_type;
    metadata.tenant_id = req.origin.tenant_id;
    metadata.user_id = req.origin.user_id;
    metadata.session_id = req.origin.session_id;
    metadata.subject = req.origin.subject;
    metadata.grade = req.origin.grade;
    metadata.topic = req.origin.topic;
    metadata.persona_id = req.origin.persona_id;
    metadata.quality_score = req.quality_score;
    metadata.created_at_ms = NowUnixMs();
    if (req.ttl) {
        metadata.expires_at_ms = metadata.created_at_ms +
            std::chrono::duration_cast<std::chrono::milliseconds>(*req.ttl).count();
    }
    return metadata;
}
}

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

    AppendPod(buf, kCacheRecordExtensionMagic);
    AppendPod(buf, kCacheRecordExtensionVersion);
    AppendPod(buf, r.metadata.entry_id);
    auto scope = static_cast<std::int32_t>(r.metadata.scope);
    auto answer_type = static_cast<std::int32_t>(r.metadata.answer_type);
    AppendPod(buf, scope);
    AppendPod(buf, answer_type);
    AppendString(buf, r.metadata.tenant_id);
    AppendString(buf, r.metadata.user_id);
    AppendString(buf, r.metadata.session_id);
    AppendString(buf, r.metadata.subject);
    AppendString(buf, r.metadata.grade);
    AppendString(buf, r.metadata.topic);
    AppendString(buf, r.metadata.persona_id);
    AppendPod(buf, r.metadata.quality_score);
    AppendPod(buf, r.metadata.created_at_ms);
    AppendPod(buf, r.metadata.expires_at_ms);
    AppendString(buf, r.metadata.fingerprint.tokenizer_version);
    AppendString(buf, r.metadata.fingerprint.embedding_model_version);
    AppendString(buf, r.metadata.fingerprint.corpus_version);
    AppendString(buf, r.metadata.fingerprint.policy_version);
    AppendPod(buf, r.metadata.fingerprint.embedding_dimension);
    AppendString(buf, r.payload_type);
    AppendString(buf, r.prompt_version);
    AppendString(buf, r.model_version);
    AppendString(buf, r.corpus_version);
    AppendString(buf, r.policy_version);
    AppendStringMap(buf, r.extra_metadata);

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
    p += response_len;

    if (p == end) {
        return r;
    }

    std::uint32_t magic = 0;
    auto status = ReadPod(p, end, magic, "cache record extension magic");
    if (!status.ok()) {
        return status;
    }
    if (magic != kCacheRecordExtensionMagic) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "invalid cache record extension magic");
    }

    std::uint32_t version = 0;
    status = ReadPod(p, end, version, "cache record extension version");
    if (!status.ok()) {
        return status;
    }
    if (version != kCacheRecordExtensionVersion) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "unsupported cache record extension version");
    }

    status = ReadPod(p, end, r.metadata.entry_id, "entry_id");
    if (!status.ok()) return status;
    std::int32_t scope = 0;
    std::int32_t answer_type = 0;
    status = ReadPod(p, end, scope, "scope");
    if (!status.ok()) return status;
    status = ReadPod(p, end, answer_type, "answer_type");
    if (!status.ok()) return status;
    r.metadata.scope = static_cast<CacheScope>(scope);
    r.metadata.answer_type = static_cast<AnswerType>(answer_type);
    if (auto s = ReadString(p, end, r.metadata.tenant_id, "tenant_id"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.metadata.user_id, "user_id"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.metadata.session_id, "session_id"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.metadata.subject, "subject"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.metadata.grade, "grade"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.metadata.topic, "topic"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.metadata.persona_id, "persona_id"); !s.ok()) return s;
    if (auto s = ReadPod(p, end, r.metadata.quality_score, "quality_score"); !s.ok()) return s;
    if (auto s = ReadPod(p, end, r.metadata.created_at_ms, "created_at_ms"); !s.ok()) return s;
    if (auto s = ReadPod(p, end, r.metadata.expires_at_ms, "expires_at_ms"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.metadata.fingerprint.tokenizer_version, "tokenizer_version"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.metadata.fingerprint.embedding_model_version, "embedding_model_version"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.metadata.fingerprint.corpus_version, "corpus_version"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.metadata.fingerprint.policy_version, "policy_version"); !s.ok()) return s;
    if (auto s = ReadPod(p, end, r.metadata.fingerprint.embedding_dimension, "embedding_dimension"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.payload_type, "payload_type"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.prompt_version, "prompt_version"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.model_version, "model_version"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.corpus_version, "corpus_version"); !s.ok()) return s;
    if (auto s = ReadString(p, end, r.policy_version, "policy_version"); !s.ok()) return s;
    if (auto s = ReadStringMap(p, end, r.extra_metadata); !s.ok()) return s;

    return r;
}

// ──────────────────────────────────────────────────────────────────────────
// SemanticCachePipeline (unchanged skeleton)
// ──────────────────────────────────────────────────────────────────────────

core::Result<std::unique_ptr<SemanticCachePipeline>> SemanticCachePipeline::Create(
    SemanticCachePipelineOptions options,
    SemanticCachePipelineDeps deps) {
    if (options.top_k == 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "top_k must be > 0");
    }
    if (options.similarity_floor < -1.0f || options.similarity_floor > 1.0f) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "similarity_floor must be within [-1, 1]");
    }
    if (!deps.risk_detector) {
        deps.risk_detector = std::make_shared<KeywordContextRiskDetector>();
    }
    if (!deps.policy_matcher) {
        deps.policy_matcher = std::make_shared<DefaultPolicyMatcher>();
    }
    if (!deps.index_manager && !deps.repository) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "semantic cache requires an index manager or vector repository");
    }
    auto self = std::unique_ptr<SemanticCachePipeline>(
        new SemanticCachePipeline(std::move(options), std::move(deps)));
    return self;
}

SemanticCachePipeline::SemanticCachePipeline(SemanticCachePipelineOptions options,
                                              SemanticCachePipelineDeps deps)
    : options_(std::move(options)), deps_(std::move(deps)) {}

SemanticCachePipeline::~SemanticCachePipeline() = default;

core::Result<CacheLookupResult> SemanticCachePipeline::Lookup(const CacheLookupRequest& req) {
    if (req.text.empty()) {
        return CacheLookupResult{};
    }
    if (req.scope == CacheScope::Global && !options_.enable_global_scope) {
        return CacheLookupResult{};
    }

    auto risk = deps_.risk_detector->Assess(req);
    if (!risk.ok()) {
        return risk.status();
    }
    if (req.scope == CacheScope::Global && risk.value().blocks_global_cache) {
        return CacheLookupResult{};
    }

    auto embedding = EncodeText(options_, deps_, req.text);
    if (!embedding.ok()) {
        return embedding.status();
    }

    if (deps_.index_manager) {
        auto hits = deps_.index_manager->SearchAllBatches(embedding.value(), options_.top_k);
        if (!hits.ok()) {
            if (hits.status().code() == core::ErrorCode::NotFound) {
                return CacheLookupResult{};
            }
            return hits.status();
        }
        for (const auto& hit : hits.value()) {
            if (hit.score < options_.similarity_floor) {
                continue;
            }
            auto matched = deps_.policy_matcher->Matches(req, hit.record.metadata);
            if (!matched.ok()) {
                return matched.status();
            }
            if (!matched.value()) {
                continue;
            }

            CacheLookupResult result;
            result.hit = true;
            result.similarity_score = hit.score;
            result.payload = BuildLookupPayload(hit.record);
            result.retrieved_at = std::chrono::system_clock::now();
            return result;
        }
        return CacheLookupResult{};
    }

    auto records = deps_.repository->Search(embedding.value(), options_.top_k);
    if (!records.ok()) {
        if (records.status().code() == core::ErrorCode::NotFound) {
            return CacheLookupResult{};
        }
        return records.status();
    }
    for (const auto& record : records.value()) {
        if (record.embedding.size() != kExpectedEmbeddingDim) {
            continue;
        }
        const float score = dot_product_unrolled<kExpectedEmbeddingDim>(embedding.value().data(), record.embedding.data());
        if (score < options_.similarity_floor) {
            continue;
        }
        auto matched = deps_.policy_matcher->Matches(req, record.metadata);
        if (!matched.ok()) {
            return matched.status();
        }
        if (!matched.value()) {
            continue;
        }
        CacheLookupResult result;
        result.hit = true;
        result.similarity_score = score;
        result.payload = BuildLookupPayload(record);
        result.retrieved_at = std::chrono::system_clock::now();
        return result;
    }
    return CacheLookupResult{};
}

core::Status SemanticCachePipeline::Store(const CacheStoreRequest& req) {
    if (req.origin.text.empty() || req.response_payload.empty()) {
        return core::Status::Ok();
    }
    if (req.origin.scope == CacheScope::Global && !options_.enable_global_scope) {
        return core::Status::Ok();
    }

    auto risk = deps_.risk_detector->Assess(req.origin);
    if (!risk.ok()) {
        return risk.status();
    }
    if (req.origin.scope == CacheScope::Global && risk.value().blocks_global_cache) {
        return core::Status::Ok();
    }

    auto embedding = EncodeText(options_, deps_, req.origin.text);
    if (!embedding.ok()) {
        return embedding.status();
    }

    storage::CacheRecord record;
    record.embedding = std::move(embedding).value();
    record.input = req.origin.text;
    record.response = req.response_payload;
    record.metadata = MetadataFromStoreRequest(req);
    record.prompt_version = req.origin.extra.contains("prompt_version") ? req.origin.extra.at("prompt_version") : std::string{};
    record.model_version = req.origin.extra.contains("model_version") ? req.origin.extra.at("model_version") : std::string{};
    record.corpus_version = req.origin.extra.contains("corpus_version") ? req.origin.extra.at("corpus_version") : std::string{};
    record.policy_version = req.origin.extra.contains("policy_version") ? req.origin.extra.at("policy_version") : std::string{};
    record.payload_type = req.origin.extra.contains("payload_type") ? req.origin.extra.at("payload_type") : std::string{};
    record.extra_metadata = req.origin.extra;
    record.metadata.fingerprint.tokenizer_version =
        req.origin.extra.contains("tokenizer_version") ? req.origin.extra.at("tokenizer_version") : std::string{};
    record.metadata.fingerprint.embedding_model_version = record.model_version;
    record.metadata.fingerprint.corpus_version = record.corpus_version;
    record.metadata.fingerprint.policy_version = record.policy_version;
    record.metadata.fingerprint.embedding_dimension = static_cast<std::int32_t>(record.embedding.size());

    if (deps_.index_manager) {
        return deps_.index_manager->AddRecord(record);
    }
    return deps_.repository->Store(std::vector<storage::CacheRecord>{std::move(record)});
}

// ──────────────────────────────────────────────────────────────────────────
// VectorIndexManager implementation
// ──────────────────────────────────────────────────────────────────────────

cache_vector::VectorIndexManager::VectorIndexManager(
    std::string user_uuid,
    std::shared_ptr<RedisConnectionPool> redis_pool,
    std::shared_ptr<storage::sqlite::SqliteConnectionPool> sqlite_pool,
    std::size_t max_cached_records)
    : max_cached_records_(max_cached_records),
      session_key_({.tenant_id = "default", .user_id = user_uuid, .session_id = "legacy"}),
      session_scoped_key_(false),
      user_uuid_(std::move(user_uuid)),
      redis_pool_(std::move(redis_pool)),
      sqlite_pool_(std::move(sqlite_pool)),
      active_timestamp_(0),
      active_count_(0) {
    // 旧构造路径保留 legacy SQLite 表和 Redis key，避免静默丢失已有数据。
    initialization_status_ = EnsureSchema();
    if (initialization_status_.ok()) {
        initialization_status_ = LoadActiveBatch();
    }
    if (initialization_status_.ok()) {
        initialization_status_ = LoadTimestampIndex();
    }
}

cache_vector::VectorIndexManager::VectorIndexManager(
    L0SessionKey session_key,
    std::shared_ptr<RedisConnectionPool> redis_pool,
    std::shared_ptr<IL0SessionBatchMetadataStore> metadata_store,
    std::size_t max_cached_records)
    : max_cached_records_(max_cached_records),
      session_key_(std::move(session_key)),
      session_scoped_key_(true),
      user_uuid_(session_key_.user_id),
      redis_pool_(std::move(redis_pool)),
      metadata_store_(std::move(metadata_store)),
      active_timestamp_(0),
      active_count_(0) {
    initialization_status_ = metadata_store_
        ? core::Status::Ok()
        : core::Status::Error(core::ErrorCode::FailedPrecondition, "L0 metadata store is missing");
    if (initialization_status_.ok()) {
        initialization_status_ = LoadActiveBatch();
    }
    if (initialization_status_.ok()) {
        initialization_status_ = LoadTimestampIndex();
    }
}

core::Status cache_vector::VectorIndexManager::EnsureSchema() {
    if (metadata_store_) {
        return core::Status::Ok();
    }
    {
        auto read_lease_result = AcquireIndexConnection(sqlite_pool_, false);
        if (!read_lease_result.ok()) {
            return read_lease_result.status();
        }
        auto read_lease = std::move(read_lease_result).value();
        auto& read_connection = read_lease.connection();
        auto active_probe = read_connection.Prepare(
            "SELECT user_uuid, timestamp, count FROM active_batch LIMIT 0");
        auto timestamp_probe = read_connection.Prepare(
            "SELECT user_uuid, timestamp FROM cache_timestamp_index LIMIT 0");
        if (active_probe.ok() && timestamp_probe.ok()) {
            // Schema 已就绪时只走 read lease，新 Session 初始化无需进入 SQLite 写队列。
            return core::Status::Ok();
        }
    }

    auto lease_result = AcquireIndexConnection(sqlite_pool_, true);
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    auto lease = std::move(lease_result).value();
    auto& connection = lease.connection();

    auto status = connection.Execute(
        "CREATE TABLE IF NOT EXISTS active_batch ("
        "user_uuid TEXT PRIMARY KEY, "
        "timestamp INTEGER NOT NULL, "
        "count INTEGER NOT NULL)");
    if (!status.ok()) {
        return status;
    }

    status = connection.Execute(
        "CREATE TABLE IF NOT EXISTS cache_timestamp_index ("
        "user_uuid TEXT NOT NULL, "
        "timestamp INTEGER NOT NULL, "
        "PRIMARY KEY (user_uuid, timestamp))");
    if (!status.ok()) {
        return status;
    }

    auto schema_probe = connection.Prepare(
        "SELECT user_uuid, timestamp FROM cache_timestamp_index LIMIT 0");
    if (schema_probe.ok()) {
        return core::Status::Ok();
    }

    // 旧表只有 timestamp 列；索引可由 Redis 批次重建，因此启动时一次性迁移，避免查询热路径执行 DDL。
    status = connection.Execute("DROP TABLE IF EXISTS cache_timestamp_index");
    if (!status.ok()) {
        return status;
    }
    return connection.Execute(
        "CREATE TABLE cache_timestamp_index ("
        "user_uuid TEXT NOT NULL, "
        "timestamp INTEGER NOT NULL, "
        "PRIMARY KEY (user_uuid, timestamp))");
}

std::size_t cache_vector::VectorIndexManager::CurrentSize() const {
    return active_count_ + timestamp_index_.size() * max_cached_records_;
}

std::string cache_vector::VectorIndexManager::BuildBatchKey(std::int64_t timestamp) const {
    if (session_scoped_key_) {
        return "cache:v2:batch:" + session_key_.tenant_id + ":" +
               session_key_.user_id + ":" + session_key_.session_id + ":" +
               std::to_string(timestamp);
    }
    return "cache:batch:" + user_uuid_ + ":" + std::to_string(timestamp);
}

core::Status cache_vector::VectorIndexManager::AddRecord(const storage::CacheRecord& record) {
    if (!initialization_status_.ok()) {
        return initialization_status_;
    }
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
    if (!initialization_status_.ok()) {
        return initialization_status_;
    }
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

    std::int64_t timestamp = 0;
    bool need_rebuild = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (timestamp_index_.empty()) {
            need_rebuild = true;
        }
    }

    // Redis SCAN 和 SQLite repair 不能在持有内存状态锁时执行，否则 repair 写回会重入同一把锁。
    if (need_rebuild) {
        auto rebuild_status = RebuildTimestampIndexFromRedis();
        if (!rebuild_status.ok()) {
            return rebuild_status;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (timestamp_index_.empty()) {
            return core::Status::Error(core::ErrorCode::NotFound, "no batches to reload");
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
        {
            std::lock_guard<std::mutex> lock(mutex_);
            timestamp_index_ = std::move(rebuilt);
            read_cursor_ = 0;
        }
        // 使用锁外快照写回 SQLite，避免 Redis repair 与 metadata writer 互相阻塞。
        static_cast<void>(StoreTimestampIndex());
    }

    return core::Status::Ok();
}

core::Result<std::vector<storage::CacheRecord>> cache_vector::VectorIndexManager::Search(
    const std::vector<float>& embedding,
    std::size_t top_k) {

    if (!initialization_status_.ok()) {
        return initialization_status_;
    }

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

core::Result<std::vector<cache_vector::ScoredCacheRecord>> cache_vector::VectorIndexManager::SearchAllBatches(
    const std::vector<float>& embedding,
    std::size_t top_k) {
    if (!initialization_status_.ok()) {
        return initialization_status_;
    }
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
    if (active_timestamp_ == 0 && timestamp_index_.empty()) {
        status = RebuildTimestampIndexFromRedis();
        if (!status.ok()) {
            return status;
        }
    }

    search_records_.clear();
    if (active_timestamp_ != 0) {
        auto result = redis_pool_->HGetAll(BuildBatchKey(active_timestamp_));
        if (!result.ok()) {
            return result.status();
        }
        for (const auto& [field, serialized] : result.value()) {
            auto record_result = DeserializeCacheRecord(serialized);
            if (record_result.ok()) {
                search_records_.push_back(std::move(record_result.value()));
            }
        }
    }

    for (auto ts : timestamp_index_) {
        auto result = redis_pool_->HGetAll(BuildBatchKey(ts));
        if (!result.ok()) {
            return result.status();
        }
        for (const auto& [field, serialized] : result.value()) {
            auto record_result = DeserializeCacheRecord(serialized);
            if (record_result.ok()) {
                search_records_.push_back(std::move(record_result.value()));
            }
        }
    }

    if (search_records_.empty()) {
        return core::Status::Error(core::ErrorCode::NotFound, "no records loaded");
    }

    std::vector<ScoredCacheRecord> scored;
    scored.reserve(search_records_.size());
    for (const auto& record : search_records_) {
        if (record.embedding.size() != kExpectedEmbeddingDim) {
            continue;
        }
        scored.push_back(ScoredCacheRecord{
            record,
            dot_product_unrolled<kExpectedEmbeddingDim>(embedding.data(), record.embedding.data())
        });
    }
    if (scored.empty()) {
        return core::Status::Error(core::ErrorCode::NotFound, "no valid records loaded");
    }

    const auto n = std::min(top_k, scored.size());
    std::partial_sort(
        scored.begin(),
        scored.begin() + n,
        scored.end(),
        [](const ScoredCacheRecord& lhs, const ScoredCacheRecord& rhs) {
            return lhs.score > rhs.score;
        });
    scored.resize(n);
    return scored;
}

core::Result<std::vector<cache_vector::SearchWithContextResult>> cache_vector::VectorIndexManager::SearchWithContext(
    const std::vector<float>& embedding,
    std::size_t top_k,
    std::size_t neighbors_per_hit) {

    if (!initialization_status_.ok()) {
        return initialization_status_;
    }

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
    if (metadata_store_) {
        auto loaded = metadata_store_->LoadActiveBatch(session_key_);
        if (!loaded.ok()) return loaded.status();
        if (!loaded.value().has_value()) {
            active_timestamp_ = 0;
            active_count_ = 0;
        } else {
            active_timestamp_ = loaded.value()->timestamp;
            active_count_ = loaded.value()->count;
        }
        return core::Status::Ok();
    }
    auto lease_result = AcquireIndexConnection(sqlite_pool_, false);
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    auto lease = std::move(lease_result).value();
    auto& connection = lease.connection();

    auto stmt_result = connection.Prepare(
        "SELECT timestamp, count FROM active_batch WHERE user_uuid = ?");
    if (!stmt_result.ok()) {
        return stmt_result.status();
    }
    auto stmt = std::move(stmt_result.value());

    auto status = stmt.BindText(1, user_uuid_);
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
    if (metadata_store_) {
        return metadata_store_->SaveActiveBatch(
            session_key_,
            L0ActiveBatchState{.timestamp = active_timestamp_, .count = active_count_});
    }
    auto lease_result = AcquireIndexConnection(sqlite_pool_, true);
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    auto lease = std::move(lease_result).value();
    auto& connection = lease.connection();

    auto stmt_result = connection.Prepare(
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

    if (metadata_store_) {
        status = metadata_store_->ClearActiveBatch(session_key_);
        if (!status.ok()) return status;
        active_timestamp_ = 0;
        active_count_ = 0;
        return core::Status::Ok();
    }

    auto lease_result = AcquireIndexConnection(sqlite_pool_, true);
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    auto lease = std::move(lease_result).value();
    auto& connection = lease.connection();
    auto stmt_result = connection.Prepare("DELETE FROM active_batch WHERE user_uuid = ?");
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
    if (metadata_store_) {
        std::vector<std::int64_t> snapshot;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot.assign(timestamp_index_.begin(), timestamp_index_.end());
        }
        return metadata_store_->ReplaceTimestampIndex(session_key_, snapshot);
    }
    auto lease_result = AcquireIndexConnection(sqlite_pool_, true);
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    auto lease = std::move(lease_result).value();
    auto& connection = lease.connection();

    storage::sqlite::SqliteTransaction txn(connection);
    auto status = txn.Begin();
    if (!status.ok()) return status;

    status = connection.Execute(
        "CREATE TABLE IF NOT EXISTS cache_timestamp_index ("
        "user_uuid TEXT NOT NULL, "
        "timestamp INTEGER NOT NULL, "
        "PRIMARY KEY (user_uuid, timestamp))");
    if (!status.ok()) {
        txn.Rollback();
        return status;
    }

    auto delete_result = connection.Prepare(
        "DELETE FROM cache_timestamp_index WHERE user_uuid = ?");
    if (!delete_result.ok()) {
        // Table might not have user_uuid column yet (old schema), recreate
        status = connection.Execute("DROP TABLE IF EXISTS cache_timestamp_index");
        if (!status.ok()) {
            txn.Rollback();
            return status;
        }
        status = connection.Execute(
            "CREATE TABLE cache_timestamp_index ("
            "user_uuid TEXT NOT NULL, "
            "timestamp INTEGER NOT NULL, "
            "PRIMARY KEY (user_uuid, timestamp))");
        if (!status.ok()) {
            txn.Rollback();
            return status;
        }
    } else {
        auto delete_statement = std::move(delete_result).value();
        if (auto bind = delete_statement.BindText(1, user_uuid_); !bind.ok()) {
            txn.Rollback();
            return bind;
        }
        auto deleted = delete_statement.Step();
        if (!deleted.ok()) {
            txn.Rollback();
            return deleted.status();
        }
    }

    auto stmt_result = connection.Prepare("INSERT OR IGNORE INTO cache_timestamp_index (user_uuid, timestamp) VALUES (?, ?)");
    if (!stmt_result.ok()) {
        txn.Rollback();
        return stmt_result.status();
    }
    auto stmt = std::move(stmt_result.value());

    std::vector<std::int64_t> timestamp_snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        timestamp_snapshot.assign(timestamp_index_.begin(), timestamp_index_.end());
    }
    for (auto ts : timestamp_snapshot) {
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
    if (metadata_store_) {
        auto loaded = metadata_store_->LoadTimestampIndex(session_key_);
        if (!loaded.ok()) return loaded.status();
        std::lock_guard<std::mutex> lock(mutex_);
        timestamp_index_.assign(loaded.value().begin(), loaded.value().end());
        backpack_timestamps_.assign(loaded.value().begin(), loaded.value().end());
        read_cursor_ = 0;
        return core::Status::Ok();
    }
    auto lease_result = AcquireIndexConnection(sqlite_pool_, false);
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    auto lease = std::move(lease_result).value();
    auto& connection = lease.connection();

    auto stmt_result = connection.Prepare(
        "SELECT timestamp FROM cache_timestamp_index WHERE user_uuid = ? ORDER BY timestamp");
    if (!stmt_result.ok()) {
        return stmt_result.status();
    }
    auto stmt = std::move(stmt_result.value());

    auto status = stmt.BindText(1, user_uuid_);
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

}
