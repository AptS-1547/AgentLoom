#include "long_term_memory_compressor.h"
#include "../core/logger_adapter.h"
#include "../semantic_cache/semantic_cache_pipeline.h"
#include "../storage/sqlite/sqlite_statement.h"
#include "../storage/sqlite/sqlite_migration.h"

#include <nlohmann/json.hpp>
#include <chrono>
#include <sstream>
#include <ctime>
#include <map>
#include <array>

using json = nlohmann::json;

namespace agent::memory {

namespace {

core::Status ApplyL3RegistryV1(storage::sqlite::SqliteConnection& connection) {
    return connection.Execute(
        "CREATE TABLE IF NOT EXISTS l3_user_registry ("
        "user_uuid TEXT PRIMARY KEY NOT NULL, "
        "first_seen_at_ms INTEGER NOT NULL, "
        "last_seen_at_ms INTEGER NOT NULL)");
}

const std::array<storage::sqlite::SqliteMigrationStep, 1> kL3RegistryMigrations{{
    {
        .version = 1,
        .name = "create_l3_user_registry",
        .checksum = "long_term_memory_v1_l3_user_registry",
        .apply = ApplyL3RegistryV1,
    },
}};

class L3RegistryMigrationSource final
    : public storage::sqlite::ISqliteMigrationSource {
public:
    std::string_view MigrationNamespace() const noexcept override {
        return "long_term_memory";
    }

    std::span<const storage::sqlite::SqliteMigrationStep>
    MigrationSteps() const noexcept override {
        return kL3RegistryMigrations;
    }
};

L3RegistryMigrationSource kL3RegistryMigrationSource;

}

static const char* FACT_EXTRACTION_PROMPT = R"(从以下会话状态记录中抽取用户的长期偏好、重要特征、关键决策。
只输出 JSON 数组，每条字符串不超过 50 字，不要任何解释。
格式：["事实1", "事实2", ...])";

LongTermMemoryCompressor::LongTermMemoryCompressor(LongTermMemoryCompressorOptions options)
    : options_(std::move(options)), logger_(core::LoggerAdapter::ForModule("memory")) {}

LongTermMemoryCompressor::~LongTermMemoryCompressor() = default;

core::Result<std::unique_ptr<LongTermMemoryCompressor>> LongTermMemoryCompressor::Create(
    LongTermMemoryCompressorOptions options) {

    if (!options.redis_pool) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "redis_pool is required");
    }
    if (!options.vector_repo) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "vector_repo is required");
    }
    if (!options.partition_registry) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "partition_registry is required");
    }
    if (options.mode == CompressorMode::Production) {
        if (!options.llm_client) {
            return core::Status::Error(core::ErrorCode::InvalidArgument,
                "llm_client is required in Production mode");
        }
        if (!options.embedding_pipeline) {
            return core::Status::Error(core::ErrorCode::InvalidArgument,
                "embedding_pipeline is required in Production mode");
        }
        if (!options.index_manager) {
            return core::Status::Error(core::ErrorCode::InvalidArgument,
                "index_manager is required in Production mode");
        }
    }

    auto compressor = std::unique_ptr<LongTermMemoryCompressor>(
        new LongTermMemoryCompressor(std::move(options)));
    auto registry_status = compressor->EnsureRegistrySchema();
    if (!registry_status.ok()) {
        return registry_status;
    }

    return compressor;
}

core::Status LongTermMemoryCompressor::EnsureRegistrySchema() const {
    if (!options_.registry_pool) {
        return core::Status::Ok();
    }
    storage::sqlite::SqliteMigrationRunner runner(options_.registry_pool);
    std::array<storage::sqlite::ISqliteMigrationSource*, 1> sources{
        &kL3RegistryMigrationSource};
    return runner.ApplyAll(sources);
}

core::Status LongTermMemoryCompressor::RegisterUser(const std::string& user_uuid) const {
    if (user_uuid.empty() || !options_.registry_pool) {
        return core::Status::Ok();
    }
    auto lease_r = options_.registry_pool->AcquireWrite();
    if (!lease_r.ok()) {
        return lease_r.status();
    }
    auto lease = std::move(lease_r).value();
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    auto stmt_r = lease->Prepare(
        "INSERT INTO l3_user_registry (user_uuid, first_seen_at_ms, last_seen_at_ms) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(user_uuid) DO UPDATE SET last_seen_at_ms = excluded.last_seen_at_ms");
    if (!stmt_r.ok()) {
        return stmt_r.status();
    }
    auto stmt = std::move(stmt_r).value();
    auto status = stmt.BindText(1, user_uuid);
    if (!status.ok()) {
        return status;
    }
    status = stmt.BindInt64(2, now_ms);
    if (!status.ok()) {
        return status;
    }
    status = stmt.BindInt64(3, now_ms);
    if (!status.ok()) {
        return status;
    }
    auto step_r = stmt.Step();
    if (!step_r.ok()) {
        return step_r.status();
    }
    return core::Status::Ok();
}

core::Result<std::vector<std::string>> LongTermMemoryCompressor::GetRegisteredUsers() {
    if (!options_.registry_pool) {
        return std::vector<std::string>{};
    }
    auto schema_status = EnsureRegistrySchema();
    if (!schema_status.ok()) {
        return schema_status;
    }
    auto lease_r = options_.registry_pool->AcquireRead();
    if (!lease_r.ok()) {
        return lease_r.status();
    }
    auto lease = std::move(lease_r).value();
    auto stmt_r = lease->Prepare(
        "SELECT user_uuid FROM l3_user_registry ORDER BY last_seen_at_ms DESC, user_uuid ASC");
    if (!stmt_r.ok()) {
        return stmt_r.status();
    }
    auto stmt = std::move(stmt_r).value();
    std::vector<std::string> users;
    while (true) {
        auto step_r = stmt.Step();
        if (!step_r.ok()) {
            return step_r.status();
        }
        if (step_r.value() == storage::sqlite::SqliteStepResult::Done) {
            break;
        }
        users.push_back(stmt.ColumnText(0));
    }
    return users;
}

core::Result<int64_t> LongTermMemoryCompressor::EnsureUserPartition(const std::string& user_uuid) {
    vector_storage::PartitionKey key;
    key.collection_id = options_.collection_id;
    key.user_id = user_uuid;
    key.memory_level = "L3";
    return options_.partition_registry->Resolve(key);
}

core::Result<std::vector<storage::CacheRecord>> LongTermMemoryCompressor::FetchDailyRecords(
    const std::string& user_uuid,
    const std::string& date
) {

    std::vector<storage::CacheRecord> all_records;

    std::string pattern = "cache:batch:" + user_uuid + ":*";

    auto scan_result = options_.redis_pool->Scan(pattern);
    if (!scan_result.ok()) {
        return scan_result.status();
    }

    std::vector<std::string> batch_keys = std::move(scan_result).value();

    if (batch_keys.empty()) {
        logger_.warn("No L0 batches found for user={}", user_uuid);
        return all_records;
    }

    auto parse_date = [](int64_t timestamp_ms) -> std::string {
        auto tp = std::chrono::system_clock::time_point(std::chrono::milliseconds(timestamp_ms));
        auto tt = std::chrono::system_clock::to_time_t(tp);
        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &tt);
#else
        localtime_r(&tt, &tm_buf);
#endif
        char buf[11];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm_buf);
        return std::string(buf);
    };

    for (const auto& key : batch_keys) {
        auto last_colon = key.rfind(':');
        if (last_colon == std::string::npos) continue;

        std::string ts_str = key.substr(last_colon + 1);
        int64_t timestamp_ms = 0;
        try {
            timestamp_ms = std::stoll(ts_str);
        } catch (const std::exception&) {
            logger_.warn("Invalid timestamp in key: {}", key);
            continue;
        }

        std::string batch_date = parse_date(timestamp_ms);
        if (batch_date != date) {
            continue;
        }

        auto hgetall_result = options_.redis_pool->HGetAll(key);
        if (!hgetall_result.ok()) {
            logger_.error("Failed to fetch records from batch {}: {}", key, hgetall_result.status().message());
            continue;
        }

        const auto& field_values = hgetall_result.value();
        for (const auto& [field, serialized] : field_values) {
            auto record_result = semantic_cache::DeserializeCacheRecord(serialized);
            if (record_result.ok()) {
                all_records.push_back(std::move(record_result).value());
            } else {
                logger_.warn("Failed to deserialize record: {}", record_result.status().message());
            }
        }

        if (static_cast<int>(all_records.size()) >= options_.max_records_per_batch) {
            logger_.info("Reached max_records_per_batch limit ({}), stopping fetch",
                options_.max_records_per_batch);
            break;
        }
    }

    return all_records;
}

core::Result<std::vector<std::string>> LongTermMemoryCompressor::CompressToFacts(
    const std::vector<storage::CacheRecord>& records) {

    if (records.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "No records to compress");
    }

    std::ostringstream oss;
    for (size_t i = 0; i < records.size(); ++i) {
        oss << "--- Record " << (i + 1) << " ---\n";
        oss << "Input: " << records[i].input << "\n";
        oss << "Response: " << records[i].response << "\n\n";
    }

    std::string input_text = oss.str();

    llm::ChatCompletionRequest req;
    req.model = "";
    req.messages = {
        {llm::ChatRole::System, FACT_EXTRACTION_PROMPT},
        {llm::ChatRole::User, input_text}
    };
    req.temperature = options_.compression_temperature;
    req.max_tokens = options_.compression_max_tokens;

    auto response_result = options_.llm_client->Complete(req);
    if (!response_result.ok()) {
        return response_result.status();
    }

    auto response = std::move(response_result).value();
    std::string raw_output = response.content;

    try {
        auto parsed = json::parse(raw_output);
        if (!parsed.is_array()) {
            return std::vector<std::string>{raw_output};
        }

        std::vector<std::string> facts;
        for (const auto& item : parsed) {
            if (item.is_string()) {
                facts.push_back(item.get<std::string>());
            }
        }
        return facts;
    } catch (const json::exception&) {
        return std::vector<std::string>{raw_output};
    }
}

core::Result<int> LongTermMemoryCompressor::StoreFacts(
    const std::string& user_uuid,
    const std::string& date,
    const std::vector<std::string>& facts,
    int source_record_count) {

    auto partition_r = EnsureUserPartition(user_uuid);
    if (!partition_r.ok()) {
        return partition_r.status();
    }
    int64_t partition_id = partition_r.value();

    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    json meta;
    meta["user_uuid"] = user_uuid;
    meta["date"] = date;
    meta["source_record_count"] = source_record_count;
    std::string meta_json = meta.dump();

    std::vector<vector_storage::EntryRecord> entries;
    entries.reserve(facts.size());

    for (size_t i = 0; i < facts.size(); ++i) {
        const auto& fact = facts[i];

        auto embed_r = options_.embedding_pipeline->Encode(fact);
        if (!embed_r.ok()) {
            logger_.error("Failed to embed fact #{}: {}", i + 1, embed_r.status().message());
            continue;
        }

        vector_storage::EntryRecord entry;
        entry.partition_id = partition_id;
        entry.cache_key = user_uuid + ":" + date + ":" + std::to_string(i);
        entry.text_hash = date + ":" + std::to_string(i);
        entry.content_hash = date + ":" + fact.substr(0, 32);
        entry.memory_hash = user_uuid + ":" + date + ":" + std::to_string(i);
        entry.vector = std::move(embed_r).value();
        entry.memory_type = "fact";
        entry.payload = fact;
        entry.extra_metadata_json = meta_json;
        entry.created_at_ms = now_ms;
        entries.push_back(std::move(entry));
    }

    if (entries.empty()) {
        return core::Status::Error(core::ErrorCode::InternalError, "All embeddings failed");
    }

    auto insert_status = options_.vector_repo->InsertEntries(entries);
    if (!insert_status.ok()) {
        return insert_status;
    }
    auto register_status = RegisterUser(user_uuid);
    if (!register_status.ok()) {
        return register_status;
    }

    if (options_.index_manager) {
        for (const auto& e : entries) {
            options_.index_manager->NotifyInserted(partition_id, e.entry_id,
                std::span<const float>(e.vector.data(), e.vector.size()));
        }
    }

    return static_cast<int>(entries.size());
}

core::Status LongTermMemoryCompressor::StoreSummary(const LongTermMemoryRecord& record) {
    std::vector<std::string> facts;
    std::istringstream stream(record.summary);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.size() > 2 && line[0] == '-' && line[1] == ' ') {
            line = line.substr(2);
        }
        if (!line.empty()) {
            facts.push_back(std::move(line));
        }
    }

    if (facts.empty()) {
        return core::Status::Ok();
    }

    if (options_.mode == CompressorMode::Testing || !options_.embedding_pipeline) {
        auto partition_r = EnsureUserPartition(record.user_uuid);
        if (!partition_r.ok()) {
            return partition_r.status();
        }
        int64_t partition_id = partition_r.value();

        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        json meta;
        meta["user_uuid"] = record.user_uuid;
        meta["date"] = record.date;
        meta["source_record_count"] = record.source_record_count;
        std::string meta_json = meta.dump();

        std::vector<vector_storage::EntryRecord> entries;
        for (size_t i = 0; i < facts.size(); ++i) {
            vector_storage::EntryRecord entry;
            entry.partition_id = partition_id;
            entry.cache_key = record.user_uuid + ":" + record.date + ":" + std::to_string(i);
            entry.text_hash = record.date + ":" + std::to_string(i);
            entry.content_hash = record.date + ":" + facts[i].substr(0, 32);
            entry.memory_hash = record.user_uuid + ":" + record.date + ":" + std::to_string(i);
            entry.vector = std::vector<float>(384, 0.0f);
            entry.memory_type = "fact";
            entry.payload = facts[i];
            entry.extra_metadata_json = meta_json;
            entry.created_at_ms = now_ms;
            entries.push_back(std::move(entry));
        }

        auto status = options_.vector_repo->InsertEntries(entries);
        if (!status.ok()) {
            return status;
        }
        return RegisterUser(record.user_uuid);
    }

    auto r = StoreFacts(record.user_uuid, record.date, facts, record.source_record_count);
    if (!r.ok()) return r.status();
    return core::Status::Ok();
}

core::Result<int> LongTermMemoryCompressor::CompressDailyMemory(
    const std::string& user_uuid,
    const std::string& date) {

    logger_.info("Starting L3 compression for user={}, date={}", user_uuid, date);

    auto records_result = FetchDailyRecords(user_uuid, date);
    if (!records_result.ok()) {
        return records_result.status();
    }

    auto records = std::move(records_result).value();
    if (records.empty()) {
        logger_.info("No L0 records found for user={}, date={}, skipping compression",
            user_uuid, date);
        return 0;
    }

    logger_.info("Fetched {} L0 records for compression", records.size());

    auto facts_result = CompressToFacts(records);
    if (!facts_result.ok()) {
        return facts_result.status();
    }

    auto facts = std::move(facts_result).value();
    logger_.info("LLM extracted {} facts", facts.size());

    auto store_result = StoreFacts(user_uuid, date, facts, static_cast<int>(records.size()));
    if (!store_result.ok()) {
        return store_result.status();
    }

    logger_.info("L3 compression complete for user={}, date={}, {} facts stored from {} records",
        user_uuid, date, store_result.value(), records.size());

    return static_cast<int>(records.size());
}

std::future<core::Result<int>> LongTermMemoryCompressor::CompressDailyMemoryAsync(
    const std::string& user_uuid,
    const std::string& date) {

    return std::async(std::launch::async, [this, user_uuid, date]() {
        return CompressDailyMemory(user_uuid, date);
    });
}

core::Result<LongTermMemoryRecord> LongTermMemoryCompressor::GetDailySummary(
    const std::string& user_uuid,
    const std::string& date) {

    auto partition_r = EnsureUserPartition(user_uuid);
    if (!partition_r.ok()) {
        return partition_r.status();
    }
    int64_t partition_id = partition_r.value();

    auto entries_r = options_.vector_repo->ListEntries(partition_id, false);
    if (!entries_r.ok()) {
        return entries_r.status();
    }

    std::vector<std::string> facts;
    int source_record_count = 0;
    int64_t latest_timestamp = 0;

    for (const auto& entry : entries_r.value()) {
        if (entry.memory_type != "fact") continue;

        try {
            auto meta = json::parse(entry.extra_metadata_json);
            if (meta.contains("date") && meta["date"].get<std::string>() == date) {
                facts.push_back(entry.payload);
                if (meta.contains("source_record_count")) {
                    source_record_count = meta["source_record_count"].get<int>();
                }
                if (entry.created_at_ms > latest_timestamp) {
                    latest_timestamp = entry.created_at_ms;
                }
            }
        } catch (const json::exception&) {
            continue;
        }
    }

    if (facts.empty()) {
        return core::Status::Error(core::ErrorCode::NotFound,
            "No L3 summary found for user=" + user_uuid + ", date=" + date);
    }

    LongTermMemoryRecord record;
    record.user_uuid = user_uuid;
    record.date = date;
    record.source_record_count = source_record_count;
    record.timestamp = latest_timestamp / 1000;

    std::ostringstream summary;
    for (const auto& fact : facts) {
        summary << "- " << fact << "\n";
    }
    record.summary = summary.str();

    return record;
}

core::Result<std::vector<LongTermMemoryRecord>> LongTermMemoryCompressor::GetUserSummaries(
    const std::string& user_uuid,
    int limit) {

    auto partition_r = EnsureUserPartition(user_uuid);
    if (!partition_r.ok()) {
        return partition_r.status();
    }
    int64_t partition_id = partition_r.value();

    auto entries_r = options_.vector_repo->ListEntries(partition_id, false);
    if (!entries_r.ok()) {
        return entries_r.status();
    }

    // Group facts by date (descending order via std::greater)
    std::map<std::string, std::vector<std::string>, std::greater<>> date_facts;
    std::map<std::string, int> date_source_counts;
    std::map<std::string, int64_t> date_timestamps;

    for (const auto& entry : entries_r.value()) {
        if (entry.memory_type != "fact") continue;

        try {
            auto meta = json::parse(entry.extra_metadata_json);
            if (!meta.contains("date")) continue;
            std::string d = meta["date"].get<std::string>();

            date_facts[d].push_back(entry.payload);
            if (meta.contains("source_record_count")) {
                date_source_counts[d] = meta["source_record_count"].get<int>();
            }
            if (entry.created_at_ms > date_timestamps[d]) {
                date_timestamps[d] = entry.created_at_ms;
            }
        } catch (const json::exception&) {
            continue;
        }
    }

    std::vector<LongTermMemoryRecord> records;
    int count = 0;
    for (const auto& [d, facts] : date_facts) {
        if (count >= limit) break;

        LongTermMemoryRecord rec;
        rec.user_uuid = user_uuid;
        rec.date = d;
        rec.source_record_count = date_source_counts[d];
        rec.timestamp = date_timestamps[d] / 1000;

        std::ostringstream summary;
        for (const auto& fact : facts) {
            summary << "- " << fact << "\n";
        }
        rec.summary = summary.str();

        records.push_back(std::move(rec));
        ++count;
    }

    return records;
}

core::Result<std::vector<vector_storage::EntryRecord>> LongTermMemoryCompressor::SearchFacts(
    const std::string& user_uuid,
    const std::string& query,
    int top_k) {

    if (!options_.embedding_pipeline || !options_.index_manager) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
            "SearchFacts requires embedding_pipeline and index_manager");
    }

    auto partition_r = EnsureUserPartition(user_uuid);
    if (!partition_r.ok()) {
        return partition_r.status();
    }
    int64_t partition_id = partition_r.value();

    auto embed_r = options_.embedding_pipeline->Encode(query);
    if (!embed_r.ok()) {
        return embed_r.status();
    }

    vector::IndexSearchOptions search_opts;
    search_opts.top_k = static_cast<size_t>(top_k);
    search_opts.include_forgotten = false;

    return options_.index_manager->Search(partition_id,
        std::span<const float>(embed_r.value().data(), embed_r.value().size()),
        search_opts);
}

}
