#include "../memory/long_term_memory_compressor.h"
#include "../storage/vector/sqlite_vector_repository.h"
#include "../storage/vector/vector_partition_registry.h"
#include "../storage/sqlite/sqlite_connection_pool.h"
#include "../semantic_cache/redis_connection_pool.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace agent::memory {

class LongTermMemoryCompressorTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto path = fs::temp_directory_path() / "agent_l3_memory_test.db";
        std::error_code ec;
        fs::remove(path, ec);
        fs::remove(path.string() + "-wal", ec);
        fs::remove(path.string() + "-shm", ec);
        test_db_path_ = path.string();

        // Setup SQLite pool for vector repository
        storage::sqlite::SqliteConnectionPoolOptions pool_opts;
        pool_opts.path = test_db_path_;
        pool_opts.read_connection_count = 2;
        pool_opts.enable_wal = true;
        sqlite_pool_ = std::make_shared<storage::sqlite::SqliteConnectionPool>(pool_opts);
        auto pool_start = sqlite_pool_->Start();
        ASSERT_TRUE(pool_start.ok()) << pool_start.message();

        // Setup vector repository
        vector_repo_ = std::make_shared<vector_storage::SqliteVectorRepository>(sqlite_pool_);
        auto schema_status = vector_repo_->EnsureSchema();
        ASSERT_TRUE(schema_status.ok()) << schema_status.message();

        // Create a test collection
        vector_storage::CollectionDescriptor coll;
        coll.name = "test_memory_v1";
        coll.embedding_model_fingerprint = "test";
        coll.tokenizer_fingerprint = "test";
        coll.pooling_strategy = "mean";
        coll.normalization = "l2";
        coll.dimension = 384;
        coll.corpus_version = "v1";
        coll.policy_version = "v1";
        auto coll_r = vector_repo_->EnsureCollection(coll);
        ASSERT_TRUE(coll_r.ok()) << coll_r.status().message();
        collection_id_ = coll_r.value();

        // Setup partition registry
        partition_registry_ = std::make_shared<vector_storage::PartitionRegistry>(vector_repo_);

        // Setup Redis pool
        semantic_cache::RedisPoolOptions redis_opts;
        redis_opts.host = "127.0.0.1";
        redis_opts.port = "6379";
        redis_opts.pool_size = 2;
        redis_pool_ = std::make_shared<semantic_cache::RedisConnectionPool>(redis_opts);
        auto redis_start = redis_pool_->Start();
        ASSERT_TRUE(redis_start.ok()) << redis_start.message();
    }

    void TearDown() override {
        redis_pool_->Shutdown();
        sqlite_pool_->Close();
        std::error_code ec;
        fs::remove(test_db_path_, ec);
        fs::remove(test_db_path_ + "-wal", ec);
        fs::remove(test_db_path_ + "-shm", ec);
    }

    std::unique_ptr<LongTermMemoryCompressor> MakeCompressor() {
        LongTermMemoryCompressorOptions opts;
        opts.redis_pool = redis_pool_;
        opts.vector_repo = vector_repo_;
        opts.partition_registry = partition_registry_;
        opts.registry_pool = sqlite_pool_;
        opts.collection_id = collection_id_;
        opts.mode = CompressorMode::Testing;
        auto r = LongTermMemoryCompressor::Create(std::move(opts));
        EXPECT_TRUE(r.ok()) << r.status().message();
        return std::move(r).value();
    }

    std::string test_db_path_;
    std::shared_ptr<storage::sqlite::SqliteConnectionPool> sqlite_pool_;
    std::shared_ptr<vector_storage::SqliteVectorRepository> vector_repo_;
    std::shared_ptr<vector_storage::PartitionRegistry> partition_registry_;
    std::shared_ptr<semantic_cache::RedisConnectionPool> redis_pool_;
    int64_t collection_id_ = 0;
};

TEST_F(LongTermMemoryCompressorTest, CreateWithValidDeps) {
    auto compressor = MakeCompressor();
    EXPECT_NE(compressor, nullptr);
}

TEST_F(LongTermMemoryCompressorTest, StoreSummaryAndRetrieve) {
    auto compressor = MakeCompressor();

    LongTermMemoryRecord record;
    record.user_uuid = "test-user-123";
    record.date = "2026-05-27";
    record.summary = "- User learned C++ template metaprogramming\n- Completed 3 exercises\n";
    record.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    record.source_record_count = 15;

    auto store_status = compressor->StoreSummary(record);
    ASSERT_TRUE(store_status.ok()) << store_status.message();

    auto retrieve_result = compressor->GetDailySummary("test-user-123", "2026-05-27");
    ASSERT_TRUE(retrieve_result.ok()) << retrieve_result.status().message();

    const auto& retrieved = retrieve_result.value();
    EXPECT_EQ(retrieved.user_uuid, "test-user-123");
    EXPECT_EQ(retrieved.date, "2026-05-27");
    EXPECT_EQ(retrieved.source_record_count, 15);
    EXPECT_TRUE(retrieved.summary.find("C++ template metaprogramming") != std::string::npos);
    EXPECT_TRUE(retrieved.summary.find("3 exercises") != std::string::npos);

    auto users = compressor->GetRegisteredUsers();
    ASSERT_TRUE(users.ok()) << users.status().message();
    ASSERT_FALSE(users.value().empty());
    EXPECT_NE(std::find(users.value().begin(), users.value().end(), "test-user-123"), users.value().end());
}

TEST_F(LongTermMemoryCompressorTest, GetUserSummariesOrderedByDate) {
    auto compressor = MakeCompressor();

    std::vector<std::string> dates = {"2026-05-25", "2026-05-26", "2026-05-27"};
    for (const auto& date : dates) {
        LongTermMemoryRecord record;
        record.user_uuid = "test-user-456";
        record.date = date;
        record.summary = "- Summary for " + date + "\n";
        record.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        record.source_record_count = 10;

        auto status = compressor->StoreSummary(record);
        ASSERT_TRUE(status.ok()) << status.message();
    }

    auto summaries_result = compressor->GetUserSummaries("test-user-456", 10);
    ASSERT_TRUE(summaries_result.ok()) << summaries_result.status().message();

    const auto& summaries = summaries_result.value();
    ASSERT_EQ(summaries.size(), 3);
    EXPECT_EQ(summaries[0].date, "2026-05-27");
    EXPECT_EQ(summaries[1].date, "2026-05-26");
    EXPECT_EQ(summaries[2].date, "2026-05-25");
}

TEST_F(LongTermMemoryCompressorTest, StoresUserUuidInFactMetadata) {
    auto compressor = MakeCompressor();

    LongTermMemoryRecord record;
    record.user_uuid = "metadata-user";
    record.date = "2026-05-27";
    record.summary = "- Likes algebra practice\n";
    record.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    record.source_record_count = 3;

    auto store_status = compressor->StoreSummary(record);
    ASSERT_TRUE(store_status.ok()) << store_status.message();

    vector_storage::PartitionKey key;
    key.collection_id = collection_id_;
    key.user_id = "metadata-user";
    key.memory_level = "L3";
    auto partition = partition_registry_->Lookup(key);
    ASSERT_TRUE(partition.ok()) << partition.status().message();
    ASSERT_TRUE(partition.value().has_value());

    auto entries = vector_repo_->ListEntries(*partition.value(), false);
    ASSERT_TRUE(entries.ok()) << entries.status().message();
    ASSERT_EQ(entries.value().size(), 1);
    EXPECT_NE(entries.value()[0].extra_metadata_json.find("\"user_uuid\":\"metadata-user\""),
              std::string::npos);
}

TEST_F(LongTermMemoryCompressorTest, GetDailySummaryNotFound) {
    auto compressor = MakeCompressor();

    auto result = compressor->GetDailySummary("nonexistent-user", "2026-01-01");
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::NotFound);
}

}  // namespace agent::memory
