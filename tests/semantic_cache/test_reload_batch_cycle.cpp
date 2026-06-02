#include "../../src/semantic_cache/semantic_cache_pipeline.h"
#include "../../src/semantic_cache/redis_connection_pool.h"
#include "../../src/storage/sqlite/sqlite_connection.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <chrono>
#include <memory>

using namespace agent::semantic_cache;
using namespace storage;
#define MAX_CACHE_RECORDS 1000
class ReloadBatchCycleTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_db_path_ = "test_reload_cycle.db";
        std::filesystem::remove(test_db_path_);

        auto conn_result = sqlite::SqliteConnection::Open(test_db_path_);
        ASSERT_TRUE(conn_result.ok());
        sqlite_conn_ = std::move(conn_result.value());

        redis_pool_ = std::make_shared<RedisConnectionPool>(
            RedisPoolOptions{
                .host = "127.0.0.1",
                .port = "5000",
                .pool_size = 2
            });
        redis_pool_->Start();

        user_uuid_ = "test_user_reload_cycle_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

        CleanupRedisKeys();
    }

    void TearDown() override {
        if (redis_pool_) {
            CleanupRedisKeys();
            redis_pool_->Shutdown();
        }
        sqlite_conn_ = sqlite::SqliteConnection();
        std::filesystem::remove(test_db_path_);
    }

    void CleanupRedisKeys() {
        std::string pattern = "cache:batch:" + user_uuid_ + ":*";
        auto result = redis_pool_->Scan(pattern);
        if (result.ok()) {
            redis_pool_->Del(result.value());
        }
    }

    std::string test_db_path_;
    sqlite::SqliteConnection sqlite_conn_;
    std::shared_ptr<RedisConnectionPool> redis_pool_;
    std::string user_uuid_;
};

TEST_F(ReloadBatchCycleTest, CyclicReloadDoesNotEmptyIndex) {
    cache_vector::VectorIndexManager manager(user_uuid_, redis_pool_, std::move(sqlite_conn_),MAX_CACHE_RECORDS);

    for (int batch = 0; batch < 3; ++batch) {
        for (int i = 0; i < MAX_CACHE_RECORDS; ++i) {
            CacheRecord record;
            record.embedding.resize(agent::semantic_cache::kExpectedEmbeddingDim, 0.1f * batch);
            record.input = "batch_" + std::to_string(batch) + "_record_" + std::to_string(i);
            record.response = "response_" + std::to_string(i);
            auto status = manager.AddRecord(record);
            ASSERT_TRUE(status.ok()) << status.message();
        }
    }

    for (int round = 0; round < 6; ++round) {
        std::vector<float> query(agent::semantic_cache::kExpectedEmbeddingDim, 0.5f);
        auto result = manager.Search(query, 5);
        ASSERT_TRUE(result.ok()) << "Round " << round << " failed: " << result.status().message();
        ASSERT_FALSE(result.value().empty()) << "Round " << round << " returned empty results";
    }

    EXPECT_GT(manager.CurrentSize(), 0);
}

TEST_F(ReloadBatchCycleTest, RebuildFromRedisWhenIndexEmpty) {
    cache_vector::VectorIndexManager manager(user_uuid_, redis_pool_, std::move(sqlite_conn_));

    for (int batch = 0; batch < 2; ++batch) {
        for (int i = 0; i < MAX_CACHE_RECORDS; ++i) {
            CacheRecord record;
            record.embedding.resize(agent::semantic_cache::kExpectedEmbeddingDim, 0.2f);
            record.input = "rebuild_test_" + std::to_string(batch) + "_" + std::to_string(i);
            record.response = "resp";
            manager.AddRecord(record);
        }
    }

    std::string new_db = "test_rebuild_fresh.db";
    std::filesystem::remove(new_db);
    auto new_conn_result = sqlite::SqliteConnection::Open(new_db);
    ASSERT_TRUE(new_conn_result.ok());

    {
        cache_vector::VectorIndexManager fresh_manager(
            user_uuid_, redis_pool_, std::move(new_conn_result.value()));

        std::vector<float> query(agent::semantic_cache::kExpectedEmbeddingDim, 0.3f);
        auto result = fresh_manager.Search(query, 3);
        ASSERT_TRUE(result.ok()) << result.status().message();
        ASSERT_FALSE(result.value().empty());
    }  // fresh_manager 析构，释放数据库文件句柄

    std::filesystem::remove(new_db);
}
