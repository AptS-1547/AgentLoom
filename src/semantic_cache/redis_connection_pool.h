#pragma once

#include "result.h"
#include <sw/redis++/redis++.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace agent::semantic_cache {

struct RedisPoolOptions {
    std::string host = "127.0.0.1";
    std::string port = "6379";
    std::string password;
    std::size_t pool_size = 4;
    std::chrono::seconds connect_timeout{5};
    std::chrono::milliseconds command_timeout{5000};
};

/// Redis connection pool backed by redis++.
/// Provides synchronous Hash, String, List, and SCAN operations with pipeline support.
/// Thread-safe: redis++ connection pool is internally synchronized.
class RedisConnectionPool {
public:
    explicit RedisConnectionPool(RedisPoolOptions options);
    ~RedisConnectionPool();

    RedisConnectionPool(const RedisConnectionPool&) = delete;
    RedisConnectionPool& operator=(const RedisConnectionPool&) = delete;

    /// Initialize connection pool. Returns error if Redis is unreachable.
    core::Status Start();

    /// Shutdown pool and close all connections.
    void Shutdown();

    bool running() const noexcept { return running_.load(); }

    // ─── Hash Operations ───────────────────────────────────────────────────

    /// HSET key field value. Returns true if field is new, false if updated.
    core::Result<bool> HSet(const std::string& key, const std::string& field, const std::string& value);

    /// HSET multiple fields via pipeline. Returns number of new fields created.
    core::Result<long long> HMSet(const std::string& key, const std::unordered_map<std::string, std::string>& field_values);

    /// HGETALL key. Returns all field-value pairs.
    core::Result<std::unordered_map<std::string, std::string>> HGetAll(const std::string& key);

    /// HDEL key field [field ...]. Returns number of fields deleted.
    core::Result<long long> HDel(const std::string& key, const std::vector<std::string>& fields);

    // ─── String Operations ─────────────────────────────────────────────────

    /// SET key value [EX seconds]. Returns OK status.
    core::Status Set(const std::string& key, const std::string& value, std::chrono::seconds ttl = std::chrono::seconds::zero());

    /// SET key value NX [EX seconds]. Returns true when the key was created.
    core::Result<bool> SetIfAbsent(const std::string& key,
                                   const std::string& value,
                                   std::chrono::seconds ttl = std::chrono::seconds::zero());

    /// GET key. Returns value or NotFound if key doesn't exist.
    core::Result<std::string> Get(const std::string& key);

    /// MGET key1 key2 ... Returns values in same order (empty string for missing keys).
    core::Result<std::vector<std::string>> MGet(const std::vector<std::string>& keys);

    /// DEL key [key ...]. Returns number of keys deleted.
    core::Result<long long> Del(const std::vector<std::string>& keys);

    // ─── List Operations ───────────────────────────────────────────────────

    /// RPUSH key element [element ...]. Returns list length after push.
    core::Result<long long> RPush(const std::string& key, const std::vector<std::string>& elements);

    /// LRANGE key start stop. Returns elements in range.
    core::Result<std::vector<std::string>> LRange(const std::string& key, long long start, long long stop);

    // ─── Scan Operations ───────────────────────────────────────────────────

    /// SCAN with pattern matching. Returns all matching keys (handles cursor internally).
    core::Result<std::vector<std::string>> Scan(const std::string& pattern);

    // ─── Pipeline Operations ───────────────────────────────────────────────

    /// Execute multiple commands in a pipeline for maximum throughput.
    /// Usage:
    ///   auto pipe = pool->CreatePipeline();
    ///   pipe.hset("key1", "f1", "v1");
    ///   pipe.hset("key2", "f2", "v2");
    ///   auto status = pool->ExecPipeline(pipe);
    class Pipeline {
    public:
        Pipeline& hset(const std::string& key, const std::string& field, const std::string& value);
        Pipeline& hmset(const std::string& key, const std::unordered_map<std::string, std::string>& fv);
        Pipeline& set(const std::string& key, const std::string& value);
        Pipeline& del(const std::string& key);
        std::size_t size() const { return commands_.size(); }

    private:
        friend class RedisConnectionPool;
        struct Command {
            enum Type { HSET, HMSET, SET, DEL } type;
            std::string key;
            std::string field;
            std::string value;
            std::unordered_map<std::string, std::string> field_values;
        };
        std::vector<Command> commands_;
    };

    Pipeline CreatePipeline();
    core::Status ExecPipeline(Pipeline& pipe);

private:
    core::Status EnsureConnected();

    RedisPoolOptions options_;
    std::unique_ptr<sw::redis::Redis> redis_;
    std::atomic<bool> running_{false};
};

}  // namespace agent::semantic_cache
