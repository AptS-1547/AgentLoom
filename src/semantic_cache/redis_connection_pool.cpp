#include "redis_connection_pool.h"
#include <iostream>

namespace agent::semantic_cache {

RedisConnectionPool::RedisConnectionPool(RedisPoolOptions options)
    : options_(std::move(options)) {}

RedisConnectionPool::~RedisConnectionPool() {
    Shutdown();
}

core::Status RedisConnectionPool::Start() {
    if (running_.exchange(true)) {
        return core::Status::Error(core::ErrorCode::AlreadyExists, "pool already running");
    }

    try {
        sw::redis::ConnectionOptions conn_opts;
        conn_opts.host = options_.host;
        conn_opts.port = std::stoi(options_.port);
        if (!options_.password.empty()) {
            conn_opts.password = options_.password;
        }
        conn_opts.socket_timeout = options_.command_timeout;
        conn_opts.connect_timeout = options_.connect_timeout;

        sw::redis::ConnectionPoolOptions pool_opts;
        pool_opts.size = options_.pool_size;
        pool_opts.wait_timeout = options_.connect_timeout;

        redis_ = std::make_unique<sw::redis::Redis>(conn_opts, pool_opts);

        redis_->ping();

        return core::Status::Ok();
    } catch (const sw::redis::Error& e) {
        running_.store(false);
        return core::Status::Error(core::ErrorCode::Unavailable,
            std::string("redis connection failed: ") + e.what());
    }
}

void RedisConnectionPool::Shutdown() {
    if (!running_.exchange(false)) {
        return;
    }
    redis_.reset();
}

core::Status RedisConnectionPool::EnsureConnected() {
    if (!running_.load()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "pool not running");
    }
    if (!redis_) {
        return core::Status::Error(core::ErrorCode::InternalError, "redis client is null");
    }
    return core::Status::Ok();
}

core::Result<bool> RedisConnectionPool::HSet(const std::string& key, const std::string& field, const std::string& value) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        bool created = redis_->hset(key, field, value);
        return created;
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("HSET failed: ") + e.what());
    }
}

core::Result<long long> RedisConnectionPool::HMSet(const std::string& key, const std::unordered_map<std::string, std::string>& field_values) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        auto pipe = redis_->pipeline();
        for (const auto& [field, value] : field_values) {
            pipe.hset(key, field, value);
        }
        auto replies = pipe.exec();

        long long new_fields = 0;
        for (size_t i = 0; i < field_values.size(); ++i) {
            new_fields += replies.get<long long>(i);
        }
        return new_fields;
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("HMSET failed: ") + e.what());
    }
}

core::Result<std::unordered_map<std::string, std::string>> RedisConnectionPool::HGetAll(const std::string& key) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        std::unordered_map<std::string, std::string> result;
        redis_->hgetall(key, std::inserter(result, result.begin()));
        return result;
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("HGETALL failed: ") + e.what());
    }
}

core::Result<long long> RedisConnectionPool::HDel(const std::string& key, const std::vector<std::string>& fields) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        long long deleted = redis_->hdel(key, fields.begin(), fields.end());
        return deleted;
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("HDEL failed: ") + e.what());
    }
}

core::Status RedisConnectionPool::Set(const std::string& key, const std::string& value, std::chrono::seconds ttl) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        if (ttl.count() > 0) {
            redis_->set(key, value, ttl);
        } else {
            redis_->set(key, value);
        }
        return core::Status::Ok();
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("SET failed: ") + e.what());
    }
}

core::Result<bool> RedisConnectionPool::SetIfAbsent(const std::string& key,
                                                    const std::string& value,
                                                    std::chrono::seconds ttl) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        bool created = false;
        if (ttl.count() > 0) {
            created = redis_->set(key, value, ttl, sw::redis::UpdateType::NOT_EXIST);
        } else {
            created = redis_->set(key, value, std::chrono::milliseconds(0), sw::redis::UpdateType::NOT_EXIST);
        }
        return created;
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("SET NX failed: ") + e.what());
    }
}

core::Result<std::string> RedisConnectionPool::Get(const std::string& key) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        auto val = redis_->get(key);
        if (!val) {
            return core::Status::Error(core::ErrorCode::NotFound, "key not found");
        }
        return *val;
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("GET failed: ") + e.what());
    }
}

core::Result<bool> RedisConnectionPool::Exists(const std::string& key) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        return redis_->exists(key) > 0;
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("EXISTS failed: ") + e.what());
    }
}

core::Result<std::vector<std::string>> RedisConnectionPool::MGet(const std::vector<std::string>& keys) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        std::vector<sw::redis::OptionalString> results;
        redis_->mget(keys.begin(), keys.end(), std::back_inserter(results));

        std::vector<std::string> values;
        values.reserve(results.size());
        for (auto& opt : results) {
            values.push_back(opt ? *opt : "");
        }
        return values;
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("MGET failed: ") + e.what());
    }
}

core::Result<long long> RedisConnectionPool::Del(const std::vector<std::string>& keys) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        long long deleted = redis_->del(keys.begin(), keys.end());
        return deleted;
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("DEL failed: ") + e.what());
    }
}

core::Result<long long> RedisConnectionPool::RPush(const std::string& key, const std::vector<std::string>& elements) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        long long length = redis_->rpush(key, elements.begin(), elements.end());
        return length;
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("RPUSH failed: ") + e.what());
    }
}

core::Result<std::vector<std::string>> RedisConnectionPool::LRange(const std::string& key, long long start, long long stop) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        std::vector<std::string> result;
        redis_->lrange(key, start, stop, std::back_inserter(result));
        return result;
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("LRANGE failed: ") + e.what());
    }
}

core::Result<std::vector<std::string>> RedisConnectionPool::Scan(const std::string& pattern) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    try {
        std::vector<std::string> keys;
        long long cursor = 0;
        do {
            cursor = redis_->scan(cursor, pattern, 100, std::back_inserter(keys));
        } while (cursor != 0);
        return keys;
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("SCAN failed: ") + e.what());
    }
}

RedisConnectionPool::Pipeline RedisConnectionPool::CreatePipeline() {
    return Pipeline();
}

RedisConnectionPool::Pipeline& RedisConnectionPool::Pipeline::hset(const std::string& key, const std::string& field, const std::string& value) {
    commands_.push_back({Command::HSET, key, field, value, {}});
    return *this;
}

RedisConnectionPool::Pipeline& RedisConnectionPool::Pipeline::hmset(const std::string& key, const std::unordered_map<std::string, std::string>& fv) {
    commands_.push_back({Command::HMSET, key, "", "", fv});
    return *this;
}

RedisConnectionPool::Pipeline& RedisConnectionPool::Pipeline::set(const std::string& key, const std::string& value) {
    commands_.push_back({Command::SET, key, "", value, {}});
    return *this;
}

RedisConnectionPool::Pipeline& RedisConnectionPool::Pipeline::del(const std::string& key) {
    commands_.push_back({Command::DEL, key, "", "", {}});
    return *this;
}

core::Status RedisConnectionPool::ExecPipeline(Pipeline& pipe) {
    auto status = EnsureConnected();
    if (!status.ok()) return status;

    if (pipe.commands_.empty()) {
        return core::Status::Ok();
    }

    try {
        auto redis_pipe = redis_->pipeline();
        for (const auto& cmd : pipe.commands_) {
            switch (cmd.type) {
                case Pipeline::Command::HSET:
                    redis_pipe.hset(cmd.key, cmd.field, cmd.value);
                    break;
                case Pipeline::Command::HMSET:
                    for (const auto& [f, v] : cmd.field_values) {
                        redis_pipe.hset(cmd.key, f, v);
                    }
                    break;
                case Pipeline::Command::SET:
                    redis_pipe.set(cmd.key, cmd.value);
                    break;
                case Pipeline::Command::DEL:
                    redis_pipe.del(cmd.key);
                    break;
            }
        }
        redis_pipe.exec();
        return core::Status::Ok();
    } catch (const sw::redis::Error& e) {
        return core::Status::Error(core::ErrorCode::InternalError,
            std::string("Pipeline exec failed: ") + e.what());
    }
}

}  // namespace agent::semantic_cache
