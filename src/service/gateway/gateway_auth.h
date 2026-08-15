#pragma once

#include "http_types.h"
#include "logger_adapter.h"
#include "result.h"

#include <cstddef>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace agent::semantic_cache {
class RedisConnectionPool;
}

namespace storage::sqlite {
class SqliteConnectionPool;
}

namespace agent::service::gateway {

struct AuthIdentity {
    std::string user_uuid = "local_user";
    std::string tenant_id = "default";
    std::string subject;
    std::string issuer;
    std::string audience;
    std::string token_id;
    std::chrono::system_clock::time_point expires_at{};
    bool authenticated = false;
};

struct GatewayAuthOptions {
    bool enabled = false;
    bool allow_dev_identity = true;
    bool require_auth_for_api = false;
    std::string cookie_name = "agent_auth";
    std::string public_key_pem;
    std::string private_key_pem;
    std::string issuer;
    std::string audience;
    std::chrono::seconds clock_skew{60};
    std::chrono::seconds token_ttl{std::chrono::hours(8)};
    bool cookie_http_only = true;
    bool cookie_secure = false;
    std::string cookie_same_site = "Lax";
    bool require_session_record = false;
    bool auto_provision_session = true;
    bool enable_dev_registration = false;
    std::string session_store_backend = "sqlite";
    std::string session_database_path;
    std::string redis_host = "127.0.0.1";
    std::string redis_port = "6379";
    std::string redis_password;
    std::size_t redis_pool_size = 8;
    std::chrono::milliseconds redis_command_timeout{5000};
    std::string redis_key_prefix = "agent:gateway:auth";
};

struct GatewayDevelopmentKeyPair {
    std::string private_key_pem;
    std::string public_key_pem;
};

struct AuthSessionRecord {
    std::string token_id;
    std::string user_uuid;
    std::string tenant_id = "default";
    std::string subject;
    std::chrono::system_clock::time_point issued_at{};
    std::chrono::system_clock::time_point expires_at{};
    std::chrono::system_clock::time_point updated_at{};
    bool revoked = false;
};

struct AuthUserRecord {
    std::string user_uuid;
    std::string tenant_id = "default";
    std::string username;
    std::string password_hash;
    std::string password_salt;
    int password_iterations = 0;
    std::string subject;
    std::chrono::system_clock::time_point created_at{};
    std::chrono::system_clock::time_point updated_at{};
    bool disabled = false;
};

class IAuthSessionStore {
public:
    virtual ~IAuthSessionStore() = default;
    virtual core::Status EnsureSchema() = 0;
    virtual core::Result<AuthSessionRecord> ResolveSession(std::string_view token_id) = 0;
    virtual core::Result<AuthUserRecord> ResolveUserByUsername(std::string_view username) = 0;
    virtual core::Status UpsertUser(const AuthUserRecord& record) = 0;
    virtual core::Status UpsertSession(const AuthSessionRecord& record) = 0;
    virtual core::Status RevokeSession(std::string_view token_id, std::string_view reason) = 0;
    virtual core::Result<std::size_t> CleanupExpired(
        std::chrono::system_clock::time_point now,
        std::size_t max_records) = 0;
};

class IGatewayAuthenticator {
public:
    virtual ~IGatewayAuthenticator() = default;
    virtual core::Result<AuthIdentity> Authenticate(const ::net::BeastHttpRequest& request) const = 0;
};

struct AuthRegistrationRequest {
    std::string username;
    std::string password;
    std::string user_uuid;
    std::string tenant_id = "default";
    std::string subject;
    std::chrono::seconds ttl{0};
};

struct AuthLoginRequest {
    std::string username;
    std::string password;
    std::chrono::seconds ttl{0};
};

struct AuthRegistrationResult {
    AuthIdentity identity;
    std::string token;
    std::string cookie_header;
    std::chrono::system_clock::time_point issued_at{};
};

class IAuthRegistrationService {
public:
    virtual ~IAuthRegistrationService() = default;
    virtual core::Result<AuthRegistrationResult> Register(const AuthRegistrationRequest& request) = 0;
    virtual core::Result<AuthRegistrationResult> Login(const AuthLoginRequest& request) = 0;
};

class JwtCookieAuthenticator final : public IGatewayAuthenticator {
public:
    explicit JwtCookieAuthenticator(GatewayAuthOptions options = {},
                                    std::shared_ptr<IAuthSessionStore> session_store = nullptr,
                                    core::LoggerAdapter logger = core::LoggerAdapter::ForModule("gateway-auth"));

    core::Result<AuthIdentity> Authenticate(const ::net::BeastHttpRequest& request) const override;

private:
    core::Result<std::string> ExtractToken(const ::net::BeastHttpRequest& request) const;
    core::Result<AuthIdentity> VerifyJwt(std::string_view token) const;
    core::Result<AuthIdentity> ResolveLocalSession(const AuthIdentity& jwt_identity) const;

    GatewayAuthOptions options_;
    std::shared_ptr<IAuthSessionStore> session_store_;
    core::LoggerAdapter logger_;
};

class JwtAuthRegistrationService final : public IAuthRegistrationService {
public:
    explicit JwtAuthRegistrationService(GatewayAuthOptions options = {},
                                        std::shared_ptr<IAuthSessionStore> session_store = nullptr,
                                        core::LoggerAdapter logger = core::LoggerAdapter::ForModule("gateway-auth"));

    core::Result<AuthRegistrationResult> Register(const AuthRegistrationRequest& request) override;
    core::Result<AuthRegistrationResult> Login(const AuthLoginRequest& request) override;

private:
    core::Result<AuthRegistrationResult> IssueForIdentity(const AuthIdentity& identity,
                                                          std::chrono::system_clock::time_point issued_at) const;
    core::Result<std::string> IssueJwt(const AuthIdentity& identity,
                                       std::chrono::system_clock::time_point issued_at) const;
    std::string BuildCookieHeader(std::string_view token,
                                  std::chrono::system_clock::time_point expires_at) const;

    GatewayAuthOptions options_;
    std::shared_ptr<IAuthSessionStore> session_store_;
    core::LoggerAdapter logger_;
};

class SqliteAuthSessionStore final : public IAuthSessionStore {
public:
    explicit SqliteAuthSessionStore(std::string database_path);
    explicit SqliteAuthSessionStore(std::shared_ptr<storage::sqlite::SqliteConnectionPool> pool);

    core::Status EnsureSchema() override;
    core::Result<AuthSessionRecord> ResolveSession(std::string_view token_id) override;
    core::Result<AuthUserRecord> ResolveUserByUsername(std::string_view username) override;
    core::Status UpsertUser(const AuthUserRecord& record) override;
    core::Status UpsertSession(const AuthSessionRecord& record) override;
    core::Status RevokeSession(std::string_view token_id, std::string_view reason) override;
    core::Result<std::size_t> CleanupExpired(
        std::chrono::system_clock::time_point now,
        std::size_t max_records) override;

private:
    std::shared_ptr<storage::sqlite::SqliteConnectionPool> pool_;
};

class RedisAuthSessionStore final : public IAuthSessionStore {
public:
    RedisAuthSessionStore(std::shared_ptr<semantic_cache::RedisConnectionPool> redis,
                          std::string key_prefix = "agent:gateway:auth");

    core::Status EnsureSchema() override;
    core::Result<AuthSessionRecord> ResolveSession(std::string_view token_id) override;
    core::Result<AuthUserRecord> ResolveUserByUsername(std::string_view username) override;
    core::Status UpsertUser(const AuthUserRecord& record) override;
    core::Status UpsertSession(const AuthSessionRecord& record) override;
    core::Status RevokeSession(std::string_view token_id, std::string_view reason) override;
    core::Result<std::size_t> CleanupExpired(
        std::chrono::system_clock::time_point now,
        std::size_t max_records) override;

private:
    std::string SessionKey(std::string_view token_id) const;
    std::string UserKey(std::string_view user_uuid) const;
    std::string UsernameKey(std::string_view username) const;

    std::shared_ptr<semantic_cache::RedisConnectionPool> redis_;
    std::string key_prefix_;
};

std::optional<std::string> ExtractCookieValue(std::string_view cookie_header, std::string_view name);
core::Result<GatewayDevelopmentKeyPair> GenerateDevelopmentRsaKeyPair();

} // namespace agent::service::gateway
