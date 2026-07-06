#include "config_section.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(GatewayAuthConfigSection, "gateway_auth")
    CONFIG_CLI_STRING(kEnabled, "--gateway-auth-enabled");
    CONFIG_CLI_STRING(kAllowDevIdentity, "--gateway-auth-allow-dev-identity");
    CONFIG_CLI_STRING(kRequireAuthForApi, "--gateway-auth-required");
    CONFIG_CLI_STRING(kCookieName, "--gateway-auth-cookie");
    CONFIG_CLI_STRING(kPublicKeyFile, "--gateway-auth-public-key-file");
    CONFIG_CLI_STRING(kPrivateKeyFile, "--gateway-auth-private-key-file");
    CONFIG_CLI_STRING(kIssuer, "--gateway-auth-issuer");
    CONFIG_CLI_STRING(kAudience, "--gateway-auth-audience");
    CONFIG_CLI_STRING(kClockSkew, "--gateway-auth-clock-skew");
    CONFIG_CLI_STRING(kTokenTtl, "--gateway-auth-token-ttl");
    CONFIG_CLI_STRING(kSessionStoreBackend, "--gateway-auth-session-store");
    CONFIG_CLI_STRING(kSessionDb, "--gateway-auth-session-db");
    CONFIG_CLI_STRING(kRedisHost, "--gateway-auth-redis-host");
    CONFIG_CLI_STRING(kRedisPort, "--gateway-auth-redis-port");
    CONFIG_CLI_STRING(kRequireSessionRecord, "--gateway-auth-require-session");
    CONFIG_CLI_STRING(kAutoProvisionSession, "--gateway-auth-auto-provision");
    CONFIG_CLI_STRING(kEnableDevRegistration, "--gateway-auth-enable-dev-registration");
    CONFIG_CLI_STRING(kCookieSecure, "--gateway-auth-cookie-secure");
    void Validate(MultimodalServerOptions& options) const override;
};

std::filesystem::path ResolveRelativeToConfig(const std::filesystem::path& path,
                                              const std::filesystem::path& config_path) {
    if (path.empty() || path.is_absolute() || config_path.empty()) {
        return path;
    }
    return config_path.parent_path() / path;
}

std::string ReadTextFile(const std::filesystem::path& path, std::string_view label) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open " + std::string(label) + ": " + path.string());
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

void GatewayAuthConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) {
        return;
    }
    SetBool(*section, Name(), "enabled", options.gateway_auth.enabled);
    SetBool(*section, Name(), "allow_dev_identity", options.gateway_auth.allow_dev_identity);
    SetBool(*section, Name(), "require_auth_for_api", options.gateway_auth.require_auth_for_api);
    SetString(*section, Name(), "cookie_name", options.gateway_auth.cookie_name);
    SetString(*section, Name(), "public_key_pem", options.gateway_auth.public_key_pem);
    SetString(*section, Name(), "public_key_file", options.gateway_auth.public_key_file);
    SetString(*section, Name(), "private_key_pem", options.gateway_auth.private_key_pem);
    SetString(*section, Name(), "private_key_file", options.gateway_auth.private_key_file);
    SetString(*section, Name(), "issuer", options.gateway_auth.issuer);
    SetString(*section, Name(), "audience", options.gateway_auth.audience);
    SetInt(*section, Name(), "clock_skew_seconds", options.gateway_auth.clock_skew_seconds, 0, 3600);
    SetInt(*section, Name(), "token_ttl_seconds", options.gateway_auth.token_ttl_seconds, 1, 86400 * 30);
    SetBool(*section, Name(), "cookie_http_only", options.gateway_auth.cookie_http_only);
    SetBool(*section, Name(), "cookie_secure", options.gateway_auth.cookie_secure);
    SetString(*section, Name(), "cookie_same_site", options.gateway_auth.cookie_same_site);
    SetBool(*section, Name(), "require_session_record", options.gateway_auth.require_session_record);
    SetBool(*section, Name(), "auto_provision_session", options.gateway_auth.auto_provision_session);
    SetBool(*section, Name(), "enable_dev_registration", options.gateway_auth.enable_dev_registration);
    SetString(*section, Name(), "session_store_backend", options.gateway_auth.session_store_backend);
    SetString(*section, Name(), "session_database_path", options.gateway_auth.session_database_path);
    SetString(*section, Name(), "redis_host", options.gateway_auth.redis_host);
    SetString(*section, Name(), "redis_port", options.gateway_auth.redis_port);
    SetString(*section, Name(), "redis_password", options.gateway_auth.redis_password);
    SetInt(*section, Name(), "redis_pool_size", options.gateway_auth.redis_pool_size, 1, 1024);
    SetInt(*section, Name(), "redis_command_timeout_ms", options.gateway_auth.redis_command_timeout_ms, 1, 60000);
    SetString(*section, Name(), "redis_key_prefix", options.gateway_auth.redis_key_prefix);
    SetBool(*section, Name(), "generate_dev_keys", options.gateway_auth.generate_dev_keys);
}

bool GatewayAuthConfigSection::LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const {
    CliArgumentParser parser(cursor);

    CONFIG_FLAG_ARG(kEnabled, options.gateway_auth.enabled = true;)
    CONFIG_FLAG_ARG(kAllowDevIdentity, options.gateway_auth.allow_dev_identity = true;)
    CONFIG_FLAG_ARG(kRequireAuthForApi, {
        options.gateway_auth.enabled = true;
        options.gateway_auth.require_auth_for_api = true;
        options.gateway_auth.allow_dev_identity = false;
    })
    CONFIG_VALUE_ARG(kCookieName, value, options.gateway_auth.cookie_name = *value;)
    CONFIG_VALUE_ARG(kPublicKeyFile, value, options.gateway_auth.public_key_file = *value;)
    CONFIG_VALUE_ARG(kPrivateKeyFile, value, options.gateway_auth.private_key_file = *value;)
    CONFIG_VALUE_ARG(kIssuer, value, options.gateway_auth.issuer = *value;)
    CONFIG_VALUE_ARG(kAudience, value, options.gateway_auth.audience = *value;)
    CONFIG_VALUE_ARG(kClockSkew, value, {
        options.gateway_auth.clock_skew_seconds = ParseNonNegativeOption(kClockSkew, *value);
    })
    CONFIG_VALUE_ARG(kTokenTtl, value, {
        options.gateway_auth.token_ttl_seconds = ParseNonNegativeOption(kTokenTtl, *value);
    })
    CONFIG_VALUE_ARG(kSessionStoreBackend, value, options.gateway_auth.session_store_backend = *value;)
    CONFIG_VALUE_ARG(kSessionDb, value, options.gateway_auth.session_database_path = *value;)
    CONFIG_VALUE_ARG(kRedisHost, value, options.gateway_auth.redis_host = *value;)
    CONFIG_VALUE_ARG(kRedisPort, value, options.gateway_auth.redis_port = *value;)
    CONFIG_FLAG_ARG(kRequireSessionRecord, {
        options.gateway_auth.require_session_record = true;
        options.gateway_auth.auto_provision_session = false;
    })
    CONFIG_FLAG_ARG(kAutoProvisionSession, options.gateway_auth.auto_provision_session = true;)
    CONFIG_FLAG_ARG(kEnableDevRegistration, options.gateway_auth.enable_dev_registration = true;)
    CONFIG_FLAG_ARG(kCookieSecure, options.gateway_auth.cookie_secure = true;)

    return false;
}

void GatewayAuthConfigSection::Validate(MultimodalServerOptions& options) const {
    auto& auth = options.gateway_auth;
    if (auth.cookie_name.empty()) {
        throw std::runtime_error("gateway_auth.cookie_name must not be empty");
    }
    if (auth.clock_skew_seconds < 0) {
        throw std::runtime_error("gateway_auth.clock_skew_seconds must be non-negative");
    }
    if (auth.token_ttl_seconds <= 0) {
        throw std::runtime_error("gateway_auth.token_ttl_seconds must be positive");
    }
    if (!auth.public_key_file.empty()) {
        const auto path = ResolveRelativeToConfig(auth.public_key_file, options.config_file_path);
        auth.public_key_pem = ReadTextFile(path, "gateway_auth.public_key_file");
    }
    if (!auth.private_key_file.empty()) {
        const auto path = ResolveRelativeToConfig(auth.private_key_file, options.config_file_path);
        auth.private_key_pem = ReadTextFile(path, "gateway_auth.private_key_file");
    }
    if (auth.enabled && auth.public_key_pem.empty() && !auth.generate_dev_keys) {
        throw std::runtime_error("gateway_auth.public_key_pem or public_key_file is required when gateway auth is enabled");
    }
    if (auth.enabled && auth.private_key_pem.empty() && !auth.generate_dev_keys) {
        throw std::runtime_error("gateway_auth.private_key_pem or private_key_file is required when gateway auth is enabled");
    }
    if (auth.session_store_backend.empty()) {
        auth.session_store_backend = "sqlite";
    }
    if (auth.session_store_backend != "sqlite" && auth.session_store_backend != "redis") {
        throw std::runtime_error("gateway_auth.session_store_backend must be sqlite or redis");
    }
    if (auth.redis_pool_size <= 0) {
        throw std::runtime_error("gateway_auth.redis_pool_size must be positive");
    }
    if (auth.redis_command_timeout_ms <= 0) {
        throw std::runtime_error("gateway_auth.redis_command_timeout_ms must be positive");
    }
    if (auth.session_store_backend == "redis" && auth.redis_key_prefix.empty()) {
        throw std::runtime_error("gateway_auth.redis_key_prefix must not be empty when Redis auth store is used");
    }
    if (auth.require_session_record && auth.session_store_backend == "sqlite" && auth.session_database_path.empty()) {
        throw std::runtime_error("gateway_auth.session_database_path is required when require_session_record is true");
    }
    if (!auth.session_database_path.empty()) {
        auth.session_database_path = ResolveRelativeToConfig(auth.session_database_path, options.config_file_path).string();
    }
}

} // namespace

REGISTER_CONFIG_SECTION(GatewayAuthConfigSection)

} // namespace server_config
