#include "config_section.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(AuthConfigSection, "auth")
    CONFIG_CLI_STRING(kToken, "--auth-token");
    CONFIG_CLI_STRING(kTokenFile, "--auth-token-file");
    CONFIG_CLI_STRING(kTokenEnv, "--auth-token-env");
    CONFIG_CLI_STRING(kHeader, "--auth-header");
    void Validate(MultimodalServerOptions& options) const override;
};

void AuthConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) {
        return;
    }
    SetString(*section, Name(), "token", options.auth.token);
    SetString(*section, Name(), "token_file", options.auth_token_file);
    SetString(*section, Name(), "token_env", options.auth_token_env);
    SetString(*section, Name(), "metadata_key", options.auth.metadata_key);
}

bool AuthConfigSection::LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const {
    CliArgumentParser parser(cursor);
    CONFIG_VALUE_ARG(kToken, value, options.auth.token = *value;)
    CONFIG_VALUE_ARG(kTokenFile, value, options.auth_token_file = *value;)
    CONFIG_VALUE_ARG(kTokenEnv, value, options.auth_token_env = *value;)
    CONFIG_VALUE_ARG(kHeader, value, options.auth.metadata_key = *value;)
    return false;
}

std::string TrimToken(std::string value) {
    if (value.size() >= 3 &&
        static_cast<unsigned char>(value[0]) == 0xEF &&
        static_cast<unsigned char>(value[1]) == 0xBB &&
        static_cast<unsigned char>(value[2]) == 0xBF) {
        value.erase(0, 3);
    }

    const auto not_space = [](unsigned char c) {
        return !std::isspace(c);
    };

    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

void ValidateResolvedToken(const std::string& token, std::string_view source) {
    if (token.empty()) {
        throw std::runtime_error("Auth token from " + std::string(source) + " is empty");
    }
    for (unsigned char c : token) {
        if (c == 0 || c < 0x21 || c == 0x7F) {
            throw std::runtime_error("Auth token from " + std::string(source) + " contains invalid characters");
        }
    }
}

std::string ReadAuthTokenFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open auth token file: " + path.string());
    }
    std::string value(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    return TrimToken(std::move(value));
}

std::optional<std::string> ReadAuthTokenEnv(const std::string& name) {
    if (name.empty()) {
        return std::nullopt;
    }
    const char* value = std::getenv(name.c_str());
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return TrimToken(value);
}

void AuthConfigSection::Validate(MultimodalServerOptions& options) const {
    if (options.auth.metadata_key.empty()) {
        throw std::runtime_error("--auth-header must not be empty");
    }

    if (!options.auth.token.empty()) {
        options.auth.token = TrimToken(options.auth.token);
        ValidateResolvedToken(options.auth.token, kToken);
        options.auth_source = "command-line";
        return;
    }

    if (!options.auth_token_file.empty()) {
        options.auth.token = ReadAuthTokenFile(options.auth_token_file);
        ValidateResolvedToken(options.auth.token, kTokenFile);
        options.auth_source = "file";
        return;
    }

    if (auto env_token = ReadAuthTokenEnv(options.auth_token_env)) {
        options.auth.token = *env_token;
        ValidateResolvedToken(options.auth.token, options.auth_token_env);
        options.auth_source = "environment";
        return;
    }

    options.auth_source = "disabled";
}

} // namespace

REGISTER_CONFIG_SECTION(AuthConfigSection)

} // namespace server_config
