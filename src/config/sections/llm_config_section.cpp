#include "config_section.h"

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

DECLARE_CONFIG_SECTION(LlmConfigSection, "llm")
    CONFIG_CLI_STRING(kBaseUrl, "--llm-base-url");
    CONFIG_CLI_STRING(kApiKeyEnv, "--llm-api-key-env");
    CONFIG_CLI_STRING(kApiKeyFile, "--llm-api-key-file");
    CONFIG_CLI_STRING(kModel, "--llm-model");
    CONFIG_CLI_STRING(kTimeout, "--llm-timeout");
    CONFIG_CLI_STRING(kMaxRetries, "--llm-max-retries");
    void Validate(MultimodalServerOptions& options) const override;
};

void LlmConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) {
        return;
    }

    SetString(*section, Name(), "base_url", options.llm.base_url);
    SetBool(*section, Name(), "enabled", options.llm.enabled);
    SetString(*section, Name(), "api_key_env", options.llm.api_key_env);
    SetString(*section, Name(), "api_key_file", options.llm.api_key_file);
    SetString(*section, Name(), "model", options.llm.model);
    SetInt(*section, Name(), "timeout_ms", options.llm.timeout_ms, 1000, 600000);
    SetInt(*section, Name(), "max_retries", options.llm.max_retries, 0, 10);
    SetBool(*section, Name(), "require_api_key", options.llm.require_api_key);
    SetBool(*section, Name(), "allow_placeholder", options.llm.allow_placeholder);
    SetBool(*section, Name(), "disable_tls_verify_on_windows", options.llm.disable_tls_verify_on_windows);
    SetString(*section, Name(), "ca_bundle_path", options.llm.ca_bundle_path);

    // Paths are stored as-is (relative); resolution happens in LlmPromptStore::Load.
    if (const Json* prompts_field = FindField(*section, Name(), "prompts")) {
        if (!prompts_field->is_object()) {
            throw std::runtime_error("llm.prompts must be an object");
        }
        for (auto it = prompts_field->begin(); it != prompts_field->end(); ++it) {
            if (!it.value().is_string()) {
                throw std::runtime_error("llm.prompts values must be strings");
            }
            options.llm.prompts[it.key()] = it.value().get<std::string>();
        }
    }
}

bool LlmConfigSection::LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const {
    CliArgumentParser parser(cursor);

    CONFIG_VALUE_ARG(kBaseUrl, value, options.llm.base_url = *value;)
    CONFIG_VALUE_ARG(kApiKeyEnv, value, options.llm.api_key_env = *value;)
    CONFIG_VALUE_ARG(kApiKeyFile, value, options.llm.api_key_file = *value;)
    CONFIG_VALUE_ARG(kModel, value, options.llm.model = *value;)
    CONFIG_VALUE_ARG(kTimeout, value, {
        options.llm.timeout_ms = ParsePositiveOption(kTimeout, *value);
    })
    CONFIG_VALUE_ARG(kMaxRetries, value, {
        options.llm.max_retries = ParseNonNegativeOption(kMaxRetries, *value);
    })

    return false;
}

// Resolve api_key: env var → file → error (if base_url is set).
// Mirrors the auth section's token resolution pattern.
namespace {

std::optional<std::string> ReadKeyFromEnv(const std::string& name) {
    if (name.empty()) return std::nullopt;
    const char* val = std::getenv(name.c_str());
    if (!val || *val == '\0') return std::nullopt;
    return std::string(val);
}

std::string ReadKeyFromFile(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open LLM api_key_file: " + path.string());
    }
    std::string key((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    // Trim whitespace / BOM
    while (!key.empty() && (std::isspace(static_cast<unsigned char>(key.back())))) {
        key.pop_back();
    }
    auto first = key.find_first_not_of(" \t\r\n");
    if (first != std::string::npos) key = key.substr(first);
    return key;
}

std::filesystem::path ResolveRelativeToConfig(const std::filesystem::path& p,
                                               const std::filesystem::path& config_path) {
    if (p.is_absolute() || config_path.empty()) {
        return p;
    }
    return config_path.parent_path() / p;
}

}  // namespace

void LlmConfigSection::Validate(MultimodalServerOptions& options) const {
    // LLM client is optional — skip validation if base_url is not configured.
    if (!options.llm.enabled || options.llm.base_url.empty()) {
        return;
    }

    if (options.llm.model.empty()) {
        throw std::runtime_error("llm.model must not be empty when base_url is set");
    }

    // api_key_env → api_key_file → error
    if (auto key = ReadKeyFromEnv(options.llm.api_key_env)) {
        options.llm.api_key = *key;
        return;
    }
    if (!options.llm.api_key_file.empty()) {
        auto path = ResolveRelativeToConfig(options.llm.api_key_file,
                                              options.config_file_path);
        options.llm.api_key = ReadKeyFromFile(path);
        if (options.llm.api_key.empty()) {
            throw std::runtime_error("LLM api_key_file resolved to an empty key");
        }
        return;
    }

    if (options.llm.require_api_key) {
        throw std::runtime_error(
            "LLM base_url is set but no API key found. "
            "Set env var " + options.llm.api_key_env +
            " or set llm.api_key_file in config.");
    }
}

} // namespace

REGISTER_CONFIG_SECTION(LlmConfigSection)

} // namespace server_config
