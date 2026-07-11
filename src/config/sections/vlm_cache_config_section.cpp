#include "config_section.h"

#include <cstddef>
#include <stdexcept>

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(VlmCacheConfigSection, "vlm_cache")
    CONFIG_CLI_STRING(kEnabled, "--vlm-cache-enabled");
    CONFIG_CLI_STRING(kPersist, "--vlm-cache-persist");
    CONFIG_CLI_STRING(kDir, "--vlm-cache-dir");
    CONFIG_CLI_STRING(kMaxEntries, "--vlm-cache-max-entries");
    CONFIG_CLI_STRING(kMaxMb, "--vlm-cache-max-mb");
    CONFIG_CLI_STRING(kTtlSeconds, "--vlm-cache-ttl-seconds");
    CONFIG_CLI_STRING(kNoStoreImages, "--no-vlm-cache-store-images");
    CONFIG_CLI_STRING(kNoStorePrompts, "--no-vlm-cache-store-prompts");
    CONFIG_CLI_STRING(kNoStaleOnFailure, "--no-vlm-cache-stale-on-failure");
    CONFIG_CLI_STRING(kNoDefault, "--no-vlm-cache-default");
    CONFIG_CLI_STRING(kPromptKvEnabled, "--vlm-prompt-kv-cache-enabled");
    CONFIG_CLI_STRING(kPromptKvBackend, "--vlm-prompt-kv-backend");
    CONFIG_CLI_STRING(kPromptKvRedisHost, "--vlm-prompt-kv-redis-host");
    CONFIG_CLI_STRING(kPromptKvRedisPort, "--vlm-prompt-kv-redis-port");
    CONFIG_CLI_STRING(kPromptKvRedisPassword, "--vlm-prompt-kv-redis-password");
    CONFIG_CLI_STRING(kPromptKvRedisPoolSize, "--vlm-prompt-kv-redis-pool-size");
    CONFIG_CLI_STRING(kPromptKvRedisTimeoutMs, "--vlm-prompt-kv-redis-timeout-ms");
    CONFIG_CLI_STRING(kPromptKvKeyPrefix, "--vlm-prompt-kv-key-prefix");
    CONFIG_CLI_STRING(kPromptKvTtlSeconds, "--vlm-prompt-kv-ttl-seconds");
    CONFIG_CLI_STRING(kPromptKvMaxMb, "--vlm-prompt-kv-max-mb");
    void Validate(MultimodalServerOptions& options) const override;
};

void VlmCacheConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) {
        return;
    }
    SetBool(*section, Name(), "enabled", options.vlm_cache.enabled);
    SetBool(*section, Name(), "persist", options.vlm_cache.persist);
    SetPath(*section, Name(), "dir", options.vlm_cache.cache_dir);
    SetSize(*section, Name(), "max_entries", options.vlm_cache.max_entries);
    SetMegabytes(*section, Name(), "max_mb", options.vlm_cache.max_bytes);
    SetInt64(*section, Name(), "ttl_seconds", options.vlm_cache.ttl_seconds, 0);
    SetBool(*section, Name(), "store_images", options.vlm_cache.store_images);
    SetBool(*section, Name(), "store_prompts", options.vlm_cache.store_prompts);
    SetBool(*section, Name(), "stale_on_failure", options.vlm_cache.allow_stale_on_failure);
    SetBool(*section, Name(), "default_allow_cache", options.vlm_cache.default_allow_cache);

    const Json* vec = FindSection(*section, "vector");
    if (vec) {
        SetBool(*vec, "vlm_cache.vector", "enabled", options.vlm_cache_vector.enabled);
        SetFloat(*vec, "vlm_cache.vector", "sim_threshold_high",
                 options.vlm_cache_vector.sim_threshold_high, 0.0f, 1.0f);
        SetFloat(*vec, "vlm_cache.vector", "sim_threshold_mid",
                 options.vlm_cache_vector.sim_threshold_mid, 0.0f, 1.0f);
        SetFloat(*vec, "vlm_cache.vector", "max_saliency_for_mid",
                 options.vlm_cache_vector.max_saliency_for_mid, 0.0f, 1.0f);
        SetSize(*vec, "vlm_cache.vector", "max_entries_per_bucket",
                options.vlm_cache_vector.max_entries_per_bucket);
        SetInt64(*vec, "vlm_cache.vector", "ttl_seconds",
                 options.vlm_cache_vector.ttl_seconds, 0);
        SetPath(*vec, "vlm_cache.vector", "dir", options.vlm_cache_vector.vector_dir);
    }

    const Json* prompt_kv = FindSection(*section, "prompt_kv");
    if (prompt_kv) {
        SetBool(*prompt_kv, "vlm_cache.prompt_kv", "enabled", options.vlm_prompt_kv_cache.enabled);
        SetString(*prompt_kv, "vlm_cache.prompt_kv", "backend", options.vlm_prompt_kv_cache.backend);
        SetString(*prompt_kv, "vlm_cache.prompt_kv", "redis_host", options.vlm_prompt_kv_cache.redis_host);
        SetInt(*prompt_kv, "vlm_cache.prompt_kv", "redis_port", options.vlm_prompt_kv_cache.redis_port, 1, 65535);
        SetString(*prompt_kv, "vlm_cache.prompt_kv", "redis_password", options.vlm_prompt_kv_cache.redis_password);
        SetSize(*prompt_kv, "vlm_cache.prompt_kv", "redis_pool_size", options.vlm_prompt_kv_cache.redis_pool_size, 1);
        SetInt(*prompt_kv, "vlm_cache.prompt_kv", "redis_command_timeout_ms", options.vlm_prompt_kv_cache.redis_command_timeout_ms, 1);
        SetString(*prompt_kv, "vlm_cache.prompt_kv", "key_prefix", options.vlm_prompt_kv_cache.key_prefix);
        SetInt64(*prompt_kv, "vlm_cache.prompt_kv", "ttl_seconds", options.vlm_prompt_kv_cache.ttl_seconds, 1);
        SetMegabytes(*prompt_kv, "vlm_cache.prompt_kv", "max_mb", options.vlm_prompt_kv_cache.max_bytes, 1);
    }
}

bool VlmCacheConfigSection::LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const {
    CliArgumentParser parser(cursor);
    CONFIG_FLAG_ARG(kEnabled, options.vlm_cache.enabled = true;)
    CONFIG_FLAG_ARG(kPersist, options.vlm_cache.persist = true;)
    CONFIG_VALUE_ARG(kDir, value, options.vlm_cache.cache_dir = *value;)
    CONFIG_VALUE_ARG(kMaxEntries, value, options.vlm_cache.max_entries = static_cast<std::size_t>(ParseNonNegativeOption(kMaxEntries, *value));)
    CONFIG_VALUE_ARG(kMaxMb, value, options.vlm_cache.max_bytes = ParseOptionalMegabytesOption(kMaxMb, *value);)
    CONFIG_VALUE_ARG(kTtlSeconds, value, options.vlm_cache.ttl_seconds = ParseNonNegativeOption(kTtlSeconds, *value);)
    CONFIG_FLAG_ARG(kNoStoreImages, options.vlm_cache.store_images = false;)
    CONFIG_FLAG_ARG(kNoStorePrompts, options.vlm_cache.store_prompts = false;)
    CONFIG_FLAG_ARG(kNoStaleOnFailure, options.vlm_cache.allow_stale_on_failure = false;)
    CONFIG_FLAG_ARG(kNoDefault, options.vlm_cache.default_allow_cache = false;)
    CONFIG_FLAG_ARG(kPromptKvEnabled, options.vlm_prompt_kv_cache.enabled = true;)
    CONFIG_VALUE_ARG(kPromptKvBackend, value, options.vlm_prompt_kv_cache.backend = *value;)
    CONFIG_VALUE_ARG(kPromptKvRedisHost, value, options.vlm_prompt_kv_cache.redis_host = *value;)
    CONFIG_VALUE_ARG(kPromptKvRedisPort, value, options.vlm_prompt_kv_cache.redis_port = ParsePositiveOption(kPromptKvRedisPort, *value);)
    CONFIG_VALUE_ARG(kPromptKvRedisPassword, value, options.vlm_prompt_kv_cache.redis_password = *value;)
    CONFIG_VALUE_ARG(kPromptKvRedisPoolSize, value, options.vlm_prompt_kv_cache.redis_pool_size = static_cast<std::size_t>(ParsePositiveOption(kPromptKvRedisPoolSize, *value));)
    CONFIG_VALUE_ARG(kPromptKvRedisTimeoutMs, value, options.vlm_prompt_kv_cache.redis_command_timeout_ms = ParsePositiveOption(kPromptKvRedisTimeoutMs, *value);)
    CONFIG_VALUE_ARG(kPromptKvKeyPrefix, value, options.vlm_prompt_kv_cache.key_prefix = *value;)
    CONFIG_VALUE_ARG(kPromptKvTtlSeconds, value, options.vlm_prompt_kv_cache.ttl_seconds = ParsePositiveOption(kPromptKvTtlSeconds, *value);)
    CONFIG_VALUE_ARG(kPromptKvMaxMb, value, options.vlm_prompt_kv_cache.max_bytes = ParseMegabytesOption(kPromptKvMaxMb, *value);)
    return false;
}

void VlmCacheConfigSection::Validate(MultimodalServerOptions& options) const {
    if (options.vlm_cache.persist && options.vlm_cache.cache_dir.empty()) {
        throw std::runtime_error("--vlm-cache-dir must not be empty when persistence is enabled");
    }
    if (options.vlm_prompt_kv_cache.enabled && options.vlm_prompt_kv_cache.key_prefix.empty()) {
        throw std::runtime_error("vlm_cache.prompt_kv.key_prefix must not be empty when prompt KV cache is enabled");
    }
    if (options.vlm_prompt_kv_cache.enabled &&
        options.vlm_prompt_kv_cache.backend != "redis" &&
        options.vlm_prompt_kv_cache.backend != "memory") {
        throw std::runtime_error("vlm_cache.prompt_kv.backend must be redis or memory");
    }
}

} // namespace

REGISTER_CONFIG_SECTION(VlmCacheConfigSection)

} // namespace server_config
