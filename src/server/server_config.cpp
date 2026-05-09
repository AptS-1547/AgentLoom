#include "server_config.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace server_config {
namespace {

using json = nlohmann::json;

class IConfigSection {
public:
    virtual ~IConfigSection() = default;
    virtual std::string_view Name() const = 0;
    virtual void Load(const json& root, MultimodalServerOptions& options) const = 0;
};

const json* FindSection(const json& root, std::string_view name) {
    auto it = root.find(std::string(name));
    if (it == root.end()) {
        return nullptr;
    }
    if (!it->is_object()) {
        throw std::runtime_error(std::string(name) + " must be an object");
    }
    return &*it;
}

const json* FindField(const json& section, std::string_view section_name, std::string_view field_name) {
    auto it = section.find(std::string(field_name));
    if (it == section.end()) {
        return nullptr;
    }
    if (it->is_null()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must not be null");
    }
    return &*it;
}

void SetString(
    const json& section,
    std::string_view section_name,
    std::string_view field_name,
    std::string& target) {
    const json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    if (!value->is_string()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be a string");
    }
    target = value->get<std::string>();
}

void SetPath(
    const json& section,
    std::string_view section_name,
    std::string_view field_name,
    std::filesystem::path& target) {
    const json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    if (!value->is_string()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be a string");
    }
    target = value->get<std::string>();
}

int64_t ReadInteger(
    const json& value,
    std::string_view section_name,
    std::string_view field_name,
    int64_t min_value,
    int64_t max_value) {
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be an integer");
    }
    int64_t parsed = 0;
    try {
        parsed = value.get<int64_t>();
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " is out of range");
    }
    if (parsed < min_value || parsed > max_value) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " is out of range");
    }
    return parsed;
}

void SetInt(
    const json& section,
    std::string_view section_name,
    std::string_view field_name,
    int& target,
    int min_value = std::numeric_limits<int>::min(),
    int max_value = std::numeric_limits<int>::max()) {
    const json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    target = static_cast<int>(ReadInteger(*value, section_name, field_name, min_value, max_value));
}

void SetInt32(
    const json& section,
    std::string_view section_name,
    std::string_view field_name,
    int32_t& target,
    int32_t min_value = std::numeric_limits<int32_t>::min(),
    int32_t max_value = std::numeric_limits<int32_t>::max()) {
    const json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    target = static_cast<int32_t>(ReadInteger(*value, section_name, field_name, min_value, max_value));
}

void SetInt64(
    const json& section,
    std::string_view section_name,
    std::string_view field_name,
    int64_t& target,
    int64_t min_value = std::numeric_limits<int64_t>::min(),
    int64_t max_value = std::numeric_limits<int64_t>::max()) {
    const json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    target = ReadInteger(*value, section_name, field_name, min_value, max_value);
}

void SetSize(
    const json& section,
    std::string_view section_name,
    std::string_view field_name,
    size_t& target,
    size_t min_value = 0,
    size_t max_value = std::numeric_limits<size_t>::max()) {
    const json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    const int64_t parsed = ReadInteger(
        *value,
        section_name,
        field_name,
        static_cast<int64_t>(std::min<size_t>(min_value, static_cast<size_t>(std::numeric_limits<int64_t>::max()))),
        static_cast<int64_t>(std::min<size_t>(max_value, static_cast<size_t>(std::numeric_limits<int64_t>::max()))));
    target = static_cast<size_t>(parsed);
}

void SetUInt32(
    const json& section,
    std::string_view section_name,
    std::string_view field_name,
    uint32_t& target,
    uint32_t min_value = 0,
    uint32_t max_value = std::numeric_limits<uint32_t>::max()) {
    const json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    target = static_cast<uint32_t>(ReadInteger(*value, section_name, field_name, min_value, max_value));
}

void SetFloat(
    const json& section,
    std::string_view section_name,
    std::string_view field_name,
    float& target,
    float min_value,
    float max_value) {
    const json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    if (!value->is_number()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be a number");
    }
    const double parsed = value->get<double>();
    if (!std::isfinite(parsed) || parsed < min_value || parsed > max_value) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " is out of range");
    }
    target = static_cast<float>(parsed);
}

void SetBool(
    const json& section,
    std::string_view section_name,
    std::string_view field_name,
    bool& target) {
    const json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    if (!value->is_boolean()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be a boolean");
    }
    target = value->get<bool>();
}

size_t MegabytesToBytes(int64_t mb, std::string_view section_name, std::string_view field_name) {
    if (mb < 0) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be non-negative");
    }
    const auto max_mb = static_cast<int64_t>(std::numeric_limits<size_t>::max() / (1024ULL * 1024ULL));
    if (mb > max_mb) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " is out of range");
    }
    return static_cast<size_t>(mb) * 1024ULL * 1024ULL;
}

void SetMegabytes(
    const json& section,
    std::string_view section_name,
    std::string_view field_name,
    size_t& target,
    int64_t min_mb = 0) {
    const json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    const int64_t parsed = ReadInteger(
        *value,
        section_name,
        field_name,
        min_mb,
        static_cast<int64_t>(std::numeric_limits<size_t>::max() / (1024ULL * 1024ULL)));
    target = MegabytesToBytes(parsed, section_name, field_name);
}

class ModelsConfigSection final : public IConfigSection {
public:
    std::string_view Name() const override {
        return "models";
    }

    void Load(const json& root, MultimodalServerOptions& options) const override {
        const json* section = FindSection(root, Name());
        if (!section) {
            return;
        }
        SetString(*section, Name(), "llm", options.llm_model);
        SetString(*section, Name(), "mmproj", options.mmproj);
        SetString(*section, Name(), "bert", options.bert_model);
        SetString(*section, Name(), "vit", options.vit_model);
        SetInt(*section, Name(), "n_gpu_layers", options.n_gpu_layers);
        SetString(*section, Name(), "provider", options.bert_runtime.execution_provider);
        SetInt(*section, Name(), "cuda_device", options.bert_runtime.cuda_device_id, 0);
    }
};

class GrpcConfigSection final : public IConfigSection {
public:
    std::string_view Name() const override {
        return "grpc";
    }

    void Load(const json& root, MultimodalServerOptions& options) const override {
        const json* section = FindSection(root, Name());
        if (!section) {
            return;
        }
        SetString(*section, Name(), "host", options.grpc.host);
        SetString(*section, Name(), "port", options.grpc.port);
        SetString(*section, Name(), "log_dir", options.grpc.log_dir);
        SetInt(*section, Name(), "num_cqs", options.grpc.grpc_num_cqs, 0);
        SetInt(*section, Name(), "min_pollers", options.grpc.grpc_min_pollers, 0);
        SetInt(*section, Name(), "max_pollers", options.grpc.grpc_max_pollers, 0);
        SetInt(*section, Name(), "max_receive_message_mb", options.grpc.max_receive_message_mb, 1);
        SetInt(*section, Name(), "max_send_message_mb", options.grpc.max_send_message_mb, 1);
        SetInt(*section, Name(), "stats_log_interval_seconds", options.grpc.stats_log_interval_seconds, 0);
        SetInt(*section, Name(), "slow_request_ms", options.grpc.slow_request_ms, 0);
    }
};

class AuthConfigSection final : public IConfigSection {
public:
    std::string_view Name() const override {
        return "auth";
    }

    void Load(const json& root, MultimodalServerOptions& options) const override {
        const json* section = FindSection(root, Name());
        if (!section) {
            return;
        }
        SetString(*section, Name(), "token", options.auth.token);
        SetString(*section, Name(), "token_file", options.auth_token_file);
        SetString(*section, Name(), "token_env", options.auth_token_env);
        SetString(*section, Name(), "metadata_key", options.auth.metadata_key);
    }
};

class LimitsConfigSection final : public IConfigSection {
public:
    std::string_view Name() const override {
        return "limits";
    }

    void Load(const json& root, MultimodalServerOptions& options) const override {
        const json* section = FindSection(root, Name());
        if (!section) {
            return;
        }
        SetMegabytes(*section, Name(), "max_image_mb", options.limits.max_image_bytes, 1);
        SetSize(*section, Name(), "max_image_bytes", options.limits.max_image_bytes, 1);
        SetSize(*section, Name(), "max_image_pixels", options.limits.max_image_pixels, 1);
        SetUInt32(*section, Name(), "max_image_width", options.limits.max_image_width, 1);
        SetUInt32(*section, Name(), "max_image_height", options.limits.max_image_height, 1);
        SetSize(*section, Name(), "max_prompt_bytes", options.limits.max_prompt_bytes, 1);
        SetInt32(*section, Name(), "min_context_size", options.limits.min_context_size, 1);
        SetInt32(*section, Name(), "max_context_size", options.limits.max_context_size, 1);
        SetInt32(*section, Name(), "max_vlm_tokens", options.limits.max_vlm_tokens, 1);
        SetFloat(*section, Name(), "max_temperature", options.limits.max_temperature, 0.0f, 100.0f);
        SetInt32(*section, Name(), "max_top_k", options.limits.max_top_k, 0);
        SetInt32(*section, Name(), "max_sequence_length", options.limits.max_sequence_length, 1);
        SetInt32(*section, Name(), "max_batch_size", options.limits.max_batch_size, 1);
        SetInt64(*section, Name(), "max_token_id", options.limits.max_token_id, 1);
        SetFloat(*section, Name(), "max_abs_personality", options.limits.max_abs_personality, 0.0f, 1000000.0f);
    }
};

class VramGuardConfigSection final : public IConfigSection {
public:
    std::string_view Name() const override {
        return "vram_guard";
    }

    void Load(const json& root, MultimodalServerOptions& options) const override {
        const json* section = FindSection(root, Name());
        if (!section) {
            return;
        }
        SetInt(*section, Name(), "monitor_interval_seconds", options.vram.monitor_interval_seconds, 0);
        SetMegabytes(*section, Name(), "warning_free_mb", options.vram.warning_free_bytes);
        SetMegabytes(*section, Name(), "unload_free_mb", options.vram.unload_free_bytes);
        SetMegabytes(*section, Name(), "min_free_before_load_mb", options.vram.min_free_before_load_bytes);
        SetBool(*section, Name(), "reload_after_unload", options.vram.reload_after_unload);
        SetBool(*section, Name(), "unload_on_oom_error", options.vram.unload_on_oom_error);
    }
};

class VlmCacheConfigSection final : public IConfigSection {
public:
    std::string_view Name() const override {
        return "vlm_cache";
    }

    void Load(const json& root, MultimodalServerOptions& options) const override {
        const json* section = FindSection(root, Name());
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
    }
};

std::vector<std::unique_ptr<IConfigSection>> BuildSections() {
    std::vector<std::unique_ptr<IConfigSection>> sections;
    sections.push_back(std::make_unique<ModelsConfigSection>());
    sections.push_back(std::make_unique<GrpcConfigSection>());
    sections.push_back(std::make_unique<AuthConfigSection>());
    sections.push_back(std::make_unique<LimitsConfigSection>());
    sections.push_back(std::make_unique<VramGuardConfigSection>());
    sections.push_back(std::make_unique<VlmCacheConfigSection>());
    return sections;
}

}

std::optional<std::filesystem::path> FindConfigPath(int argc, char** argv) {
    std::optional<std::filesystem::path> path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--config requires a value");
            }
            path = argv[++i];
        } else if (arg.rfind("--config=", 0) == 0) {
            path = arg.substr(std::string("--config=").size());
        }
    }
    return path;
}

void LoadConfigFile(const std::filesystem::path& path, MultimodalServerOptions& options) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open config file: " + path.string());
    }

    json root;
    try {
        file >> root;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse config file " + path.string() + ": " + e.what());
    }

    if (!root.is_object()) {
        throw std::runtime_error("Config file root must be an object");
    }

    for (const auto& section : BuildSections()) {
        section->Load(root, options);
    }
}

}
