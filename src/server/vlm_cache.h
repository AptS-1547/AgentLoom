#pragma once

#include "llama_runner.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vlm_cache {

struct Options {
    bool enabled = false;
    bool persist = false;
    std::filesystem::path cache_dir = "cache/vlm";
    size_t max_entries = 512;
    size_t max_bytes = 1024ULL * 1024ULL * 1024ULL;
    int64_t ttl_seconds = 3600;
    bool store_images = true;
    bool store_prompts = true;
    bool allow_stale_on_failure = true;
    bool default_allow_cache = true;
};

struct KeyInfo {
    std::string cache_key;
    std::string image_sha256;
    std::string prompt_sha256;
};

struct Result {
    std::string cache_key;
    std::string image_sha256;
    std::string prompt_sha256;
    std::string text;
    float image_encode_ms = 0.0f;
    float prompt_eval_ms = 0.0f;
    float eval_ms = 0.0f;
    int32_t prompt_tokens = 0;
    int32_t generated_tokens = 0;
    int64_t created_at_ms = 0;
    int64_t last_hit_at_ms = 0;
    uint64_t hit_count = 0;
};

struct StoreRecord {
    KeyInfo key;
    std::string model_fingerprint;
    std::string session_id;
    std::string request_id;
    std::string task_type;
    std::string prompt;
    std::vector<uint8_t> image_data;
    llm::GenerateParams params;
    Result result;
};

std::string Sha256Hex(const void* data, size_t size);
std::string Sha256Hex(std::string_view data);

class VLMCache {
public:
    explicit VLMCache(Options options);

    KeyInfo BuildKey(
        const std::vector<uint8_t>& image_data,
        std::string_view prompt,
        const llm::GenerateParams& params,
        std::string_view model_fingerprint) const;

    std::optional<Result> Get(const std::string& cache_key);
    std::optional<Result> GetLatestFallback(
        std::string_view session_id,
        std::string_view task_type,
        std::string_view model_fingerprint);
    void Put(StoreRecord record);

    const Options& options() const;
    size_t size() const;

private:
    struct Entry {
        StoreRecord record;
        size_t bytes = 0;
    };

    Result TouchAndMakeResult(Entry& entry, int64_t now_ms);
    Result MakeResult(const Entry& entry) const;
    size_t EstimateBytes(const StoreRecord& record) const;
    bool IsExpired(const Entry& entry, int64_t now_ms) const;
    void EvictLocked();
    void LoadPersisted();
    void PersistEntry(const Entry& entry);
    std::optional<Entry> ReadEntry(const std::filesystem::path& path);
    void RemovePersistedResult(const std::string& cache_key);

    Options options_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> entries_;
    size_t current_bytes_ = 0;
};

}
