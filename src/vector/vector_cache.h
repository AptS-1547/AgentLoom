#pragma once

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

struct VectorOptions {
    bool enabled = false;
    float sim_threshold_high = 0.97f;
    float sim_threshold_mid = 0.93f;
    float max_saliency_for_mid = 0.15f;
    size_t max_entries_per_bucket = 256;
    int64_t ttl_seconds = 3600;
    bool persist = false;
    std::filesystem::path vector_dir = "cache/vlm/vectors";
};

struct VectorEntry {
    std::string cache_key;
    std::string model_fingerprint;
    std::string session_id;
    std::vector<float> embedding;
    int64_t created_at_ms = 0;
};

struct VectorHit {
    std::string cache_key;
    float similarity = 0.0f;
    bool tentative = false;
    bool same_session = false;
};

enum class VectorSessionScope {
    PreferSameSession,
    SameSessionOnly,
    CrossSessionOnly,
};

class VectorIndex {
public:
    explicit VectorIndex(VectorOptions options);

    void Put(std::string_view bucket_key, VectorEntry entry);

    std::optional<VectorHit> Query(
        std::string_view bucket_key,
        std::string_view model_fingerprint,
        const std::vector<float>& embedding,
        float saliency_hint,
        std::string_view preferred_session_id = {},
        VectorSessionScope session_scope = VectorSessionScope::PreferSameSession);

    void Evict(std::string_view bucket_key, const std::string& cache_key);

    const VectorOptions& options() const;
    size_t size() const;

private:
    static float CosineSim(const std::vector<float>& a, const std::vector<float>& b);
    static void Normalize(std::vector<float>& v);
    void EvictExpiredLocked(std::vector<VectorEntry>& bucket, int64_t now_ms);

    VectorOptions options_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::vector<VectorEntry>> buckets_;
};

} // namespace vlm_cache
