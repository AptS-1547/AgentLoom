#include "vector_cache.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace vlm_cache {
namespace {

int64_t NowMs() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

} // namespace

VectorIndex::VectorIndex(VectorOptions options)
    : options_(std::move(options)) {
    if (options_.persist) {
        spdlog::warn("[VectorIndex] persist=true but persistence is not yet implemented; "
                 "vector index will be memory-only");
    }
}

void VectorIndex::Normalize(std::vector<float>& v) {
    float norm_sq = 0.0f;
    for (float x : v) norm_sq += x * x;
    if (norm_sq < 1e-12f) return;
    const float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (float& x : v) x *= inv_norm;
}

float VectorIndex::CosineSim(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

void VectorIndex::Put(std::string_view bucket_key, VectorEntry entry) {
    if (!options_.enabled || entry.embedding.empty()) return;

    Normalize(entry.embedding);
    if (entry.created_at_ms <= 0) {
        entry.created_at_ms = NowMs();
    }

    std::lock_guard lock(mutex_);
    auto& bucket = buckets_[std::string(bucket_key)];

    for (auto& existing : bucket) {
        if (existing.cache_key == entry.cache_key) {
            existing = std::move(entry);
            return;
        }
    }

    bucket.push_back(std::move(entry));

    if (bucket.size() > options_.max_entries_per_bucket) {
        EvictExpiredLocked(bucket, NowMs());
    }
    if (bucket.size() > options_.max_entries_per_bucket) {
        bucket.erase(bucket.begin());
    }
}

std::optional<VectorHit> VectorIndex::Query(
    std::string_view bucket_key,
    std::string_view model_fingerprint,
    const std::vector<float>& embedding,
    float saliency_hint) {
    if (!options_.enabled || embedding.empty()) return std::nullopt;

    std::vector<float> query = embedding;
    Normalize(query);

    std::lock_guard lock(mutex_);
    auto it = buckets_.find(std::string(bucket_key));
    if (it == buckets_.end() || it->second.empty()) return std::nullopt;

    const int64_t now = NowMs();
    float best_sim = 0.0f;
    const VectorEntry* best_entry = nullptr;

    for (const auto& entry : it->second) {
        if (!model_fingerprint.empty() && entry.model_fingerprint != model_fingerprint) {
            continue;
        }
        if (options_.ttl_seconds > 0 && entry.created_at_ms > 0) {
            if (now - entry.created_at_ms > options_.ttl_seconds * 1000) {
                continue;
            }
        }
        const float sim = CosineSim(query, entry.embedding);
        if (sim > best_sim) {
            best_sim = sim;
            best_entry = &entry;
        }
    }

    if (!best_entry) return std::nullopt;

    if (best_sim >= options_.sim_threshold_high) {
        spdlog::debug("[VectorIndex] high-confidence hit: sim={:.4f} key={}", best_sim, best_entry->cache_key);
        return VectorHit{best_entry->cache_key, best_sim, false};
    }

    if (best_sim >= options_.sim_threshold_mid) {
        const bool saliency_ok = (saliency_hint <= 0.0f) ||
                                  (saliency_hint < options_.max_saliency_for_mid);
        if (saliency_ok) {
            spdlog::debug("[VectorIndex] tentative hit: sim={:.4f} saliency={:.3f} key={}",
                      best_sim, saliency_hint, best_entry->cache_key);
            return VectorHit{best_entry->cache_key, best_sim, true};
        }
        spdlog::debug("[VectorIndex] mid-band rejected by saliency: sim={:.4f} saliency={:.3f}",
                  best_sim, saliency_hint);
    }

    return std::nullopt;
}

void VectorIndex::Evict(std::string_view bucket_key, const std::string& cache_key) {
    std::lock_guard lock(mutex_);
    auto it = buckets_.find(std::string(bucket_key));
    if (it == buckets_.end()) return;

    auto& bucket = it->second;
    bucket.erase(
        std::remove_if(bucket.begin(), bucket.end(),
                       [&](const VectorEntry& e) { return e.cache_key == cache_key; }),
        bucket.end());
}

const VectorOptions& VectorIndex::options() const {
    return options_;
}

size_t VectorIndex::size() const {
    std::lock_guard lock(mutex_);
    size_t total = 0;
    for (const auto& [_, bucket] : buckets_) {
        total += bucket.size();
    }
    return total;
}

void VectorIndex::EvictExpiredLocked(std::vector<VectorEntry>& bucket, int64_t now_ms) {
    if (options_.ttl_seconds <= 0) return;
    const int64_t ttl_ms = options_.ttl_seconds * 1000;
    bucket.erase(
        std::remove_if(bucket.begin(), bucket.end(),
                       [&](const VectorEntry& e) {
                           return e.created_at_ms > 0 && now_ms - e.created_at_ms > ttl_ms;
                       }),
        bucket.end());
}

} // namespace vlm_cache
