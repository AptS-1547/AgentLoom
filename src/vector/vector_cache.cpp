#include "vector_cache.h"

#include "../core/logger_adapter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace vlm_cache {
namespace {

static core::LoggerAdapter logger = core::LoggerAdapter::ForModule("vector");

int64_t NowMs() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

} // namespace

VectorIndex::VectorIndex(VectorOptions options)
    : options_(std::move(options)) {
    if (options_.persist) {
        logger.warn("[VectorIndex] persist=true but persistence is not yet implemented; "
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
    float saliency_hint,
    std::string_view preferred_session_id,
    VectorSessionScope session_scope) {
    if (!options_.enabled || embedding.empty()) return std::nullopt;

    std::vector<float> query = embedding;
    Normalize(query);

    std::lock_guard lock(mutex_);
    auto it = buckets_.find(std::string(bucket_key));
    if (it == buckets_.end() || it->second.empty()) return std::nullopt;

    const int64_t now = NowMs();
    float best_same_session_sim = 0.0f;
    float best_cross_session_sim = 0.0f;
    const VectorEntry* best_same_session = nullptr;
    const VectorEntry* best_cross_session = nullptr;

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
        const bool same_session =
            !preferred_session_id.empty() && entry.session_id == preferred_session_id;
        if ((session_scope == VectorSessionScope::SameSessionOnly && !same_session) ||
            (session_scope == VectorSessionScope::CrossSessionOnly && same_session)) {
            continue;
        }
        if (same_session) {
            if (sim > best_same_session_sim) {
                best_same_session_sim = sim;
                best_same_session = &entry;
            }
        } else if (sim > best_cross_session_sim) {
            best_cross_session_sim = sim;
            best_cross_session = &entry;
        }
    }

    const auto make_hit = [&](const VectorEntry* entry, float similarity, bool same_session)
        -> std::optional<VectorHit> {
        if (!entry) {
            return std::nullopt;
        }
        if (similarity >= options_.sim_threshold_high) {
            logger.debug("[VectorIndex] high-confidence hit: sim={:.4f} key={}", similarity, entry->cache_key);
            return VectorHit{entry->cache_key, similarity, false, same_session};
        }
        if (similarity >= options_.sim_threshold_mid) {
            const bool saliency_ok = (saliency_hint <= 0.0f) ||
                                      (saliency_hint < options_.max_saliency_for_mid);
            if (saliency_ok) {
                logger.debug("[VectorIndex] tentative hit: sim={:.4f} saliency={:.3f} key={}",
                             similarity, saliency_hint, entry->cache_key);
                return VectorHit{entry->cache_key, similarity, true, same_session};
            }
            logger.debug("[VectorIndex] mid-band rejected by saliency: sim={:.4f} saliency={:.3f}",
                         similarity, saliency_hint);
        }
        return std::nullopt;
    };

    if (session_scope != VectorSessionScope::CrossSessionOnly) {
        if (auto hit = make_hit(best_same_session, best_same_session_sim, true)) {
            return hit;
        }
    }
    if (session_scope != VectorSessionScope::SameSessionOnly) {
        return make_hit(best_cross_session, best_cross_session_sim, false);
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
