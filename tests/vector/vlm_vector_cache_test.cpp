#include "vector_cache.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace {

vlm_cache::VectorOptions TestOptions() {
    vlm_cache::VectorOptions options;
    options.enabled = true;
    options.sim_threshold_high = 0.99f;
    options.sim_threshold_mid = 0.90f;
    options.max_saliency_for_mid = 0.5f;
    options.ttl_seconds = 0;
    return options;
}

vlm_cache::VectorEntry Entry(
    std::string cache_key,
    std::string session_id,
    std::vector<float> embedding) {
    vlm_cache::VectorEntry entry;
    entry.cache_key = std::move(cache_key);
    entry.model_fingerprint = "model";
    entry.session_id = std::move(session_id);
    entry.embedding = std::move(embedding);
    return entry;
}

TEST(VlmVectorCacheTest, PrefersQualifiedSameSessionCandidate) {
    vlm_cache::VectorIndex index(TestOptions());
    index.Put("prompt", Entry("same", "session-a", {0.95f, 0.3122499f}));
    index.Put("prompt", Entry("cross", "session-b", {1.0f, 0.0f}));

    auto hit = index.Query(
        "prompt",
        "model",
        {1.0f, 0.0f},
        0.0f,
        "session-a");

    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->cache_key, "same");
    EXPECT_TRUE(hit->same_session);
    EXPECT_TRUE(hit->tentative);
}

TEST(VlmVectorCacheTest, FallsBackToCrossSessionWhenSameSessionIsBelowThreshold) {
    vlm_cache::VectorIndex index(TestOptions());
    index.Put("prompt", Entry("same", "session-a", {0.0f, 1.0f}));
    index.Put("prompt", Entry("cross", "session-b", {1.0f, 0.0f}));

    auto hit = index.Query(
        "prompt",
        "model",
        {1.0f, 0.0f},
        0.0f,
        "session-a");

    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->cache_key, "cross");
    EXPECT_FALSE(hit->same_session);
    EXPECT_FALSE(hit->tentative);
}

TEST(VlmVectorCacheTest, SessionScopesDoNotMixCandidates) {
    vlm_cache::VectorIndex index(TestOptions());
    index.Put("prompt", Entry("same", "session-a", {1.0f, 0.0f}));
    index.Put("prompt", Entry("cross", "session-b", {1.0f, 0.0f}));

    auto same = index.Query(
        "prompt",
        "model",
        {1.0f, 0.0f},
        0.0f,
        "session-a",
        vlm_cache::VectorSessionScope::SameSessionOnly);
    auto cross = index.Query(
        "prompt",
        "model",
        {1.0f, 0.0f},
        0.0f,
        "session-a",
        vlm_cache::VectorSessionScope::CrossSessionOnly);

    ASSERT_TRUE(same.has_value());
    ASSERT_TRUE(cross.has_value());
    EXPECT_EQ(same->cache_key, "same");
    EXPECT_TRUE(same->same_session);
    EXPECT_EQ(cross->cache_key, "cross");
    EXPECT_FALSE(cross->same_session);
}

} // namespace
