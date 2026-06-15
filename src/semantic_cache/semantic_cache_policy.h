#pragma once

#include "semantic_cache_types.h"
#include "result.h"

#include <chrono>
#include <string>

namespace agent::semantic_cache {

/// Fingerprint identifying the policy / model / corpus version that an
/// entry was produced under.  Two entries with different fingerprints must
/// not be considered interchangeable, even if their texts are similar.
struct CacheEntryFingerprint {
    std::string tokenizer_version;
    std::string embedding_model_version;
    std::string corpus_version;
    std::string policy_version;
    std::int32_t embedding_dimension = 0;
};

/// Metadata describing a candidate entry returned by the index.  Carried
/// alongside the vector match so the policy layer can decide whether the
/// match is reusable for the current request.
struct CacheEntryMetadata {
    std::int64_t entry_id = 0;
    CacheScope scope = CacheScope::Global;
    AnswerType answer_type = AnswerType::Generic;

    std::string tenant_id;
    std::string user_id;
    std::string session_id;

    std::string subject;
    std::string grade;
    std::string topic;
    std::string persona_id;

    float quality_score = 0.0f;
    std::int64_t created_at_ms = 0;
    std::int64_t expires_at_ms = 0;

    CacheEntryFingerprint fingerprint;
};

/// Decides whether a candidate entry is safe to reuse for a given request.
///
/// Rules to enforce (see roadmap §4 / §5):
///   - global semantic cache != user/session memory
///   - tenant / subject / grade / topic / persona compatibility
///   - fingerprint (tokenizer / model / corpus / policy / dim) match
///   - quality / expiry / answer_type alignment
///   - private user memory may only hit user-scoped entries
class IPolicyMatcher {
public:
    virtual ~IPolicyMatcher() = default;

    /// Returns Ok with `true` if the entry is reusable for the request.
    /// Implementations should fail closed: when in doubt, return `false`.
    virtual core::Result<bool> Matches(const CacheLookupRequest& req,
                                        const CacheEntryMetadata& entry) = 0;
};

struct DefaultPolicyMatcherOptions {
    CacheEntryFingerprint expected_fingerprint;
    float min_quality_score = 0.0f;
    bool require_fingerprint = false;
};

class DefaultPolicyMatcher final : public IPolicyMatcher {
public:
    explicit DefaultPolicyMatcher(DefaultPolicyMatcherOptions options = {});

    core::Result<bool> Matches(const CacheLookupRequest& req,
                               const CacheEntryMetadata& entry) override;

private:
    DefaultPolicyMatcherOptions options_;
};

}  // namespace agent::semantic_cache
