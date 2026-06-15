#include "semantic_cache_policy.h"

#include <utility>

namespace agent::semantic_cache {

namespace {

bool IsEmptyFingerprint(const CacheEntryFingerprint& fp) {
    return fp.tokenizer_version.empty() &&
           fp.embedding_model_version.empty() &&
           fp.corpus_version.empty() &&
           fp.policy_version.empty() &&
           fp.embedding_dimension == 0;
}

bool SameFingerprint(const CacheEntryFingerprint& lhs, const CacheEntryFingerprint& rhs) {
    return lhs.tokenizer_version == rhs.tokenizer_version &&
           lhs.embedding_model_version == rhs.embedding_model_version &&
           lhs.corpus_version == rhs.corpus_version &&
           lhs.policy_version == rhs.policy_version &&
           lhs.embedding_dimension == rhs.embedding_dimension;
}

bool CompatibleScope(const CacheLookupRequest& req, const CacheEntryMetadata& entry) {
    if (req.scope == CacheScope::Global) {
        return entry.scope == CacheScope::Global;
    }
    if (req.scope == CacheScope::Tenant) {
        return entry.scope == CacheScope::Global ||
               (entry.scope == CacheScope::Tenant && entry.tenant_id == req.tenant_id);
    }
    if (req.scope == CacheScope::User) {
        return entry.scope == CacheScope::Global ||
               (entry.scope == CacheScope::Tenant && entry.tenant_id == req.tenant_id) ||
               (entry.scope == CacheScope::User && entry.user_id == req.user_id);
    }
    if (req.scope == CacheScope::Session) {
        return entry.scope == CacheScope::Session &&
               entry.user_id == req.user_id &&
               entry.session_id == req.session_id;
    }
    if (req.scope == CacheScope::Persona) {
        return entry.scope == CacheScope::Persona && entry.persona_id == req.persona_id;
    }
    return false;
}

bool EqualWhenBothSet(const std::string& lhs, const std::string& rhs) {
    return lhs.empty() || rhs.empty() || lhs == rhs;
}

} // namespace

DefaultPolicyMatcher::DefaultPolicyMatcher(DefaultPolicyMatcherOptions options)
    : options_(std::move(options)) {}

core::Result<bool> DefaultPolicyMatcher::Matches(const CacheLookupRequest& req,
                                                 const CacheEntryMetadata& entry) {
    if (options_.require_fingerprint &&
        !SameFingerprint(options_.expected_fingerprint, entry.fingerprint)) {
        return false;
    }
    if (!options_.require_fingerprint &&
        !IsEmptyFingerprint(options_.expected_fingerprint) &&
        !IsEmptyFingerprint(entry.fingerprint) &&
        !SameFingerprint(options_.expected_fingerprint, entry.fingerprint)) {
        return false;
    }
    if (!CompatibleScope(req, entry)) {
        return false;
    }
    if (entry.answer_type != AnswerType::Generic &&
        req.answer_type != AnswerType::Generic &&
        entry.answer_type != req.answer_type) {
        return false;
    }
    if (!EqualWhenBothSet(req.tenant_id, entry.tenant_id) ||
        !EqualWhenBothSet(req.subject, entry.subject) ||
        !EqualWhenBothSet(req.grade, entry.grade) ||
        !EqualWhenBothSet(req.topic, entry.topic) ||
        !EqualWhenBothSet(req.persona_id, entry.persona_id)) {
        return false;
    }
    if (entry.quality_score > 0.0f && entry.quality_score < options_.min_quality_score) {
        return false;
    }
    if (entry.expires_at_ms > 0) {
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
        if (entry.expires_at_ms <= now_ms) {
            return false;
        }
    }
    return true;
}

}  // namespace agent::semantic_cache
