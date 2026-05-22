#pragma once

#include "context_risk_detector.h"
#include "isemantic_cache.h"
#include "semantic_cache_policy.h"
#include "semantic_cache_types.h"
#include "result.h"

#include <memory>

namespace agent::semantic_cache {

/// Dependencies injected into the pipeline.  Forward-declared opaque types
/// keep the public header free of Faiss / Ort / tokenizer details.
///
/// TODO(orange): pick the concrete handle types when wiring this up — likely
///   - tokenizer:        agent::vector::HfTokenizer (or pool)
///   - embedding:        agent::vector::OnnxTextEmbeddingModel
///   - index manager:    agent::vector::VectorIndexManager
///   - repository:       agent::storage::IVectorRepository
///
/// They can be stored as raw references / shared_ptr / unique_ptr — pick
/// whatever matches the lifetime story you want.  This header just owns the
/// composition; concrete types live in the .cpp.
struct SemanticCachePipelineDeps {
    std::shared_ptr<IContextRiskDetector> risk_detector;
    std::shared_ptr<IPolicyMatcher> policy_matcher;
    // TODO(orange): add tokenizer / embedding / index / repository handles.
};

struct SemanticCachePipelineOptions {
    std::size_t top_k = 8;
    float similarity_floor = 0.85f;
    bool enable_global_scope = true;
};

/// Pipeline implementation of ISemanticCache.
///
/// Encapsulates the lookup chain described in NEXT_RUNTIME_ROADMAP §5:
///   precheck → tokenize → embed → vector search → metadata filter →
///   policy filter → payload load.
///
/// Construction is intentionally factory-style so wiring failures surface
/// as core::Status rather than exceptions in the ctor.
class SemanticCachePipeline : public ISemanticCache {
public:
    static core::Result<std::unique_ptr<SemanticCachePipeline>> Create(
        SemanticCachePipelineOptions options,
        SemanticCachePipelineDeps deps);

    ~SemanticCachePipeline() override;

    core::Result<CacheLookupResult> Lookup(const CacheLookupRequest& req) override;
    core::Status Store(const CacheStoreRequest& req) override;

private:
    SemanticCachePipeline(SemanticCachePipelineOptions options,
                           SemanticCachePipelineDeps deps);

    SemanticCachePipelineOptions options_;
    SemanticCachePipelineDeps deps_;
};

}  // namespace agent::semantic_cache
