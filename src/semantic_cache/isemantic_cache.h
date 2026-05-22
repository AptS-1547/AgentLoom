#pragma once

#include "semantic_cache_types.h"
#include "result.h"

namespace agent::semantic_cache {

/// Abstract semantic cache facade.
///
/// Implementations should encapsulate the full lookup pipeline:
///   1. precheck (context risk, scope policy)
///   2. tokenize + embed query text
///   3. top-k vector search
///   4. metadata / quality / scope filters
///   5. payload fetch
///
/// Failure to find a safe match returns an Ok result with `hit == false`.
/// Non-ok status is reserved for infrastructure failures.
class ISemanticCache {
public:
    virtual ~ISemanticCache() = default;

    virtual core::Result<CacheLookupResult> Lookup(const CacheLookupRequest& req) = 0;

    virtual core::Status Store(const CacheStoreRequest& req) = 0;
};

}  // namespace agent::semantic_cache
