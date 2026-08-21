#pragma once

#include "semantic_cache_types.h"
#include "result.h"

#include <functional>
#include <vector>

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

/// 可选的异步语义缓存扩展。同步 ISemanticCache 保持兼容，只有具备独立
/// embedding/IO 调度能力的实现才需要实现该接口。
class IAsyncSemanticCache {
public:
    using LookupCompletion = std::function<void(core::Result<CacheLookupResult>)>;
    using StoreCompletion = std::function<void(core::Status)>;

    virtual ~IAsyncSemanticCache() = default;
    /// 返回非 OK 表示请求未被接纳，之后不得调用 completion；返回 OK 后必须恰好完成一次。
    virtual core::Status LookupAsync(CacheLookupRequest request,
                                     LookupCompletion completion) = 0;

    /// 异步接纳一条记忆；旧实现可返回 FailedPrecondition，由上层选择兼容路径。
    virtual core::Status StoreAsync(CacheStoreRequest request,
                                    StoreCompletion completion) {
        static_cast<void>(request);
        static_cast<void>(completion);
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "asynchronous semantic cache store is not supported");
    }
};

}  // namespace agent::semantic_cache
