#pragma once

#include "http_client.h"
#include "thread_pool.h"

#include <future>
#include <memory>

namespace agent::net {

/// Thread-pool-backed concurrent HTTP client.
///
/// Wraps any `IHttpClient` and dispatches each `ExecuteAsync` call as an
/// independent task on the supplied `ThreadPool`.  Each call owns its own
/// Beast io_context, stream, and buffer — no shared mutable state, no locks
/// needed on the underlying client.
///
/// Useful for fire-and-forget LLM maintenance calls and parallel batch
/// requests where the caller can wait on multiple futures at once.
class ConcurrentHttpClient {
public:
    /// `client` must outlive this object.  `pool` must outlive this object.
    ConcurrentHttpClient(IHttpClient& client, core::ThreadPool& pool)
        : client_(client), pool_(pool) {}

    /// Submit a request to the thread pool and return a future.
    /// The future resolves with the response or a non-ok status on failure.
    /// Returns `ResourceExhausted` immediately if the pool queue is full.
    std::future<core::Result<HttpClientResponse>> ExecuteAsync(HttpClientRequest req) {
        auto promise = std::make_shared<std::promise<core::Result<HttpClientResponse>>>();
        auto fut = promise->get_future();

        auto st = pool_.Submit([this, req = std::move(req),
                                 promise](core::ThreadPoolContext&) mutable -> core::Status {
            auto result = client_.Execute(req);
            promise->set_value(std::move(result));
            return core::Status::Ok();
        });

        if (!st.ok()) {
            promise->set_value(st);
        }

        return fut;
    }

private:
    IHttpClient& client_;
    core::ThreadPool& pool_;
};

}  // namespace agent::net
