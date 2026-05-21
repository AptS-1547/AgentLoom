#pragma once

#include "http_client.h"
#include "../tls/tls_context.h"

#include <memory>

namespace agent::net {

/// Options for constructing a `BeastHttpClient`.
struct BeastHttpClientOptions {
    /// Required for any https:// URL.  Ignored for http://.
    /// Pass null to disable HTTPS entirely (http-only client).
    std::shared_ptr<TlsContext> tls_context;
};

/// Beast-based HTTP/HTTPS client.
///
/// One request per `Execute` call: DNS resolve → TCP connect →
/// (TLS handshake) → write request → read response → close.  Synchronous —
/// the call blocks until the response is complete or the timeout fires.
///
/// Not designed for high-throughput service-to-service calls.  Use for
/// occasional outbound requests (LLM completions, webhook publishing,
/// metrics push) where simplicity beats reuse.
class BeastHttpClient : public IHttpClient {
public:
    static core::Result<std::unique_ptr<BeastHttpClient>> Create(BeastHttpClientOptions options);

    ~BeastHttpClient() override;

    core::Result<HttpClientResponse> Execute(const HttpClientRequest& req) override;

private:
    explicit BeastHttpClient(BeastHttpClientOptions options);

    BeastHttpClientOptions options_;
};

}  // namespace agent::net
