#pragma once

#include "result.h"

#include <cstdint>
#include <string>
#include <vector>

namespace agent::net {

struct HttpHeader {
    std::string name;
    std::string value;
};

struct HttpClientRequest {
    std::string method = "POST";
    /// Full URL including scheme, host, optional port and path.
    /// Examples: "https://api.deepseek.com/v1/chat/completions",
    ///           "http://127.0.0.1:8080/echo".
    std::string url;
    std::vector<HttpHeader> headers;
    std::string body;
    std::int32_t timeout_ms = 60000;
};

struct HttpClientResponse {
    int status = 0;
    std::vector<HttpHeader> headers;
    std::string body;
};

/// Outbound HTTP/HTTPS client interface.
///
/// Implementations execute one request per call: DNS resolve → connect →
/// (TLS handshake) → write request → read full response → close.  No connection
/// reuse, no auto-redirect.  Caller is responsible for retry / backoff (see
/// retry_policy.h) and for picking a sensible `timeout_ms`.
class IHttpClient {
public:
    virtual ~IHttpClient() = default;

    /// Run a single request.  Returns a non-ok Status for transport-level
    /// failures (DNS, connect, TLS handshake, timeout, malformed response);
    /// returns `Ok` with `status` populated for any HTTP-level outcome,
    /// including 4xx/5xx.
    virtual core::Result<HttpClientResponse> Execute(const HttpClientRequest& req) = 0;
};

}  // namespace agent::net

