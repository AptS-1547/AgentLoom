#pragma once

#include "tls_options.h"
#include "result.h"

#include <boost/asio/ssl/context.hpp>
#include <openssl/ssl.h>
#include <memory>
#include <string>

namespace agent::net {

/// Wraps a `boost::asio::ssl::context` configured for client-side TLS.
///
/// Built once at startup and shared across many connections — TLS contexts
/// are designed to be reused.  Holds the OpenSSL `SSL_CTX*` underneath.
class TlsContext {
public:
    /// Build a client-side TLS context with the given options.
    /// Returns `InvalidArgument` if `ca_bundle_path` is set but unreadable.
    static core::Result<std::shared_ptr<TlsContext>> CreateClient(const TlsClientOptions& options);

    /// Underlying boost::asio::ssl::context.  Callers wire SSL streams against it.
    boost::asio::ssl::context& asio_context() noexcept { return ctx_; }

    /// Apply per-connection setup that depends on the target hostname:
    /// sets SNI and (when verify_mode is Default) configures hostname checking.
    /// Should be invoked on the SSL stream's native handle before the handshake.
    ///
    /// Returns `InvalidArgument` if `ssl` is null or `hostname` is empty,
    /// `InternalError` if OpenSSL rejects the value.
    core::Status PrepareConnection(SSL* ssl, const std::string& hostname) const;

    TlsVerifyMode verify_mode() const noexcept { return options_.verify_mode; }

private:
    TlsContext(boost::asio::ssl::context ctx, TlsClientOptions options);

    boost::asio::ssl::context ctx_;
    TlsClientOptions options_;
};

}  // namespace agent::net
