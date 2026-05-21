#include "tls_context.h"

#include <openssl/ssl.h>
#include <openssl/x509v3.h>
#include <openssl/err.h>

#include <utility>

namespace agent::net {

namespace {

boost::asio::ssl::context BuildBaseContext(TlsMinVersion min_version) {
    namespace ssl = boost::asio::ssl;
    // tls_client picks the highest version supported by the linked OpenSSL; we
    // raise the floor via SSL_CTX_set_min_proto_version below.
    ssl::context ctx(ssl::context::tls_client);
    ctx.set_options(ssl::context::default_workarounds |
                    ssl::context::no_sslv2 |
                    ssl::context::no_sslv3 |
                    ssl::context::no_tlsv1 |
                    ssl::context::no_tlsv1_1);
    long min = (min_version == TlsMinVersion::Tls13) ? TLS1_3_VERSION : TLS1_2_VERSION;
    SSL_CTX_set_min_proto_version(ctx.native_handle(), min);
    return ctx;
}

}  // namespace

TlsContext::TlsContext(boost::asio::ssl::context ctx, TlsClientOptions options)
    : ctx_(std::move(ctx)), options_(std::move(options)) {}

core::Result<std::shared_ptr<TlsContext>> TlsContext::CreateClient(const TlsClientOptions& options) {
    namespace ssl = boost::asio::ssl;
    auto ctx = BuildBaseContext(options.min_version);

    if (options.verify_mode == TlsVerifyMode::Default) {
        ctx.set_verify_mode(ssl::verify_peer);
        if (!options.ca_bundle_path.empty()) {
            boost::system::error_code ec;
            ctx.load_verify_file(options.ca_bundle_path, ec);
            if (ec) {
                return core::Status(core::ErrorCode::InvalidArgument,
                    "Failed to load CA bundle: " + ec.message());
            }
        } else {
            boost::system::error_code ec;
            ctx.set_default_verify_paths(ec);
            if (ec) {
                return core::Status(core::ErrorCode::InternalError,
                    "set_default_verify_paths failed: " + ec.message());
            }
        }
    } else {
        ctx.set_verify_mode(ssl::verify_none);
    }

    auto self = std::shared_ptr<TlsContext>(new TlsContext(std::move(ctx), options));
    return self;
}

core::Status TlsContext::PrepareConnection(SSL* ssl, const std::string& hostname) const {
    if (!ssl) {
        return core::Status(core::ErrorCode::InvalidArgument, "null SSL handle");
    }
    if (hostname.empty()) {
        return core::Status(core::ErrorCode::InvalidArgument, "empty hostname");
    }

    // SNI — required by virtually every modern API gateway.
    if (SSL_set_tlsext_host_name(ssl, hostname.c_str()) != 1) {
        unsigned long err = ERR_get_error();
        char buf[256] = {0};
        ERR_error_string_n(err, buf, sizeof(buf));
        return core::Status(core::ErrorCode::InternalError,
            std::string("SSL_set_tlsext_host_name failed: ") + buf);
    }

    if (options_.verify_mode == TlsVerifyMode::Default) {
        X509_VERIFY_PARAM* param = SSL_get0_param(ssl);
        X509_VERIFY_PARAM_set_hostflags(param, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
        if (X509_VERIFY_PARAM_set1_host(param, hostname.c_str(), hostname.size()) != 1) {
            return core::Status(core::ErrorCode::InternalError,
                "X509_VERIFY_PARAM_set1_host failed");
        }
    }
    return core::Status::Ok();
}

}  // namespace agent::net
