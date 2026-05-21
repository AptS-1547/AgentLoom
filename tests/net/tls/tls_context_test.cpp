#include "tls_context.h"

#include <openssl/ssl.h>
#include <openssl/x509v3.h>

#include <gtest/gtest.h>

using agent::net::TlsContext;
using agent::net::TlsClientOptions;
using agent::net::TlsMinVersion;
using agent::net::TlsVerifyMode;

namespace {

// Helper: create a free-standing SSL* from a TlsContext for testing
// PrepareConnection without spinning up sockets.
struct TestSsl {
    explicit TestSsl(TlsContext& ctx) {
        ssl = SSL_new(ctx.asio_context().native_handle());
    }
    ~TestSsl() { if (ssl) SSL_free(ssl); }
    SSL* ssl = nullptr;
};

}  // namespace

TEST(TlsContextTest, CreateClientTls12Default) {
    TlsClientOptions opts;
    opts.min_version = TlsMinVersion::Tls12;
    auto r = TlsContext::CreateClient(opts);
    ASSERT_TRUE(r.ok());
    auto ctx = std::move(r).value();
    EXPECT_NE(ctx.get(), nullptr);
    EXPECT_EQ(ctx->verify_mode(), TlsVerifyMode::Default);

    long got = SSL_CTX_get_min_proto_version(ctx->asio_context().native_handle());
    EXPECT_EQ(got, TLS1_2_VERSION);
}

TEST(TlsContextTest, CreateClientTls13) {
    TlsClientOptions opts;
    opts.min_version = TlsMinVersion::Tls13;
    auto r = TlsContext::CreateClient(opts);
    ASSERT_TRUE(r.ok());
    auto ctx = std::move(r).value();

    long got = SSL_CTX_get_min_proto_version(ctx->asio_context().native_handle());
    EXPECT_EQ(got, TLS1_3_VERSION);
}

TEST(TlsContextTest, CreateClientVerifyNone) {
    TlsClientOptions opts;
    opts.verify_mode = TlsVerifyMode::None;
    auto r = TlsContext::CreateClient(opts);
    ASSERT_TRUE(r.ok());
    auto ctx = std::move(r).value();
    EXPECT_EQ(ctx->verify_mode(), TlsVerifyMode::None);

    int mode = SSL_CTX_get_verify_mode(ctx->asio_context().native_handle());
    EXPECT_EQ(mode, SSL_VERIFY_NONE);
}

TEST(TlsContextTest, CreateClientFailsOnMissingCaBundle) {
    TlsClientOptions opts;
    opts.ca_bundle_path = "C:/no/such/ca/bundle.pem";
    auto r = TlsContext::CreateClient(opts);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), core::ErrorCode::InvalidArgument);
}

TEST(TlsContextTest, PrepareConnectionSetsSniAndHostCheck) {
    TlsClientOptions opts;
    auto r = TlsContext::CreateClient(opts);
    ASSERT_TRUE(r.ok());
    auto ctx = std::move(r).value();

    TestSsl t(*ctx);
    ASSERT_NE(t.ssl, nullptr);

    auto status = ctx->PrepareConnection(t.ssl, "api.example.com");
    ASSERT_TRUE(status.ok()) << status.message();

    const char* sni = SSL_get_servername(t.ssl, TLSEXT_NAMETYPE_host_name);
    ASSERT_NE(sni, nullptr);
    EXPECT_STREQ(sni, "api.example.com");

    X509_VERIFY_PARAM* param = SSL_get0_param(t.ssl);
    ASSERT_NE(param, nullptr);
    // OpenSSL ≥ 3.0 exposes the expected host list via X509_VERIFY_PARAM_get0_host.
    const char* host0 = X509_VERIFY_PARAM_get0_host(param, 0);
    ASSERT_NE(host0, nullptr);
    EXPECT_STREQ(host0, "api.example.com");
}

TEST(TlsContextTest, PrepareConnectionRejectsEmptyHostname) {
    TlsClientOptions opts;
    auto r = TlsContext::CreateClient(opts);
    ASSERT_TRUE(r.ok());
    auto ctx = std::move(r).value();

    TestSsl t(*ctx);
    ASSERT_NE(t.ssl, nullptr);

    auto status = ctx->PrepareConnection(t.ssl, "");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), core::ErrorCode::InvalidArgument);
}

TEST(TlsContextTest, PrepareConnectionRejectsNullHandle) {
    TlsClientOptions opts;
    auto r = TlsContext::CreateClient(opts);
    ASSERT_TRUE(r.ok());
    auto ctx = std::move(r).value();

    auto status = ctx->PrepareConnection(nullptr, "api.example.com");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), core::ErrorCode::InvalidArgument);
}

TEST(TlsContextTest, VerifyNoneSkipsHostCheckInPrepare) {
    TlsClientOptions opts;
    opts.verify_mode = TlsVerifyMode::None;
    auto r = TlsContext::CreateClient(opts);
    ASSERT_TRUE(r.ok());
    auto ctx = std::move(r).value();

    TestSsl t(*ctx);
    ASSERT_NE(t.ssl, nullptr);

    // Still sets SNI, but should NOT install hostname verify.
    auto status = ctx->PrepareConnection(t.ssl, "anything.test");
    ASSERT_TRUE(status.ok());

    const char* sni = SSL_get_servername(t.ssl, TLSEXT_NAMETYPE_host_name);
    EXPECT_STREQ(sni, "anything.test");

    X509_VERIFY_PARAM* param = SSL_get0_param(t.ssl);
    const char* host0 = X509_VERIFY_PARAM_get0_host(param, 0);
    EXPECT_EQ(host0, nullptr);  // no host check installed
}
