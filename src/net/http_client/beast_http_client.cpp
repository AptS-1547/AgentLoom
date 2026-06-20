#include "beast_http_client.h"
#include "url_parser.h"

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>

#include <chrono>
#include <exception>
#include <utility>

namespace agent::net {

namespace beast = boost::beast;
namespace http = boost::beast::http;
namespace asio = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

namespace {

core::Status MakeStatus(const beast::error_code& ec, std::string_view stage) {
    return core::Status(core::ErrorCode::InternalError,
        std::string(stage) + ": " + ec.message());
}

HttpClientResponse ConvertResponse(http::response<http::string_body>& res) {
    HttpClientResponse out;
    out.status = static_cast<int>(res.result_int());
    for (auto const& field : res) {
        out.headers.push_back(HttpHeader{
            std::string(field.name_string()),
            std::string(field.value())
        });
    }
    out.body = std::move(res.body());
    return out;
}

http::verb VerbFromMethod(const std::string& method) {
    auto v = http::string_to_verb(method);
    if (v == http::verb::unknown) {
        // Beast string_to_verb is case-insensitive; an unknown method falls
        // back to verb::unknown which still serializes the string literal.
        return http::verb::unknown;
    }
    return v;
}

http::request<http::string_body> BuildRequest(const ParsedUrl& url,
                                               const HttpClientRequest& req) {
    http::request<http::string_body> hreq;
    auto verb = VerbFromMethod(req.method);
    if (verb == http::verb::unknown) {
        // Fall back to using the raw method literal.  Beast supports this via
        // method_string(); the verb stays "unknown" but serialization uses
        // the literal.
        hreq.method_string(req.method);
    } else {
        hreq.method(verb);
    }
    hreq.target(url.target);
    hreq.version(11);
    hreq.set(http::field::host, url.port == 80 || url.port == 443
        ? url.host
        : url.host + ":" + std::to_string(url.port));
    hreq.set(http::field::user_agent, "agent-http-client/1.0");
    bool has_content_type = false;
    for (auto const& h : req.headers) {
        hreq.set(h.name, h.value);
        // case-insensitive comparison
        if (h.name.size() == 12) {
            bool match = true;
            const char* want = "content-type";
            for (size_t i = 0; i < 12; ++i) {
                if (std::tolower(static_cast<unsigned char>(h.name[i])) != want[i]) {
                    match = false; break;
                }
            }
            if (match) has_content_type = true;
        }
    }
    if (!req.body.empty() && !has_content_type) {
        hreq.set(http::field::content_type, "application/octet-stream");
    }
    hreq.body() = req.body;
    hreq.prepare_payload();
    return hreq;
}

// Run a single async op to completion or kill the stream when the deadline
// fires.  Returns the error code reported by the op (or operation_aborted
// when we cancelled it).
template <typename Stream, typename StartFn>
beast::error_code RunWithDeadline(asio::io_context& ioc,
                                  Stream& lowest_stream,
                                  std::chrono::milliseconds timeout,
                                  StartFn start_op) {
    beast::error_code op_ec = asio::error::would_block;
    bool done = false;
    start_op([&](beast::error_code ec) {
        op_ec = ec;
        done = true;
    });
    ioc.restart();
    ioc.run_for(timeout);
    if (!done) {
        // Force-close so the pending handler resolves promptly.
        beast::error_code ignore;
        lowest_stream.socket().close(ignore);
        ioc.run();
        if (op_ec == asio::error::would_block) {
            op_ec = beast::error::timeout;
        }
    }
    return op_ec;
}

core::Status FromOpErr(const beast::error_code& ec, std::string_view stage) {
    if (ec == beast::error::timeout || ec == asio::error::operation_aborted) {
        return core::Status(core::ErrorCode::Timeout,
            std::string(stage) + ": timed out");
    }
    return core::Status(core::ErrorCode::InternalError,
        std::string(stage) + ": " + ec.message());
}

core::Result<HttpClientResponse> ExecutePlain(const ParsedUrl& url,
                                              const HttpClientRequest& req) {
    asio::io_context ioc;
    tcp::resolver resolver(ioc);
    beast::tcp_stream stream(ioc);
    const auto timeout = std::chrono::milliseconds(req.timeout_ms);

    beast::error_code ec;
    auto results = resolver.resolve(url.host, std::to_string(url.port), ec);
    if (ec) return MakeStatus(ec, "DNS resolve");

    ec = RunWithDeadline(ioc, stream, timeout, [&](auto cb) {
        stream.async_connect(results, [cb = std::move(cb)](beast::error_code e, auto const&) {
            cb(e);
        });
    });
    if (ec) return FromOpErr(ec, "TCP connect");

    auto hreq = BuildRequest(url, req);
    ec = RunWithDeadline(ioc, stream, timeout, [&](auto cb) {
        http::async_write(stream, hreq, [cb = std::move(cb)](beast::error_code e, std::size_t) {
            cb(e);
        });
    });
    if (ec) return FromOpErr(ec, "HTTP write");

    beast::flat_buffer buffer;
    http::response<http::string_body> hres;
    ec = RunWithDeadline(ioc, stream, timeout, [&](auto cb) {
        http::async_read(stream, buffer, hres,
            [cb = std::move(cb)](beast::error_code e, std::size_t) { cb(e); });
    });
    if (ec) return FromOpErr(ec, "HTTP read");

    beast::error_code ignore;
    stream.socket().shutdown(tcp::socket::shutdown_both, ignore);
    return ConvertResponse(hres);
}

core::Result<HttpClientResponse> ExecuteTls(const ParsedUrl& url,
                                            const HttpClientRequest& req,
                                            TlsContext& tls) {
    asio::io_context ioc;
    tcp::resolver resolver(ioc);
    beast::ssl_stream<beast::tcp_stream> stream(ioc, tls.asio_context());
    const auto timeout = std::chrono::milliseconds(req.timeout_ms);
    auto& lowest = beast::get_lowest_layer(stream);

    if (auto st = tls.PrepareConnection(stream.native_handle(), url.host); !st.ok()) {
        return st;
    }

    beast::error_code ec;
    auto results = resolver.resolve(url.host, std::to_string(url.port), ec);
    if (ec) return MakeStatus(ec, "DNS resolve");

    ec = RunWithDeadline(ioc, lowest, timeout, [&](auto cb) {
        lowest.async_connect(results, [cb = std::move(cb)](beast::error_code e, auto const&) {
            cb(e);
        });
    });
    if (ec) return FromOpErr(ec, "TCP connect");

    ec = RunWithDeadline(ioc, lowest, timeout, [&](auto cb) {
        stream.async_handshake(ssl::stream_base::client,
            [cb = std::move(cb)](beast::error_code e) { cb(e); });
    });
    if (ec) return FromOpErr(ec, "TLS handshake");

    auto hreq = BuildRequest(url, req);
    ec = RunWithDeadline(ioc, lowest, timeout, [&](auto cb) {
        http::async_write(stream, hreq,
            [cb = std::move(cb)](beast::error_code e, std::size_t) { cb(e); });
    });
    if (ec) return FromOpErr(ec, "HTTPS write");

    beast::flat_buffer buffer;
    http::response<http::string_body> hres;
    ec = RunWithDeadline(ioc, lowest, timeout, [&](auto cb) {
        http::async_read(stream, buffer, hres,
            [cb = std::move(cb)](beast::error_code e, std::size_t) { cb(e); });
    });
    if (ec) return FromOpErr(ec, "HTTPS read");

    // Best-effort close; ignore any error from cancelled writes during shutdown.
    beast::error_code ignore;
    lowest.socket().close(ignore);
    return ConvertResponse(hres);
}

}  // namespace

core::Result<std::unique_ptr<BeastHttpClient>> BeastHttpClient::Create(
    BeastHttpClientOptions options) {
    auto self = std::unique_ptr<BeastHttpClient>(new BeastHttpClient(std::move(options)));
    return self;
}

BeastHttpClient::BeastHttpClient(BeastHttpClientOptions options)
    : options_(std::move(options)) {}

BeastHttpClient::~BeastHttpClient() = default;

core::Result<HttpClientResponse> BeastHttpClient::Execute(const HttpClientRequest& req) {
    try {
        if (req.timeout_ms <= 0) {
            return core::Status(core::ErrorCode::InvalidArgument, "timeout_ms must be > 0");
        }

        auto url_r = ParseUrl(req.url);
        if (!url_r) return url_r.status();
        auto url = std::move(url_r).value();

        if (url.scheme == UrlScheme::Https) {
            if (!options_.tls_context) {
                return core::Status(core::ErrorCode::InvalidArgument,
                    "https URL but client has no TLS context");
            }
            return ExecuteTls(url, req, *options_.tls_context);
        }
        return ExecutePlain(url, req);
    } catch (const std::exception& e) {
        return core::Status(core::ErrorCode::InternalError,
            std::string("HTTP client exception: ") + e.what());
    } catch (...) {
        return core::Status(core::ErrorCode::InternalError,
            "HTTP client exception: unknown error");
    }
}

}  // namespace agent::net
