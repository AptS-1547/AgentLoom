#pragma once

#include "protocol_types.h"

#include <boost/beast/http.hpp>
#include <boost/version.hpp>

#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace net {

#if defined(AGENTLOOM_BUILT_BOOST_VERSION)
static_assert(BOOST_VERSION == AGENTLOOM_BUILT_BOOST_VERSION,
              "AgentLoom and its consumer must use the same Boost headers");
#endif
static_assert(static_cast<unsigned>(boost::beast::http::field::connection) == 59u,
              "AgentLoom requires the Boost.Beast 1.85 HTTP field layout");

namespace beast = boost::beast;
namespace http = beast::http;

using BeastHttpRequest = http::request<http::string_body>;
using BeastHttpResponse = http::response<http::string_body>;

struct HttpRequest {
    BeastHttpRequest message;
    ConnectionContext connection;

    static HttpRequest FromBeast(BeastHttpRequest message, ConnectionContext connection = {}) noexcept {
        return {std::move(message), std::move(connection)};
    }
};

struct HttpResponse {
    BeastHttpResponse message;
    bool close_after_write = false;

    static HttpResponse FromBeast(BeastHttpResponse message, bool close_after_write = false) noexcept {
        return {std::move(message), close_after_write};
    }

    static HttpResponse Empty(http::status status) {
        HttpResponse response;
        response.message.result(status);
        response.message.version(11);
        response.message.body().clear();
        response.message.prepare_payload();
        return response;
    }

    static HttpResponse Text(http::status status,
                             std::string body,
                             std::string_view content_type = "text/plain; charset=utf-8") {
        HttpResponse response;
        response.message.result(status);
        response.message.version(11);
        response.message.set(http::field::content_type, content_type);
        response.message.body() = std::move(body);
        response.message.prepare_payload();
        return response;
    }

    static HttpResponse Json(http::status status, std::string body) {
        return Text(status, std::move(body), "application/json; charset=utf-8");
    }
};

struct HttpHandlerResult {
    core::Status status = core::Status::Ok();
    HttpResponse response;
    ConnectionCloseInfo close_info;
};

using HttpResponseCallback = std::function<void(HttpHandlerResult)>;
using HttpHandler = std::function<void(HttpRequest, HttpResponseCallback)>;
using HttpAccessController = std::function<void(const BeastHttpRequest&, const ConnectionContext&, AccessCompletion)>;

} // namespace net
