#include "static_file_handler.h"

#include <boost/beast/core/error.hpp>

#include <algorithm>

namespace net {

namespace {

bool StartsWithDotDot(const std::filesystem::path& path) {
    for (const auto& part : path) {
        if (part == "..") {
            return true;
        }
    }
    return false;
}

http::response<http::string_body> MakeStringResponse(const BeastHttpRequest& request,
                                                     http::status status,
                                                     std::string body,
                                                     std::string_view content_type = "text/plain; charset=utf-8") {
    http::response<http::string_body> response{status, request.version()};
    response.set(http::field::server, "AgentBackendPredict");
    response.set(http::field::content_type, content_type);
    response.keep_alive(request.keep_alive());
    response.body() = std::move(body);
    response.prepare_payload();
    return response;
}

} // namespace

StaticFileHandler::StaticFileHandler(StaticFileOptions options)
    : options_(std::move(options)) {}

http::message_generator StaticFileHandler::Handle(const BeastHttpRequest& request) const {
    if (request.method() != http::verb::get && request.method() != http::verb::head) {
        auto response = MakeStringResponse(request, http::status::method_not_allowed, "method not allowed");
        response.set(http::field::allow, "GET, HEAD");
        return response;
    }

    auto path_result = ResolveTarget(std::string_view(request.target().data(), request.target().size()));
    if (!path_result.ok()) {
        return ErrorResponse(request, http::status::bad_request, path_result.status().message());
    }

    auto path = std::move(path_result).value();
    if (!std::filesystem::exists(path) && options_.spa_fallback) {
        path = ResolveFallbackPath();
    }
    if (!std::filesystem::exists(path) || std::filesystem::is_directory(path)) {
        return ErrorResponse(request, http::status::not_found, "not found");
    }

    beast::error_code ec;
    http::file_body::value_type body;
    body.open(path.string().c_str(), beast::file_mode::scan, ec);
    if (ec == beast::errc::no_such_file_or_directory) {
        return ErrorResponse(request, http::status::not_found, "not found");
    }
    if (ec) {
        return ErrorResponse(request, http::status::internal_server_error, ec.message());
    }

    const auto size = body.size();
    if (request.method() == http::verb::head) {
        http::response<http::empty_body> response{http::status::ok, request.version()};
        response.set(http::field::server, "AgentBackendPredict");
        response.set(http::field::content_type, MimeType(path.string()));
        response.content_length(size);
        response.keep_alive(request.keep_alive());
        return response;
    }

    http::response<http::file_body> response{
        std::piecewise_construct,
        std::make_tuple(std::move(body)),
        std::make_tuple(http::status::ok, request.version())};
    response.set(http::field::server, "AgentBackendPredict");
    response.set(http::field::content_type, MimeType(path.string()));
    response.content_length(size);
    response.keep_alive(request.keep_alive());
    return response;
}

core::Result<std::filesystem::path> StaticFileHandler::ResolveTarget(std::string_view target) const {
    if (options_.root.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "static root is empty");
    }
    if (target.empty() || target.front() != '/') {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "target must be absolute");
    }

    auto path_target = target.substr(1);
    const auto query_pos = path_target.find('?');
    if (query_pos != std::string_view::npos) {
        path_target = path_target.substr(0, query_pos);
    }

    std::filesystem::path relative = std::string(path_target);
    if (relative.empty()) {
        relative = options_.index_file;
    }
    if (StartsWithDotDot(relative) || relative.is_absolute()) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "path traversal is denied");
    }
    if (!relative.empty()) {
        auto last = relative.filename();
        if (last.empty() || last == "." || last == "..") {
            relative /= options_.index_file;
        }
    }

    return options_.root / relative;
}

http::message_generator StaticFileHandler::ErrorResponse(const BeastHttpRequest& request,
                                                         http::status status,
                                                         std::string body) const {
    return MakeStringResponse(request, status, std::move(body));
}

std::filesystem::path StaticFileHandler::ResolveFallbackPath() const {
    return options_.root / options_.index_file;
}

std::string_view MimeType(std::string_view path) noexcept {
    const auto extension_pos = path.rfind('.');
    const auto extension = extension_pos == std::string_view::npos ? std::string_view{} : path.substr(extension_pos);
    if (extension == ".htm" || extension == ".html") {
        return "text/html; charset=utf-8";
    }
    if (extension == ".css") {
        return "text/css; charset=utf-8";
    }
    if (extension == ".js" || extension == ".mjs") {
        return "text/javascript; charset=utf-8";
    }
    if (extension == ".json") {
        return "application/json; charset=utf-8";
    }
    if (extension == ".png") {
        return "image/png";
    }
    if (extension == ".jpg" || extension == ".jpeg") {
        return "image/jpeg";
    }
    if (extension == ".gif") {
        return "image/gif";
    }
    if (extension == ".svg") {
        return "image/svg+xml";
    }
    if (extension == ".wasm") {
        return "application/wasm";
    }
    if (extension == ".ico") {
        return "image/x-icon";
    }
    return "application/octet-stream";
}

} // namespace net
