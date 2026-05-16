#pragma once

#include "http_types.h"
#include "result.h"

#include <boost/beast/http.hpp>

#include <filesystem>
#include <string>
#include <string_view>

namespace net {

struct StaticFileOptions {
    std::filesystem::path root;
    std::string index_file = "index.html";
    bool spa_fallback = true;
};

class StaticFileHandler {
public:
    explicit StaticFileHandler(StaticFileOptions options);

    http::message_generator Handle(const BeastHttpRequest& request) const;
    core::Result<std::filesystem::path> ResolveTarget(std::string_view target) const;

private:
    http::message_generator ErrorResponse(const BeastHttpRequest& request,
                                          http::status status,
                                          std::string body) const;
    std::filesystem::path ResolveFallbackPath() const;

    StaticFileOptions options_;
};

std::string_view MimeType(std::string_view path) noexcept;

} // namespace net
