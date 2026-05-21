#pragma once

#include "result.h"

#include <cstdint>
#include <string>

namespace agent::net {

enum class UrlScheme {
    Http,
    Https,
};

struct ParsedUrl {
    UrlScheme scheme = UrlScheme::Https;
    std::string host;
    /// Resolved port: defaults to 80 / 443 when the URL omits one.
    std::uint16_t port = 0;
    /// Path including any query string, leading '/' guaranteed.  "/" if absent.
    std::string target = "/";
};

/// Parse "scheme://host[:port][/path][?query]".  Userinfo, fragments, and
/// non-http(s) schemes are rejected with `InvalidArgument`.
core::Result<ParsedUrl> ParseUrl(const std::string& url);

}  // namespace agent::net
