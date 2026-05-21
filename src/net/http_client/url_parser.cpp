#include "url_parser.h"

#include <algorithm>
#include <cctype>
#include <charconv>

namespace agent::net {

namespace {

bool IEquals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

}  // namespace

core::Result<ParsedUrl> ParseUrl(const std::string& url) {
    if (url.empty()) {
        return core::Status(core::ErrorCode::InvalidArgument, "empty URL");
    }

    auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        return core::Status(core::ErrorCode::InvalidArgument, "URL missing scheme");
    }
    std::string_view scheme(url.data(), scheme_end);

    ParsedUrl out;
    std::uint16_t default_port = 0;
    if (IEquals(scheme, "https")) {
        out.scheme = UrlScheme::Https;
        default_port = 443;
    } else if (IEquals(scheme, "http")) {
        out.scheme = UrlScheme::Http;
        default_port = 80;
    } else {
        return core::Status(core::ErrorCode::InvalidArgument,
            "unsupported scheme: " + std::string(scheme));
    }

    const std::size_t authority_begin = scheme_end + 3;
    if (authority_begin >= url.size()) {
        return core::Status(core::ErrorCode::InvalidArgument, "URL missing host");
    }
    if (url.find('@', authority_begin) != std::string::npos) {
        // Reject userinfo to avoid surprising credentials handling.
        return core::Status(core::ErrorCode::InvalidArgument,
            "userinfo in URL is not supported");
    }
    if (url.find('#', authority_begin) != std::string::npos) {
        return core::Status(core::ErrorCode::InvalidArgument,
            "URL fragment is not supported");
    }

    // path begins at first '/' after authority; query (?) belongs to path.
    std::size_t path_begin = url.find('/', authority_begin);
    std::string_view authority;
    if (path_begin == std::string::npos) {
        authority = std::string_view(url.data() + authority_begin,
                                     url.size() - authority_begin);
        out.target = "/";
    } else {
        authority = std::string_view(url.data() + authority_begin,
                                     path_begin - authority_begin);
        out.target.assign(url, path_begin, std::string::npos);
    }

    if (authority.empty()) {
        return core::Status(core::ErrorCode::InvalidArgument, "URL missing host");
    }

    auto colon = authority.rfind(':');
    if (colon == std::string_view::npos) {
        out.host.assign(authority);
        out.port = default_port;
    } else {
        out.host.assign(authority.substr(0, colon));
        auto port_sv = authority.substr(colon + 1);
        if (port_sv.empty()) {
            return core::Status(core::ErrorCode::InvalidArgument, "empty port");
        }
        unsigned int port = 0;
        auto* begin = port_sv.data();
        auto* end = begin + port_sv.size();
        auto [p, ec] = std::from_chars(begin, end, port);
        if (ec != std::errc{} || p != end || port == 0 || port > 65535) {
            return core::Status(core::ErrorCode::InvalidArgument,
                "invalid port: " + std::string(port_sv));
        }
        out.port = static_cast<std::uint16_t>(port);
    }

    if (out.host.empty()) {
        return core::Status(core::ErrorCode::InvalidArgument, "URL missing host");
    }
    return out;
}

}  // namespace agent::net
