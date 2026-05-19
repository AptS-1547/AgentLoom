#pragma once

#include "http_types.h"

#include <boost/beast/http.hpp>

#include <cstddef>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace net {

struct HttpRequestFilterDecision {
    bool allowed = true;
    http::status status = http::status::ok;
    std::string reason;

    static HttpRequestFilterDecision Allow() {
        return {};
    }

    static HttpRequestFilterDecision Deny(http::status status, std::string reason) {
        return {false, status, std::move(reason)};
    }
};

struct HttpRequestFilterOptions {
    bool enabled = false;
    std::size_t max_target_bytes = 4096;
    std::size_t max_header_field_bytes = 8192;
    std::size_t max_header_value_bytes = 8192;
    std::size_t max_total_header_bytes = 64 * 1024;
    bool reject_control_chars = true;
    // Heuristic regex checks for obvious scanner traffic and commodity attacks.
    // This is a lightweight guardrail, not a complete WAF or SQL parser.
    bool reject_suspicious_patterns = true;
    std::vector<std::string> extra_patterns;
};

class HttpRequestFilter {
public:
    explicit HttpRequestFilter(HttpRequestFilterOptions options = {});

    const HttpRequestFilterOptions& options() const noexcept {
        return options_;
    }

    HttpRequestFilterDecision Check(const BeastHttpRequest& request) const;

private:
    HttpRequestFilterDecision CheckSizeLimits(const BeastHttpRequest& request) const;
    HttpRequestFilterDecision CheckControlChars(const BeastHttpRequest& request) const;
    HttpRequestFilterDecision CheckSuspiciousPatterns(const BeastHttpRequest& request) const;
    bool ContainsControlChars(std::string_view value) const noexcept;
    bool MatchesSuspiciousPattern(std::string_view value) const;

    HttpRequestFilterOptions options_;
    std::vector<std::regex> suspicious_patterns_;
};

} // namespace net
