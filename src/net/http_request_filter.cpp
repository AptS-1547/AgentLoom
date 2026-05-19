#include "http_request_filter.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <string>

namespace net {

namespace {

constexpr std::array<std::string_view, 12> kDefaultSuspiciousPatterns = {
    R"((\bunion\b\s+(?:all\s+)?\bselect\b))",
    R"((?:\bor\b|\band\b)\s+['"]?\d+['"]?\s*=\s*['"]?\d+)",
    R"((?:--|#|/\*))",
    R"((\bdrop\b\s+\btable\b|\binsert\b\s+\binto\b|\bdelete\b\s+\bfrom\b|\bupdate\b\s+\w+\s+\bset\b))",
    R"((\bxp_cmdshell\b|\binformation_schema\b))",
    R"(<\s*script\b)",
    R"((?:\.\./|\.\.\\))",
    R"((?:%00|\\x00|\\0))",
    R"((?:%u[0-9a-f]{4}))",
    R"((?:%[0-9a-f]{2}){12,})",
    R"((?:\\x[0-9a-f]{2}){8,})",
    R"((?:a{2048,}|%41{512,}))",
};

std::regex MakeRegex(std::string_view pattern) {
    return std::regex(std::string(pattern), std::regex::icase | std::regex::optimize);
}

std::string ToString(boost::beast::string_view value) {
    return std::string(value.data(), value.size());
}

} // namespace

HttpRequestFilter::HttpRequestFilter(HttpRequestFilterOptions options)
    : options_(std::move(options)) {
    if (options_.reject_suspicious_patterns) {
        suspicious_patterns_.reserve(kDefaultSuspiciousPatterns.size() + options_.extra_patterns.size());
        for (auto pattern : kDefaultSuspiciousPatterns) {
            suspicious_patterns_.push_back(MakeRegex(pattern));
        }
        for (const auto& pattern : options_.extra_patterns) {
            suspicious_patterns_.push_back(MakeRegex(pattern));
        }
    }
}

HttpRequestFilterDecision HttpRequestFilter::Check(const BeastHttpRequest& request) const {
    auto size_decision = CheckSizeLimits(request);
    if (!size_decision.allowed) {
        return size_decision;
    }

    if (options_.reject_control_chars) {
        auto control_decision = CheckControlChars(request);
        if (!control_decision.allowed) {
            return control_decision;
        }
    }

    if (options_.reject_suspicious_patterns) {
        auto pattern_decision = CheckSuspiciousPatterns(request);
        if (!pattern_decision.allowed) {
            return pattern_decision;
        }
    }

    return HttpRequestFilterDecision::Allow();
}

HttpRequestFilterDecision HttpRequestFilter::CheckSizeLimits(const BeastHttpRequest& request) const {
    if (request.target().size() > options_.max_target_bytes) {
        return HttpRequestFilterDecision::Deny(http::status::uri_too_long, "request target is too long");
    }

    std::size_t total_header_bytes = 0;
    for (const auto& field : request) {
        const auto name_size = field.name_string().size();
        const auto value_size = field.value().size();
        if (name_size > options_.max_header_field_bytes || value_size > options_.max_header_value_bytes) {
            return HttpRequestFilterDecision::Deny(http::status::request_header_fields_too_large,
                                                  "request header field is too large");
        }
        total_header_bytes += name_size + value_size;
        if (total_header_bytes > options_.max_total_header_bytes) {
            return HttpRequestFilterDecision::Deny(http::status::request_header_fields_too_large,
                                                  "request headers are too large");
        }
    }

    return HttpRequestFilterDecision::Allow();
}

HttpRequestFilterDecision HttpRequestFilter::CheckControlChars(const BeastHttpRequest& request) const {
    if (ContainsControlChars(ToString(request.target()))) {
        return HttpRequestFilterDecision::Deny(http::status::bad_request, "request target contains control characters");
    }
    for (const auto& field : request) {
        if (ContainsControlChars(ToString(field.name_string())) || ContainsControlChars(ToString(field.value()))) {
            return HttpRequestFilterDecision::Deny(http::status::bad_request, "request header contains control characters");
        }
    }
    return HttpRequestFilterDecision::Allow();
}

HttpRequestFilterDecision HttpRequestFilter::CheckSuspiciousPatterns(const BeastHttpRequest& request) const {
    if (MatchesSuspiciousPattern(ToString(request.target()))) {
        return HttpRequestFilterDecision::Deny(http::status::forbidden, "request target matched a security filter");
    }

    for (const auto& field : request) {
        if (MatchesSuspiciousPattern(ToString(field.value()))) {
            return HttpRequestFilterDecision::Deny(http::status::forbidden, "request header matched a security filter");
        }
    }

    if (MatchesSuspiciousPattern(request.body())) {
        return HttpRequestFilterDecision::Deny(http::status::forbidden, "request body matched a security filter");
    }

    return HttpRequestFilterDecision::Allow();
}

bool HttpRequestFilter::ContainsControlChars(std::string_view value) const noexcept {
    return std::any_of(value.begin(), value.end(), [](unsigned char ch) {
        return std::iscntrl(ch) != 0 && ch != '\t';
    });
}

bool HttpRequestFilter::MatchesSuspiciousPattern(std::string_view value) const {
    if (value.empty()) {
        return false;
    }
    const auto input = std::string(value);
    return std::any_of(suspicious_patterns_.begin(), suspicious_patterns_.end(), [&](const std::regex& pattern) {
        return std::regex_search(input, pattern);
    });
}

} // namespace net
