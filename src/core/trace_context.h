#pragma once

#include <chrono>
#include <cstdint>
#include <random>
#include <string>

namespace core {

struct TraceContext {
    std::string trace_id;
    std::string span_id;
    std::string user_uuid;
};

inline thread_local TraceContext* current_trace = nullptr;

class TraceScope {
public:
    explicit TraceScope(TraceContext& ctx) noexcept
        : prev_(current_trace) {
        current_trace = &ctx;
    }

    ~TraceScope() noexcept {
        current_trace = prev_;
    }

    TraceScope(const TraceScope&) = delete;
    TraceScope& operator=(const TraceScope&) = delete;

private:
    TraceContext* prev_;
};

inline std::string GenerateTraceId() {
    static thread_local std::mt19937_64 rng(
        static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()));

    const std::uint64_t hi = rng();
    const std::uint64_t lo = rng();

    static constexpr char hex_chars[] = "0123456789abcdef";
    std::string id;
    id.reserve(32);
    for (int i = 60; i >= 0; i -= 4) {
        id.push_back(hex_chars[(hi >> i) & 0xF]);
    }
    for (int i = 60; i >= 0; i -= 4) {
        id.push_back(hex_chars[(lo >> i) & 0xF]);
    }
    return id;
}

inline std::string_view CurrentTraceId() noexcept {
    if (current_trace && !current_trace->trace_id.empty()) {
        return current_trace->trace_id;
    }
    return "-";
}

} // namespace core
