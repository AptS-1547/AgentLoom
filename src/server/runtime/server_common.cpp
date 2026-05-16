/**
 * @file server_common.cpp
 * @brief gRPC 服务端公共基础设施实现
 */

#include "server_common.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace server_common {

ProcessMemoryUsage ReadProcessMemoryUsage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);

    if (!GetProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
            sizeof(counters))) {
        return {};
    }

    ProcessMemoryUsage usage;
    usage.working_set_bytes = static_cast<size_t>(counters.WorkingSetSize);
    usage.private_usage_bytes = static_cast<size_t>(counters.PrivateUsage);
    usage.peak_working_set_bytes = static_cast<size_t>(counters.PeakWorkingSetSize);
    return usage;
#elif defined(__linux__)
    const auto read_status_value_bytes = [](const char* key) -> std::optional<size_t> {
        std::ifstream status("/proc/self/status");
        if (!status) {
            return std::nullopt;
        }

        std::string label;
        size_t value_kb = 0;
        std::string unit;
        while (status >> label >> value_kb >> unit) {
            if (label == key) {
                return value_kb * 1024;
            }
            std::string rest_of_line;
            std::getline(status, rest_of_line);
        }

        return std::nullopt;
    };

    ProcessMemoryUsage usage;
    if (const auto working_set = read_status_value_bytes("VmRSS:")) {
        usage.working_set_bytes = *working_set;
    }
    if (const auto peak_working_set = read_status_value_bytes("VmHWM:")) {
        usage.peak_working_set_bytes = *peak_working_set;
    }
    if (const auto private_rss = read_status_value_bytes("RssAnon:")) {
        usage.private_usage_bytes = *private_rss;
    } else if (const auto virtual_size = read_status_value_bytes("VmSize:")) {
        usage.private_usage_bytes = *virtual_size;
    }
    return usage;
#else
    return {};
#endif
}

void RuntimeStats::BeginRequest(size_t sample_count, bool is_batch) {
    inflight_requests_.fetch_add(1, std::memory_order_relaxed);
    total_rpc_requests_.fetch_add(1, std::memory_order_relaxed);
    total_samples_.fetch_add(static_cast<int64_t>(sample_count), std::memory_order_relaxed);
    if (is_batch) {
        total_predict_batch_calls_.fetch_add(1, std::memory_order_relaxed);
    } else {
        total_predict_calls_.fetch_add(1, std::memory_order_relaxed);
    }
    UpdateMax(max_batch_size_, static_cast<int64_t>(sample_count));
}

void RuntimeStats::EndRequest(bool success, int64_t latency_us, const std::string& error_message) {
    inflight_requests_.fetch_sub(1, std::memory_order_relaxed);
    total_latency_us_.fetch_add(latency_us, std::memory_order_relaxed);
    UpdateMax(max_latency_us_, latency_us);
    if (!success) {
        total_errors_.fetch_add(1, std::memory_order_relaxed);
        if (!error_message.empty()) {
            std::lock_guard<std::mutex> lock(last_error_mutex_);
            last_error_ = error_message;
        }
    }
}

RuntimeStatsSnapshot RuntimeStats::Snapshot() const {
    RuntimeStatsSnapshot snapshot;
    snapshot.uptime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started_at_
    ).count();
    snapshot.inflight_requests = inflight_requests_.load(std::memory_order_relaxed);
    snapshot.total_rpc_requests = total_rpc_requests_.load(std::memory_order_relaxed);
    snapshot.total_predict_calls = total_predict_calls_.load(std::memory_order_relaxed);
    snapshot.total_predict_batch_calls = total_predict_batch_calls_.load(std::memory_order_relaxed);
    snapshot.total_samples = total_samples_.load(std::memory_order_relaxed);
    snapshot.total_errors = total_errors_.load(std::memory_order_relaxed);
    snapshot.max_batch_size = max_batch_size_.load(std::memory_order_relaxed);

    const int64_t total_latency_us = total_latency_us_.load(std::memory_order_relaxed);
    snapshot.total_latency_us = total_latency_us;
    if (snapshot.total_rpc_requests > 0) {
        snapshot.average_latency_ms =
            static_cast<double>(total_latency_us) /
            static_cast<double>(snapshot.total_rpc_requests) /
            1000.0;
    }
    snapshot.max_latency_ms =
        static_cast<double>(max_latency_us_.load(std::memory_order_relaxed)) / 1000.0;

    const ProcessMemoryUsage memory_usage = ReadProcessMemoryUsage();
    snapshot.working_set_mb =
        static_cast<double>(memory_usage.working_set_bytes) / (1024.0 * 1024.0);
    snapshot.private_usage_mb =
        static_cast<double>(memory_usage.private_usage_bytes) / (1024.0 * 1024.0);
    snapshot.peak_working_set_mb =
        static_cast<double>(memory_usage.peak_working_set_bytes) / (1024.0 * 1024.0);

    std::lock_guard<std::mutex> lock(last_error_mutex_);
    snapshot.last_error = last_error_;
    return snapshot;
}

ScopedRequestStats::ScopedRequestStats(
    RuntimeStats& stats,
    std::string method_name,
    size_t sample_count,
    bool is_batch,
    int slow_request_ms)
    : stats_(stats),
      method_name_(std::move(method_name)),
      sample_count_(sample_count),
      slow_request_ms_(slow_request_ms),
      started_at_(std::chrono::steady_clock::now()) {
    stats_.BeginRequest(sample_count_, is_batch);
}

void ScopedRequestStats::MarkSuccess() {
    success_ = true;
}

void ScopedRequestStats::MarkFailure(const std::string& error_message) {
    success_ = false;
    error_message_ = error_message;
}

ScopedRequestStats::~ScopedRequestStats() {
    const int64_t latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started_at_
    ).count();
    stats_.EndRequest(success_, latency_us, error_message_);

    if (slow_request_ms_ > 0 && latency_us >= static_cast<int64_t>(slow_request_ms_) * 1000) {
        if (!error_message_.empty()) {
            spdlog::warn(
                "[ServerSlowRequest] method={} samples={} latency_ms={:.3f} success={} error={}",
                method_name_,
                sample_count_,
                static_cast<double>(latency_us) / 1000.0,
                success_,
                error_message_
            );
        } else {
            spdlog::warn(
                "[ServerSlowRequest] method={} samples={} latency_ms={:.3f} success={}",
                method_name_,
                sample_count_,
                static_cast<double>(latency_us) / 1000.0,
                success_
            );
        }
    }
}

void LogStatsPeriodically(
    const RuntimeStats& stats,
    const std::string& bind_address,
    std::stop_token stop_token,
    int interval_seconds) {
    if (interval_seconds <= 0) {
        return;
    }

    RuntimeStatsSnapshot previous_snapshot;

    while (!stop_token.stop_requested()) {
        for (int second = 0; second < interval_seconds; ++second) {
            if (stop_token.stop_requested()) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        const RuntimeStatsSnapshot snapshot = stats.Snapshot();
        const int64_t interval_rpc = snapshot.total_rpc_requests - previous_snapshot.total_rpc_requests;
        const int64_t interval_samples = snapshot.total_samples - previous_snapshot.total_samples;
        const int64_t interval_errors = snapshot.total_errors - previous_snapshot.total_errors;
        const int64_t interval_latency_us = snapshot.total_latency_us - previous_snapshot.total_latency_us;
        double interval_average_latency_ms = 0.0;
        if (interval_rpc > 0) {
            interval_average_latency_ms =
                static_cast<double>(interval_latency_us) /
                static_cast<double>(interval_rpc) /
                1000.0;
        }

        spdlog::info(
            "[ServerStats] bind={} uptime_ms={} inflight={} total_rpc={} interval_rpc={} "
            "total_samples={} interval_samples={} total_errors={} interval_errors={} "
            "avg_latency_ms={:.3f} interval_avg_latency_ms={:.3f} max_latency_ms={:.3f} "
            "max_batch_size={} working_set_mb={:.2f} private_usage_mb={:.2f} peak_working_set_mb={:.2f}",
            bind_address,
            snapshot.uptime_ms,
            snapshot.inflight_requests,
            snapshot.total_rpc_requests,
            interval_rpc,
            snapshot.total_samples,
            interval_samples,
            snapshot.total_errors,
            interval_errors,
            snapshot.average_latency_ms,
            interval_average_latency_ms,
            snapshot.max_latency_ms,
            snapshot.max_batch_size,
            snapshot.working_set_mb,
            snapshot.private_usage_mb,
            snapshot.peak_working_set_mb
        );
        if (!snapshot.last_error.empty()) {
            spdlog::warn("[ServerStats] last_error={}", snapshot.last_error);
        }

        previous_snapshot = snapshot;
    }
}

int ParseIntValue(const std::string& flag, const char* value) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer for " + flag + ": " + value);
    }
}

int ResolveDefaultGrpcNumCqs() {
    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    return std::clamp(static_cast<int>(hw_threads / 8), 1, 4);
}

int ResolveDefaultGrpcMaxPollers() {
    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    return std::clamp(static_cast<int>(hw_threads / 2), 2, 16);
}

} // namespace server_common
