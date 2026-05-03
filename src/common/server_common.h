/**
 * @file server_common.h
 * @brief gRPC 服务端公共基础设施（统计、内存监控、参数解析）
 */

#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <Psapi.h>
#pragma comment(lib, "Psapi.lib")
#endif

#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>

// 日志宏
#define LOG_INFO(...)    spdlog::info(__VA_ARGS__)
#define LOG_WARN(...)    spdlog::warn(__VA_ARGS__)
#define LOG_ERROR(...)   spdlog::error(__VA_ARGS__)
#define LOG_DEBUG(...)   spdlog::debug(__VA_ARGS__)

namespace server_common {

inline void UpdateMax(std::atomic<int64_t>& target, int64_t candidate) {
    int64_t current = target.load(std::memory_order_relaxed);
    while (candidate > current &&
           !target.compare_exchange_weak(
               current,
               candidate,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

struct RuntimeStatsSnapshot {
    int64_t uptime_ms = 0;
    int64_t inflight_requests = 0;
    int64_t total_rpc_requests = 0;
    int64_t total_predict_calls = 0;
    int64_t total_predict_batch_calls = 0;
    int64_t total_samples = 0;
    int64_t total_errors = 0;
    int64_t total_latency_us = 0;
    double average_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    int64_t max_batch_size = 0;
    double working_set_mb = 0.0;
    double private_usage_mb = 0.0;
    double peak_working_set_mb = 0.0;
    std::string last_error;
};

struct ProcessMemoryUsage {
    size_t working_set_bytes = 0;
    size_t private_usage_bytes = 0;
    size_t peak_working_set_bytes = 0;
};

ProcessMemoryUsage ReadProcessMemoryUsage();

class RuntimeStats {
public:
    RuntimeStats() = default;

    void BeginRequest(size_t sample_count, bool is_batch);
    void EndRequest(bool success, int64_t latency_us, const std::string& error_message);
    RuntimeStatsSnapshot Snapshot() const;

private:
    const std::chrono::steady_clock::time_point started_at_ = std::chrono::steady_clock::now();
    std::atomic<int64_t> inflight_requests_{0};
    std::atomic<int64_t> total_rpc_requests_{0};
    std::atomic<int64_t> total_predict_calls_{0};
    std::atomic<int64_t> total_predict_batch_calls_{0};
    std::atomic<int64_t> total_samples_{0};
    std::atomic<int64_t> total_errors_{0};
    std::atomic<int64_t> total_latency_us_{0};
    std::atomic<int64_t> max_latency_us_{0};
    std::atomic<int64_t> max_batch_size_{0};
    mutable std::mutex last_error_mutex_;
    std::string last_error_;
};

class ScopedRequestStats {
public:
    ScopedRequestStats(
        RuntimeStats& stats,
        std::string method_name,
        size_t sample_count,
        bool is_batch,
        int slow_request_ms);

    void MarkSuccess();
    void MarkFailure(const std::string& error_message);
    ~ScopedRequestStats();

    ScopedRequestStats(const ScopedRequestStats&) = delete;
    ScopedRequestStats& operator=(const ScopedRequestStats&) = delete;

private:
    RuntimeStats& stats_;
    std::string method_name_;
    size_t sample_count_ = 0;
    int slow_request_ms_ = 0;
    std::chrono::steady_clock::time_point started_at_;
    bool success_ = false;
    std::string error_message_;
};

void LogStatsPeriodically(
    const RuntimeStats& stats,
    const std::string& bind_address,
    std::stop_token stop_token,
    int interval_seconds);

struct GrpcServerOptions {
    std::string host = "127.0.0.1";
    std::string port = "50051";
    std::string log_dir = "logs";
    int grpc_num_cqs = 0;
    int grpc_min_pollers = 0;
    int grpc_max_pollers = 0;
    int max_receive_message_mb = 16;
    int max_send_message_mb = 16;
    int stats_log_interval_seconds = 30;
    int slow_request_ms = 250;
};

bool IsFlag(const std::string& arg);
int ParseIntValue(const std::string& flag, const char* value);
int ResolveDefaultGrpcNumCqs();
int ResolveDefaultGrpcMaxPollers();
void ParseGrpcServerOptions(int argc, char** argv, int start_index, GrpcServerOptions& options);

} // namespace server_common
