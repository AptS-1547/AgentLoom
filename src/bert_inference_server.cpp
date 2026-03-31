/**
 * @file bert_inference_server.cpp
 * @brief BERT gRPC 推理服务
 */

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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <stdexcept>
#include <string>
#include <thread>

#include <grpc/grpc.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>
#include <spdlog/spdlog.h>

#include "bert_inference.grpc.pb.h"
#include "logger.h"
#include "onnx_model.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

using bert_inference::BERTInference;
using bert_inference::PredictRequest;
using bert_inference::PredictResponse;
using bert_inference::PredictBatchRequest;
using bert_inference::PredictBatchResponse;

namespace {

void UpdateMax(std::atomic<int64_t>& target, int64_t candidate) {
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

class RuntimeStats {
public:
    RuntimeStats() = default;

    void BeginRequest(size_t sample_count, bool is_batch) {
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

    void EndRequest(bool success, int64_t latency_us, const std::string& error_message) {
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

    RuntimeStatsSnapshot Snapshot() const {
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
        int slow_request_ms)
        : stats_(stats),
          method_name_(std::move(method_name)),
          sample_count_(sample_count),
          slow_request_ms_(slow_request_ms),
          started_at_(std::chrono::steady_clock::now()) {
        stats_.BeginRequest(sample_count_, is_batch);
    }

    void MarkSuccess() {
        success_ = true;
    }

    void MarkFailure(const std::string& error_message) {
        success_ = false;
        error_message_ = error_message;
    }

    ~ScopedRequestStats() {
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

} // namespace

namespace bert {

/**
 * gRPC 服务实现
 */
class BERTInferenceServiceImpl final : public BERTInference::Service {
public:
    BERTInferenceServiceImpl(
        std::unique_ptr<OnnxBERTModel> model,
        RuntimeStats& stats,
        int slow_request_ms)
        : model_(std::move(model)),
          stats_(stats),
          slow_request_ms_(slow_request_ms) {}

    Status Predict(ServerContext* context,
                   const PredictRequest* request,
                   PredictResponse* response) override {
        (void)context;
        ScopedRequestStats request_stats(stats_, "Predict", 1, false, slow_request_ms_);

        // 验证输入
        if (request->input_ids_size() == 0 || request->attention_mask_size() == 0) {
            response->set_error("Empty input");
            request_stats.MarkFailure("Empty input");
            return Status(StatusCode::INVALID_ARGUMENT, "Empty input");
        }

        if (request->input_ids_size() != request->attention_mask_size()) {
            response->set_error("input_ids and attention_mask size mismatch");
            request_stats.MarkFailure("input_ids and attention_mask size mismatch");
            return Status(StatusCode::INVALID_ARGUMENT, "Size mismatch");
        }

        if (request->personality_size() != 11) {
            response->set_error("personality must have 11 elements");
            request_stats.MarkFailure("personality must have 11 elements");
            return Status(StatusCode::INVALID_ARGUMENT, "Invalid personality size");
        }

        // 转换输入
        std::vector<int64_t> input_ids(request->input_ids().begin(), request->input_ids().end());
        std::vector<int64_t> attention_mask(request->attention_mask().begin(), request->attention_mask().end());
        std::vector<float> personality(request->personality().begin(), request->personality().end());

        // 推理
        auto result = model_->Predict(input_ids, attention_mask, personality);

        if (!result.success) {
            response->set_error(result.error_message);
            request_stats.MarkFailure(result.error_message);
            return Status(StatusCode::INTERNAL, result.error_message);
        }

        // 填充响应
        response->mutable_emotion_logits()->Reserve(static_cast<int>(result.emotion_logits.size()));
        for (float v : result.emotion_logits) {
            response->add_emotion_logits(v);
        }
        response->mutable_behavior_logits()->Reserve(static_cast<int>(result.behavior_logits.size()));
        for (float v : result.behavior_logits) {
            response->add_behavior_logits(v);
        }
        response->mutable_tone_logits()->Reserve(static_cast<int>(result.tone_logits.size()));
        for (float v : result.tone_logits) {
            response->add_tone_logits(v);
        }
        response->set_intensity(result.intensity);
        response->mutable_response_length_logits()->Reserve(
            static_cast<int>(result.response_length_logits.size())
        );
        for (float v : result.response_length_logits) {
            response->add_response_length_logits(v);
        }

        request_stats.MarkSuccess();
        return Status::OK;
    }

    Status PredictBatch(ServerContext* context,
                        const PredictBatchRequest* request,
                        PredictBatchResponse* response) override {
        (void)context;

        size_t batch_size = request->batch_size();
        size_t seq_len = request->seq_length();
        ScopedRequestStats request_stats(
            stats_,
            "PredictBatch",
            batch_size,
            true,
            slow_request_ms_
        );

        if (batch_size == 0) {
            response->set_error("Empty batch");
            request_stats.MarkFailure("Empty batch");
            return Status(StatusCode::INVALID_ARGUMENT, "Empty batch");
        }

        // 验证输入尺寸
        size_t expected_input_size = batch_size * seq_len;
        size_t expected_personality_size = batch_size * 11;

        if (static_cast<size_t>(request->input_ids_size()) != expected_input_size ||
            static_cast<size_t>(request->attention_mask_size()) != expected_input_size ||
            static_cast<size_t>(request->personality_size()) != expected_personality_size) {
            response->set_error("Input size mismatch");
            request_stats.MarkFailure("Input size mismatch");
            return Status(StatusCode::INVALID_ARGUMENT, "Input size mismatch");
        }

        // 转换输入
        std::vector<int64_t> input_ids(request->input_ids().begin(), request->input_ids().end());
        std::vector<int64_t> attention_mask(request->attention_mask().begin(), request->attention_mask().end());
        std::vector<float> personality(request->personality().begin(), request->personality().end());

        // 批量推理
        auto results = model_->PredictBatch(input_ids, attention_mask, personality, batch_size, seq_len);

        if (results.empty()) {
            response->set_error("Batch inference failed");
            request_stats.MarkFailure("Batch inference failed");
            return Status(StatusCode::INTERNAL, "Batch inference failed");
        }

        // 填充响应（展平）
        response->mutable_emotion_logits()->Reserve(static_cast<int>(batch_size * results.front().emotion_logits.size()));
        response->mutable_behavior_logits()->Reserve(static_cast<int>(batch_size * results.front().behavior_logits.size()));
        response->mutable_tone_logits()->Reserve(static_cast<int>(batch_size * results.front().tone_logits.size()));
        response->mutable_intensity()->Reserve(static_cast<int>(batch_size));
        response->mutable_response_length_logits()->Reserve(
            static_cast<int>(batch_size * results.front().response_length_logits.size())
        );
        for (const auto& result : results) {
            for (float v : result.emotion_logits) {
                response->add_emotion_logits(v);
            }
            for (float v : result.behavior_logits) {
                response->add_behavior_logits(v);
            }
            for (float v : result.tone_logits) {
                response->add_tone_logits(v);
            }
            response->add_intensity(result.intensity);
            for (float v : result.response_length_logits) {
                response->add_response_length_logits(v);
            }
        }

        request_stats.MarkSuccess();
        return Status::OK;
    }

private:
    std::unique_ptr<OnnxBERTModel> model_;
    RuntimeStats& stats_;
    int slow_request_ms_ = 0;
};

} // namespace bert

namespace {

struct ServerOptions {
    bert::ModelRuntimeOptions model_runtime;
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

bool IsFlag(const std::string& arg) {
    return arg.rfind("--", 0) == 0;
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

ServerOptions ParseServerOptions(int argc, char** argv) {
    ServerOptions options;
    bool port_assigned = false;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];

        if (!port_assigned && !IsFlag(arg)) {
            options.port = arg;
            port_assigned = true;
            continue;
        }

        if (arg == "--provider") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--provider requires a value");
            }
            options.model_runtime.execution_provider = argv[++i];
        } else if (arg == "--host") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--host requires a value");
            }
            options.host = argv[++i];
        } else if (arg == "--log-dir") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--log-dir requires a value");
            }
            options.log_dir = argv[++i];
        } else if (arg == "--cuda-device") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--cuda-device requires a value");
            }
            options.model_runtime.cuda_device_id = ParseIntValue(arg, argv[++i]);
        } else if (arg == "--intra-op") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--intra-op requires a value");
            }
            options.model_runtime.intra_op_num_threads = ParseIntValue(arg, argv[++i]);
        } else if (arg == "--inter-op") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--inter-op requires a value");
            }
            options.model_runtime.inter_op_num_threads = ParseIntValue(arg, argv[++i]);
        } else if (arg == "--disable-mem-pattern") {
            options.model_runtime.enable_mem_pattern = false;
        } else if (arg == "--disable-cpu-mem-arena") {
            options.model_runtime.enable_cpu_mem_arena = false;
        } else if (arg == "--no-provider-fallback") {
            options.model_runtime.allow_cpu_fallback = false;
        } else if (arg == "--grpc-num-cqs") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--grpc-num-cqs requires a value");
            }
            options.grpc_num_cqs = ParseIntValue(arg, argv[++i]);
        } else if (arg == "--grpc-min-pollers") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--grpc-min-pollers requires a value");
            }
            options.grpc_min_pollers = ParseIntValue(arg, argv[++i]);
        } else if (arg == "--grpc-max-pollers") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--grpc-max-pollers requires a value");
            }
            options.grpc_max_pollers = ParseIntValue(arg, argv[++i]);
        } else if (arg == "--max-recv-mb") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--max-recv-mb requires a value");
            }
            options.max_receive_message_mb = ParseIntValue(arg, argv[++i]);
        } else if (arg == "--max-send-mb") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--max-send-mb requires a value");
            }
            options.max_send_message_mb = ParseIntValue(arg, argv[++i]);
        } else if (arg == "--stats-log-interval-seconds") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--stats-log-interval-seconds requires a value");
            }
            options.stats_log_interval_seconds = ParseIntValue(arg, argv[++i]);
        } else if (arg == "--slow-request-ms") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--slow-request-ms requires a value");
            }
            options.slow_request_ms = ParseIntValue(arg, argv[++i]);
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (options.grpc_num_cqs <= 0) {
        options.grpc_num_cqs = ResolveDefaultGrpcNumCqs();
    }
    if (options.grpc_min_pollers <= 0) {
        options.grpc_min_pollers = 1;
    }
    if (options.grpc_max_pollers <= 0) {
        options.grpc_max_pollers = ResolveDefaultGrpcMaxPollers();
    }
    if (options.grpc_max_pollers < options.grpc_min_pollers) {
        options.grpc_max_pollers = options.grpc_min_pollers;
    }
    if (options.stats_log_interval_seconds < 0) {
        options.stats_log_interval_seconds = 0;
    }
    if (options.slow_request_ms < 0) {
        options.slow_request_ms = 0;
    }

    return options;
}

} // namespace

void PrintUsage(const char* program) {
    std::cerr << "Usage: " << program << " <model_path> [port] [options]" << std::endl;
    std::cerr << "  model_path: Path to joint_model.onnx" << std::endl;
    std::cerr << "  port: Server port (default: 50051)" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --host <ip>                    Bind address (default: 127.0.0.1)" << std::endl;
    std::cerr << "  --log-dir <path>              Persistent log directory (default: ./logs)" << std::endl;
    std::cerr << "  --provider <auto|cpu|cuda>   Execution provider preference (default: auto)" << std::endl;
    std::cerr << "  --cuda-device <id>           CUDA device id (default: 0)" << std::endl;
    std::cerr << "  --intra-op <n>               ONNX Runtime intra-op threads" << std::endl;
    std::cerr << "  --inter-op <n>               ONNX Runtime inter-op threads" << std::endl;
    std::cerr << "  --disable-mem-pattern        Disable ORT memory pattern cache" << std::endl;
    std::cerr << "  --disable-cpu-mem-arena      Disable ORT CPU memory arena" << std::endl;
    std::cerr << "  --no-provider-fallback       Fail instead of falling back from CUDA to CPU" << std::endl;
    std::cerr << "  --grpc-num-cqs <n>           gRPC sync server completion queues" << std::endl;
    std::cerr << "  --grpc-min-pollers <n>       gRPC sync server min pollers" << std::endl;
    std::cerr << "  --grpc-max-pollers <n>       gRPC sync server max pollers" << std::endl;
    std::cerr << "  --max-recv-mb <n>            gRPC max receive message size in MB" << std::endl;
    std::cerr << "  --max-send-mb <n>            gRPC max send message size in MB" << std::endl;
    std::cerr << "  --stats-log-interval-seconds <n>  Periodic stats log interval (default: 30, 0 disables)" << std::endl;
    std::cerr << "  --slow-request-ms <n>        Log requests slower than this threshold (default: 250)" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        PrintUsage(argv[0]);
        return 1;
    }

    std::filesystem::path model_path;
#ifdef _WIN32
    const int needed = MultiByteToWideChar(CP_ACP, 0, argv[1], -1, nullptr, 0);
    if (needed == 0) {
        std::cerr << "Conversion failed" << std::endl;
        return 1;
    }

    std::wstring wide_model_path(static_cast<size_t>(needed), L'\0');
    MultiByteToWideChar(CP_ACP, 0, argv[1], -1, wide_model_path.data(), needed);
    wide_model_path.pop_back();
    model_path = std::filesystem::path(wide_model_path);
#else
    model_path = std::filesystem::path(argv[1]);
#endif

    ServerOptions options;
    try {
        options = ParseServerOptions(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[Server] " << e.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

    logging::LoggerOptions log_options;
    log_options.log_dir = options.log_dir;
    if (!logging::Initialize(log_options)) {
        return 1;
    }

    std::string server_address = options.host + ":" + options.port;
    const auto log_file = logging::GetLogFilePath(log_options);

    spdlog::info("[Server] Persistent logging enabled: {}", log_file.string());
    spdlog::info("[Server] Loading ONNX model...");
    spdlog::info("[Server] Bind address: {}", server_address);
    spdlog::info(
        "[Server] Provider preference: {}, CUDA device: {}, gRPC CQs: {}, pollers: {}-{}, "
        "stats_log_interval_seconds: {}, slow_request_ms: {}",
        options.model_runtime.execution_provider,
        options.model_runtime.cuda_device_id,
        options.grpc_num_cqs,
        options.grpc_min_pollers,
        options.grpc_max_pollers,
        options.stats_log_interval_seconds,
        options.slow_request_ms
    );

    // 加载模型
    auto model = std::make_unique<bert::OnnxBERTModel>();
    if (!model->LoadModel(model_path, options.model_runtime)) {
        spdlog::error("[Server] Failed to load model");
        logging::Shutdown();
        return 1;
    }

    spdlog::info("[Server] Model loaded successfully");
    spdlog::info("[Server] {}", model->GetInfo());
    spdlog::info("[Server] Starting gRPC server on {}", server_address);

    // 创建服务
    RuntimeStats stats;
    bert::BERTInferenceServiceImpl service(std::move(model), stats, options.slow_request_ms);

    grpc::EnableDefaultHealthCheckService(true);
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(options.max_receive_message_mb * 1024 * 1024);
    builder.SetMaxSendMessageSize(options.max_send_message_mb * 1024 * 1024);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::NUM_CQS, options.grpc_num_cqs);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::MIN_POLLERS, options.grpc_min_pollers);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::MAX_POLLERS, options.grpc_max_pollers);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    if (server == nullptr) {
        spdlog::error("[Server] Failed to start gRPC server");
        logging::Shutdown();
        return 1;
    }

    if (grpc::HealthCheckServiceInterface* health = server->GetHealthCheckService();
        health != nullptr) {
        health->SetServingStatus("bert_inference.BERTInference", true);
        health->SetServingStatus(true);
        spdlog::info(
            "[Server] Health check service enabled for grpc.health.v1.Health "
            "(service=bert_inference.BERTInference)"
        );
    }

    spdlog::info("[Server] Ready");

    std::jthread stats_thread(
        [&stats, server_address, interval = options.stats_log_interval_seconds](std::stop_token stop_token) {
            LogStatsPeriodically(stats, server_address, stop_token, interval);
        }
    );

    // 等待终止
    server->Wait();
    spdlog::info("[Server] Shutdown complete");
    logging::Shutdown();

    return 0;
}
