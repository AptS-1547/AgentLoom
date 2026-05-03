#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "bert_inference.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using bert_inference::BERTInference;
using bert_inference::PredictRequest;
using bert_inference::PredictResponse;

namespace {

struct BenchmarkOptions {
    std::string target = "localhost:50051";
    int concurrency = 8;
    int requests_per_worker = 100;
    int seq_len = 64;
    int warmup_requests = 10;
};

struct Summary {
    double total_seconds = 0.0;
    size_t total_requests = 0;
    size_t success_requests = 0;
    size_t failed_requests = 0;
    double average_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double throughput_rps = 0.0;
};

class BERTClient {
public:
    explicit BERTClient(const std::shared_ptr<Channel>& channel)
        : stub_(BERTInference::NewStub(channel)) {}

    bool Predict(const std::vector<int64_t>& input_ids,
                 const std::vector<int64_t>& attention_mask,
                 const std::vector<float>& personality) {
        PredictRequest request;
        for (auto value : input_ids) {
            request.add_input_ids(value);
        }
        for (auto value : attention_mask) {
            request.add_attention_mask(value);
        }
        for (auto value : personality) {
            request.add_personality(value);
        }

        PredictResponse response;
        ClientContext context;
        Status status = stub_->Predict(&context, request, &response);
        return status.ok() && response.error().empty();
    }

private:
    std::unique_ptr<BERTInference::Stub> stub_;
};

int ParseInt(const std::string& flag, const char* value) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer for " + flag + ": " + value);
    }
}

BenchmarkOptions ParseArgs(int argc, char** argv) {
    BenchmarkOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--target") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--target requires a value");
            }
            options.target = argv[++i];
        } else if (arg == "--concurrency") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--concurrency requires a value");
            }
            options.concurrency = ParseInt(arg, argv[++i]);
        } else if (arg == "--requests") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--requests requires a value");
            }
            options.requests_per_worker = ParseInt(arg, argv[++i]);
        } else if (arg == "--seq-len") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--seq-len requires a value");
            }
            options.seq_len = ParseInt(arg, argv[++i]);
        } else if (arg == "--warmup") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--warmup requires a value");
            }
            options.warmup_requests = ParseInt(arg, argv[++i]);
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (options.concurrency <= 0 || options.requests_per_worker <= 0 || options.seq_len <= 0) {
        throw std::runtime_error("concurrency, requests, and seq-len must be > 0");
    }
    if (options.warmup_requests < 0) {
        throw std::runtime_error("warmup must be >= 0");
    }

    return options;
}

double Percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double index = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(index);
    const size_t upper = std::min(values.size() - 1, lower + 1);
    const double weight = index - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

Summary RunBenchmark(const BenchmarkOptions& options) {
    auto channel = grpc::CreateChannel(options.target, grpc::InsecureChannelCredentials());

    std::vector<int64_t> input_ids(static_cast<size_t>(options.seq_len), 101);
    std::vector<int64_t> attention_mask(static_cast<size_t>(options.seq_len), 1);
    std::vector<float> personality(11, 0.5f);

    // Warmup on a single client to stabilize lazy initialization.
    BERTClient warmup_client(channel);
    for (int i = 0; i < options.warmup_requests; ++i) {
        warmup_client.Predict(input_ids, attention_mask, personality);
    }

    std::mutex start_mutex;
    std::condition_variable start_cv;
    bool start_flag = false;

    std::vector<std::thread> workers;
    std::vector<std::vector<double>> latencies(static_cast<size_t>(options.concurrency));
    std::vector<size_t> failures(static_cast<size_t>(options.concurrency), 0);

    for (int worker_index = 0; worker_index < options.concurrency; ++worker_index) {
        workers.emplace_back([&, worker_index]() {
            BERTClient client(channel);

            {
                std::unique_lock<std::mutex> lock(start_mutex);
                start_cv.wait(lock, [&]() { return start_flag; });
            }

            auto& worker_latencies = latencies[static_cast<size_t>(worker_index)];
            worker_latencies.reserve(static_cast<size_t>(options.requests_per_worker));

            for (int request_index = 0; request_index < options.requests_per_worker; ++request_index) {
                const auto start = std::chrono::steady_clock::now();
                const bool ok = client.Predict(input_ids, attention_mask, personality);
                const auto end = std::chrono::steady_clock::now();

                const double elapsed_ms =
                    std::chrono::duration<double, std::milli>(end - start).count();
                worker_latencies.push_back(elapsed_ms);

                if (!ok) {
                    ++failures[static_cast<size_t>(worker_index)];
                }
            }
        });
    }

    const auto begin = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(start_mutex);
        start_flag = true;
    }
    start_cv.notify_all();

    for (auto& worker : workers) {
        worker.join();
    }
    const auto end = std::chrono::steady_clock::now();

    std::vector<double> all_latencies;
    all_latencies.reserve(static_cast<size_t>(options.concurrency * options.requests_per_worker));

    size_t failed_requests = 0;
    for (size_t i = 0; i < latencies.size(); ++i) {
        all_latencies.insert(all_latencies.end(), latencies[i].begin(), latencies[i].end());
        failed_requests += failures[i];
    }

    Summary summary;
    summary.total_seconds = std::chrono::duration<double>(end - begin).count();
    summary.total_requests = all_latencies.size();
    summary.failed_requests = failed_requests;
    summary.success_requests = summary.total_requests - summary.failed_requests;

    if (!all_latencies.empty()) {
        summary.average_ms = std::accumulate(all_latencies.begin(), all_latencies.end(), 0.0) /
                             static_cast<double>(all_latencies.size());
        summary.p50_ms = Percentile(all_latencies, 0.50);
        summary.p95_ms = Percentile(all_latencies, 0.95);
        summary.p99_ms = Percentile(all_latencies, 0.99);
    }

    if (summary.total_seconds > 0.0) {
        summary.throughput_rps =
            static_cast<double>(summary.success_requests) / summary.total_seconds;
    }

    return summary;
}

void PrintUsage(const char* program) {
    std::cerr << "Usage: " << program
              << " [--target host:port] [--concurrency N] [--requests N]"
              << " [--seq-len N] [--warmup N]" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const BenchmarkOptions options = ParseArgs(argc, argv);

        std::cout << "Benchmark target: " << options.target
                  << ", concurrency=" << options.concurrency
                  << ", requests/worker=" << options.requests_per_worker
                  << ", seq_len=" << options.seq_len
                  << ", warmup=" << options.warmup_requests << std::endl;

        const Summary summary = RunBenchmark(options);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total requests: " << summary.total_requests
                  << ", success=" << summary.success_requests
                  << ", failed=" << summary.failed_requests << std::endl;
        std::cout << "Elapsed: " << summary.total_seconds << " s" << std::endl;
        std::cout << "Throughput: " << summary.throughput_rps << " req/s" << std::endl;
        std::cout << "Average: " << summary.average_ms << " ms" << std::endl;
        std::cout << "p50: " << summary.p50_ms << " ms" << std::endl;
        std::cout << "p95: " << summary.p95_ms << " ms" << std::endl;
        std::cout << "p99: " << summary.p99_ms << " ms" << std::endl;
        return summary.failed_requests == 0 ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }
}
