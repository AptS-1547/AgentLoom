#include "../../src/semantic_cache/semantic_cache_pipeline.h"
#include "../../src/core/result.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <random>
#include <cmath>

using namespace agent::semantic_cache;
using namespace storage;

struct BenchmarkResult {
    std::string operation;
    std::size_t iterations;
    double avg_ms;
    double min_ms;
    double max_ms;
    double p50_ms;
    double p95_ms;
    double p99_ms;
};

std::vector<double> CalculatePercentiles(std::vector<double> latencies) {
    std::sort(latencies.begin(), latencies.end());
    std::size_t n = latencies.size();
    return {
        latencies[n / 2],
        latencies[static_cast<std::size_t>(n * 0.95)],
        latencies[static_cast<std::size_t>(n * 0.99)]
    };
}

BenchmarkResult RunBenchmark(const std::string& operation, std::function<void()> fn, std::size_t iterations) {
    std::vector<double> latencies;
    latencies.reserve(iterations);

    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double total_ms = 0.0;

    for (std::size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    auto percentiles = CalculatePercentiles(latencies);

    return BenchmarkResult{
        operation,
        iterations,
        total_ms / iterations,
        min_ms,
        max_ms,
        percentiles[0],
        percentiles[1],
        percentiles[2]
    };
}

void PrintResult(const BenchmarkResult& result) {
    std::cout << "\n=== " << result.operation << " ===\n";
    std::cout << "Iterations: " << result.iterations << "\n";
    std::cout << "Average:    " << result.avg_ms << " ms\n";
    std::cout << "Min:        " << result.min_ms << " ms\n";
    std::cout << "Max:        " << result.max_ms << " ms\n";
    std::cout << "P50:        " << result.p50_ms << " ms\n";
    std::cout << "P95:        " << result.p95_ms << " ms\n";
    std::cout << "P99:        " << result.p99_ms << " ms\n";
    std::cout << "Throughput: " << (1000.0 / result.avg_ms) << " ops/sec\n";
}

std::vector<float> GenerateRandomEmbedding(std::size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> embedding(dim);
    float norm = 0.0f;

    for (auto& val : embedding) {
        val = dist(rng);
        norm += val * val;
    }

    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (auto& val : embedding) {
            val /= norm;
        }
    }

    return embedding;
}

int main(int argc, char** argv) {
    std::size_t num_samples = (argc >= 2) ? std::stoul(argv[1]) : 10000;

    std::cout << "=== L0 Semantic Cache Core Benchmark ===\n";
    std::cout << "Samples: " << num_samples << "\n\n";

    std::mt19937 rng(42);

    std::cout << "--- Phase 1: Generate Test Data ---\n";
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(num_samples);

    for (std::size_t i = 0; i < num_samples; ++i) {
        embeddings.push_back(GenerateRandomEmbedding(384, rng));
    }
    std::cout << "Generated " << embeddings.size() << " random 384-dim embeddings\n";

    std::cout << "\n--- Phase 2: SIMD Dot Product Benchmark ---\n";
    auto simd_result = RunBenchmark(
        "SIMD Dot Product (384 dim)",
        [&]() {
            static std::size_t idx = 0;
            float score = dot_product_unrolled<384>(
                embeddings[0].data(),
                embeddings[(idx % (embeddings.size() - 1)) + 1].data()
            );
            (void)score;
            idx++;
        },
        100000
    );
    PrintResult(simd_result);

    std::cout << "\n--- Phase 3: Serialization Benchmark ---\n";
    std::vector<CacheRecord> records;
    records.reserve(std::min(num_samples, std::size_t(1000)));

    for (std::size_t i = 0; i < std::min(num_samples, std::size_t(1000)); ++i) {
        CacheRecord record;
        record.embedding = embeddings[i];
        record.input = "测试输入" + std::to_string(i);
        record.response = "测试响应" + std::to_string(i);
        records.push_back(record);
    }

    std::vector<std::string> serialized_records;
    serialized_records.reserve(records.size());

    auto serialize_result = RunBenchmark(
        "CacheRecord Serialization",
        [&]() {
            static std::size_t idx = 0;
            serialized_records.push_back(SerializeCacheRecord(records[idx % records.size()]));
            idx++;
        },
        records.size()
    );
    PrintResult(serialize_result);

    auto deserialize_result = RunBenchmark(
        "CacheRecord Deserialization",
        [&]() {
            static std::size_t idx = 0;
            auto result = DeserializeCacheRecord(serialized_records[idx % serialized_records.size()]);
            idx++;
        },
        serialized_records.size()
    );
    PrintResult(deserialize_result);

    std::cout << "\n--- Phase 4: Top-K Search Simulation ---\n";
    std::size_t search_size = std::min(embeddings.size(), std::size_t(10000));
    std::vector<float> query_embedding = embeddings[0];

    auto search_result = RunBenchmark(
        "Top-K Search (k=8, n=" + std::to_string(search_size) + ")",
        [&]() {
            std::vector<std::pair<float, std::size_t>> scores;
            scores.reserve(search_size);

            for (std::size_t i = 0; i < search_size; ++i) {
                float score = dot_product_unrolled<384>(
                    query_embedding.data(),
                    embeddings[i % embeddings.size()].data()
                );
                scores.emplace_back(score, i);
            }

            std::partial_sort(
                scores.begin(),
                scores.begin() + 8,
                scores.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; }
            );
        },
        100
    );
    PrintResult(search_result);

    std::cout << "\n=== Benchmark Complete ===\n";
    std::cout << "\nSummary:\n";
    std::cout << "- SIMD Dot Product: " << simd_result.avg_ms << " ms/op (~"
              << (simd_result.avg_ms * 1000.0) << " µs)\n";
    std::cout << "- Serialization:    " << serialize_result.avg_ms << " ms/op\n";
    std::cout << "- Deserialization:  " << deserialize_result.avg_ms << " ms/op\n";
    std::cout << "- Top-K Search:     " << search_result.avg_ms << " ms/op\n";

    return 0;
}
