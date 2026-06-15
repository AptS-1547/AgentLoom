#include "../../src/semantic_cache/semantic_cache_pipeline.h"
#include "../../src/vector/hf_tokenizer.h"
#include "../../src/vector/onnx_text_embedding_model.h"
#include "../../src/core/result.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <functional>
#include <limits>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <fstream>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

using namespace agent::semantic_cache;
using namespace storage;

struct MemorySnapshot {
    size_t working_set_mb;
    size_t private_bytes_mb;
    size_t peak_working_set_mb;
};

MemorySnapshot GetMemoryUsage() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));

    return MemorySnapshot{
        pmc.WorkingSetSize / (1024 * 1024),
        pmc.PrivateUsage / (1024 * 1024),
        pmc.PeakWorkingSetSize / (1024 * 1024)
    };
}

void PrintMemoryUsage(const std::string& phase, const MemorySnapshot& mem) {
    std::cout << "[" << phase << "] Memory: "
              << "Working Set = " << mem.working_set_mb << " MB, "
              << "Private = " << mem.private_bytes_mb << " MB, "
              << "Peak = " << mem.peak_working_set_mb << " MB\n";
}

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

std::vector<std::string> LoadTestQueries(const std::string& dataset_path, std::size_t max_samples) {
    std::vector<std::string> queries;
    queries.reserve(max_samples);

    std::ifstream file(dataset_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open dataset file: " << dataset_path << "\n";
        std::cerr << "Using synthetic test data instead.\n";

        // Fallback to synthetic data
        for (std::size_t i = 0; i < max_samples; ++i) {
            queries.push_back("这是一个测试查询句子，用于性能基准测试 " + std::to_string(i));
        }
        return queries;
    }

    std::string line;
    while (std::getline(file, line) && queries.size() < max_samples) {
        if (!line.empty() && line.size() > 10) {
            queries.push_back(line);
        }
    }

    if (queries.empty()) {
        std::cerr << "Warning: No valid queries loaded from dataset.\n";
        queries.push_back("默认测试查询");
    }

    return queries;
}

int main(int argc, char** argv) {
    std::string tokenizer_path = "D:/Users/21405/source/repos/AgentBackendPredict/onnx_models/minilm/tokenizer.json";
    std::string model_path = "D:/Users/21405/source/repos/AgentBackendPredict/onnx_models/minilm/model.onnx";
    std::string dataset_path = (argc >= 2) ? argv[1] : "";
    std::size_t num_samples = (argc >= 3) ? std::stoul(argv[2]) : 1000;
    std::string execution_provider = (argc >= 4) ? argv[3] : "auto";

    std::cout << "=== Full Pipeline Benchmark ===\n";
    std::cout << "Tokenizer:  " << tokenizer_path << "\n";
    std::cout << "Model:      " << model_path << "\n";
    std::cout << "Samples:    " << num_samples << "\n";
    std::cout << "Provider:   " << execution_provider << "\n\n";

    auto mem_start = GetMemoryUsage();
    PrintMemoryUsage("Startup", mem_start);

    // Load tokenizer
    std::cout << "\n--- Phase 1: Load HfTokenizer ---\n";
    auto tokenizer_result = vector::HfTokenizer::LoadFromFile(tokenizer_path);
    if (!tokenizer_result.ok()) {
        std::cerr << "Failed to load tokenizer: " << tokenizer_result.status().message() << "\n";
        return 1;
    }
    auto tokenizer = std::move(tokenizer_result.value());
    std::cout << "Tokenizer loaded successfully\n";
    PrintMemoryUsage("After Tokenizer", GetMemoryUsage());

    // Load embedding model
    std::cout << "\n--- Phase 2: Load ONNX Embedding Model ---\n";
    vector::EmbeddingModelOptions options;
    options.model_path = model_path;
    options.pooling = vector::PoolingStrategy::Mean;
    options.normalize = true;
    options.execution_provider = execution_provider;
    options.allow_cpu_fallback = true;

    auto model_result = vector::OnnxTextEmbeddingModel::Load(options);
    if (!model_result.ok()) {
        std::cerr << "Failed to load embedding model: " << model_result.status().message() << "\n";
        return 1;
    }
    auto model = std::move(model_result.value());
    std::cout << "Embedding model loaded successfully\n";
    std::cout << "Active provider: " << model->GetActiveExecutionProvider() << "\n";
    PrintMemoryUsage("After Model Load", GetMemoryUsage());

    // Load test queries
    std::cout << "\n--- Phase 3: Load Test Queries ---\n";
    std::vector<std::string> queries;
    if (!dataset_path.empty()) {
        queries = LoadTestQueries(dataset_path, num_samples);
    } else {
        for (std::size_t i = 0; i < num_samples; ++i) {
            queries.push_back("这是一个测试查询句子，用于性能基准测试 " + std::to_string(i));
        }
    }
    std::cout << "Loaded " << queries.size() << " test queries\n";
    PrintMemoryUsage("After Load Queries", GetMemoryUsage());

    // Benchmark tokenization
    std::cout << "\n--- Phase 4: Tokenization Benchmark ---\n";
    std::vector<vector::TokenizedBatch> token_ids_list;
    token_ids_list.reserve(queries.size());

    auto tokenize_result = RunBenchmark(
        "HfTokenizer::Encode",
        [&]() {
            static std::size_t idx = 0;
            auto result = tokenizer.Encode(queries[idx % queries.size()]);
            if (result.ok()) {
                if (token_ids_list.size() < queries.size()) {
                    token_ids_list.push_back(result.value());
                }
            }
            idx++;
        },
        std::min(queries.size(), std::size_t(1000))
    );
    PrintResult(tokenize_result);
    PrintMemoryUsage("After Tokenization", GetMemoryUsage());

    // Benchmark embedding
    std::cout << "\n--- Phase 5: Embedding Benchmark ---\n";
    std::vector<vector::EmbeddingBatch> embeddings;
    embeddings.reserve(token_ids_list.size());

    auto embed_result = RunBenchmark(
        "OnnxTextEmbeddingModel::Embed",
        [&]() {
            static std::size_t idx = 0;
            auto result = model->Embed(token_ids_list[idx % token_ids_list.size()]);
            if (result.ok()) {
                if (embeddings.size() < token_ids_list.size()) {
                    embeddings.push_back(result.value());
                }
            }
            idx++;
        },
        std::min(token_ids_list.size(), std::size_t(1000))
    );
    PrintResult(embed_result);
    PrintMemoryUsage("After Embedding", GetMemoryUsage());

    // Benchmark batch embedding
    std::cout << "\n--- Phase 5.5: Batch Embedding Benchmark ---\n";
    std::vector<std::size_t> batch_sizes = {2, 4, 8, 16};

    for (auto batch_size : batch_sizes) {
        if (queries.size() < batch_size) {
            std::cout << "Skipping batch_size=" << batch_size << " (insufficient data)\n";
            continue;
        }

        // Prepare batch of text queries
        std::vector<std::string_view> batch_texts;
        batch_texts.reserve(batch_size);
        for (std::size_t i = 0; i < batch_size; ++i) {
            batch_texts.push_back(queries[i]);
        }

        // Tokenize batch
        auto tokenize_batch_result = tokenizer.EncodeBatch(batch_texts);
        if (!tokenize_batch_result.ok()) {
            std::cerr << "Failed to tokenize batch: " << tokenize_batch_result.status().message() << "\n";
            continue;
        }
        auto batched_tokens = tokenize_batch_result.value();

        auto batch_result = RunBenchmark(
            "Batch Embedding (batch=" + std::to_string(batch_size) + ")",
            [&]() {
                auto result = model->Embed(batched_tokens);
                (void)result;
            },
            100
        );
        PrintResult(batch_result);

        // Calculate per-item throughput
        double per_item_ms = batch_result.avg_ms / batch_size;
        double per_item_qps = 1000.0 / per_item_ms;
        std::cout << "  Per-item latency: " << per_item_ms << " ms\n";
        std::cout << "  Per-item throughput: " << per_item_qps << " QPS\n";
    }
    PrintMemoryUsage("After Batch Embedding", GetMemoryUsage());

    // Benchmark SIMD dot product
    std::cout << "\n--- Phase 6: SIMD Dot Product Benchmark ---\n";
    if (embeddings.size() >= 2) {
        auto simd_result = RunBenchmark(
            "SIMD Dot Product (384 dim)",
            [&]() {
                static std::size_t idx = 0;
                float score = dot_product_unrolled<384>(
                    embeddings[0].embeddings.data(),
                    embeddings[(idx % (embeddings.size() - 1)) + 1].embeddings.data()
                );
                (void)score;
                idx++;
            },
            100000
        );
        PrintResult(simd_result);
    }

    // Benchmark serialization
    std::cout << "\n--- Phase 7: Serialization Benchmark ---\n";
    std::vector<CacheRecord> records;
    records.reserve(std::min(embeddings.size(), std::size_t(1000)));

    for (std::size_t i = 0; i < std::min(embeddings.size(), std::size_t(1000)); ++i) {
        CacheRecord record;
        record.embedding = embeddings[i].embeddings;
        record.input = queries[i];
        record.response = "响应" + std::to_string(i);
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

    // Benchmark Top-K search
    std::cout << "\n--- Phase 8: Top-K Search Simulation ---\n";
    std::size_t search_size = std::min(embeddings.size(), std::size_t(10000));
    if (!embeddings.empty()) {
        std::vector<float> query_embedding = embeddings[0].embeddings;

        auto search_result = RunBenchmark(
            "Top-K Search (k=8, n=" + std::to_string(search_size) + ")",
            [&]() {
                std::vector<std::pair<float, std::size_t>> scores;
                scores.reserve(search_size);

                for (std::size_t i = 0; i < search_size; ++i) {
                    float score = dot_product_unrolled<384>(
                        query_embedding.data(),
                        embeddings[i % embeddings.size()].embeddings.data()
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
    }

    // Summary
    std::cout << "\n=== Benchmark Complete ===\n";
    auto mem_final = GetMemoryUsage();
    PrintMemoryUsage("Final", mem_final);

    std::cout << "\nMemory Delta from Start:\n";
    std::cout << "  Working Set: +" << (mem_final.working_set_mb - mem_start.working_set_mb) << " MB\n";
    std::cout << "  Private:     +" << (mem_final.private_bytes_mb - mem_start.private_bytes_mb) << " MB\n";
    std::cout << "  Peak:        " << mem_final.peak_working_set_mb << " MB\n";

    std::cout << "\nEnd-to-End Pipeline Summary:\n";
    std::cout << "- Tokenization:     " << tokenize_result.avg_ms << " ms/op\n";
    std::cout << "- Embedding:        " << embed_result.avg_ms << " ms/op\n";
    std::cout << "- Serialization:    " << serialize_result.avg_ms << " ms/op\n";
    std::cout << "\nTotal E2E latency:  " << (tokenize_result.avg_ms + embed_result.avg_ms) << " ms\n";

    return 0;
}
