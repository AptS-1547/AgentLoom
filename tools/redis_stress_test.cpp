// Redis++ connection pool stress test
// Tests performance under high load with WSL Redis
//
// Run: .\build\x64-Release-Tests\Release\redis_stress_test.exe

#include "../../src/semantic_cache/redis_connection_pool.h"
#include "../../src/semantic_cache/semantic_cache_pipeline.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <random>
#include <atomic>

using namespace agent::semantic_cache;
using namespace std::chrono;

struct BenchmarkResult {
    std::string name;
    size_t operations;
    double elapsed_ms;
    double ops_per_sec;
    double avg_latency_us;
};

void PrintResult(const BenchmarkResult& r) {
    std::cout << "\n[" << r.name << "]\n";
    std::cout << "  Operations:   " << r.operations << "\n";
    std::cout << "  Elapsed:      " << r.elapsed_ms << " ms\n";
    std::cout << "  Throughput:   " << r.ops_per_sec << " ops/sec\n";
    std::cout << "  Avg Latency:  " << r.avg_latency_us << " us\n";
}

// Benchmark 1: Simple SET/GET operations
BenchmarkResult BenchmarkSetGet(RedisConnectionPool& pool, size_t count) {
    auto start = steady_clock::now();

    for (size_t i = 0; i < count; ++i) {
        std::string key = "stress:test:" + std::to_string(i);
        std::string value = "value_" + std::to_string(i);

        auto status = pool.Set(key, value);
        if (!status.ok()) {
            std::cerr << "SET failed: " << status.message() << "\n";
            break;
        }

        auto result = pool.Get(key);
        if (!result.ok()) {
            std::cerr << "GET failed: " << result.status().message() << "\n";
            break;
        }
    }

    auto elapsed = duration_cast<milliseconds>(steady_clock::now() - start).count();
    double ops_per_sec = (count * 2.0 * 1000.0) / elapsed;  // SET + GET
    double avg_latency = (elapsed * 1000.0) / (count * 2.0);

    return {"SET/GET Sequential", count * 2, static_cast<double>(elapsed), ops_per_sec, avg_latency};
}

// Benchmark 2: HSET/HGETALL with binary payloads
BenchmarkResult BenchmarkHashOperations(RedisConnectionPool& pool, size_t batch_count, size_t records_per_batch) {
    auto start = steady_clock::now();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t batch = 0; batch < batch_count; ++batch) {
        std::string key = "stress:batch:" + std::to_string(batch);
        std::unordered_map<std::string, std::string> field_values;

        for (size_t i = 0; i < records_per_batch; ++i) {
            storage::CacheRecord record;
            record.embedding.resize(384);
            for (auto& v : record.embedding) v = dist(gen);
            record.input = "input_" + std::to_string(i);
            record.response = "response_" + std::to_string(i);

            std::string serialized = SerializeCacheRecord(record);
            field_values[std::to_string(i)] = serialized;
        }

        auto result = pool.HMSet(key, field_values);
        if (!result.ok()) {
            std::cerr << "HMSET failed: " << result.status().message() << "\n";
            break;
        }

        auto get_result = pool.HGetAll(key);
        if (!get_result.ok()) {
            std::cerr << "HGETALL failed: " << get_result.status().message() << "\n";
            break;
        }
    }

    auto elapsed = duration_cast<milliseconds>(steady_clock::now() - start).count();
    size_t total_ops = batch_count * 2;  // HMSET + HGETALL
    size_t total_records = batch_count * records_per_batch * 2;
    double ops_per_sec = (total_ops * 1000.0) / elapsed;
    double avg_latency = (elapsed * 1000.0) / total_ops;

    std::cout << "  (Total records: " << total_records << ")\n";

    return {"HSET/HGETALL Binary Payloads", total_ops, static_cast<double>(elapsed), ops_per_sec, avg_latency};
}

// Benchmark 3: Pipeline batch operations
BenchmarkResult BenchmarkPipeline(RedisConnectionPool& pool, size_t batch_count, size_t ops_per_batch) {
    auto start = steady_clock::now();

    for (size_t batch = 0; batch < batch_count; ++batch) {
        auto pipe = pool.CreatePipeline();

        for (size_t i = 0; i < ops_per_batch; ++i) {
            std::string key = "stress:pipe:" + std::to_string(batch) + ":" + std::to_string(i);
            pipe.set(key, "value_" + std::to_string(i));
        }

        auto status = pool.ExecPipeline(pipe);
        if (!status.ok()) {
            std::cerr << "Pipeline exec failed: " << status.message() << "\n";
            break;
        }
    }

    auto elapsed = duration_cast<milliseconds>(steady_clock::now() - start).count();
    size_t total_ops = batch_count * ops_per_batch;
    double ops_per_sec = (total_ops * 1000.0) / elapsed;
    double avg_latency = (elapsed * 1000.0) / batch_count;

    return {"Pipeline Batch SET", total_ops, static_cast<double>(elapsed), ops_per_sec, avg_latency};
}

// Benchmark 4: Concurrent operations (multi-threaded)
BenchmarkResult BenchmarkConcurrent(RedisConnectionPool& pool, size_t thread_count, size_t ops_per_thread) {
    std::atomic<size_t> completed{0};
    std::atomic<size_t> errors{0};

    auto start = steady_clock::now();

    std::vector<std::thread> threads;
    for (size_t t = 0; t < thread_count; ++t) {
        threads.emplace_back([&pool, t, ops_per_thread, &completed, &errors]() {
            for (size_t i = 0; i < ops_per_thread; ++i) {
                std::string key = "stress:thread:" + std::to_string(t) + ":" + std::to_string(i);
                auto status = pool.Set(key, "value");
                if (status.ok()) {
                    completed++;
                } else {
                    errors++;
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    auto elapsed = duration_cast<milliseconds>(steady_clock::now() - start).count();
    size_t total_ops = thread_count * ops_per_thread;
    double ops_per_sec = (completed.load() * 1000.0) / elapsed;
    double avg_latency = (elapsed * 1000.0) / total_ops;

    std::cout << "  (Completed: " << completed << ", Errors: " << errors << ")\n";

    return {"Concurrent Multi-threaded SET", completed.load(), static_cast<double>(elapsed), ops_per_sec, avg_latency};
}

// Benchmark 5: SCAN performance
BenchmarkResult BenchmarkScan(RedisConnectionPool& pool) {
    auto start = steady_clock::now();

    auto result = pool.Scan("stress:*");

    auto elapsed = duration_cast<milliseconds>(steady_clock::now() - start).count();

    size_t key_count = result.ok() ? result.value().size() : 0;
    double ops_per_sec = key_count > 0 ? (key_count * 1000.0) / elapsed : 0;

    std::cout << "  (Keys found: " << key_count << ")\n";

    return {"SCAN Pattern Matching", 1, static_cast<double>(elapsed), ops_per_sec, static_cast<double>(elapsed * 1000.0)};
}

int main() {
    std::cout << "=== Redis++ Connection Pool Stress Test ===\n";
    std::cout << "Target: WSL Redis (127.0.0.1:5000)\n\n";

    RedisPoolOptions opts;
    opts.host = "127.0.0.1";
    opts.port = "5000";
    opts.pool_size = 8;
    opts.command_timeout = std::chrono::milliseconds(5000);

    auto pool = std::make_shared<RedisConnectionPool>(opts);
    auto status = pool->Start();

    if (!status.ok()) {
        std::cerr << "Failed to start Redis pool: " << status.message() << "\n";
        return 1;
    }

    std::cout << "Redis pool started (pool_size=" << opts.pool_size << ")\n";

    // Warm up
    std::cout << "\nWarming up...\n";
    for (int i = 0; i < 100; ++i) {
        pool->Set("warmup:" + std::to_string(i), "value");
    }

    std::vector<BenchmarkResult> results;

    // Test 1: Sequential SET/GET
    std::cout << "\n[Test 1] Sequential SET/GET...\n";
    results.push_back(BenchmarkSetGet(*pool, 1000));
    PrintResult(results.back());

    // Test 2: Hash operations with binary payloads
    std::cout << "\n[Test 2] Hash operations with binary payloads (384-dim vectors)...\n";
    results.push_back(BenchmarkHashOperations(*pool, 50, 100));  // 50 batches * 100 records
    PrintResult(results.back());

    // Test 3: Pipeline batch operations
    std::cout << "\n[Test 3] Pipeline batch operations...\n";
    results.push_back(BenchmarkPipeline(*pool, 100, 50));  // 100 batches * 50 ops
    PrintResult(results.back());

    // Test 4: Concurrent multi-threaded
    std::cout << "\n[Test 4] Concurrent multi-threaded operations...\n";
    results.push_back(BenchmarkConcurrent(*pool, 8, 500));  // 8 threads * 500 ops
    PrintResult(results.back());

    // Test 5: SCAN performance
    std::cout << "\n[Test 5] SCAN pattern matching...\n";
    results.push_back(BenchmarkScan(*pool));
    PrintResult(results.back());

    // Summary
    std::cout << "\n=== Summary ===\n";
    for (const auto& r : results) {
        std::cout << r.name << ": " << r.ops_per_sec << " ops/sec\n";
    }

    // Cleanup
    std::cout << "\nCleaning up test keys...\n";
    auto keys_result = pool->Scan("stress:*");
    if (keys_result.ok()) {
        auto keys = keys_result.value();
        if (!keys.empty()) {
            pool->Del(keys);
            std::cout << "Deleted " << keys.size() << " keys\n";
        }
    }
    keys_result = pool->Scan("warmup:*");
    if (keys_result.ok()) {
        auto keys = keys_result.value();
        if (!keys.empty()) {
            pool->Del(keys);
        }
    }

    pool->Shutdown();

    std::cout << "\nStress test completed!\n";
    return 0;
}
