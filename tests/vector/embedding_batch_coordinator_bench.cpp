#include "embedding_batch_coordinator.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

class SimulatedEmbeddingProvider final : public vector::IEmbeddingBatchProvider {
public:
    core::Result<vector::EmbeddingBatch> EncodeBatch(
        std::span<const std::string_view> texts) const override {
        std::this_thread::sleep_for(std::chrono::microseconds(200 + texts.size() * 20));
        vector::EmbeddingBatch batch;
        batch.batch_size = texts.size();
        batch.dimension = 8;
        batch.embeddings.resize(batch.batch_size * batch.dimension, 0.5f);
        return batch;
    }
};

std::size_t ParseSize(const char* value, std::size_t fallback) {
    if (!value || *value == '\0') {
        return fallback;
    }
    const auto parsed = std::strtoull(value, nullptr, 10);
    return parsed == 0 ? fallback : static_cast<std::size_t>(parsed);
}

} // namespace

int main(int argc, char** argv) {
    const auto request_count = ParseSize(argc > 1 ? argv[1] : nullptr, 10000);
    const auto session_count = ParseSize(argc > 2 ? argv[2] : nullptr, 1000);
    const auto batch_size = ParseSize(argc > 3 ? argv[3] : nullptr, 16);
    const auto worker_count = ParseSize(argc > 4 ? argv[4] : nullptr, 4);

    core::ThreadPool compute_pool({
        .worker_count = worker_count,
        .queue_capacity = request_count,
        .name = "embedding-batch-bench",
    });
    auto status = compute_pool.Start();
    if (!status.ok()) {
        std::cerr << "compute pool start failed: " << status.message() << '\n';
        return 1;
    }

    auto provider = std::make_shared<SimulatedEmbeddingProvider>();
    vector::EmbeddingBatchCoordinator coordinator(
        compute_pool,
        provider,
        vector::EmbeddingBatchCoordinatorOptions{
            .max_pending_requests = request_count,
            .max_batch_size = batch_size,
            .max_batch_wait = std::chrono::milliseconds(2),
            .max_inflight_batches = worker_count,
        });
    status = coordinator.Start();
    if (!status.ok()) {
        std::cerr << "coordinator start failed: " << status.message() << '\n';
        compute_pool.Shutdown(false);
        return 1;
    }

    std::mutex mutex;
    std::condition_variable condition;
    std::atomic<std::size_t> completed{0};
    std::atomic<std::size_t> failed{0};
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < request_count; ++index) {
        status = coordinator.Submit({
            .session_id = "session-" + std::to_string(index % session_count),
            .tenant_id = "tenant-" + std::to_string(index % 16),
            .user_uuid = "user-" + std::to_string(index % session_count),
            .trace_id = "bench-" + std::to_string(index),
            .text = "embedding benchmark request " + std::to_string(index),
            .completion = [&](core::Result<std::vector<float>> result) {
                if (!result.ok()) {
                    failed.fetch_add(1, std::memory_order_relaxed);
                }
                if (completed.fetch_add(1, std::memory_order_acq_rel) + 1 == request_count) {
                    std::lock_guard lock(mutex);
                    condition.notify_all();
                }
            },
        });
        if (!status.ok()) {
            std::cerr << "submit failed at " << index << ": " << status.message() << '\n';
            coordinator.Shutdown();
            compute_pool.Shutdown(true);
            return 1;
        }
    }

    {
        std::unique_lock lock(mutex);
        if (!condition.wait_for(lock, std::chrono::seconds(30), [&] {
                return completed.load(std::memory_order_acquire) == request_count;
            })) {
            std::cerr << "benchmark timed out\n";
            coordinator.Shutdown();
            compute_pool.Shutdown(true);
            return 1;
        }
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const auto snapshot = coordinator.Snapshot();
    coordinator.Shutdown();
    compute_pool.Shutdown(true);

    std::cout << "{\n"
              << "  \"requests\": " << request_count << ",\n"
              << "  \"sessions\": " << session_count << ",\n"
              << "  \"configuredBatchSize\": " << batch_size << ",\n"
              << "  \"workers\": " << worker_count << ",\n"
              << "  \"elapsedSeconds\": " << elapsed << ",\n"
              << "  \"throughputRps\": " << request_count / elapsed << ",\n"
              << "  \"submittedBatches\": " << snapshot.submitted_batches << ",\n"
              << "  \"partialBatches\": " << snapshot.partial_batches << ",\n"
              << "  \"maxObservedBatchSize\": " << snapshot.max_observed_batch_size << ",\n"
              << "  \"failedRequests\": " << failed.load(std::memory_order_relaxed) << '\n'
              << "}\n";
    return failed.load(std::memory_order_relaxed) == 0 ? 0 : 1;
}
