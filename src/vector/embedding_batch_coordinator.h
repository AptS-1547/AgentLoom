#pragma once

#include "embedding_pipeline.h"
#include "logger_adapter.h"
#include "thread_pool.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vector {

/// 可替换的批量 embedding 执行边界，便于单元测试和未来接入其他推理后端。
class IEmbeddingBatchProvider {
public:
    virtual ~IEmbeddingBatchProvider() = default;

    virtual core::Result<EmbeddingBatch> EncodeBatch(
        std::span<const std::string_view> texts) const = 0;
};

class EmbeddingPipelineBatchProvider final : public IEmbeddingBatchProvider {
public:
    explicit EmbeddingPipelineBatchProvider(std::shared_ptr<EmbeddingPipeline> pipeline);

    core::Result<EmbeddingBatch> EncodeBatch(
        std::span<const std::string_view> texts) const override;

private:
    std::shared_ptr<EmbeddingPipeline> pipeline_;
};

struct EmbeddingBatchCoordinatorOptions {
    std::size_t max_pending_requests = 1024;
    std::size_t max_batch_size = 16;
    std::chrono::milliseconds max_batch_wait{2};
    std::size_t max_inflight_batches = 1;
    /// 非空时，保序后的单条 continuation 投递到该池执行；其生命周期必须长于协调器。
    core::ThreadPool* completion_pool = nullptr;
};

struct EmbeddingBatchRequest {
    std::string session_id;
    std::string tenant_id;
    std::string user_uuid;
    std::string trace_id;
    std::string text;
    std::function<void(core::Result<std::vector<float>>)> completion;
};

struct EmbeddingBatchCoordinatorSnapshot {
    bool running = false;
    std::size_t pending_requests = 0;
    std::size_t inflight_batches = 0;
    std::uint64_t submitted_requests = 0;
    std::uint64_t completed_requests = 0;
    std::uint64_t rejected_requests = 0;
    std::uint64_t failed_requests = 0;
    std::uint64_t submitted_batches = 0;
    std::uint64_t partial_batches = 0;
    std::uint64_t total_batch_wait_us = 0;
    std::uint64_t total_model_time_us = 0;
    std::size_t max_observed_batch_size = 0;
};

/// 有界 micro-batch 协调器。
///
/// 协调线程只负责收集和刷新请求，不占用 compute_pool worker；compute_pool
/// 仅执行已经形成的 EncodeBatch work item。完成结果按照每个 session 的
/// admission 顺序释放，避免不同 batch 完成顺序造成隐性乱序。
class EmbeddingBatchCoordinator final {
public:
    EmbeddingBatchCoordinator(
        core::ThreadPool& compute_pool,
        std::shared_ptr<IEmbeddingBatchProvider> provider,
        EmbeddingBatchCoordinatorOptions options = {},
        core::LoggerAdapter logger = core::LoggerAdapter::ForModule("embedding-batch"));
    ~EmbeddingBatchCoordinator();

    EmbeddingBatchCoordinator(const EmbeddingBatchCoordinator&) = delete;
    EmbeddingBatchCoordinator& operator=(const EmbeddingBatchCoordinator&) = delete;

    core::Status Start();
    void Shutdown() noexcept;
    core::Status Submit(EmbeddingBatchRequest request);
    EmbeddingBatchCoordinatorSnapshot Snapshot() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace vector
