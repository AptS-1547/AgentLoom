#include "embedding_batch_coordinator.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vector {

EmbeddingPipelineBatchProvider::EmbeddingPipelineBatchProvider(
    std::shared_ptr<EmbeddingPipeline> pipeline)
    : pipeline_(std::move(pipeline)) {}

core::Result<EmbeddingBatch> EmbeddingPipelineBatchProvider::EncodeBatch(
    std::span<const std::string_view> texts) const {
    if (!pipeline_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                   "embedding pipeline is missing");
    }
    return pipeline_->EncodeBatch(texts);
}

struct EmbeddingBatchCoordinator::Impl final
    : public std::enable_shared_from_this<EmbeddingBatchCoordinator::Impl> {
    struct PendingRequest {
        EmbeddingBatchRequest request;
        std::uint64_t session_sequence = 0;
        std::chrono::steady_clock::time_point admitted_at;
        std::atomic<bool> completed{false};
    };

    struct ReadyResult {
        std::shared_ptr<PendingRequest> request;
        core::Result<std::vector<float>> result;
    };

    struct SessionOrderState {
        std::uint64_t next_admission = 1;
        std::uint64_t next_delivery = 1;
        std::size_t outstanding = 0;
        bool delivery_inflight = false;
        std::map<std::uint64_t, ReadyResult> completed;
    };

    Impl(core::ThreadPool& compute_pool,
         std::shared_ptr<IEmbeddingBatchProvider> provider,
         EmbeddingBatchCoordinatorOptions options,
         core::LoggerAdapter logger)
        : compute_pool(compute_pool),
          provider(std::move(provider)),
          options(options),
          logger(logger.valid() ? std::move(logger)
                                : core::LoggerAdapter::ForModule("embedding-batch")) {}

    core::Status Start() {
        if (!provider) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                       "embedding batch provider is missing");
        }
        if (options.max_pending_requests == 0 || options.max_batch_size == 0 ||
            options.max_batch_wait.count() <= 0 || options.max_inflight_batches == 0) {
            return core::Status::Error(core::ErrorCode::InvalidArgument,
                                       "embedding batch options must be positive");
        }
        if (options.completion_pool &&
            !options.completion_pool->serializes_concurrency_key_until_completion()) {
            return core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "embedding completion pool must serialize concurrency keys");
        }
        bool expected = false;
        if (!running.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return core::Status::Ok();
        }
        auto self = shared_from_this();
        dispatcher = std::jthread([self = std::move(self)](std::stop_token stop_token) {
            self->DispatchLoop(stop_token);
        });
        return core::Status::Ok();
    }

    void Shutdown() noexcept {
        if (!running.exchange(false, std::memory_order_acq_rel)) {
            return;
        }
        std::vector<std::shared_ptr<PendingRequest>> pending;
        {
            std::lock_guard lock(mutex);
            stopping = true;
            while (!queue.empty()) {
                pending.push_back(std::move(queue.front()));
                queue.pop_front();
            }
        }
        condition.notify_all();
        if (dispatcher.joinable()) {
            dispatcher.request_stop();
            dispatcher.join();
        }
        for (auto& request : pending) {
            CompleteRequest(request, core::Status::Error(
                core::ErrorCode::Cancelled,
                "embedding batch coordinator was shut down"));
        }
        std::unique_lock lock(mutex);
        // completion 可能仍在 IO pool 排队；等到每个 Session 的逐条回调真正返回，
        // 才允许协调器析构，避免 continuation 捕获的业务对象提前失效。
        drained.wait(lock, [this] {
            return inflight_batches == 0 && sessions.empty();
        });
    }

    core::Status Submit(EmbeddingBatchRequest request) {
        if (request.text.empty() || !request.completion) {
            return core::Status::Error(core::ErrorCode::InvalidArgument,
                                       "embedding batch text and completion are required");
        }
        if (request.session_id.empty()) {
            return core::Status::Error(core::ErrorCode::InvalidArgument,
                                       "embedding batch session_id is required");
        }
        if (!running.load(std::memory_order_acquire)) {
            return core::Status::Error(core::ErrorCode::Unavailable,
                                       "embedding batch coordinator is not running");
        }

        auto pending_request = std::make_shared<PendingRequest>();
        pending_request->request = std::move(request);
        pending_request->admitted_at = std::chrono::steady_clock::now();
        {
            std::lock_guard lock(mutex);
            if (stopping) {
                return core::Status::Error(core::ErrorCode::Cancelled,
                                           "embedding batch coordinator is stopping");
            }
            if (queue.size() >= options.max_pending_requests) {
                rejected_requests.fetch_add(1, std::memory_order_relaxed);
                return core::Status::Error(core::ErrorCode::ResourceExhausted,
                                           "embedding batch pending queue is full");
            }
            auto& session = sessions[pending_request->request.session_id];
            pending_request->session_sequence = session.next_admission++;
            ++session.outstanding;
            queue.push_back(std::move(pending_request));
            submitted_requests.fetch_add(1, std::memory_order_relaxed);
        }
        condition.notify_one();
        return core::Status::Ok();
    }

    EmbeddingBatchCoordinatorSnapshot Snapshot() const {
        EmbeddingBatchCoordinatorSnapshot snapshot;
        snapshot.running = running.load(std::memory_order_acquire);
        {
            std::lock_guard lock(mutex);
            snapshot.pending_requests = queue.size();
            snapshot.inflight_batches = inflight_batches;
        }
        snapshot.submitted_requests = submitted_requests.load(std::memory_order_relaxed);
        snapshot.completed_requests = completed_requests.load(std::memory_order_relaxed);
        snapshot.rejected_requests = rejected_requests.load(std::memory_order_relaxed);
        snapshot.failed_requests = failed_requests.load(std::memory_order_relaxed);
        snapshot.submitted_batches = submitted_batches.load(std::memory_order_relaxed);
        snapshot.partial_batches = partial_batches.load(std::memory_order_relaxed);
        snapshot.total_batch_wait_us = total_batch_wait_us.load(std::memory_order_relaxed);
        snapshot.total_model_time_us = total_model_time_us.load(std::memory_order_relaxed);
        snapshot.max_observed_batch_size = max_observed_batch_size.load(std::memory_order_relaxed);
        return snapshot;
    }

private:
    void DispatchLoop(std::stop_token stop_token) noexcept {
        while (!stop_token.stop_requested()) {
            std::vector<std::shared_ptr<PendingRequest>> batch;
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, [this, &stop_token] {
                    return stopping || stop_token.stop_requested() || !queue.empty();
                });
                if (stopping || stop_token.stop_requested()) {
                    break;
                }

                const auto deadline = queue.front()->admitted_at + options.max_batch_wait;
                condition.wait_until(lock, deadline, [this, &stop_token] {
                    return stopping || stop_token.stop_requested() ||
                           queue.size() >= options.max_batch_size;
                });
                if (stopping || stop_token.stop_requested()) {
                    break;
                }
                condition.wait(lock, [this, &stop_token] {
                    return stopping || stop_token.stop_requested() ||
                           inflight_batches < options.max_inflight_batches;
                });
                if (stopping || stop_token.stop_requested()) {
                    break;
                }

                const auto count = std::min(options.max_batch_size, queue.size());
                batch.reserve(count);
                for (std::size_t i = 0; i < count; ++i) {
                    batch.push_back(std::move(queue.front()));
                    queue.pop_front();
                }
                ++inflight_batches;
            }
            SubmitBatch(std::move(batch));
        }
    }

    void SubmitBatch(std::vector<std::shared_ptr<PendingRequest>> batch) noexcept {
        const auto batch_count = batch.size();
        const auto batch_started = std::chrono::steady_clock::now();
        std::uint64_t wait_us = 0;
        for (const auto& request : batch) {
            wait_us += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    batch_started - request->admitted_at).count());
        }
        total_batch_wait_us.fetch_add(wait_us, std::memory_order_relaxed);
        submitted_batches.fetch_add(1, std::memory_order_relaxed);
        if (batch.size() < options.max_batch_size) {
            partial_batches.fetch_add(1, std::memory_order_relaxed);
        }
        UpdateMaximum(max_observed_batch_size, batch.size());

        std::vector<std::string_view> texts;
        texts.reserve(batch.size());
        for (const auto& request : batch) {
            texts.push_back(request->request.text);
        }

        auto self = shared_from_this();
        const auto status = compute_pool.Submit(
            [self = std::move(self), batch = std::move(batch), texts = std::move(texts)](
                core::ThreadPoolContext&) mutable -> core::Status {
                const auto started = std::chrono::steady_clock::now();
                core::Result<EmbeddingBatch> encoded = [&]() -> core::Result<EmbeddingBatch> {
                    try {
                        return self->provider->EncodeBatch(texts);
                    } catch (const std::exception& exception) {
                        return core::Status::Error(core::ErrorCode::InternalError, exception.what());
                    } catch (...) {
                        return core::Status::Error(core::ErrorCode::Unknown,
                                                  "embedding batch provider threw an unknown exception");
                    }
                }();
                self->total_model_time_us.fetch_add(static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - started).count()),
                    std::memory_order_relaxed);

                if (!encoded.ok()) {
                    self->failed_requests.fetch_add(batch.size(), std::memory_order_relaxed);
                    for (const auto& request : batch) {
                        self->CompleteRequest(request, encoded.status());
                    }
                    self->BatchFinished();
                    return encoded.status();
                }
                const auto dimension = encoded.value().dimension;
                const auto expected_embeddings = dimension == 0 ||
                        encoded.value().batch_size >
                            std::numeric_limits<std::size_t>::max() / dimension
                    ? 0
                    : encoded.value().batch_size * dimension;
                if (encoded.value().batch_size != batch.size() ||
                    dimension == 0 ||
                    encoded.value().embeddings.size() != expected_embeddings) {
                    const auto mismatch = core::Status::Error(
                        core::ErrorCode::InternalError,
                        "embedding batch provider returned an unexpected batch size");
                    self->failed_requests.fetch_add(batch.size(), std::memory_order_relaxed);
                    for (const auto& request : batch) {
                        self->CompleteRequest(request, mismatch);
                    }
                    self->BatchFinished();
                    return mismatch;
                }
                for (std::size_t index = 0; index < batch.size(); ++index) {
                    const auto row = encoded.value().row(index);
                    self->CompleteRequest(
                        batch[index],
                        std::vector<float>(row.begin(), row.end()));
                }
                self->BatchFinished();
                return core::Status::Ok();
            },
            {},
            "embedding-batch");
        if (!status.ok()) {
            failed_requests.fetch_add(batch_count, std::memory_order_relaxed);
            // Submit() 失败时 work item 未接管 batch；此处仍需逐条完成原始请求。
            for (const auto& request : batch) {
                CompleteRequest(request, status);
            }
            BatchFinished();
        }
    }

    void CompleteRequest(const std::shared_ptr<PendingRequest>& request,
                         core::Result<std::vector<float>> result) noexcept {
        if (request->completed.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        std::optional<ReadyResult> ready;
        {
            std::lock_guard lock(mutex);
            auto session = sessions.find(request->request.session_id);
            if (session == sessions.end()) {
                return;
            }
            session->second.completed.emplace(
                request->session_sequence,
                ReadyResult{request, std::move(result)});
            if (!session->second.delivery_inflight) {
                auto next = session->second.completed.find(session->second.next_delivery);
                if (next != session->second.completed.end()) {
                    ready.emplace(std::move(next->second));
                    session->second.completed.erase(next);
                    ++session->second.next_delivery;
                    session->second.delivery_inflight = true;
                }
            }
        }
        if (ready) {
            DispatchCompletion(std::move(*ready));
        }
    }

    void DispatchCompletion(ReadyResult item) noexcept {
        if (options.completion_pool) {
            const auto fallback_request = item.request;
            core::ThreadPoolTaskMetadata metadata;
            metadata.concurrency_key = item.request->request.session_id;
            metadata.tenant_key = item.request->request.tenant_id;
            if (!item.request->request.user_uuid.empty()) {
                metadata.fairness_key = item.request->request.tenant_id + ":" +
                    item.request->request.user_uuid;
            }
            auto completion = std::make_shared<ReadyResult>(std::move(item));
            auto self = shared_from_this();
            const auto status = options.completion_pool->Submit(
                [self = std::move(self), completion](core::ThreadPoolContext&) mutable -> core::Status {
                    self->InvokeCompletion(std::move(*completion));
                    return core::Status::Ok();
                },
                {},
                "embedding-continuation",
                std::move(metadata));
            if (status.ok()) {
                return;
            }
            failed_requests.fetch_add(1, std::memory_order_relaxed);
            InvokeCompletion(ReadyResult{fallback_request, status});
            return;
        }
        InvokeCompletion(std::move(item));
    }

    void InvokeCompletion(ReadyResult item) noexcept {
        try {
            item.request->request.completion(std::move(item.result));
            completed_requests.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::exception& exception) {
            logger.error("[embedding-batch] completion exception: {}", exception.what());
        } catch (...) {
            logger.error("[embedding-batch] completion exception: unknown");
        }
        MarkDelivered(item.request->request.session_id);
    }

    void MarkDelivered(const std::string& session_id) noexcept {
        std::optional<ReadyResult> next_ready;
        {
            std::lock_guard lock(mutex);
            auto session = sessions.find(session_id);
            if (session == sessions.end()) {
                return;
            }
            if (session->second.outstanding > 0) {
                --session->second.outstanding;
            }
            auto next = session->second.completed.find(session->second.next_delivery);
            if (next != session->second.completed.end()) {
                next_ready.emplace(std::move(next->second));
                session->second.completed.erase(next);
                ++session->second.next_delivery;
            } else {
                session->second.delivery_inflight = false;
                if (session->second.outstanding == 0 && session->second.completed.empty()) {
                    sessions.erase(session);
                }
            }
            drained.notify_all();
        }
        if (next_ready) {
            DispatchCompletion(std::move(*next_ready));
        }
    }

    void BatchFinished() noexcept {
        std::lock_guard lock(mutex);
        if (inflight_batches > 0) {
            --inflight_batches;
        }
        drained.notify_all();
        condition.notify_all();
    }

    static void UpdateMaximum(std::atomic<std::size_t>& target, std::size_t value) noexcept {
        auto current = target.load(std::memory_order_relaxed);
        while (current < value &&
               !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
        }
    }

    core::ThreadPool& compute_pool;
    std::shared_ptr<IEmbeddingBatchProvider> provider;
    EmbeddingBatchCoordinatorOptions options;
    core::LoggerAdapter logger;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::condition_variable drained;
    std::deque<std::shared_ptr<PendingRequest>> queue;
    std::unordered_map<std::string, SessionOrderState> sessions;
    std::jthread dispatcher;
    std::size_t inflight_batches = 0;
    bool stopping = false;
    std::atomic<bool> running{false};
    std::atomic<std::uint64_t> submitted_requests{0};
    std::atomic<std::uint64_t> completed_requests{0};
    std::atomic<std::uint64_t> rejected_requests{0};
    std::atomic<std::uint64_t> failed_requests{0};
    std::atomic<std::uint64_t> submitted_batches{0};
    std::atomic<std::uint64_t> partial_batches{0};
    std::atomic<std::uint64_t> total_batch_wait_us{0};
    std::atomic<std::uint64_t> total_model_time_us{0};
    std::atomic<std::size_t> max_observed_batch_size{0};
};

EmbeddingBatchCoordinator::EmbeddingBatchCoordinator(
    core::ThreadPool& compute_pool,
    std::shared_ptr<IEmbeddingBatchProvider> provider,
    EmbeddingBatchCoordinatorOptions options,
    core::LoggerAdapter logger)
    : impl_(std::make_shared<Impl>(
          compute_pool, std::move(provider), options, std::move(logger))) {}

EmbeddingBatchCoordinator::~EmbeddingBatchCoordinator() {
    Shutdown();
}

core::Status EmbeddingBatchCoordinator::Start() {
    return impl_->Start();
}

void EmbeddingBatchCoordinator::Shutdown() noexcept {
    impl_->Shutdown();
}

core::Status EmbeddingBatchCoordinator::Submit(EmbeddingBatchRequest request) {
    return impl_->Submit(std::move(request));
}

EmbeddingBatchCoordinatorSnapshot EmbeddingBatchCoordinator::Snapshot() const {
    return impl_->Snapshot();
}

} // namespace vector
