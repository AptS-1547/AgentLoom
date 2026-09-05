#include "cloud_task_coordinator.h"

#include "blocking_queue.h"
#include "logger_adapter.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <future>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace agent::llm {

namespace {

class AsyncCloudTaskOperation final : public ICloudTaskOperation,
                                      public std::enable_shared_from_this<AsyncCloudTaskOperation> {
public:
    AsyncCloudTaskOperation(std::shared_ptr<IAsyncLlmClient> client,
                            std::vector<CloudTaskChunk> chunks,
                            std::shared_ptr<ICloudTaskResultSink> sink,
                            std::size_t max_inflight,
                            std::function<void(bool)> on_chunk_completed,
                            std::function<void()> on_finished)
        : client_(std::move(client)),
          chunks_(std::move(chunks)),
          sink_(std::move(sink)),
          ordered_({.capacity = std::max<std::size_t>(chunks_.size(), 1), .first_sequence = 1}),
          operations_(chunks_.size()),
          completed_(chunks_.size(), false),
          max_inflight_(std::max<std::size_t>(max_inflight, 1)),
          on_chunk_completed_(std::move(on_chunk_completed)),
          on_finished_(std::move(on_finished)) {}

    ~AsyncCloudTaskOperation() override {
        Cancel();
        if (drain_thread_.joinable()) {
            if (drain_thread_.get_id() == std::this_thread::get_id()) {
                drain_thread_.detach();
            } else {
                drain_thread_.join();
            }
        }
    }

    void Start() {
        auto seal_status = ordered_.Seal(chunks_.back().chunk_sequence);
        if (!seal_status.ok()) {
            std::lock_guard lock(mutex_);
            terminal_ = true;
            terminal_status_ = seal_status;
            return;
        }
        auto self = shared_from_this();
        drain_thread_ = std::thread([self] { self->Drain(); });
        LaunchAvailable();
    }

    void Cancel() noexcept override {
        std::vector<std::shared_ptr<IAsyncLlmOperation>> to_cancel;
        {
            std::lock_guard lock(mutex_);
            if (!terminal_) {
                terminal_ = true;
                terminal_status_ = core::Status::Error(
                    core::ErrorCode::Cancelled,
                    "cloud task cancelled");
            }
            to_cancel.reserve(operations_.size());
            for (auto& operation : operations_) {
                if (operation) {
                    to_cancel.push_back(std::move(operation));
                }
            }
        }
        for (const auto& operation : to_cancel) {
            operation->Cancel();
        }
    }

private:
    void OnProvider(std::size_t index, core::Result<ChatCompletionResponse> response) noexcept {
        CloudTaskChunkResult result{
            .task = chunks_[index].task,
            .session_sequence = chunks_[index].session_sequence,
            .chunk_sequence = chunks_[index].chunk_sequence,
            .attempt = chunks_[index].attempt,
        };
        if (response.ok()) {
            result.response = std::move(response).value();
        } else {
            result.status = response.status();
        }
        const bool successful = result.status.ok();

        {
            std::lock_guard lock(mutex_);
            if (completed_[index]) {
                return;
            }
            completed_[index] = true;
            operations_[index].reset();
            if (inflight_ > 0) {
                --inflight_;
            }
            ++completed_chunks_;
            if (result.status.ok()) {
                ++successful_chunks_;
            } else {
                ++failed_chunks_;
            }
        }
        bool accept_result = false;
        {
            std::lock_guard lock(mutex_);
            accept_result = !terminal_;
        }
        if (accept_result) {
            const auto status = ordered_.Admit(result.chunk_sequence, std::move(result));
            if (!status.ok()) {
                std::lock_guard lock(mutex_);
                terminal_ = true;
                terminal_status_ = status;
            }
        }
        if (on_chunk_completed_) {
            on_chunk_completed_(successful);
        }
        LaunchAvailable();
    }

    void LaunchAvailable() noexcept {
        while (true) {
            std::size_t index = 0;
            {
                std::lock_guard lock(mutex_);
                if (terminal_ || next_index_ >= chunks_.size() || inflight_ >= max_inflight_) {
                    return;
                }
                index = next_index_++;
                ++inflight_;
            }

            auto self = shared_from_this();
            auto submitted = client_->CompleteAsync(
                chunks_[index].request,
                [self, index](core::Result<ChatCompletionResponse> response) {
                    self->OnProvider(index, std::move(response));
                });
            if (!submitted.ok()) {
                OnProvider(index, submitted.status());
                continue;
            }

            bool cancel = false;
            {
                std::lock_guard lock(mutex_);
                if (completed_[index] || terminal_) {
                    cancel = true;
                } else {
                    operations_[index] = std::move(submitted).value();
                }
            }
            if (cancel) {
                submitted.value()->Cancel();
            }
        }
    }

    void Drain() noexcept {
        std::size_t delivered = 0;
        while (delivered < chunks_.size()) {
            auto item = ordered_.WaitTake(std::chrono::seconds(1));
            if (!item.ok()) {
                std::lock_guard lock(mutex_);
                if (terminal_) {
                    break;
                }
                continue;
            }
            if (!item.value().value.has_value()) {
                break;
            }

            core::Status sink_status = core::Status::Ok();
            try {
                sink_status = sink_->OnChunk(std::move(item).value().value.value());
            } catch (const std::exception& exception) {
                sink_status = core::Status::Error(
                    core::ErrorCode::InternalError,
                    std::string("cloud task result sink threw: ") + exception.what());
            } catch (...) {
                sink_status = core::Status::Error(
                    core::ErrorCode::Unknown,
                    "cloud task result sink threw an unknown exception");
            }
            ++delivered;
            if (!sink_status.ok()) {
                Cancel();
                break;
            }
        }

        CloudTaskSummary summary;
        summary.task = chunks_.front().task;
        summary.chunk_count = chunks_.size();
        {
            std::lock_guard lock(mutex_);
            summary.completed_chunks = completed_chunks_;
            summary.successful_chunks = successful_chunks_;
            summary.failed_chunks = failed_chunks_;
            summary.status = terminal_ ? terminal_status_ : core::Status::Ok();
        }
        try {
            sink_->OnCompleted(std::move(summary));
        } catch (...) {
        }
        if (on_finished_) {
            on_finished_();
        }
    }

    std::shared_ptr<IAsyncLlmClient> client_;
    std::vector<CloudTaskChunk> chunks_;
    std::shared_ptr<ICloudTaskResultSink> sink_;
    core::OrderedBitmapWindow<CloudTaskChunkResult> ordered_;
    std::mutex mutex_;
    std::vector<std::shared_ptr<IAsyncLlmOperation>> operations_;
    std::vector<bool> completed_;
    std::thread drain_thread_;
    std::size_t next_index_ = 0;
    std::size_t inflight_ = 0;
    std::size_t max_inflight_ = 1;
    std::size_t completed_chunks_ = 0;
    std::size_t successful_chunks_ = 0;
    std::size_t failed_chunks_ = 0;
    bool terminal_ = false;
    core::Status terminal_status_ = core::Status::Ok();
    std::function<void(bool)> on_chunk_completed_;
    std::function<void()> on_finished_;
};

class FutureCloudTaskSink final : public ICloudTaskResultSink {
public:
    using Value = core::Result<std::vector<CloudTaskChunkResult>>;

    explicit FutureCloudTaskSink(std::shared_ptr<std::promise<Value>> promise)
        : promise_(std::move(promise)) {}

    core::Status OnChunk(CloudTaskChunkResult result) override {
        results_.push_back(std::move(result));
        return core::Status::Ok();
    }

    core::Status OnCompleted(CloudTaskSummary summary) override {
        if (!summary.status.ok()) {
            promise_->set_value(summary.status);
        } else {
            promise_->set_value(std::move(results_));
        }
        return core::Status::Ok();
    }

private:
    std::shared_ptr<std::promise<Value>> promise_;
    std::vector<CloudTaskChunkResult> results_;
};

}

class CloudTaskCoordinator::Impl
    : public std::enable_shared_from_this<CloudTaskCoordinator::Impl> {
public:
    Impl(std::shared_ptr<ILlmClient> llm_client,
         CloudTaskCoordinatorOptions options,
         core::LoggerAdapter logger,
         std::shared_ptr<IAsyncLlmClient> async_llm_client = nullptr)
        : llm_client_(std::move(llm_client)),
          async_llm_client_(std::move(async_llm_client)),
          options_(std::move(options)),
          logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("cloud-task")),
          async_workers_({
              .worker_count = 1,
              .queue_capacity = options_.max_pending_tasks,
              .name = options_.thread_pool_name + "-orchestrator",
          }) {}

    ~Impl() {
        async_workers_.Shutdown(true);
    }

    core::Result<std::vector<CloudTaskChunkResult>> Execute(std::vector<CloudTaskChunk> chunks) {
        std::lock_guard execute_lock(execute_mutex_);
        auto validation = Validate(chunks);
        if (!validation.ok()) {
            rejected_chunks_.fetch_add(chunks.size(), std::memory_order_relaxed);
            return validation;
        }
        if (!llm_client_) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "cloud task LLM client is missing");
        }

        running_.store(true, std::memory_order_release);
        submitted_chunks_.fetch_add(chunks.size(), std::memory_order_relaxed);
        core::BlockingQueue<std::size_t> backlog(options_.queue_capacity == 0 ? chunks.size() : options_.queue_capacity);
        core::OrderedBitmapWindow<CloudTaskChunkResult> ordered({
            .capacity = std::max<std::size_t>(chunks.size(), 1),
            .first_sequence = 1,
        });
        core::ThreadPool workers({
            .worker_count = options_.worker_count,
            .queue_capacity = options_.worker_count,
            .name = options_.thread_pool_name,
        });
        auto status = workers.Start();
        if (!status.ok()) {
            running_.store(false, std::memory_order_release);
            return status;
        }

        for (std::size_t worker = 0; worker < options_.worker_count; ++worker) {
            status = workers.Submit([&backlog, &ordered, &chunks, this] {
                while (true) {
                    auto index = backlog.WaitPop();
                    if (!index.ok()) {
                        return core::Status::Ok();
                    }
                    const auto& chunk = chunks[index.value()];
                    CloudTaskChunkResult result{
                        .task = chunk.task,
                        .session_sequence = chunk.session_sequence,
                        .chunk_sequence = chunk.chunk_sequence,
                        .attempt = chunk.attempt,
                    };
                    try {
                        auto response = llm_client_->Complete(chunk.request);
                        if (response.ok()) {
                            result.response = std::move(response).value();
                            result.status = core::Status::Ok();
                            successful_chunks_.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            result.status = response.status();
                            failed_chunks_.fetch_add(1, std::memory_order_relaxed);
                        }
                    } catch (const std::exception& exception) {
                        static_cast<void>(exception);
                        result.status = core::Status::Error(
                            core::ErrorCode::InternalError,
                            "cloud task client failed unexpectedly");
                        failed_chunks_.fetch_add(1, std::memory_order_relaxed);
                    } catch (...) {
                        result.status = core::Status::Error(core::ErrorCode::Unknown, "cloud task client failed unexpectedly");
                        failed_chunks_.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (!result.status.ok()) {
                        logger_.warn(
                            "[cloud-task] chunk failed task={} sequence={} code={} message={}",
                            result.task.task_id,
                            result.chunk_sequence,
                            static_cast<int>(result.status.code()),
                            result.status.message());
                    }
                    const auto admit_status = ordered.Admit(result.chunk_sequence, std::move(result));
                    if (!admit_status.ok()) {
                        logger_.warn(
                            "[cloud-task] ordered result admission failed code={} message={}",
                            static_cast<int>(admit_status.code()), admit_status.message());
                    }
                    completed_chunks_.fetch_add(1, std::memory_order_relaxed);
                }
            }, {}, "cloud-task-worker");
            if (!status.ok()) {
                backlog.Close(true);
                workers.Shutdown(false);
                running_.store(false, std::memory_order_release);
                return status;
            }
        }

        for (std::size_t index = 0; index < chunks.size(); ++index) {
            status = backlog.Push(index);
            if (!status.ok()) {
                rejected_chunks_.fetch_add(chunks.size() - index, std::memory_order_relaxed);
                break;
            }
        }
        backlog.Close();
        workers.Shutdown(true);
        running_.store(false, std::memory_order_release);

        if (!status.ok()) {
            return status;
        }
        status = ordered.Seal(chunks.back().chunk_sequence);
        if (!status.ok()) {
            return status;
        }

        std::vector<CloudTaskChunkResult> results;
        results.reserve(chunks.size());
        for (std::size_t index = 0; index < chunks.size(); ++index) {
            auto item = ordered.TryTake();
            if (!item.ok()) {
                return item.status();
            }
            if (!item.value().value.has_value()) {
                return item.value().terminal_status;
            }
            results.push_back(std::move(item).value().value.value());
        }
        return results;
    }

    core::Result<std::future<core::Result<std::vector<CloudTaskChunkResult>>>> ExecuteAsync(
        std::vector<CloudTaskChunk> chunks) {
        if (async_llm_client_) {
            using AsyncResult = core::Result<std::vector<CloudTaskChunkResult>>;
            auto promise = std::make_shared<std::promise<AsyncResult>>();
            auto future = promise->get_future();
            auto sink = std::make_shared<FutureCloudTaskSink>(promise);
            auto operation = StartAsync(std::move(chunks), std::move(sink));
            if (!operation.ok()) {
                return operation.status();
            }
            return future;
        }
        if (options_.max_pending_tasks == 0) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "cloud task max_pending_tasks must be positive");
        }
        {
            std::lock_guard lock(async_workers_mutex_);
            if (!async_workers_.running()) {
                auto start_status = async_workers_.Start();
                if (!start_status.ok()) {
                    return start_status;
                }
            }
        }

        using AsyncResult = core::Result<std::vector<CloudTaskChunkResult>>;
        auto promise = std::make_shared<std::promise<AsyncResult>>();
        auto future = promise->get_future();
        auto owned_chunks = std::make_shared<std::vector<CloudTaskChunk>>(std::move(chunks));
        auto submit_status = async_workers_.Submit(
            [this, promise, owned_chunks] {
                promise->set_value(Execute(std::move(*owned_chunks)));
                return core::Status::Ok();
            },
            {},
            "cloud-task-execution");
        if (!submit_status.ok()) {
            return submit_status;
        }
        return future;
    }

    core::Result<std::shared_ptr<ICloudTaskOperation>> StartAsync(
        std::vector<CloudTaskChunk> chunks,
        std::shared_ptr<ICloudTaskResultSink> sink);

    CloudTaskCoordinatorSnapshot Snapshot() const {
        return {
            .submitted_chunks = submitted_chunks_.load(std::memory_order_relaxed),
            .completed_chunks = completed_chunks_.load(std::memory_order_relaxed),
            .successful_chunks = successful_chunks_.load(std::memory_order_relaxed),
            .failed_chunks = failed_chunks_.load(std::memory_order_relaxed),
            .rejected_chunks = rejected_chunks_.load(std::memory_order_relaxed),
            .running = running_.load(std::memory_order_acquire) ||
                       active_tasks_.load(std::memory_order_acquire) != 0,
        };
    }

private:
    core::Status Validate(const std::vector<CloudTaskChunk>& chunks) const {
        if (options_.worker_count == 0 || options_.max_chunks == 0 || options_.max_pending_tasks == 0) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "cloud task coordinator options are invalid");
        }
        if (chunks.empty()) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "cloud task chunks are empty");
        }
        if (chunks.size() > options_.max_chunks) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "cloud task chunk capacity reached");
        }
        for (std::size_t index = 0; index < chunks.size(); ++index) {
            const auto& chunk = chunks[index];
            if (chunk.task.task_id.empty() || chunk.task.session_id.empty() ||
                chunk.chunk_sequence != index + 1 || chunk.final_chunk_sequence != chunks.size()) {
                return core::Status::Error(core::ErrorCode::InvalidArgument, "cloud task chunks must use contiguous sequence metadata");
            }
            if (chunk.request.messages.empty()) {
                return core::Status::Error(core::ErrorCode::InvalidArgument, "cloud task chunk request is empty");
            }
        }
        return core::Status::Ok();
    }

    std::shared_ptr<ILlmClient> llm_client_;
    std::shared_ptr<IAsyncLlmClient> async_llm_client_;
    CloudTaskCoordinatorOptions options_;
    core::LoggerAdapter logger_;
    std::mutex execute_mutex_;
    std::mutex async_workers_mutex_;
    core::ThreadPool async_workers_;
    std::atomic<std::size_t> submitted_chunks_{0};
    std::atomic<std::size_t> completed_chunks_{0};
    std::atomic<std::size_t> successful_chunks_{0};
    std::atomic<std::size_t> failed_chunks_{0};
    std::atomic<std::size_t> rejected_chunks_{0};
    std::atomic<bool> running_{false};
    std::atomic<std::size_t> active_tasks_{0};
};

core::Result<std::shared_ptr<ICloudTaskOperation>> CloudTaskCoordinator::Impl::StartAsync(
    std::vector<CloudTaskChunk> chunks,
    std::shared_ptr<ICloudTaskResultSink> sink) {
    auto validation = Validate(chunks);
    if (!validation.ok()) {
        rejected_chunks_.fetch_add(chunks.size(), std::memory_order_relaxed);
        return validation;
    }
    if (!async_llm_client_) {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "cloud task asynchronous LLM client is missing");
    }
    if (!sink) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "cloud task result sink is missing");
    }

    auto active_tasks = active_tasks_.load(std::memory_order_acquire);
    while (true) {
        if (active_tasks >= options_.max_pending_tasks) {
            rejected_chunks_.fetch_add(chunks.size(), std::memory_order_relaxed);
            return core::Status::Error(
                core::ErrorCode::ResourceExhausted,
                "cloud task active task capacity reached");
        }
        if (active_tasks_.compare_exchange_weak(
                active_tasks,
                active_tasks + 1,
                std::memory_order_acq_rel)) {
            break;
        }
    }
    submitted_chunks_.fetch_add(chunks.size(), std::memory_order_relaxed);
    auto owner = shared_from_this();
    auto operation = std::make_shared<AsyncCloudTaskOperation>(
        async_llm_client_,
        std::move(chunks),
        std::move(sink),
        options_.worker_count,
        [owner](bool successful) {
            owner->completed_chunks_.fetch_add(1, std::memory_order_relaxed);
            if (successful) {
                owner->successful_chunks_.fetch_add(1, std::memory_order_relaxed);
            } else {
                owner->failed_chunks_.fetch_add(1, std::memory_order_relaxed);
            }
        },
        [owner] { owner->active_tasks_.fetch_sub(1, std::memory_order_acq_rel); });
    operation->Start();
    return std::static_pointer_cast<ICloudTaskOperation>(std::move(operation));
}

CloudTaskCoordinator::CloudTaskCoordinator(
    std::shared_ptr<ILlmClient> llm_client,
    CloudTaskCoordinatorOptions options,
    core::LoggerAdapter logger)
    : impl_(std::make_shared<Impl>(std::move(llm_client), std::move(options), std::move(logger))) {}

CloudTaskCoordinator::CloudTaskCoordinator(
    std::shared_ptr<IAsyncLlmClient> llm_client,
    CloudTaskCoordinatorOptions options,
    core::LoggerAdapter logger)
    : impl_(std::make_shared<Impl>(
          nullptr,
          std::move(options),
          std::move(logger),
          std::move(llm_client))) {}

CloudTaskCoordinator::~CloudTaskCoordinator() = default;

core::Result<std::vector<CloudTaskChunkResult>> CloudTaskCoordinator::Execute(
    std::vector<CloudTaskChunk> chunks) {
    return impl_->Execute(std::move(chunks));
}

core::Result<std::future<core::Result<std::vector<CloudTaskChunkResult>>>>
CloudTaskCoordinator::ExecuteAsync(std::vector<CloudTaskChunk> chunks) {
    return impl_->ExecuteAsync(std::move(chunks));
}

core::Result<std::shared_ptr<ICloudTaskOperation>> CloudTaskCoordinator::StartAsync(
    std::vector<CloudTaskChunk> chunks,
    std::shared_ptr<ICloudTaskResultSink> sink) {
    return impl_->StartAsync(std::move(chunks), std::move(sink));
}

CloudTaskCoordinatorSnapshot CloudTaskCoordinator::Snapshot() const {
    return impl_->Snapshot();
}

}
