#include "cloud_task_coordinator.h"

#include "blocking_queue.h"
#include "logger_adapter.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <future>
#include <mutex>
#include <utility>

namespace agent::llm {

class CloudTaskCoordinator::Impl {
public:
    Impl(std::shared_ptr<ILlmClient> llm_client,
         CloudTaskCoordinatorOptions options,
         core::LoggerAdapter logger)
        : llm_client_(std::move(llm_client)),
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

    CloudTaskCoordinatorSnapshot Snapshot() const {
        return {
            .submitted_chunks = submitted_chunks_.load(std::memory_order_relaxed),
            .completed_chunks = completed_chunks_.load(std::memory_order_relaxed),
            .successful_chunks = successful_chunks_.load(std::memory_order_relaxed),
            .failed_chunks = failed_chunks_.load(std::memory_order_relaxed),
            .rejected_chunks = rejected_chunks_.load(std::memory_order_relaxed),
            .running = running_.load(std::memory_order_acquire),
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
};

CloudTaskCoordinator::CloudTaskCoordinator(
    std::shared_ptr<ILlmClient> llm_client,
    CloudTaskCoordinatorOptions options,
    core::LoggerAdapter logger)
    : impl_(std::make_unique<Impl>(std::move(llm_client), std::move(options), std::move(logger))) {}

CloudTaskCoordinator::~CloudTaskCoordinator() = default;

core::Result<std::vector<CloudTaskChunkResult>> CloudTaskCoordinator::Execute(
    std::vector<CloudTaskChunk> chunks) {
    return impl_->Execute(std::move(chunks));
}

core::Result<std::future<core::Result<std::vector<CloudTaskChunkResult>>>>
CloudTaskCoordinator::ExecuteAsync(std::vector<CloudTaskChunk> chunks) {
    return impl_->ExecuteAsync(std::move(chunks));
}

CloudTaskCoordinatorSnapshot CloudTaskCoordinator::Snapshot() const {
    return impl_->Snapshot();
}

} // namespace agent::llm
