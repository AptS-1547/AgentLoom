#include "async_grpc_runtime.h"

namespace grpc_runtime {
namespace detail {

AsyncGrpcRuntimeState::AsyncGrpcRuntimeState(
    std::shared_ptr<core::ThreadPool> thread_pool_value,
    AsyncGrpcRuntimeOptions options_value,
    core::LoggerAdapter logger_value)
    : thread_pool(std::move(thread_pool_value)),
      options(std::move(options_value)),
      logger(logger_value.valid()
                 ? std::move(logger_value)
                 : core::LoggerAdapter::ForModule("async-grpc-runtime")) {}

bool AsyncGrpcRuntimeState::TryAcquire() noexcept {
    auto current = inflight_calls_.load(std::memory_order_relaxed);
    while (current < options.max_inflight_calls) {
        if (inflight_calls_.compare_exchange_weak(
                current,
                current + 1,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            accepted_calls_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

void AsyncGrpcRuntimeState::Release() noexcept {
    inflight_calls_.fetch_sub(1, std::memory_order_acq_rel);
}

void AsyncGrpcRuntimeState::MarkRejected() noexcept {
    rejected_calls_.fetch_add(1, std::memory_order_relaxed);
}

void AsyncGrpcRuntimeState::MarkSubmissionFailure() noexcept {
    submission_failures_.fetch_add(1, std::memory_order_relaxed);
}

void AsyncGrpcRuntimeState::MarkFinished(
    const core::Status& status,
    bool cancelled,
    bool deadline_exceeded) noexcept {
    completed_calls_.fetch_add(1, std::memory_order_relaxed);
    if (!status.ok()) {
        failed_calls_.fetch_add(1, std::memory_order_relaxed);
    }
    if (cancelled) {
        cancelled_calls_.fetch_add(1, std::memory_order_relaxed);
    }
    if (deadline_exceeded) {
        deadline_exceeded_calls_.fetch_add(1, std::memory_order_relaxed);
    }
}

AsyncGrpcRuntimeSnapshot AsyncGrpcRuntimeState::Snapshot() const noexcept {
    return {
        inflight_calls_.load(std::memory_order_relaxed),
        accepted_calls_.load(std::memory_order_relaxed),
        rejected_calls_.load(std::memory_order_relaxed),
        completed_calls_.load(std::memory_order_relaxed),
        failed_calls_.load(std::memory_order_relaxed),
        cancelled_calls_.load(std::memory_order_relaxed),
        deadline_exceeded_calls_.load(std::memory_order_relaxed),
        submission_failures_.load(std::memory_order_relaxed)};
}

} // namespace detail

core::Result<std::shared_ptr<AsyncGrpcRuntime>> AsyncGrpcRuntime::Create(
    std::shared_ptr<core::ThreadPool> thread_pool,
    AsyncGrpcRuntimeOptions options,
    core::LoggerAdapter logger) {
    if (!thread_pool) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "async gRPC runtime requires a thread pool");
    }
    if (options.max_inflight_calls == 0) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "async gRPC max_inflight_calls must be greater than zero");
    }
    return std::shared_ptr<AsyncGrpcRuntime>(new AsyncGrpcRuntime(
        std::move(thread_pool),
        std::move(options),
        std::move(logger)));
}

AsyncGrpcRuntime::AsyncGrpcRuntime(
    std::shared_ptr<core::ThreadPool> thread_pool,
    AsyncGrpcRuntimeOptions options,
    core::LoggerAdapter logger)
    : state_(std::make_shared<detail::AsyncGrpcRuntimeState>(
          std::move(thread_pool),
          std::move(options),
          std::move(logger))) {}

AsyncGrpcRuntimeSnapshot AsyncGrpcRuntime::Snapshot() const noexcept {
    return state_->Snapshot();
}

} // namespace grpc_runtime
