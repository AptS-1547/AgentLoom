#pragma once

#include "exception.h"
#include "grpc_status.h"
#include "logger_adapter.h"
#include "result.h"
#include "thread_pool.h"
#include "trace_context.h"

#include <grpcpp/server_context.h>
#include <grpcpp/support/server_callback.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <stop_token>
#include <string>
#include <utility>

namespace grpc_runtime {

struct AsyncGrpcCallContext {
    std::string trace_id;
    std::string method_name;
    std::chrono::system_clock::time_point deadline;
    std::stop_token stop_token;
};

struct AsyncGrpcRuntimeOptions {
    std::size_t max_inflight_calls = 256;
    std::string task_name = "grpc-unary";
};

struct AsyncGrpcRuntimeSnapshot {
    std::size_t inflight_calls = 0;
    std::uint64_t accepted_calls = 0;
    std::uint64_t rejected_calls = 0;
    std::uint64_t completed_calls = 0;
    std::uint64_t failed_calls = 0;
    std::uint64_t cancelled_calls = 0;
    std::uint64_t deadline_exceeded_calls = 0;
    std::uint64_t submission_failures = 0;
};

template <typename Request, typename Response>
class IAsyncUnaryRpcHandler {
public:
    virtual ~IAsyncUnaryRpcHandler() = default;

    virtual core::Status Handle(const AsyncGrpcCallContext& context,
                                const Request& request,
                                Response& response) = 0;
};

namespace detail {

class AsyncGrpcRuntimeState final {
public:
    AsyncGrpcRuntimeState(std::shared_ptr<core::ThreadPool> thread_pool,
                          AsyncGrpcRuntimeOptions options,
                          core::LoggerAdapter logger);

    bool TryAcquire() noexcept;
    void Release() noexcept;
    void MarkRejected() noexcept;
    void MarkSubmissionFailure() noexcept;
    void MarkFinished(const core::Status& status, bool cancelled, bool deadline_exceeded) noexcept;
    AsyncGrpcRuntimeSnapshot Snapshot() const noexcept;

    std::shared_ptr<core::ThreadPool> thread_pool;
    AsyncGrpcRuntimeOptions options;
    core::LoggerAdapter logger;

private:
    std::atomic<std::size_t> inflight_calls_{0};
    std::atomic<std::uint64_t> accepted_calls_{0};
    std::atomic<std::uint64_t> rejected_calls_{0};
    std::atomic<std::uint64_t> completed_calls_{0};
    std::atomic<std::uint64_t> failed_calls_{0};
    std::atomic<std::uint64_t> cancelled_calls_{0};
    std::atomic<std::uint64_t> deadline_exceeded_calls_{0};
    std::atomic<std::uint64_t> submission_failures_{0};
};

template <typename Request, typename Response>
class UnaryReactor final : public grpc::ServerUnaryReactor {
public:
    UnaryReactor(std::shared_ptr<AsyncGrpcRuntimeState> state,
                 grpc::CallbackServerContext& grpc_context,
                 const Request& request,
                 Response& response,
                 std::shared_ptr<IAsyncUnaryRpcHandler<Request, Response>> handler,
                 std::string method_name,
                 core::ThreadPoolTaskMetadata task_metadata)
        : state_(std::move(state)),
          call_state_(std::make_shared<CallState>(
              state_, grpc_context, request, response, std::move(handler), std::move(method_name))) {
        grpc_context.AddInitialMetadata("x-trace-id", call_state_->context().trace_id);
        grpc_context.AddTrailingMetadata("x-trace-id", call_state_->context().trace_id);
        call_state_->SetFinishCallback([this](core::Status status) {
            FinishCall(std::move(status));
        });
        Start(std::move(task_metadata));
    }

    void OnCancel() override {
        call_state_->RequestStop();
        call_state_->FinishQueued(core::Status::Error(
            core::ErrorCode::Cancelled,
            "async gRPC call cancelled"));
    }

    void OnDone() override {
        if (admitted_) {
            state_->Release();
        }
        delete this;
    }

private:
    class CallState final {
    public:
        CallState(std::shared_ptr<AsyncGrpcRuntimeState> runtime_state,
                  grpc::CallbackServerContext& grpc_context,
                  const Request& request,
                  Response& response,
                  std::shared_ptr<IAsyncUnaryRpcHandler<Request, Response>> handler,
                  std::string method_name)
            : runtime_state_(std::move(runtime_state)),
              grpc_context_(grpc_context),
              request_(request),
              response_(response),
              handler_(std::move(handler)),
              call_context_{ResolveTraceId(grpc_context_),
                            std::move(method_name),
                            grpc_context_.deadline(),
                            stop_source_.get_token()} {}

        const AsyncGrpcCallContext& context() const noexcept {
            return call_context_;
        }

        bool has_handler() const noexcept {
            return static_cast<bool>(handler_);
        }

        void SetFinishCallback(std::function<void(core::Status)> callback) {
            finish_callback_ = std::move(callback);
        }

        void RequestStop() noexcept {
            stop_source_.request_stop();
        }

        void FinishQueued(core::Status status) noexcept {
            Phase expected = Phase::Queued;
            if (phase_.compare_exchange_strong(
                    expected, Phase::Finishing, std::memory_order_acq_rel)) {
                Finish(std::move(status));
            }
        }

        void Run() noexcept {
            Phase expected = Phase::Queued;
            if (!phase_.compare_exchange_strong(
                    expected, Phase::Running, std::memory_order_acq_rel)) {
                return;
            }

            core::Status status = CancellationStatus();
            if (status.ok()) {
                try {
                    core::TraceContext trace_context{call_context_.trace_id, {}, {}};
                    core::TraceScope trace_scope(trace_context);
                    status = handler_->Handle(call_context_, request_, response_);
                } catch (const core::AppException& exception) {
                    status = exception.status();
                } catch (const std::exception& exception) {
                    runtime_state_->logger.error(
                        "[AsyncGrpcException] method={} trace_id={} error={}",
                        call_context_.method_name,
                        call_context_.trace_id,
                        exception.what());
                    status = core::Status::Error(
                        core::ErrorCode::InternalError,
                        "unhandled async gRPC handler exception");
                } catch (...) {
                    runtime_state_->logger.error(
                        "[AsyncGrpcException] method={} trace_id={} error=unknown",
                        call_context_.method_name,
                        call_context_.trace_id);
                    status = core::Status::Error(
                        core::ErrorCode::InternalError,
                        "unknown async gRPC handler exception");
                }
            }
            if (status.ok()) {
                status = CancellationStatus();
            }

            expected = Phase::Running;
            if (phase_.compare_exchange_strong(
                    expected, Phase::Finishing, std::memory_order_acq_rel)) {
                Finish(std::move(status));
            }
        }

        void FinishIfTaskDiscarded() noexcept {
            FinishQueued(core::Status::Error(
                core::ErrorCode::Unavailable,
                "async gRPC worker task was discarded"));
        }

    private:
        enum class Phase {
            Queued,
            Running,
            Finishing
        };

        core::Status CancellationStatus() const {
            if (std::chrono::system_clock::now() >= call_context_.deadline) {
                return core::Status::Error(
                    core::ErrorCode::Timeout,
                    "async gRPC deadline exceeded");
            }
            if (call_context_.stop_token.stop_requested() || grpc_context_.IsCancelled()) {
                return core::Status::Error(
                    core::ErrorCode::Cancelled,
                    "async gRPC call cancelled");
            }
            return core::Status::Ok();
        }

        void Finish(core::Status status) noexcept {
            const bool deadline_exceeded = status.code() == core::ErrorCode::Timeout;
            const bool cancelled = status.code() == core::ErrorCode::Cancelled;
            runtime_state_->MarkFinished(status, cancelled, deadline_exceeded);
            if (!status.ok()) {
                runtime_state_->logger.warn(
                    "[AsyncGrpcFailed] method={} trace_id={} error={}",
                    call_context_.method_name,
                    call_context_.trace_id,
                    status.message());
            }
            finish_callback_(std::move(status));
        }

        std::shared_ptr<AsyncGrpcRuntimeState> runtime_state_;
        grpc::CallbackServerContext& grpc_context_;
        const Request& request_;
        Response& response_;
        std::shared_ptr<IAsyncUnaryRpcHandler<Request, Response>> handler_;
        std::stop_source stop_source_;
        AsyncGrpcCallContext call_context_;
        std::function<void(core::Status)> finish_callback_;
        std::atomic<Phase> phase_{Phase::Queued};
    };

    class QueuedTaskGuard final {
    public:
        explicit QueuedTaskGuard(std::shared_ptr<CallState> call_state)
            : call_state_(std::move(call_state)) {}

        ~QueuedTaskGuard() {
            if (armed_.load(std::memory_order_acquire)) {
                call_state_->FinishIfTaskDiscarded();
            }
        }

        void Arm() noexcept {
            armed_.store(true, std::memory_order_release);
        }

    private:
        std::shared_ptr<CallState> call_state_;
        std::atomic<bool> armed_{false};
    };

    void Start(core::ThreadPoolTaskMetadata task_metadata) {
        if (!call_state_->has_handler()) {
            call_state_->FinishQueued(core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "async gRPC handler is null"));
            return;
        }
        if (!state_->TryAcquire()) {
            state_->MarkRejected();
            call_state_->FinishQueued(core::Status::Error(
                core::ErrorCode::ResourceExhausted,
                "async gRPC in-flight limit reached"));
            return;
        }
        admitted_ = true;

        auto queue_guard = std::make_shared<QueuedTaskGuard>(call_state_);
        const auto submit_status = state_->thread_pool->Submit(
            [call_state = call_state_, queue_guard](
                core::ThreadPoolContext&) -> core::Status {
                call_state->Run();
                return core::Status::Ok();
            },
            {},
            state_->options.task_name + ":" + call_state_->context().method_name,
            std::move(task_metadata));
        if (!submit_status.ok()) {
            state_->MarkSubmissionFailure();
            call_state_->FinishQueued(core::Status::Error(
                core::ErrorCode::Unavailable,
                "async gRPC worker submission failed: " + submit_status.message()));
            return;
        }
        queue_guard->Arm();
    }

    void FinishCall(core::Status status) noexcept {
        Finish(ToGrpcStatus(status));
    }

    std::shared_ptr<AsyncGrpcRuntimeState> state_;
    std::shared_ptr<CallState> call_state_;
    bool admitted_ = false;
};

} // namespace detail

class AsyncGrpcRuntime final {
public:
    static core::Result<std::shared_ptr<AsyncGrpcRuntime>> Create(
        std::shared_ptr<core::ThreadPool> thread_pool,
        AsyncGrpcRuntimeOptions options = {},
        core::LoggerAdapter logger = {});

    AsyncGrpcRuntime(const AsyncGrpcRuntime&) = delete;
    AsyncGrpcRuntime& operator=(const AsyncGrpcRuntime&) = delete;

    template <typename Request, typename Response>
    grpc::ServerUnaryReactor* StartUnary(
        grpc::CallbackServerContext& context,
        const Request& request,
        Response& response,
        std::shared_ptr<IAsyncUnaryRpcHandler<Request, Response>> handler,
        std::string method_name,
        core::ThreadPoolTaskMetadata task_metadata = {}) const {
        return new detail::UnaryReactor<Request, Response>(
            state_,
            context,
            request,
            response,
            std::move(handler),
            std::move(method_name),
            std::move(task_metadata));
    }

    AsyncGrpcRuntimeSnapshot Snapshot() const noexcept;

private:
    AsyncGrpcRuntime(std::shared_ptr<core::ThreadPool> thread_pool,
                     AsyncGrpcRuntimeOptions options,
                     core::LoggerAdapter logger);

    std::shared_ptr<detail::AsyncGrpcRuntimeState> state_;
};

} // namespace grpc_runtime
