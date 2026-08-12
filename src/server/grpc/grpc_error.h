#pragma once

#include "exception.h"
#include "grpc_status.h"
#include "logger_adapter.h"
#include "result.h"
#include "server_common.h"
#include "trace_context.h"

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/status_code_enum.h>

#include <chrono>
#include <cstddef>
#include <exception>
#include <string>
#include <utility>

namespace grpc_error {

using grpc_runtime::ToGrpcStatus;

struct RpcLogContext {
    std::string request_id;
    std::string session_id;
    std::string task_type;
};

class RpcCall {
public:
    RpcCall(grpc::ServerContext& context,
            server_common::RuntimeStats& stats,
            std::string method_name,
            std::size_t sample_count,
            bool is_batch,
            int slow_request_ms,
            RpcLogContext log_context = {},
            core::LoggerAdapter logger = {});

    template <typename Fn>
    grpc::Status Run(Fn&& fn) {
        try {
            return std::forward<Fn>(fn)();
        } catch (const core::AppException& exception) {
            return Failure(exception.status());
        } catch (const std::exception& exception) {
            return UnexpectedException(exception);
        } catch (...) {
            return UnknownException();
        }
    }

    grpc::Status Success();
    grpc::Status Failure(const core::Status& status);
    grpc::Status Failure(const core::Status& status, grpc::StatusCode override_code);
    core::Status CancellationStatus() const;

    const std::string& trace_id() const noexcept {
        return trace_context_.trace_id;
    }

private:
    grpc::Status FinishFailure(const core::Status& status, grpc::StatusCode grpc_code);
    grpc::Status UnexpectedException(const std::exception& exception);
    grpc::Status UnknownException();
    double ElapsedMilliseconds() const;

    grpc::ServerContext& context_;
    std::string method_name_;
    RpcLogContext log_context_;
    core::LoggerAdapter logger_;
    core::TraceContext trace_context_;
    core::TraceScope trace_scope_;
    server_common::ScopedRequestStats request_stats_;
    std::chrono::steady_clock::time_point started_at_;
};

} // namespace grpc_error
