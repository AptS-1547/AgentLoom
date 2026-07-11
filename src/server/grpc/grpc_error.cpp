#include "grpc_error.h"

#include <grpcpp/support/status_code_enum.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <string_view>
#include <typeinfo>

namespace grpc_error {

namespace {

grpc::StatusCode ToGrpcStatusCode(core::ErrorCode code) {
    switch (code) {
    case core::ErrorCode::Ok:
        return grpc::StatusCode::OK;
    case core::ErrorCode::InvalidArgument:
        return grpc::StatusCode::INVALID_ARGUMENT;
    case core::ErrorCode::OutOfMemory:
        return grpc::StatusCode::RESOURCE_EXHAUSTED;
    case core::ErrorCode::NotFound:
        return grpc::StatusCode::NOT_FOUND;
    case core::ErrorCode::Timeout:
        return grpc::StatusCode::DEADLINE_EXCEEDED;
    case core::ErrorCode::Cancelled:
        return grpc::StatusCode::CANCELLED;
    case core::ErrorCode::AlreadyExists:
        return grpc::StatusCode::ALREADY_EXISTS;
    case core::ErrorCode::PermissionDenied:
        return grpc::StatusCode::PERMISSION_DENIED;
    case core::ErrorCode::FailedPrecondition:
        return grpc::StatusCode::FAILED_PRECONDITION;
    case core::ErrorCode::Unimplemented:
        return grpc::StatusCode::UNIMPLEMENTED;
    case core::ErrorCode::ResourceExhausted:
        return grpc::StatusCode::RESOURCE_EXHAUSTED;
    case core::ErrorCode::Unavailable:
        return grpc::StatusCode::UNAVAILABLE;
    case core::ErrorCode::InternalError:
    case core::ErrorCode::Unknown:
    default:
        return grpc::StatusCode::INTERNAL;
    }
}

std::string_view ErrorCodeName(core::ErrorCode code) noexcept {
    switch (code) {
    case core::ErrorCode::Ok: return "OK";
    case core::ErrorCode::Unknown: return "UNKNOWN";
    case core::ErrorCode::InvalidArgument: return "INVALID_ARGUMENT";
    case core::ErrorCode::OutOfMemory: return "OUT_OF_MEMORY";
    case core::ErrorCode::NotFound: return "NOT_FOUND";
    case core::ErrorCode::Timeout: return "TIMEOUT";
    case core::ErrorCode::Cancelled: return "CANCELLED";
    case core::ErrorCode::AlreadyExists: return "ALREADY_EXISTS";
    case core::ErrorCode::PermissionDenied: return "PERMISSION_DENIED";
    case core::ErrorCode::FailedPrecondition: return "FAILED_PRECONDITION";
    case core::ErrorCode::Unimplemented: return "UNIMPLEMENTED";
    case core::ErrorCode::ResourceExhausted: return "RESOURCE_EXHAUSTED";
    case core::ErrorCode::Unavailable: return "UNAVAILABLE";
    case core::ErrorCode::InternalError: return "INTERNAL_ERROR";
    }
    return "UNKNOWN";
}

std::string_view GrpcStatusCodeName(grpc::StatusCode code) noexcept {
    switch (code) {
    case grpc::StatusCode::OK: return "OK";
    case grpc::StatusCode::CANCELLED: return "CANCELLED";
    case grpc::StatusCode::UNKNOWN: return "UNKNOWN";
    case grpc::StatusCode::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case grpc::StatusCode::DEADLINE_EXCEEDED: return "DEADLINE_EXCEEDED";
    case grpc::StatusCode::NOT_FOUND: return "NOT_FOUND";
    case grpc::StatusCode::ALREADY_EXISTS: return "ALREADY_EXISTS";
    case grpc::StatusCode::PERMISSION_DENIED: return "PERMISSION_DENIED";
    case grpc::StatusCode::RESOURCE_EXHAUSTED: return "RESOURCE_EXHAUSTED";
    case grpc::StatusCode::FAILED_PRECONDITION: return "FAILED_PRECONDITION";
    case grpc::StatusCode::ABORTED: return "ABORTED";
    case grpc::StatusCode::OUT_OF_RANGE: return "OUT_OF_RANGE";
    case grpc::StatusCode::UNIMPLEMENTED: return "UNIMPLEMENTED";
    case grpc::StatusCode::INTERNAL: return "INTERNAL";
    case grpc::StatusCode::UNAVAILABLE: return "UNAVAILABLE";
    case grpc::StatusCode::DATA_LOSS: return "DATA_LOSS";
    case grpc::StatusCode::UNAUTHENTICATED: return "UNAUTHENTICATED";
    case grpc::StatusCode::DO_NOT_USE: return "DO_NOT_USE";
    }
    return "UNKNOWN";
}

std::string MetadataValue(const grpc::ServerContext& context, std::string_view key) {
    const auto& metadata = context.client_metadata();
    const grpc::string_ref lookup_key(key.data(), key.size());
    const auto it = metadata.find(lookup_key);
    if (it == metadata.end()) {
        return {};
    }
    return std::string(it->second.data(), it->second.length());
}

std::string NormalizeLogIdentifier(std::string value) {
    constexpr std::size_t kMaxIdentifierLength = 128;
    if (value.empty() || value.size() > kMaxIdentifierLength) {
        return {};
    }
    const auto valid = std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return std::isalnum(ch) != 0 || ch == '-' || ch == '_' || ch == '.' || ch == ':';
    });
    return valid ? std::move(value) : std::string{};
}

std::string ResolveTraceId(const grpc::ServerContext& context) {
    auto trace_id = NormalizeLogIdentifier(MetadataValue(context, "x-trace-id"));
    if (trace_id.empty()) {
        trace_id = NormalizeLogIdentifier(MetadataValue(context, "x-request-id"));
    }
    if (trace_id.empty()) {
        trace_id = core::GenerateTraceId();
    }
    return trace_id;
}

const char* IdentifierOrDash(const std::string& value) noexcept {
    return value.empty() ? "-" : value.c_str();
}

bool IsExpectedClientFailure(core::ErrorCode code) noexcept {
    switch (code) {
    case core::ErrorCode::InvalidArgument:
    case core::ErrorCode::NotFound:
    case core::ErrorCode::AlreadyExists:
    case core::ErrorCode::PermissionDenied:
    case core::ErrorCode::FailedPrecondition:
    case core::ErrorCode::Unimplemented:
    case core::ErrorCode::Cancelled:
        return true;
    default:
        return false;
    }
}

} // namespace

grpc::Status ToGrpcStatus(const core::Status& status) {
    if (status.ok()) {
        return grpc::Status::OK;
    }
    return grpc::Status(ToGrpcStatusCode(status.code()), status.message());
}

grpc::Status ToGrpcStatus(const core::Status& status, grpc::StatusCode override_code) {
    if (status.ok()) {
        return grpc::Status::OK;
    }
    return grpc::Status(override_code, status.message());
}

RpcCall::RpcCall(grpc::ServerContext& context,
                 server_common::RuntimeStats& stats,
                 std::string method_name,
                 std::size_t sample_count,
                 bool is_batch,
                 int slow_request_ms,
                 RpcLogContext log_context,
                 core::LoggerAdapter logger)
    : context_(context),
      method_name_(std::move(method_name)),
      log_context_(std::move(log_context)),
      logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("inference-grpc")),
      trace_context_{ResolveTraceId(context_), {}, {}},
      trace_scope_(trace_context_),
      request_stats_(stats, method_name_, sample_count, is_batch, slow_request_ms, trace_context_.trace_id),
      started_at_(std::chrono::steady_clock::now()) {
    log_context_.request_id = NormalizeLogIdentifier(std::move(log_context_.request_id));
    log_context_.session_id = NormalizeLogIdentifier(std::move(log_context_.session_id));
    log_context_.task_type = NormalizeLogIdentifier(std::move(log_context_.task_type));
    context_.AddInitialMetadata("x-trace-id", trace_context_.trace_id);
    context_.AddTrailingMetadata("x-trace-id", trace_context_.trace_id);
    logger_.debug(
        "[InferenceRpcStarted] method={} trace_id={} request_id={} session_id={} task_type={}",
        method_name_,
        trace_context_.trace_id,
        IdentifierOrDash(log_context_.request_id),
        IdentifierOrDash(log_context_.session_id),
        IdentifierOrDash(log_context_.task_type));
}

grpc::Status RpcCall::Success() {
    if (const auto cancelled = CancellationStatus(); !cancelled.ok()) {
        return Failure(cancelled);
    }
    request_stats_.MarkSuccess();
    logger_.debug(
        "[InferenceRpcCompleted] method={} trace_id={} request_id={} session_id={} task_type={} latency_ms={:.3f}",
        method_name_,
        trace_context_.trace_id,
        IdentifierOrDash(log_context_.request_id),
        IdentifierOrDash(log_context_.session_id),
        IdentifierOrDash(log_context_.task_type),
        ElapsedMilliseconds());
    return grpc::Status::OK;
}

grpc::Status RpcCall::Failure(const core::Status& status) {
    return FinishFailure(status, ToGrpcStatusCode(status.code()));
}

grpc::Status RpcCall::Failure(const core::Status& status, grpc::StatusCode override_code) {
    return FinishFailure(status, override_code);
}

core::Status RpcCall::CancellationStatus() const {
    if (!context_.IsCancelled()) {
        return core::Status::Ok();
    }
    if (context_.deadline() <= std::chrono::system_clock::now()) {
        return core::Status::Error(core::ErrorCode::Timeout, "inference RPC deadline exceeded");
    }
    return core::Status::Error(core::ErrorCode::Cancelled, "inference RPC cancelled by client");
}

grpc::Status RpcCall::FinishFailure(const core::Status& status, grpc::StatusCode grpc_code) {
    if (status.ok()) {
        return Success();
    }
    request_stats_.MarkFailure(status.message());
    if (IsExpectedClientFailure(status.code())) {
        logger_.warn(
            "[InferenceRpcFailed] method={} trace_id={} request_id={} session_id={} task_type={} "
            "core_code={} grpc_code={} cancelled={} latency_ms={:.3f} error={}",
            method_name_,
            trace_context_.trace_id,
            IdentifierOrDash(log_context_.request_id),
            IdentifierOrDash(log_context_.session_id),
            IdentifierOrDash(log_context_.task_type),
            ErrorCodeName(status.code()),
            GrpcStatusCodeName(grpc_code),
            context_.IsCancelled(),
            ElapsedMilliseconds(),
            status.message());
    } else {
        logger_.error(
            "[InferenceRpcFailed] method={} trace_id={} request_id={} session_id={} task_type={} "
            "core_code={} grpc_code={} cancelled={} latency_ms={:.3f} error={}",
            method_name_,
            trace_context_.trace_id,
            IdentifierOrDash(log_context_.request_id),
            IdentifierOrDash(log_context_.session_id),
            IdentifierOrDash(log_context_.task_type),
            ErrorCodeName(status.code()),
            GrpcStatusCodeName(grpc_code),
            context_.IsCancelled(),
            ElapsedMilliseconds(),
            status.message());
    }
    return ToGrpcStatus(status, grpc_code);
}

grpc::Status RpcCall::UnexpectedException(const std::exception& exception) {
    const auto status = core::Status::Error(
        core::ErrorCode::InternalError,
        "Unhandled inference RPC exception");
    request_stats_.MarkFailure(status.message());
    logger_.error(
        "[InferenceRpcException] method={} trace_id={} request_id={} session_id={} task_type={} "
        "exception_type={} latency_ms={:.3f}",
        method_name_,
        trace_context_.trace_id,
        IdentifierOrDash(log_context_.request_id),
        IdentifierOrDash(log_context_.session_id),
        IdentifierOrDash(log_context_.task_type),
        typeid(exception).name(),
        ElapsedMilliseconds());
    return ToGrpcStatus(status);
}

grpc::Status RpcCall::UnknownException() {
    const auto status = core::Status::Error(
        core::ErrorCode::InternalError,
        "Unknown inference RPC exception");
    request_stats_.MarkFailure(status.message());
    logger_.error(
        "[InferenceRpcException] method={} trace_id={} request_id={} session_id={} task_type={} "
        "exception_type=unknown latency_ms={:.3f}",
        method_name_,
        trace_context_.trace_id,
        IdentifierOrDash(log_context_.request_id),
        IdentifierOrDash(log_context_.session_id),
        IdentifierOrDash(log_context_.task_type),
        ElapsedMilliseconds());
    return ToGrpcStatus(status);
}

double RpcCall::ElapsedMilliseconds() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - started_at_)
        .count();
}

} // namespace grpc_error
