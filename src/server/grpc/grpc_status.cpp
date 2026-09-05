#include "grpc_status.h"

#include "trace_context.h"

#include <algorithm>
#include <cctype>
#include <string_view>

namespace grpc_runtime {

grpc::StatusCode ToGrpcStatusCode(core::ErrorCode code) noexcept {
    switch (code) {
    case core::ErrorCode::Ok: return grpc::StatusCode::OK;
    case core::ErrorCode::InvalidArgument: return grpc::StatusCode::INVALID_ARGUMENT;
    case core::ErrorCode::OutOfMemory: return grpc::StatusCode::RESOURCE_EXHAUSTED;
    case core::ErrorCode::NotFound: return grpc::StatusCode::NOT_FOUND;
    case core::ErrorCode::Timeout: return grpc::StatusCode::DEADLINE_EXCEEDED;
    case core::ErrorCode::Cancelled: return grpc::StatusCode::CANCELLED;
    case core::ErrorCode::AlreadyExists: return grpc::StatusCode::ALREADY_EXISTS;
    case core::ErrorCode::PermissionDenied: return grpc::StatusCode::PERMISSION_DENIED;
    case core::ErrorCode::FailedPrecondition: return grpc::StatusCode::FAILED_PRECONDITION;
    case core::ErrorCode::Unimplemented: return grpc::StatusCode::UNIMPLEMENTED;
    case core::ErrorCode::ResourceExhausted: return grpc::StatusCode::RESOURCE_EXHAUSTED;
    case core::ErrorCode::Unavailable: return grpc::StatusCode::UNAVAILABLE;
    case core::ErrorCode::DataLoss: return grpc::StatusCode::DATA_LOSS;
    case core::ErrorCode::InternalError:
    case core::ErrorCode::Unknown:
    default: return grpc::StatusCode::INTERNAL;
    }
}

namespace {

core::ErrorCode ToCoreErrorCode(grpc::StatusCode code) noexcept {
    switch (code) {
    case grpc::StatusCode::OK: return core::ErrorCode::Ok;
    case grpc::StatusCode::CANCELLED: return core::ErrorCode::Cancelled;
    case grpc::StatusCode::INVALID_ARGUMENT:
    case grpc::StatusCode::OUT_OF_RANGE: return core::ErrorCode::InvalidArgument;
    case grpc::StatusCode::DEADLINE_EXCEEDED: return core::ErrorCode::Timeout;
    case grpc::StatusCode::NOT_FOUND: return core::ErrorCode::NotFound;
    case grpc::StatusCode::ALREADY_EXISTS: return core::ErrorCode::AlreadyExists;
    case grpc::StatusCode::PERMISSION_DENIED:
    case grpc::StatusCode::UNAUTHENTICATED: return core::ErrorCode::PermissionDenied;
    case grpc::StatusCode::RESOURCE_EXHAUSTED: return core::ErrorCode::ResourceExhausted;
    case grpc::StatusCode::FAILED_PRECONDITION:
    case grpc::StatusCode::ABORTED: return core::ErrorCode::FailedPrecondition;
    case grpc::StatusCode::UNIMPLEMENTED: return core::ErrorCode::Unimplemented;
    case grpc::StatusCode::UNAVAILABLE: return core::ErrorCode::Unavailable;
    case grpc::StatusCode::INTERNAL: return core::ErrorCode::InternalError;
    case grpc::StatusCode::DATA_LOSS: return core::ErrorCode::DataLoss;
    case grpc::StatusCode::UNKNOWN:
    case grpc::StatusCode::DO_NOT_USE:
    default: return core::ErrorCode::Unknown;
    }
}

std::string MetadataValue(const grpc::ServerContextBase& context, std::string_view key) {
    const auto& metadata = context.client_metadata();
    const grpc::string_ref lookup_key(key.data(), key.size());
    const auto it = metadata.find(lookup_key);
    if (it == metadata.end()) {
        return {};
    }
    return std::string(it->second.data(), it->second.length());
}

std::string NormalizeIdentifier(std::string value) {
    constexpr std::size_t kMaxIdentifierLength = 128;
    if (value.empty() || value.size() > kMaxIdentifierLength) {
        return {};
    }
    const auto valid = std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return std::isalnum(ch) != 0 || ch == '-' || ch == '_' || ch == '.' || ch == ':';
    });
    return valid ? std::move(value) : std::string{};
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

core::Status ToCoreStatus(const grpc::Status& status) {
    if (status.ok()) {
        return core::Status::Ok();
    }
    return core::Status::Error(ToCoreErrorCode(status.error_code()), status.error_message());
}

std::string ResolveTraceId(const grpc::ServerContextBase& context) {
    auto trace_id = NormalizeIdentifier(MetadataValue(context, "x-trace-id"));
    if (trace_id.empty()) {
        trace_id = NormalizeIdentifier(MetadataValue(context, "x-request-id"));
    }
    return trace_id.empty() ? core::GenerateTraceId() : trace_id;
}

} // namespace grpc_runtime
