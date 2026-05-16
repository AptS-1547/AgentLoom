#include "grpc_error.h"

#include <grpcpp/support/status_code_enum.h>

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

} // namespace

grpc::Status ToGrpcStatus(const core::Status& status) {
    if (status.ok()) {
        return grpc::Status::OK;
    }
    return grpc::Status(ToGrpcStatusCode(status.code()), status.message());
}

grpc::Status ToGrpcStatus(const std::exception& exception) {
    return grpc::Status(grpc::StatusCode::INTERNAL, exception.what());
}

grpc::Status Internal(std::string message) {
    return grpc::Status(grpc::StatusCode::INTERNAL, std::move(message));
}

} // namespace grpc_error
