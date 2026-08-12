#pragma once

#include "result.h"

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>

#include <string>

namespace grpc_runtime {

grpc::StatusCode ToGrpcStatusCode(core::ErrorCode code) noexcept;
grpc::Status ToGrpcStatus(const core::Status& status);
grpc::Status ToGrpcStatus(const core::Status& status, grpc::StatusCode override_code);
core::Status ToCoreStatus(const grpc::Status& status);

std::string ResolveTraceId(const grpc::ServerContextBase& context);

} // namespace grpc_runtime
