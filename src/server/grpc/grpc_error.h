#pragma once

#include "exception.h"
#include "result.h"

#include <grpcpp/support/status.h>

#include <exception>
#include <functional>
#include <string>
#include <utility>

namespace grpc_error {

grpc::Status ToGrpcStatus(const core::Status& status);
grpc::Status ToGrpcStatus(const std::exception& exception);
grpc::Status Internal(std::string message);

template <typename Fn>
grpc::Status GuardRpc(Fn&& fn) {
    try {
        return std::forward<Fn>(fn)();
    } catch (const core::AppException& e) {
        return ToGrpcStatus(e.status());
    } catch (const std::exception& e) {
        return ToGrpcStatus(e);
    } catch (...) {
        return Internal("Unknown server error");
    }
}

} // namespace grpc_error
