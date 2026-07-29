#pragma once

#include "result.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

namespace net {

enum class ConnectionCloseReason {
    RemoteClosed,
    IdleTimeout,
    ResponseTimeout,
    AccessDenied,
    BackpressureLimit,
    ProtocolError,
    ServerShutdown,
    InternalError
};

struct ConnectionCloseInfo {
    ConnectionCloseReason reason = ConnectionCloseReason::RemoteClosed;
    core::Status status = core::Status::Ok();
    std::string detail;
    std::uint64_t connection_id = 0;
    std::string target;

    static ConnectionCloseInfo Remote(std::string detail = {}) {
        return {ConnectionCloseReason::RemoteClosed, core::Status::Ok(), std::move(detail)};
    }

    static ConnectionCloseInfo Timeout(std::string detail = {}) {
        return {ConnectionCloseReason::ResponseTimeout,
                core::Status::Error(core::ErrorCode::Timeout, detail),
                std::move(detail)};
    }

    static ConnectionCloseInfo IdleTimeout(std::string detail = {}) {
        return {ConnectionCloseReason::IdleTimeout,
                core::Status::Error(core::ErrorCode::Timeout, detail),
                std::move(detail)};
    }

    static ConnectionCloseInfo AccessDenied(std::string detail = {}) {
        return {ConnectionCloseReason::AccessDenied,
                core::Status::Error(core::ErrorCode::PermissionDenied, detail),
                std::move(detail)};
    }

    static ConnectionCloseInfo Backpressure(std::string detail = {}) {
        return {ConnectionCloseReason::BackpressureLimit,
                core::Status::Error(core::ErrorCode::ResourceExhausted, detail),
                std::move(detail)};
    }

    static ConnectionCloseInfo Shutdown(std::string detail = {}) {
        return {ConnectionCloseReason::ServerShutdown,
                core::Status::Error(core::ErrorCode::Cancelled, detail),
                std::move(detail)};
    }
};

struct ConnectionContext {
    std::uint64_t connection_id = 0;
    std::string remote_address;
    std::chrono::steady_clock::time_point connected_at = std::chrono::steady_clock::now();
};

enum class AccessDecisionType {
    Allow,
    Deny
};

struct AccessDecision {
    AccessDecisionType type = AccessDecisionType::Allow;
    unsigned http_status = 200;
    core::Status status = core::Status::Ok();
    std::string reason;

    static AccessDecision Allow() {
        return {};
    }

    static AccessDecision Deny(unsigned http_status, std::string reason) {
        return {AccessDecisionType::Deny,
                http_status,
                core::Status::Error(core::ErrorCode::PermissionDenied, reason),
                std::move(reason)};
    }

    bool allowed() const noexcept {
        return type == AccessDecisionType::Allow;
    }
};

using AccessCompletion = std::function<void(AccessDecision)>;

template <typename Request>
using AccessController = std::function<void(const Request&, AccessCompletion)>;

} // namespace net
