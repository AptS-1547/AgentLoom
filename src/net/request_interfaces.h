#pragma once

#include "http_types.h"
#include "websocket_types.h"

#include <functional>
#include <memory>

namespace core {
class RawMemoryPool;
class ThreadPool;
} // namespace core

namespace net {

class IHttpRequest {
public:
    virtual ~IHttpRequest() = default;

    virtual const BeastHttpRequest& message() const noexcept = 0;
    virtual const ConnectionContext& connection() const noexcept = 0;
    virtual core::RawMemoryPool& memory_pool() noexcept = 0;
    virtual core::ThreadPool* task_pool() const noexcept = 0;
    // Respond is thread-safe with respect to the session executor: callers may
    // complete work asynchronously and respond from another thread.
    virtual core::Status Respond(http::message_generator response) = 0;
    virtual void Close(ConnectionCloseInfo close_info) = 0;
};

class IWebSocketStreamRequest {
public:
    virtual ~IWebSocketStreamRequest() = default;

    virtual const BeastHttpRequest& handshake_request() const noexcept = 0;
    virtual const ConnectionContext& connection() const noexcept = 0;
    virtual WebSocketMessage& message() noexcept = 0;
    virtual const WebSocketMessage& message() const noexcept = 0;
    virtual core::RawMemoryPool& memory_pool() noexcept = 0;
    virtual core::ThreadPool* task_pool() const noexcept = 0;
    // Send enqueues outbound frames and may return ResourceExhausted when
    // websocket outbound backpressure limits are reached.
    virtual core::Status Send(WebSocketFrame frame) = 0;
    virtual void Close(ConnectionCloseInfo close_info) = 0;
};

using IHttpRequestHandler = std::function<void(std::shared_ptr<IHttpRequest>)>;
using IWebSocketStreamHandler = std::function<void(std::shared_ptr<IWebSocketStreamRequest>)>;

} // namespace net
