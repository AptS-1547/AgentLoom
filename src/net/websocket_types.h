#pragma once

#include "backpressure_queue.h"
#include "protocol_types.h"
#include "shared_buffer.h"

#include <boost/beast/websocket.hpp>

#include <chrono>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace net {

namespace beast = boost::beast;
namespace websocket = beast::websocket;

enum class WebSocketMessageKind {
    Text,
    Binary
};

struct WebSocketOptions {
    std::size_t max_frame_bytes = 1024 * 1024;
    std::size_t max_message_bytes = 16 * 1024 * 1024;
    std::size_t outbound_max_items = 256;
    std::size_t outbound_max_bytes = 16 * 1024 * 1024;
    std::size_t compression_min_bytes = 64 * 1024;
    bool enable_compression = true;
    std::chrono::milliseconds response_timeout{30000};
    std::chrono::milliseconds idle_timeout{60000};

    websocket::permessage_deflate CompressionOptions(bool server_role = true) const noexcept {
        websocket::permessage_deflate option;
        option.server_enable = server_role && enable_compression;
        option.client_enable = !server_role && enable_compression;
        option.server_no_context_takeover = true;
        option.client_no_context_takeover = true;
        option.msg_size_threshold = compression_min_bytes;
        return option;
    }
};

struct WebSocketFrame {
    WebSocketMessageKind kind = WebSocketMessageKind::Binary;
    bool final_fragment = true;
    bool compressed = false;
    SharedBuffer payload;

    std::size_t size() const noexcept {
        return payload.size();
    }

    void reset() noexcept {
        payload.reset();
        final_fragment = true;
        compressed = false;
        kind = WebSocketMessageKind::Binary;
    }
};

struct WebSocketMessage {
    WebSocketMessageKind kind = WebSocketMessageKind::Binary;
    bool final_fragment = true;
    bool compressed = false;
    std::vector<SharedBuffer> fragments;
    std::size_t total_bytes = 0;

    bool fragmented() const noexcept {
        return fragments.size() > 1;
    }

    std::size_t fragment_count() const noexcept {
        return fragments.size();
    }
};

class WebSocketMessageAssembler {
public:
    explicit WebSocketMessageAssembler(WebSocketOptions options = {})
        : options_(options) {}

    core::Status AppendFrame(WebSocketFrame frame) {
        if (complete_) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "message is already complete");
        }
        if (frame.size() > options_.max_frame_bytes) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "websocket frame is too large");
        }
        if (started_ && frame.kind != kind_) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "fragment kind changed");
        }

        const auto next_total = total_bytes_ + frame.size();
        if (next_total > options_.max_message_bytes) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "websocket message is too large");
        }

        if (!started_) {
            kind_ = frame.kind;
            compressed_ = frame.compressed;
            started_ = true;
        }

        total_bytes_ = next_total;
        complete_ = frame.final_fragment;
        fragments_.push_back(std::move(frame.payload));
        return core::Status::Ok();
    }

    bool complete() const noexcept {
        return complete_;
    }

    core::Result<WebSocketMessage> TakeMessage() {
        if (!complete_) {
            return core::Status::Error(core::ErrorCode::Unavailable, "websocket message is incomplete");
        }

        WebSocketMessage message;
        message.kind = kind_;
        message.final_fragment = true;
        message.compressed = compressed_;
        message.fragments = std::move(fragments_);
        message.total_bytes = total_bytes_;
        Reset();
        return message;
    }

    void Reset() noexcept {
        kind_ = WebSocketMessageKind::Binary;
        started_ = false;
        complete_ = false;
        compressed_ = false;
        total_bytes_ = 0;
        fragments_.clear();
    }

private:
    WebSocketOptions options_;
    WebSocketMessageKind kind_ = WebSocketMessageKind::Binary;
    bool started_ = false;
    bool complete_ = false;
    bool compressed_ = false;
    std::size_t total_bytes_ = 0;
    std::vector<SharedBuffer> fragments_;
};

inline bool ShouldCompressPayload(std::size_t payload_size, const WebSocketOptions& options) noexcept {
    return options.enable_compression &&
           options.compression_min_bytes > 0 &&
           payload_size >= options.compression_min_bytes;
}

using WebSocketOutboundQueue = BackpressureQueue<WebSocketFrame>;

inline WebSocketOutboundQueue MakeWebSocketOutboundQueue(const WebSocketOptions& options) {
    return WebSocketOutboundQueue(
        BackpressureOptions{options.outbound_max_items, options.outbound_max_bytes},
        [](const WebSocketFrame& frame) {
            return frame.size();
        });
}

struct WebSocketControlEvent {
    websocket::frame_type type = websocket::frame_type::close;
    std::string payload;
};

class WebSocketSessionHandle {
public:
    virtual ~WebSocketSessionHandle() = default;
    virtual core::Status Send(WebSocketFrame frame) = 0;
    virtual void Close(ConnectionCloseInfo close_info) = 0;
};

using WebSocketMessageHandler = std::function<void(WebSocketSessionHandle&, WebSocketMessage)>;
using WebSocketCloseHandler = std::function<void(const ConnectionCloseInfo&)>;

} // namespace net
