#include "websocket_session.h"

#include <algorithm>
#include <utility>

namespace net {

namespace {

WebSocketMessageKind WebSocketKindFromStream(const websocket::stream<beast::tcp_stream>& stream) noexcept {
    return stream.got_text() ? WebSocketMessageKind::Text : WebSocketMessageKind::Binary;
}

core::Status MessageTooLargeStatus(std::string message = "websocket message is too large") {
    return core::Status::Error(core::ErrorCode::ResourceExhausted, std::move(message));
}

} // namespace

class WebSocketSession::WebSocketStreamRequest final : public IWebSocketStreamRequest {
public:
    WebSocketStreamRequest(std::shared_ptr<WebSocketSession> session, WebSocketMessage message)
        : session_(std::move(session)),
          message_(std::move(message)) {}

    const ConnectionContext& connection() const noexcept override {
        return session_->lease_.context();
    }

    WebSocketMessage& message() noexcept override {
        return message_;
    }

    const WebSocketMessage& message() const noexcept override {
        return message_;
    }

    core::RawMemoryPool& memory_pool() noexcept override {
        return session_->lease_.memory_pool();
    }

    core::ThreadPool* task_pool() const noexcept override {
        return session_->lease_.task_pool();
    }

    core::Status Send(WebSocketFrame frame) override {
        return session_->Send(std::move(frame));
    }

    void Close(ConnectionCloseInfo close_info) override {
        session_->Close(std::move(close_info));
    }

private:
    std::shared_ptr<WebSocketSession> session_;
    WebSocketMessage message_;
};

WebSocketSession::WebSocketSession(tcp::socket socket,
                                   BeastHttpRequest request,
                                   ConnectionLease lease,
                                   WebSocketSessionOptions options,
                                   WebSocketSessionCallbacks callbacks)
    : stream_(std::move(socket)),
      request_(std::move(request)),
      lease_(std::move(lease)),
      options_(std::move(options)),
      callbacks_(std::move(callbacks)),
      outbound_queue_(MakeWebSocketOutboundQueue(options_.websocket)) {}

void WebSocketSession::Run() {
    auto compression = options_.websocket.CompressionOptions(true);
    stream_.set_option(compression);
    stream_.auto_fragment(true);
    stream_.read_message_max(options_.read_buffer_limit);
    stream_.write_buffer_bytes(16 * 1024);
    stream_.control_callback([self = shared_from_this()](websocket::frame_type type, beast::string_view payload) {
        self->OnControl(type, payload);
    });
    stream_.next_layer().expires_after(options_.request_timeout);
    stream_.async_accept(
        request_,
        beast::bind_front_handler(&WebSocketSession::OnAccept, shared_from_this()));
}

core::Status WebSocketSession::Send(WebSocketFrame frame) {
    WebSocketFrame owned_frame;
    owned_frame.kind = frame.kind;
    owned_frame.final_fragment = frame.final_fragment;
    owned_frame.compressed = frame.compressed;
    if (!frame.payload.empty()) {
        auto payload_result = SharedBuffer::Copy(memory_pool_, frame.payload.view());
        if (!payload_result.ok()) {
            return payload_result.status();
        }
        owned_frame.payload = std::move(payload_result).value();
    }

    auto status = outbound_queue_.TryPush(std::move(owned_frame));
    if (!status.ok()) {
        Close(ConnectionCloseInfo::Backpressure(status.message()));
        return status;
    }
    asio::post(stream_.get_executor(), [self = shared_from_this()] {
        self->DoWrite();
    });
    return core::Status::Ok();
}

void WebSocketSession::Close(ConnectionCloseInfo close_info) {
    asio::post(stream_.get_executor(), [self = shared_from_this(), close_info = std::move(close_info)]() mutable {
        self->DoClose(std::move(close_info));
    });
}

void WebSocketSession::OnAccept(beast::error_code ec) {
    stream_.next_layer().expires_never();
    if (ec) {
        NotifyClose({ConnectionCloseReason::ProtocolError,
                     core::Status::Error(core::ErrorCode::InvalidArgument, ec.message()),
                     ec.message()});
        return;
    }
    if (callbacks_.accept_handler) {
        callbacks_.accept_handler(*this);
    }
    DoReadSome();
}

void WebSocketSession::DoReadSome() {
    if (closing_) {
        return;
    }

    const auto max_frame_bytes = options_.websocket.max_frame_bytes;
    const auto max_message_bytes = options_.websocket.max_message_bytes;
    if (max_frame_bytes == 0 || max_message_bytes == 0) {
        DoClose({ConnectionCloseReason::InternalError,
                 core::Status::Error(core::ErrorCode::InvalidArgument, "websocket read limits must be positive"),
                 "websocket read limits must be positive"});
        return;
    }

    const auto remaining_message_capacity =
        current_message_bytes_ < max_message_bytes ? max_message_bytes - current_message_bytes_ : std::size_t{0};
    const auto read_capacity = discarding_oversized_message_
                                   ? max_frame_bytes
                                   : std::min(max_frame_bytes, std::max<std::size_t>(1, remaining_message_capacity));
    auto buffer_result = SharedBuffer::AllocateCapacity(memory_pool_, read_capacity);
    if (!buffer_result.ok()) {
        DoClose({ConnectionCloseReason::InternalError, buffer_result.status(), buffer_result.status().message()});
        return;
    }

    read_buffer_ = std::move(buffer_result).value();
    stream_.async_read_some(
        asio::buffer(read_buffer_.data(), read_buffer_.capacity()),
        beast::bind_front_handler(&WebSocketSession::OnReadSome, shared_from_this()));
}

void WebSocketSession::OnReadSome(beast::error_code ec, std::size_t bytes_transferred) {
    if (ec == websocket::error::closed) {
        NotifyClose(ConnectionCloseInfo::Remote("websocket closed"));
        return;
    }
    if (ec == websocket::error::message_too_big) {
        DispatchReadError(MessageTooLargeStatus(ec.message()), 0, true);
        return;
    }
    if (ec) {
        NotifyClose({ConnectionCloseReason::ProtocolError,
                     core::Status::Error(core::ErrorCode::InvalidArgument, ec.message()),
                     ec.message()});
        return;
    }

    auto resize_status = read_buffer_.resize(bytes_transferred);
    if (!resize_status.ok()) {
        DoClose({ConnectionCloseReason::InternalError, resize_status, resize_status.message()});
        return;
    }

    const auto max_message_bytes = options_.websocket.max_message_bytes;
    const auto final_fragment = stream_.is_message_done();
    const auto next_message_bytes = current_message_bytes_ + bytes_transferred;

    if (discarding_oversized_message_) {
        current_message_bytes_ = next_message_bytes;
        if (final_fragment) {
            current_message_bytes_ = 0;
            current_message_kind_ = WebSocketMessageKind::Binary;
            discarding_oversized_message_ = false;
        }
        DoReadSome();
        return;
    }

    if (current_message_bytes_ == 0) {
        current_message_kind_ = WebSocketKindFromStream(stream_);
    }

    if (bytes_transferred > max_message_bytes - current_message_bytes_ ||
        (next_message_bytes == max_message_bytes && !final_fragment)) {
        DispatchReadError(MessageTooLargeStatus(), next_message_bytes, final_fragment);
        if (!final_fragment) {
            discarding_oversized_message_ = true;
            current_message_bytes_ = next_message_bytes;
        } else {
            current_message_bytes_ = 0;
            current_message_kind_ = WebSocketMessageKind::Binary;
        }
        DoReadSome();
        return;
    }

    current_message_bytes_ = next_message_bytes;

    WebSocketMessage message;
    message.kind = current_message_kind_;
    message.final_fragment = final_fragment;
    message.fragments.push_back(std::move(read_buffer_));
    message.total_bytes = message.fragments.front().size();

    if (message.final_fragment) {
        current_message_bytes_ = 0;
        current_message_kind_ = WebSocketMessageKind::Binary;
    }

    DispatchMessage(std::move(message));
    DoReadSome();
}

void WebSocketSession::DoWrite() {
    if (writing_ || closing_) {
        return;
    }

    auto frame_result = outbound_queue_.TryPop();
    if (!frame_result.ok()) {
        return;
    }

    auto frame = std::move(frame_result).value();
    current_write_ = std::move(frame);
    stream_.text(current_write_.kind == WebSocketMessageKind::Text);
    writing_ = true;
    stream_.async_write(
        asio::buffer(current_write_.payload.data(), current_write_.payload.size()),
        beast::bind_front_handler(&WebSocketSession::OnWrite, shared_from_this()));
}

void WebSocketSession::OnWrite(beast::error_code ec, std::size_t) {
    writing_ = false;
    current_write_.reset();
    if (ec) {
        NotifyClose({ConnectionCloseReason::InternalError,
                     core::Status::Error(core::ErrorCode::InternalError, ec.message()),
                     ec.message()});
        return;
    }
    DoWrite();
}

void WebSocketSession::DoClose(ConnectionCloseInfo close_info) {
    if (closing_) {
        return;
    }
    closing_ = true;
    NotifyClose(close_info);
    websocket::close_reason reason;
    reason.reason = close_info.detail;
    stream_.async_close(reason, [self = shared_from_this()](beast::error_code) {});
}

void WebSocketSession::NotifyClose(const ConnectionCloseInfo& close_info) {
    if (close_notified_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    lease_.Close(close_info);
    if (callbacks_.close_handler) {
        callbacks_.close_handler(close_info);
    }
}

void WebSocketSession::OnControl(websocket::frame_type type, beast::string_view payload) {
    if (type == websocket::frame_type::close) {
        NotifyClose(ConnectionCloseInfo::Remote(std::string(payload)));
    }
}

void WebSocketSession::DispatchMessage(WebSocketMessage message) {
    if (callbacks_.stream_handler) {
        callbacks_.stream_handler(std::make_shared<WebSocketStreamRequest>(shared_from_this(), std::move(message)));
    } else if (callbacks_.message_handler) {
        callbacks_.message_handler(*this, std::move(message));
    }
}

void WebSocketSession::DispatchReadError(core::Status status, std::size_t bytes_transferred, bool final_fragment) {
    WebSocketMessage message;
    message.kind = current_message_kind_;
    message.final_fragment = final_fragment;
    message.status = std::move(status);
    message.total_bytes = bytes_transferred;
    DispatchMessage(std::move(message));
}

} // namespace net
