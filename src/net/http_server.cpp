#include "http_server.h"

#include "exception.h"

#include <algorithm>
#include <deque>
#include <exception>
#include <utility>

namespace net {

namespace {

std::size_t ResolveIoThreads(std::size_t requested) noexcept {
    if (requested > 0) {
        return requested;
    }
    const auto hardware = std::thread::hardware_concurrency();
    return std::max<std::size_t>(1, hardware == 0 ? 1 : hardware);
}

std::string RemoteAddress(const tcp::socket& socket) {
    boost::system::error_code ec;
    auto endpoint = socket.remote_endpoint(ec);
    if (ec) {
        return {};
    }
    return endpoint.address().to_string(ec);
}

http::message_generator MakeStatusResponse(const BeastHttpRequest& request,
                                           http::status status,
                                           std::string body) {
    auto response = HttpResponse::Text(status, std::move(body));
    response.message.version(request.version());
    response.message.keep_alive(request.keep_alive());
    response.message.set(http::field::server, "AgentBackendPredict");
    response.message.prepare_payload();
    return std::move(response.message);
}

http::message_generator MakeStatusResponse(unsigned version,
                                           bool keep_alive,
                                           http::status status,
                                           std::string body) {
    auto response = HttpResponse::Text(status, std::move(body));
    response.message.version(version);
    response.message.keep_alive(keep_alive);
    response.message.set(http::field::server, "AgentBackendPredict");
    response.message.prepare_payload();
    return std::move(response.message);
}

http::status StatusFromAccessDecision(const AccessDecision& decision) noexcept {
    auto status = static_cast<http::status>(decision.http_status);
    if (status == http::status::unknown) {
        return http::status::forbidden;
    }
    return status;
}

WebSocketMessageKind WebSocketKindFromStream(const websocket::stream<beast::tcp_stream>& stream) noexcept {
    return stream.got_text() ? WebSocketMessageKind::Text : WebSocketMessageKind::Binary;
}

} 

class HttpServer::Listener : public std::enable_shared_from_this<Listener> {
public:
    void OnAccept(beast::error_code ec, tcp::socket socket);

    Listener(HttpServer& server, tcp::endpoint endpoint)
        : server_(server),
          acceptor_(asio::make_strand(server.io_context_)) {
        beast::error_code ec;
        acceptor_.open(endpoint.protocol(), ec);
        if (ec) {
            throw core::AppException(core::Status::Error(core::ErrorCode::InternalError, ec.message()));
        }
#if !defined(_WIN32)
        acceptor_.set_option(asio::socket_base::reuse_address(true), ec);
        if (ec) {
            throw core::AppException(core::Status::Error(core::ErrorCode::InternalError, ec.message()));
        }
#endif
        acceptor_.bind(endpoint, ec);
        if (ec) {
            throw core::AppException(core::Status::Error(core::ErrorCode::InternalError, ec.message()));
        }
        acceptor_.listen(asio::socket_base::max_listen_connections, ec);
        if (ec) {
            throw core::AppException(core::Status::Error(core::ErrorCode::InternalError, ec.message()));
        }
    }

    void Run() {
        DoAccept();
    }

    void Stop() {
        beast::error_code ec;
        acceptor_.cancel(ec);
        acceptor_.close(ec);
    }

    std::uint16_t port() const noexcept {
        beast::error_code ec;
        auto endpoint = acceptor_.local_endpoint(ec);
        return ec ? 0 : endpoint.port();
    }

private:
    void DoAccept() {
        acceptor_.async_accept(
            asio::make_strand(server_.io_context_),
            beast::bind_front_handler(&Listener::OnAccept, shared_from_this()));
    }

    HttpServer& server_;
    tcp::acceptor acceptor_;
};


void HttpServer::HttpSession::DoRead() {
    request_ = {};
    buffer_.consume(buffer_.size());
    stream_.expires_after(server_.options_.request_timeout);
    http::async_read(
        stream_,
        buffer_,
        request_,
        beast::bind_front_handler(&HttpSession::OnRead, shared_from_this()));
}

void HttpServer::HttpSession::OnRead(beast::error_code ec, std::size_t) {
    if (ec == http::error::end_of_stream) {
        Close(ConnectionCloseInfo::Remote());
        return;
    }
    if (ec == beast::error::timeout) {
        Close(ConnectionCloseInfo::Timeout("http request timeout"));
        return;
    }
    if (ec) {
        Close({ConnectionCloseReason::ProtocolError,
               core::Status::Error(core::ErrorCode::InvalidArgument, ec.message()),
               ec.message()});
        return;
    }

    if (request_.body().size() > server_.options_.request_body_limit) {
        Send(MakeStatusResponse(request_, http::status::payload_too_large, "payload too large"));
        return;
    }

    auto access_controller = server_.AccessControllerSnapshot();
    if (access_controller) {
        access_controller(request_, lease_.context(), [self = shared_from_this()](AccessDecision decision) {
            asio::post(self->stream_.get_executor(), [self, decision = std::move(decision)]() mutable {
                self->OnAccessDecision(std::move(decision));
            });
        });
        return;
    }

    OnAccessDecision(AccessDecision::Allow());
}

void HttpServer::HttpSession::OnAccessDecision(AccessDecision decision) {
    if (!decision.allowed()) {
        auto body = decision.reason.empty() ? std::string("access denied") : decision.reason;
        Send(MakeStatusResponse(request_, StatusFromAccessDecision(decision), std::move(body)));
        return;
    }

    if (websocket::is_upgrade(request_) && TryUpgradeWebSocket()) {
        return;
    }

    auto request_handler = server_.HttpRequestHandlerSnapshot();
    if (request_handler) {
        std::shared_ptr<IHttpRequest> request =
            std::make_shared<HttpServerRequest>(shared_from_this(), std::move(request_));
        request_handler(std::move(request));
        return;
    }

    auto handler = server_.HttpHandlerSnapshot();
    if (handler) {
        auto request = HttpRequest::FromBeast(std::move(request_), lease_.context());
        handler(std::move(request), HttpGeneratorCallback([self = shared_from_this()](http::message_generator response) mutable {
            asio::post(self->stream_.get_executor(), [self, response = std::move(response)]() mutable {
                self->Send(std::move(response));
            });
        }));
        return;
    }

    if (auto static_files = server_.StaticFileHandlerSnapshot()) {
        Send(static_files->Handle(request_));
        return;
    }

    Send(MakeStatusResponse(request_, http::status::not_found, "not found"));
}

void HttpServer::HttpSession::Send(http::message_generator message) {
    const auto keep_alive = message.keep_alive();
    stream_.expires_after(server_.options_.request_timeout);
    beast::async_write(
        stream_,
        std::move(message),
        beast::bind_front_handler(&HttpSession::OnWrite, shared_from_this(), keep_alive));
}

void HttpServer::HttpSession::OnWrite(bool keep_alive, beast::error_code ec, std::size_t) {
    if (ec == beast::error::timeout) {
        Close(ConnectionCloseInfo::Timeout("http response timeout"));
        return;
    }
    if (ec) {
        Close({ConnectionCloseReason::InternalError,
               core::Status::Error(core::ErrorCode::InternalError, ec.message()),
               ec.message()});
        return;
    }
    lease_.Touch();

    if (!keep_alive) {
        Close(ConnectionCloseInfo::Remote("http keep-alive disabled"));
        return;
    }
    DoRead();
}

void HttpServer::HttpSession::Close(ConnectionCloseInfo close_info) {
    beast::error_code ec;
    stream_.socket().shutdown(tcp::socket::shutdown_send, ec);
    stream_.socket().close(ec);
    lease_.Close(std::move(close_info));
}

core::Status HttpServer::HttpSession::SendFromRequest(http::message_generator response) {
    asio::post(stream_.get_executor(), [self = shared_from_this(), response = std::move(response)]() mutable {
        self->Send(std::move(response));
    });
    return core::Status::Ok();
}

void HttpServer::HttpSession::CloseFromRequest(ConnectionCloseInfo close_info) {
    asio::post(stream_.get_executor(), [self = shared_from_this(), close_info = std::move(close_info)]() mutable {
        self->Close(std::move(close_info));
    });
}

const ConnectionContext& HttpServer::HttpSession::connection() const noexcept {
    return lease_.context();
}

core::RawMemoryPool& HttpServer::HttpSession::memory_pool() noexcept {
    return lease_.memory_pool();
}

core::ThreadPool* HttpServer::HttpSession::task_pool() const noexcept {
    return lease_.task_pool();
}

void HttpServer::Listener::OnAccept(beast::error_code ec, tcp::socket socket) {
    if (!server_.running_.load(std::memory_order_acquire)) {
        return;
    }
    if (!ec) {
        ConnectionContext connection{server_.NextConnectionId(), RemoteAddress(socket)};
        auto lease_result = server_.connection_pool_.Acquire(ProtocolConnectionKind::Http, std::move(connection));
        if (lease_result.ok()) {
            std::make_shared<HttpSession>(server_, std::move(socket), std::move(lease_result).value())->Run();
        } else {
            beast::error_code close_ec;
            socket.shutdown(tcp::socket::shutdown_both, close_ec);
            socket.close(close_ec);
        }
    }
    DoAccept();
}

class HttpServer::WebSocketSession : public WebSocketSessionHandle,
                                     public std::enable_shared_from_this<WebSocketSession> {
public:
    WebSocketSession(HttpServer& server,
                     tcp::socket socket,
                     BeastHttpRequest request,
                     ConnectionLease lease,
                     WebSocketMessageHandler handler)
        : server_(server),
          stream_(std::move(socket)),
          request_(std::move(request)),
          lease_(std::move(lease)),
          handler_(std::move(handler)),
          outbound_queue_(MakeWebSocketOutboundQueue(server_.options_.websocket)) {}

    void Run() {
        auto compression = server_.options_.websocket.CompressionOptions(true);
        stream_.set_option(compression);
        stream_.auto_fragment(true);
        stream_.read_message_max(server_.options_.websocket_read_buffer_limit);
        stream_.write_buffer_bytes(16 * 1024);
        stream_.control_callback([self = shared_from_this()](websocket::frame_type type, beast::string_view payload) {
            self->OnControl(type, payload);
        });
        stream_.next_layer().expires_after(server_.options_.request_timeout);
        stream_.async_accept(
            request_,
            beast::bind_front_handler(&WebSocketSession::OnAccept, shared_from_this()));
    }

    core::Status Send(WebSocketFrame frame) override {
        auto status = outbound_queue_.TryPush(std::move(frame));
        if (!status.ok()) {
            Close(ConnectionCloseInfo::Backpressure(status.message()));
            return status;
        }
        asio::post(stream_.get_executor(), [self = shared_from_this()] {
            self->DoWrite();
        });
        return core::Status::Ok();
    }

    void Close(ConnectionCloseInfo close_info) override {
        asio::post(stream_.get_executor(), [self = shared_from_this(), close_info = std::move(close_info)]() mutable {
            self->DoClose(std::move(close_info));
        });
    }

private:
    class WebSocketStreamRequest final : public IWebSocketStreamRequest {
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

    void OnAccept(beast::error_code ec) {
        stream_.next_layer().expires_never();
        if (ec) {
            NotifyClose({ConnectionCloseReason::ProtocolError,
                         core::Status::Error(core::ErrorCode::InvalidArgument, ec.message()),
                         ec.message()});
            return;
        }
        if (auto accept_handler = server_.WebSocketAcceptHandlerSnapshot()) {
            accept_handler(*this);
        }
        DoReadSome();
    }

    void DoReadSome() {
        if (closing_) {
            return;
        }

        const auto max_frame_bytes = server_.options_.websocket.max_frame_bytes;
        const auto max_message_bytes = server_.options_.websocket.max_message_bytes;
        if (max_frame_bytes == 0 || max_message_bytes == 0) {
            DoClose({ConnectionCloseReason::InternalError,
                     core::Status::Error(core::ErrorCode::InvalidArgument, "websocket read limits must be positive"),
                     "websocket read limits must be positive"});
            return;
        }
        if (current_message_bytes_ >= max_message_bytes) {
            DoClose({ConnectionCloseReason::BackpressureLimit,
                     core::Status::Error(core::ErrorCode::ResourceExhausted, "websocket message is too large"),
                     "websocket message is too large"});
            return;
        }

        const auto read_capacity = std::min(max_frame_bytes, max_message_bytes - current_message_bytes_);
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

    void OnReadSome(beast::error_code ec, std::size_t bytes_transferred) {
        if (ec == websocket::error::closed) {
            NotifyClose(ConnectionCloseInfo::Remote("websocket closed"));
            return;
        }
        if (ec == websocket::error::message_too_big) {
            DoClose({ConnectionCloseReason::BackpressureLimit,
                     core::Status::Error(core::ErrorCode::ResourceExhausted, ec.message()),
                     ec.message()});
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

        const auto max_message_bytes = server_.options_.websocket.max_message_bytes;
        const auto final_fragment = stream_.is_message_done();
        const auto next_message_bytes = current_message_bytes_ + bytes_transferred;
        if (bytes_transferred > max_message_bytes - current_message_bytes_ ||
            (next_message_bytes == max_message_bytes && !final_fragment)) {
            DoClose({ConnectionCloseReason::BackpressureLimit,
                     core::Status::Error(core::ErrorCode::ResourceExhausted, "websocket message is too large"),
                     "websocket message is too large"});
            return;
        }

        if (current_message_bytes_ == 0) {
            current_message_kind_ = WebSocketKindFromStream(stream_);
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

        if (auto stream_handler = server_.WebSocketStreamHandlerSnapshot(request_.target())) {
            stream_handler(std::make_shared<WebSocketStreamRequest>(shared_from_this(), std::move(message)));
        } else if (handler_) {
            handler_(*this, std::move(message));
        }

        DoReadSome();
    }

    void DoWrite() {
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

    void OnWrite(beast::error_code ec, std::size_t) {
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

    void DoClose(ConnectionCloseInfo close_info) {
        if (closing_) {
            return;
        }
        closing_ = true;
        NotifyClose(close_info);
        websocket::close_reason reason;
        reason.reason = close_info.detail;
        stream_.async_close(reason, [self = shared_from_this()](beast::error_code) {});
    }

    void NotifyClose(const ConnectionCloseInfo& close_info) {
        if (close_notified_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        lease_.Close(close_info);
        if (auto close_handler = server_.WebSocketCloseHandlerSnapshot()) {
            close_handler(close_info);
        }
    }

    void OnControl(websocket::frame_type type, beast::string_view payload) {
        if (type == websocket::frame_type::close) {
            NotifyClose(ConnectionCloseInfo::Remote(std::string(payload)));
        }
    }

    HttpServer& server_;
    websocket::stream<beast::tcp_stream> stream_;
    BeastHttpRequest request_;
    ConnectionLease lease_;
    WebSocketMessageHandler handler_;
    WebSocketOutboundQueue outbound_queue_;
    core::BucketMemoryPool memory_pool_;
    SharedBuffer read_buffer_;
    std::size_t current_message_bytes_ = 0;
    WebSocketMessageKind current_message_kind_ = WebSocketMessageKind::Binary;
    WebSocketFrame current_write_;
    bool writing_ = false;
    bool closing_ = false;
    std::atomic<bool> close_notified_{false};
};

bool HttpServer::HttpSession::TryUpgradeWebSocket() {
    const auto target = std::string_view(request_.target().data(), request_.target().size());
    auto handler = server_.WebSocketHandlerSnapshot(target);
    auto stream_handler = server_.WebSocketStreamHandlerSnapshot(target);
    if (!handler && !stream_handler) {
        return false;
    }

    auto status = lease_.SetKind(ProtocolConnectionKind::WebSocket);
    if (!status.ok()) {
        Send(MakeStatusResponse(request_, http::status::service_unavailable, status.message()));
        return true;
    }

    std::make_shared<WebSocketSession>(server_, stream_.release_socket(), std::move(request_), std::move(lease_), std::move(handler))->Run();
    return true;
}

HttpServer::HttpServer(HttpServerOptions options)
    : options_(std::move(options)),
      io_context_(static_cast<int>(ResolveIoThreads(options_.io_threads))),
      connection_pool_(options_.connection_pool) {
    if (options_.static_files) {
        static_file_handler_ = std::make_shared<StaticFileHandler>(*options_.static_files);
    }
}

HttpServer::~HttpServer() {
    Stop();
}

core::Status HttpServer::Start() {
    if (running_.exchange(true, std::memory_order_acq_rel)) {
        return core::Status::Ok();
    }

    try {
        auto address = asio::ip::make_address(options_.address);
        auto endpoint = tcp::endpoint(address, options_.port);
        work_guard_ = std::make_unique<asio::executor_work_guard<asio::io_context::executor_type>>(io_context_.get_executor());
        listener_ = std::make_shared<Listener>(*this, endpoint);
        bound_port_ = listener_->port();
        listener_->Run();

        const auto thread_count = ResolveIoThreads(options_.io_threads);
        io_threads_.reserve(thread_count);
        for (std::size_t i = 0; i < thread_count; ++i) {
            io_threads_.emplace_back([this](std::stop_token) {
                io_context_.run();
            });
        }
    } catch (const core::AppException& e) {
        running_.store(false, std::memory_order_release);
        return e.status();
    } catch (const std::exception& e) {
        running_.store(false, std::memory_order_release);
        return core::Status::Error(core::ErrorCode::InternalError, e.what());
    }

    return core::Status::Ok();
}

void HttpServer::Stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    if (listener_) {
        asio::post(io_context_, [listener = listener_] {
            listener->Stop();
        });
    }
    if (work_guard_) {
        work_guard_->reset();
    }
    io_context_.stop();
    for (auto& thread : io_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    io_threads_.clear();
    listener_.reset();
    connection_pool_.CloseAll(ConnectionCloseInfo::Shutdown("http server stopped"));
}

void HttpServer::SetHttpHandler(HttpGeneratorHandler handler) {
    std::lock_guard lock(handler_mutex_);
    http_handler_ = std::move(handler);
}

void HttpServer::SetHttpRequestHandler(IHttpRequestHandler handler) {
    std::lock_guard lock(handler_mutex_);
    http_request_handler_ = std::move(handler);
}

void HttpServer::SetAccessController(HttpAccessController controller) {
    std::lock_guard lock(handler_mutex_);
    access_controller_ = std::move(controller);
}

void HttpServer::SetStaticFiles(StaticFileOptions options) {
    std::lock_guard lock(handler_mutex_);
    options_.static_files = options;
    static_file_handler_ = std::make_shared<StaticFileHandler>(std::move(options));
}

void HttpServer::SetWebSocketHandler(std::string path, WebSocketMessageHandler handler) {
    std::lock_guard lock(handler_mutex_);
    websocket_path_ = std::move(path);
    websocket_handler_ = std::move(handler);
}

void HttpServer::SetWebSocketStreamHandler(std::string path, IWebSocketStreamHandler handler) {
    std::lock_guard lock(handler_mutex_);
    websocket_stream_path_ = std::move(path);
    websocket_stream_handler_ = std::move(handler);
}

void HttpServer::SetWebSocketAcceptHandler(WebSocketAcceptHandler handler) {
    std::lock_guard lock(handler_mutex_);
    websocket_accept_handler_ = std::move(handler);
}

void HttpServer::SetWebSocketCloseHandler(WebSocketCloseHandler handler) {
    std::lock_guard lock(handler_mutex_);
    websocket_close_handler_ = std::move(handler);
}

bool HttpServer::running() const noexcept {
    return running_.load(std::memory_order_acquire);
}

std::uint16_t HttpServer::port() const noexcept {
    return bound_port_;
}

ConnectionPoolStats HttpServer::ConnectionStats() const {
    return connection_pool_.Stats();
}

std::vector<ConnectionSnapshot> HttpServer::ConnectionSnapshots() const {
    return connection_pool_.Snapshots();
}

std::uint64_t HttpServer::NextConnectionId() noexcept {
    return next_connection_id_.fetch_add(1, std::memory_order_relaxed);
}

HttpGeneratorHandler HttpServer::HttpHandlerSnapshot() const {
    std::lock_guard lock(handler_mutex_);
    return http_handler_;
}

IHttpRequestHandler HttpServer::HttpRequestHandlerSnapshot() const {
    std::lock_guard lock(handler_mutex_);
    return http_request_handler_;
}

HttpAccessController HttpServer::AccessControllerSnapshot() const {
    std::lock_guard lock(handler_mutex_);
    return access_controller_;
}

WebSocketMessageHandler HttpServer::WebSocketHandlerSnapshot(std::string_view path) const {
    std::lock_guard lock(handler_mutex_);
    return path == websocket_path_ ? websocket_handler_ : WebSocketMessageHandler{};
}

IWebSocketStreamHandler HttpServer::WebSocketStreamHandlerSnapshot(std::string_view path) const {
    std::lock_guard lock(handler_mutex_);
    return path == websocket_stream_path_ ? websocket_stream_handler_ : IWebSocketStreamHandler{};
}

WebSocketAcceptHandler HttpServer::WebSocketAcceptHandlerSnapshot() const {
    std::lock_guard lock(handler_mutex_);
    return websocket_accept_handler_;
}

WebSocketCloseHandler HttpServer::WebSocketCloseHandlerSnapshot() const {
    std::lock_guard lock(handler_mutex_);
    return websocket_close_handler_;
}

std::shared_ptr<StaticFileHandler> HttpServer::StaticFileHandlerSnapshot() const {
    std::lock_guard lock(handler_mutex_);
    return static_file_handler_;
}

} // namespace net
