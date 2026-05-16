#pragma once

#include "http_types.h"
#include "protocol_types.h"
#include "static_file_handler.h"
#include "websocket_types.h"

#include <boost/asio.hpp>
#include <boost/beast.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace net {

namespace asio = boost::asio;
using tcp = asio::ip::tcp;

struct HttpServerOptions {
    std::string address = "0.0.0.0";
    unsigned short port = 8080;
    std::size_t io_threads = 1;
    std::chrono::seconds request_timeout{30};
    std::chrono::seconds websocket_idle_timeout{60};
    std::size_t request_body_limit = 16 * 1024 * 1024;
    std::size_t websocket_read_buffer_limit = 16 * 1024 * 1024;
    WebSocketOptions websocket;
    std::optional<StaticFileOptions> static_files;
};

using HttpGeneratorCallback = std::function<void(http::message_generator)>;
using HttpGeneratorHandler = std::function<void(HttpRequest, HttpGeneratorCallback)>;
using WebSocketAcceptHandler = std::function<void(WebSocketSessionHandle&)>;

class HttpServer {
public:
    explicit HttpServer(HttpServerOptions options = {});
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    core::Status Start();
    void Stop();

    void SetHttpHandler(HttpGeneratorHandler handler);
    void SetAccessController(HttpAccessController controller);
    void SetStaticFiles(StaticFileOptions options);
    void SetWebSocketHandler(std::string path, WebSocketMessageHandler handler);
    void SetWebSocketAcceptHandler(WebSocketAcceptHandler handler);
    void SetWebSocketCloseHandler(WebSocketCloseHandler handler);

    bool running() const noexcept;
    std::uint16_t port() const noexcept;

private:
    class Listener;
    class HttpSession;
    class WebSocketSession;

    friend class Listener;
    friend class HttpSession;
    friend class WebSocketSession;

    std::uint64_t NextConnectionId() noexcept;
    HttpGeneratorHandler HttpHandlerSnapshot() const;
    HttpAccessController AccessControllerSnapshot() const;
    WebSocketMessageHandler WebSocketHandlerSnapshot(std::string_view path) const;
    WebSocketAcceptHandler WebSocketAcceptHandlerSnapshot() const;
    WebSocketCloseHandler WebSocketCloseHandlerSnapshot() const;
    std::shared_ptr<StaticFileHandler> StaticFileHandlerSnapshot() const;

    HttpServerOptions options_;
    asio::io_context io_context_;
    std::unique_ptr<asio::executor_work_guard<asio::io_context::executor_type>> work_guard_;
    std::shared_ptr<Listener> listener_;
    std::vector<std::jthread> io_threads_;
    mutable std::mutex handler_mutex_;
    HttpGeneratorHandler http_handler_;
    HttpAccessController access_controller_;
    std::string websocket_path_ = "/ws";
    WebSocketMessageHandler websocket_handler_;
    WebSocketAcceptHandler websocket_accept_handler_;
    WebSocketCloseHandler websocket_close_handler_;
    std::shared_ptr<StaticFileHandler> static_file_handler_;
    std::atomic<bool> running_{false};
    std::atomic<std::uint64_t> next_connection_id_{1};
    std::uint16_t bound_port_ = 0;
};

} // namespace net
