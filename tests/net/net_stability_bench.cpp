#include "http_server.h"

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
using tcp = asio::ip::tcp;
using namespace std::chrono_literals;

struct Options {
    std::string address = "127.0.0.1";
    std::size_t server_io_threads = 2;
    int http_clients = 8;
    int http_requests_per_client = 20;
    int ws_clients = 4;
    int ws_messages_per_client = 4;
    std::size_t http_body_bytes = 64 * 1024;
    std::size_t ws_message_bytes = 8 * 1024 * 1024;
    std::size_t chunk_bytes = 16 * 1024;
    int chunk_delay_ms = 1;
    int timeout_seconds = 60;
};

struct Stats {
    std::atomic<std::uint64_t> http_requests{0};
    std::atomic<std::uint64_t> http_ok{0};
    std::atomic<std::uint64_t> http_failed{0};
    std::atomic<std::uint64_t> http_bytes_sent{0};
    std::atomic<std::uint64_t> ws_messages{0};
    std::atomic<std::uint64_t> ws_ok{0};
    std::atomic<std::uint64_t> ws_failed{0};
    std::atomic<std::uint64_t> ws_bytes_sent{0};
    std::atomic<std::uint64_t> server_http_bytes{0};
    std::atomic<std::uint64_t> server_ws_bytes{0};
    std::atomic<std::uint64_t> server_ws_read_errors{0};
    std::mutex errors_mutex;
    std::vector<std::string> errors;
};

int ParsePositiveInt(const std::string& value, std::string_view name) {
    const auto parsed = std::stoi(value);
    if (parsed <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return parsed;
}

std::size_t ParsePositiveSize(const std::string& value, std::string_view name) {
    const auto parsed = std::stoull(value);
    if (parsed == 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<std::size_t>(parsed);
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](std::string_view name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--http-clients") {
            options.http_clients = ParsePositiveInt(require_value(arg), arg);
        } else if (arg == "--http-requests") {
            options.http_requests_per_client = ParsePositiveInt(require_value(arg), arg);
        } else if (arg == "--ws-clients") {
            options.ws_clients = ParsePositiveInt(require_value(arg), arg);
        } else if (arg == "--ws-messages") {
            options.ws_messages_per_client = ParsePositiveInt(require_value(arg), arg);
        } else if (arg == "--http-body-bytes") {
            options.http_body_bytes = ParsePositiveSize(require_value(arg), arg);
        } else if (arg == "--ws-message-bytes") {
            options.ws_message_bytes = ParsePositiveSize(require_value(arg), arg);
        } else if (arg == "--chunk-bytes") {
            options.chunk_bytes = ParsePositiveSize(require_value(arg), arg);
        } else if (arg == "--chunk-delay-ms") {
            options.chunk_delay_ms = ParsePositiveInt(require_value(arg), arg);
        } else if (arg == "--server-io-threads") {
            options.server_io_threads = ParsePositiveSize(require_value(arg), arg);
        } else if (arg == "--timeout-seconds") {
            options.timeout_seconds = ParsePositiveInt(require_value(arg), arg);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: net_stability_bench [options]\n"
                         "  --http-clients N\n"
                         "  --http-requests N\n"
                         "  --ws-clients N\n"
                         "  --ws-messages N\n"
                         "  --http-body-bytes N\n"
                         "  --ws-message-bytes N\n"
                         "  --chunk-bytes N\n"
                         "  --chunk-delay-ms N\n"
                         "  --server-io-threads N\n"
                         "  --timeout-seconds N\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return options;
}

std::string FormatError(std::string_view stage, beast::error_code ec) {
    std::ostringstream oss;
    oss << stage << ": [" << ec.category().name() << ":" << ec.value() << "] " << ec.message();
    return oss.str();
}

void RecordError(Stats& stats, std::string message) {
    std::lock_guard lock(stats.errors_mutex);
    if (stats.errors.size() < 16) {
        stats.errors.push_back(std::move(message));
    }
}

std::string Payload(std::size_t size, char seed) {
    std::string payload(size, seed);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<char>('a' + ((static_cast<unsigned char>(seed) + i) % 26));
    }
    return payload;
}

void WriteSlow(tcp::socket& socket,
               std::string_view data,
               std::size_t chunk_bytes,
               std::chrono::milliseconds delay) {
    for (std::size_t offset = 0; offset < data.size();) {
        const auto n = std::min(chunk_bytes, data.size() - offset);
        asio::write(socket, asio::buffer(data.data() + offset, n));
        offset += n;
        if (delay.count() > 0 && offset < data.size()) {
            std::this_thread::sleep_for(delay);
        }
    }
}

void RunSlowHttpClient(const Options& options, unsigned short port, int client_id, Stats& stats) {
    try {
        asio::io_context io;
        tcp::resolver resolver(io);
        const auto endpoints = resolver.resolve(options.address, std::to_string(port));
        const auto delay = std::chrono::milliseconds(options.chunk_delay_ms);
        const auto body = Payload(options.http_body_bytes, static_cast<char>('a' + client_id));
        beast::tcp_stream stream(io);

        for (int i = 0; i < options.http_requests_per_client; ++i) {
            stream.connect(endpoints);

            std::ostringstream header;
            header << "POST /slow-upload HTTP/1.1\r\n"
                   << "Host: " << options.address << "\r\n"
                   << "User-Agent: net_stability_bench\r\n"
                   << "Content-Type: application/octet-stream\r\n"
                   << "Content-Length: " << body.size() << "\r\n"
                   << "Connection: close\r\n\r\n";

            stats.http_requests.fetch_add(1, std::memory_order_relaxed);
            WriteSlow(stream.socket(), header.str(), options.chunk_bytes, 0ms);
            WriteSlow(stream.socket(), body, options.chunk_bytes, delay);
            stats.http_bytes_sent.fetch_add(body.size(), std::memory_order_relaxed);

            beast::flat_buffer buffer;
            http::response<http::string_body> response;
            stream.expires_after(std::chrono::seconds(options.timeout_seconds));
            beast::error_code ec;
            http::read(stream, buffer, response, ec);
            if (ec) {
                stats.http_failed.fetch_add(1, std::memory_order_relaxed);
                RecordError(stats, FormatError("http read", ec));
                continue;
            }
            if (response.result() == http::status::ok) {
                stats.http_ok.fetch_add(1, std::memory_order_relaxed);
            } else {
                stats.http_failed.fetch_add(1, std::memory_order_relaxed);
                RecordError(stats, "http non-ok: " + std::to_string(static_cast<unsigned>(response.result_int())));
            }

            beast::error_code ignored;
            stream.socket().shutdown(tcp::socket::shutdown_both, ignored);
            stream.socket().close(ignored);
        }
    } catch (const std::exception& e) {
        stats.http_failed.fetch_add(1, std::memory_order_relaxed);
        RecordError(stats, std::string("http client exception: ") + e.what());
    }
}

void RunLargeWebSocketClient(const Options& options, unsigned short port, int client_id, Stats& stats) {
    try {
        asio::io_context io;
        tcp::resolver resolver(io);
        const auto endpoints = resolver.resolve(options.address, std::to_string(port));
        const auto delay = std::chrono::milliseconds(options.chunk_delay_ms);
        const auto payload = Payload(options.ws_message_bytes, static_cast<char>('k' + client_id));
        websocket::stream<beast::tcp_stream> ws(asio::make_strand(io));

        ws.next_layer().connect(endpoints);
        ws.handshake(options.address, "/ws-large");
        ws.binary(true);

        for (int message_index = 0; message_index < options.ws_messages_per_client; ++message_index) {
            stats.ws_messages.fetch_add(1, std::memory_order_relaxed);
            for (std::size_t offset = 0; offset < payload.size();) {
                const auto n = std::min(options.chunk_bytes, payload.size() - offset);
                const bool final = offset + n >= payload.size();
                ws.write_some(final, asio::buffer(payload.data() + offset, n));
                offset += n;
                if (delay.count() > 0 && !final) {
                    std::this_thread::sleep_for(delay);
                }
            }
            stats.ws_bytes_sent.fetch_add(payload.size(), std::memory_order_relaxed);

            beast::flat_buffer buffer;
            beast::error_code ec;
            ws.read(buffer, ec);
            if (ec) {
                stats.ws_failed.fetch_add(1, std::memory_order_relaxed);
                RecordError(stats, FormatError("websocket read ack", ec));
                continue;
            }
            const auto ack = beast::buffers_to_string(buffer.data());
            if (ack == "ok:" + std::to_string(payload.size())) {
                stats.ws_ok.fetch_add(1, std::memory_order_relaxed);
            } else {
                stats.ws_failed.fetch_add(1, std::memory_order_relaxed);
                RecordError(stats, "unexpected websocket ack: " + ack);
            }
        }

        beast::error_code ec;
        ws.close(websocket::close_code::normal, ec);
    } catch (const std::exception& e) {
        stats.ws_failed.fetch_add(1, std::memory_order_relaxed);
        RecordError(stats, std::string("websocket client exception: ") + e.what());
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = ParseOptions(argc, argv);
        Stats stats;

        net::HttpServerOptions server_options;
        server_options.address = options.address;
        server_options.port = 0;
        server_options.io_threads = options.server_io_threads;
        server_options.request_timeout = std::chrono::seconds(options.timeout_seconds);
        server_options.request_body_limit = std::max<std::size_t>(options.http_body_bytes * 2, 1024 * 1024);
        server_options.websocket.max_frame_bytes = std::max<std::size_t>(options.chunk_bytes, 1024);
        server_options.websocket.max_message_bytes = std::max<std::size_t>(options.ws_message_bytes * 2, 1024 * 1024);
        server_options.websocket_read_buffer_limit = server_options.websocket.max_message_bytes;

        net::HttpServer server(server_options);
        server.SetHttpRequestHandler([&](std::shared_ptr<net::IHttpRequest> request) {
            stats.server_http_bytes.fetch_add(request->message().body().size(), std::memory_order_relaxed);
            request->Respond(net::HttpResponse::Text(net::http::status::ok,
                                                     std::to_string(request->message().body().size()))
                                 .message);
        });

        std::mutex ws_mutex;
        std::unordered_map<std::uint64_t, std::uint64_t> ws_connection_bytes;
        core::BucketMemoryPool ws_response_pool;
        server.SetWebSocketStreamHandler("/ws-large", [&](std::shared_ptr<net::IWebSocketStreamRequest> request) {
            auto& message = request->message();
            if (!message.ok()) {
                stats.server_ws_read_errors.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            std::uint64_t completed_bytes = 0;
            {
                std::lock_guard lock(ws_mutex);
                auto& current_bytes = ws_connection_bytes[request->connection().connection_id];
                current_bytes += message.total_bytes;
                if (message.final_fragment) {
                    completed_bytes = current_bytes;
                    ws_connection_bytes.erase(request->connection().connection_id);
                }
            }
            stats.server_ws_bytes.fetch_add(message.total_bytes, std::memory_order_relaxed);

            if (message.final_fragment) {
                const auto ack_text = "ok:" + std::to_string(completed_bytes);
                auto ack = net::SharedBuffer::Copy(ws_response_pool, ack_text);
                if (!ack.ok()) {
                    RecordError(stats, ack.status().message());
                    return;
                }
                auto status = request->Send(net::WebSocketFrame{
                    net::WebSocketMessageKind::Text,
                    true,
                    false,
                    std::move(ack).value()});
                if (!status.ok()) {
                    RecordError(stats, status.message());
                }
            }
        });

        auto start_status = server.Start();
        if (!start_status.ok()) {
            std::cerr << "failed to start server: " << start_status.message() << "\n";
            return 1;
        }

        const auto start = std::chrono::steady_clock::now();
        std::vector<std::jthread> clients;
        clients.reserve(static_cast<std::size_t>(options.http_clients + options.ws_clients));
        for (int i = 0; i < options.http_clients; ++i) {
            clients.emplace_back([&, i](std::stop_token) {
                RunSlowHttpClient(options, server.port(), i, stats);
            });
        }
        for (int i = 0; i < options.ws_clients; ++i) {
            clients.emplace_back([&, i](std::stop_token) {
                RunLargeWebSocketClient(options, server.port(), i, stats);
            });
        }
        clients.clear();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);

        server.Stop();

        const auto http_requests = stats.http_requests.load(std::memory_order_relaxed);
        const auto http_ok = stats.http_ok.load(std::memory_order_relaxed);
        const auto http_failed = stats.http_failed.load(std::memory_order_relaxed);
        const auto ws_messages = stats.ws_messages.load(std::memory_order_relaxed);
        const auto ws_ok = stats.ws_ok.load(std::memory_order_relaxed);
        const auto ws_failed = stats.ws_failed.load(std::memory_order_relaxed);
        const auto elapsed_seconds = std::max(0.001, static_cast<double>(elapsed.count()) / 1000.0);

        std::cout << "Net stability benchmark\n";
        std::cout << "  server_port       : " << server.port() << "\n";
        std::cout << "  server_io_threads : " << options.server_io_threads << "\n";
        std::cout << "  chunk_bytes       : " << options.chunk_bytes << "\n";
        std::cout << "  chunk_delay_ms    : " << options.chunk_delay_ms << "\n";
        std::cout << "  elapsed_ms        : " << elapsed.count() << "\n";
        std::cout << "  http_clients      : " << options.http_clients << "\n";
        std::cout << "  http_requests     : " << http_requests << "\n";
        std::cout << "  http_ok           : " << http_ok << "\n";
        std::cout << "  http_failed       : " << http_failed << "\n";
        std::cout << "  http_bytes_sent   : " << stats.http_bytes_sent.load(std::memory_order_relaxed) << "\n";
        std::cout << "  server_http_bytes : " << stats.server_http_bytes.load(std::memory_order_relaxed) << "\n";
        std::cout << "  ws_clients        : " << options.ws_clients << "\n";
        std::cout << "  ws_messages       : " << ws_messages << "\n";
        std::cout << "  ws_ok             : " << ws_ok << "\n";
        std::cout << "  ws_failed         : " << ws_failed << "\n";
        std::cout << "  ws_bytes_sent     : " << stats.ws_bytes_sent.load(std::memory_order_relaxed) << "\n";
        std::cout << "  server_ws_bytes   : " << stats.server_ws_bytes.load(std::memory_order_relaxed) << "\n";
        std::cout << "  server_ws_errors  : " << stats.server_ws_read_errors.load(std::memory_order_relaxed) << "\n";
        std::cout << "  aggregate_mib_s   : "
                  << static_cast<double>(stats.http_bytes_sent.load(std::memory_order_relaxed) +
                                         stats.ws_bytes_sent.load(std::memory_order_relaxed)) /
                         (1024.0 * 1024.0) / elapsed_seconds
                  << "\n";

        {
            std::lock_guard lock(stats.errors_mutex);
            if (!stats.errors.empty()) {
                std::cout << "  sample_errors:\n";
                for (const auto& error : stats.errors) {
                    std::cout << "    " << error << "\n";
                }
            }
        }

        return http_failed == 0 && ws_failed == 0 &&
                       stats.server_ws_read_errors.load(std::memory_order_relaxed) == 0
                   ? 0
                   : 1;
    } catch (const std::exception& e) {
        std::cerr << "net_stability_bench failed: " << e.what() << "\n";
        return 2;
    }
}
