#include <boost/asio.hpp>
#include <boost/beast.hpp>
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
#include <vector>

namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace websocket = beast::websocket;
using tcp = asio::ip::tcp;

struct Options {
    std::string host = "127.0.0.1";
    std::string port = "18081";
    std::string target = "/ws-download";
    int clients = 1;
    std::size_t bytes_per_client = 1024ull * 1024 * 1024;
    std::size_t chunk_bytes = 256 * 1024;
    int slow_read_delay_ms = 0;
};

struct Stats {
    std::atomic<std::uint64_t> completed_clients{0};
    std::atomic<std::uint64_t> failed_clients{0};
    std::atomic<std::uint64_t> bytes_received{0};
    std::atomic<std::uint64_t> frames_received{0};
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
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(arg + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--host") {
            options.host = value();
        } else if (arg == "--port") {
            options.port = value();
        } else if (arg == "--target") {
            options.target = value();
        } else if (arg == "--clients") {
            options.clients = ParsePositiveInt(value(), arg);
        } else if (arg == "--bytes-per-client") {
            options.bytes_per_client = ParsePositiveSize(value(), arg);
        } else if (arg == "--chunk-bytes") {
            options.chunk_bytes = ParsePositiveSize(value(), arg);
        } else if (arg == "--slow-read-delay-ms") {
            const auto parsed = std::stoi(value());
            if (parsed < 0) {
                throw std::invalid_argument(arg + " must be non-negative");
            }
            options.slow_read_delay_ms = parsed;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ws_download_client [--host HOST] [--port PORT] "
                         "[--target PATH] [--clients N] [--bytes-per-client N] "
                         "[--chunk-bytes N] [--slow-read-delay-ms N]\n";
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

void RunClient(const Options& options, int client_id, Stats& stats) {
    try {
        asio::io_context io;
        tcp::resolver resolver(io);
        auto endpoints = resolver.resolve(options.host, options.port);

        websocket::stream<beast::tcp_stream> ws(asio::make_strand(io));
        ws.next_layer().connect(endpoints);
        ws.binary(true);
        ws.handshake(options.host, options.target);

        const auto request = "GET " + std::to_string(options.bytes_per_client) + " " + std::to_string(options.chunk_bytes);
        ws.text(true);
        ws.write(asio::buffer(request));

        std::uint64_t received = 0;
        std::uint64_t frames = 0;
        const auto delay = std::chrono::milliseconds(options.slow_read_delay_ms);
        while (received < options.bytes_per_client) {
            beast::flat_buffer buffer;
            beast::error_code ec;
            ws.read(buffer, ec);
            if (ec) {
                stats.failed_clients.fetch_add(1, std::memory_order_relaxed);
                RecordError(stats, "client " + std::to_string(client_id) + " " + FormatError("read", ec));
                return;
            }

            const auto n = buffer.size();
            received += n;
            ++frames;
            stats.bytes_received.fetch_add(n, std::memory_order_relaxed);
            stats.frames_received.fetch_add(1, std::memory_order_relaxed);
            if (delay.count() > 0 && received < options.bytes_per_client) {
                std::this_thread::sleep_for(delay);
            }
            if (received < options.bytes_per_client) {
                ws.text(true);
                ws.write(asio::buffer(std::string_view("NEXT")));
            }
        }

        if (received != options.bytes_per_client) {
            stats.failed_clients.fetch_add(1, std::memory_order_relaxed);
            RecordError(stats, "client " + std::to_string(client_id) + " received unexpected byte count");
            return;
        }

        beast::error_code ec;
        ws.close(websocket::close_code::normal, ec);
        stats.completed_clients.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception& e) {
        stats.failed_clients.fetch_add(1, std::memory_order_relaxed);
        RecordError(stats, "client " + std::to_string(client_id) + " exception: " + e.what());
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = ParseOptions(argc, argv);
        Stats stats;
        const auto start = std::chrono::steady_clock::now();

        std::vector<std::jthread> clients;
        clients.reserve(static_cast<std::size_t>(options.clients));
        for (int i = 0; i < options.clients; ++i) {
            clients.emplace_back([&, i](std::stop_token) {
                RunClient(options, i, stats);
            });
        }
        clients.clear();

        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        const auto elapsed_seconds = std::max(0.001, static_cast<double>(elapsed.count()) / 1000.0);
        const auto bytes = stats.bytes_received.load(std::memory_order_relaxed);

        std::cout << "WebSocket download client\n";
        std::cout << "  target             : ws://" << options.host << ":" << options.port << options.target << "\n";
        std::cout << "  clients            : " << options.clients << "\n";
        std::cout << "  bytes_per_client   : " << options.bytes_per_client << "\n";
        std::cout << "  chunk_bytes        : " << options.chunk_bytes << "\n";
        std::cout << "  slow_read_delay_ms : " << options.slow_read_delay_ms << "\n";
        std::cout << "  completed_clients  : " << stats.completed_clients.load(std::memory_order_relaxed) << "\n";
        std::cout << "  failed_clients     : " << stats.failed_clients.load(std::memory_order_relaxed) << "\n";
        std::cout << "  frames_received    : " << stats.frames_received.load(std::memory_order_relaxed) << "\n";
        std::cout << "  bytes_received     : " << bytes << "\n";
        std::cout << "  elapsed_ms         : " << elapsed.count() << "\n";
        std::cout << "  mib_s              : " << static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsed_seconds << "\n";

        {
            std::lock_guard lock(stats.errors_mutex);
            if (!stats.errors.empty()) {
                std::cout << "  sample_errors:\n";
                for (const auto& error : stats.errors) {
                    std::cout << "    " << error << "\n";
                }
            }
        }

        return stats.failed_clients.load(std::memory_order_relaxed) == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "ws_download_client failed: " << e.what() << "\n";
        return 2;
    }
}
