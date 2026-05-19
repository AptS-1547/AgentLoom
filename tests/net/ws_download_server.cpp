#include "http_server.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

namespace {

using namespace std::chrono_literals;

std::atomic_bool g_stop_requested{false};

void OnSignal(int) {
    g_stop_requested.store(true, std::memory_order_release);
}

struct Options {
    std::string address = "127.0.0.1";
    unsigned short port = 18081;
    std::size_t io_threads = 4;
    std::size_t chunk_bytes = 256 * 1024;
    std::size_t max_download_bytes = 2ull * 1024 * 1024 * 1024;
    std::size_t outbound_max_items = 1024;
    std::size_t outbound_max_bytes = 512ull * 1024 * 1024;
};

struct Stats {
    std::atomic<std::uint64_t> accepted{0};
    std::atomic<std::uint64_t> closed{0};
    std::atomic<std::uint64_t> requests{0};
    std::atomic<std::uint64_t> frames_sent{0};
    std::atomic<std::uint64_t> bytes_sent{0};
    std::atomic<std::uint64_t> send_errors{0};
};

unsigned short ParsePort(const std::string& value) {
    const auto number = std::stoul(value);
    if (number == 0 || number > 65535) {
        throw std::out_of_range("port must be in range 1..65535");
    }
    return static_cast<unsigned short>(number);
}

std::size_t ParseSize(const std::string& value, std::string_view name) {
    const auto number = std::stoull(value);
    if (number == 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<std::size_t>(number);
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

        if (arg == "--address") {
            options.address = value();
        } else if (arg == "--port") {
            options.port = ParsePort(value());
        } else if (arg == "--io-threads") {
            options.io_threads = ParseSize(value(), arg);
        } else if (arg == "--chunk-bytes") {
            options.chunk_bytes = ParseSize(value(), arg);
        } else if (arg == "--max-download-bytes") {
            options.max_download_bytes = ParseSize(value(), arg);
        } else if (arg == "--outbound-max-items") {
            options.outbound_max_items = ParseSize(value(), arg);
        } else if (arg == "--outbound-max-bytes") {
            options.outbound_max_bytes = ParseSize(value(), arg);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ws_download_server [--address ADDR] [--port PORT] "
                         "[--io-threads N] [--chunk-bytes N] [--max-download-bytes N] "
                         "[--outbound-max-items N] [--outbound-max-bytes N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return options;
}

std::string_view FirstFragmentView(const net::WebSocketMessage& message) {
    if (message.fragments.empty()) {
        return {};
    }
    return message.fragments.front().view();
}

bool ParseGetRequest(std::string_view command, std::size_t& bytes, std::size_t& chunk_bytes) {
    std::istringstream input{std::string(command)};
    std::string verb;
    input >> verb >> bytes >> chunk_bytes;
    return verb == "GET" && bytes > 0 && chunk_bytes > 0;
}

bool IsNextRequest(std::string_view command) {
    return command == "NEXT";
}

std::string MakeChunk(std::size_t size, std::uint64_t sequence) {
    std::string chunk(size, '\0');
    for (std::size_t i = 0; i < chunk.size(); ++i) {
        chunk[i] = static_cast<char>((sequence + i) & 0xff);
    }
    return chunk;
}

struct DownloadState {
    std::size_t remaining = 0;
    std::size_t chunk_bytes = 0;
    std::uint64_t sequence = 0;
};

} // namespace

int main(int argc, char** argv) {
    try {
        std::signal(SIGINT, OnSignal);
        std::signal(SIGTERM, OnSignal);

        const auto options = ParseOptions(argc, argv);
        Stats stats;

        net::HttpServerOptions server_options;
        server_options.address = options.address;
        server_options.port = options.port;
        server_options.io_threads = options.io_threads;
        server_options.websocket.max_frame_bytes = std::max<std::size_t>(options.chunk_bytes, 1024);
        server_options.websocket.max_message_bytes = 1024 * 1024;
        server_options.websocket.outbound_max_items = options.outbound_max_items;
        server_options.websocket.outbound_max_bytes = options.outbound_max_bytes;

        net::HttpServer server(server_options);
        server.SetWebSocketAcceptHandler([&](net::WebSocketSessionHandle&) {
            stats.accepted.fetch_add(1, std::memory_order_relaxed);
        });
        server.SetWebSocketCloseHandler([&](const net::ConnectionCloseInfo&) {
            stats.closed.fetch_add(1, std::memory_order_relaxed);
        });
        std::mutex state_mutex;
        std::unordered_map<std::uint64_t, DownloadState> downloads;

        auto send_next_chunk = [&](net::IWebSocketStreamRequest& request) {
            DownloadState state;
            {
                std::lock_guard lock(state_mutex);
                auto it = downloads.find(request.connection().connection_id);
                if (it == downloads.end() || it->second.remaining == 0) {
                    return;
                }
                state = it->second;
            }

            const auto n = std::min(state.chunk_bytes, state.remaining);
            auto chunk = MakeChunk(n, state.sequence);
            core::BucketMemoryPool command_pool;
            auto payload = net::SharedBuffer::Copy(command_pool, chunk);
            if (!payload.ok()) {
                stats.send_errors.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            auto status = request.Send(net::WebSocketFrame{
                net::WebSocketMessageKind::Binary,
                true,
                false,
                std::move(payload).value()});
            if (!status.ok()) {
                stats.send_errors.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            {
                std::lock_guard lock(state_mutex);
                auto it = downloads.find(request.connection().connection_id);
                if (it != downloads.end()) {
                    it->second.remaining -= n;
                    ++it->second.sequence;
                    if (it->second.remaining == 0) {
                        downloads.erase(it);
                    }
                }
            }
            stats.frames_sent.fetch_add(1, std::memory_order_relaxed);
            stats.bytes_sent.fetch_add(n, std::memory_order_relaxed);
        };

        server.SetWebSocketCloseHandler([&](const net::ConnectionCloseInfo&) {
            stats.closed.fetch_add(1, std::memory_order_relaxed);
        });
        server.SetWebSocketStreamHandler("/ws-download", [&](std::shared_ptr<net::IWebSocketStreamRequest> request) {
            auto& message = request->message();
            if (!message.ok() || !message.final_fragment) {
                return;
            }

            const auto command = FirstFragmentView(message);
            std::size_t bytes = 0;
            std::size_t requested_chunk = 0;
            if (ParseGetRequest(command, bytes, requested_chunk)) {
                bytes = std::min(bytes, options.max_download_bytes);
                const auto chunk_bytes = std::min<std::size_t>(std::max<std::size_t>(requested_chunk, 1), options.chunk_bytes);
                {
                    std::lock_guard lock(state_mutex);
                    downloads[request->connection().connection_id] = DownloadState{bytes, chunk_bytes, 0};
                }
                stats.requests.fetch_add(1, std::memory_order_relaxed);
                send_next_chunk(*request);
                return;
            }

            if (IsNextRequest(command)) {
                send_next_chunk(*request);
            }
        });

        auto status = server.Start();
        if (!status.ok()) {
            std::cerr << "failed to start server: " << status.message() << "\n";
            return 1;
        }

        std::cout << "WebSocket download server running\n";
        std::cout << "  url: ws://" << options.address << ":" << server.port() << "/ws-download\n";
        std::cout << "  io_threads: " << options.io_threads << "\n";
        std::cout << "  chunk_bytes: " << options.chunk_bytes << "\n";
        std::cout << "  outbound_max_bytes: " << options.outbound_max_bytes << "\n";
        std::cout << "Press Ctrl+C to stop.\n";

        auto last_time = std::chrono::steady_clock::now();
        std::uint64_t last_bytes = 0;
        while (!g_stop_requested.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(1s);
            const auto now = std::chrono::steady_clock::now();
            const auto bytes = stats.bytes_sent.load(std::memory_order_relaxed);
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
            const auto delta = bytes - last_bytes;
            const auto mib_s = elapsed > 0 ? static_cast<double>(delta) / (1024.0 * 1024.0) / (static_cast<double>(elapsed) / 1000.0) : 0.0;
            last_time = now;
            last_bytes = bytes;
            std::cout << "stats accepted=" << stats.accepted.load(std::memory_order_relaxed)
                      << " closed=" << stats.closed.load(std::memory_order_relaxed)
                      << " requests=" << stats.requests.load(std::memory_order_relaxed)
                      << " frames=" << stats.frames_sent.load(std::memory_order_relaxed)
                      << " bytes=" << bytes
                      << " send_errors=" << stats.send_errors.load(std::memory_order_relaxed)
                      << " mib_s=" << mib_s << "\n";
        }

        server.Stop();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ws_download_server failed: " << e.what() << "\n";
        return 2;
    }
}
