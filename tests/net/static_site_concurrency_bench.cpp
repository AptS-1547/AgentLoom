#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/http.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;

struct Options {
    std::string host = "127.0.0.1";
    std::string port = "18080";
    int concurrency = 32;
    int requests_per_session = 200;
    int io_threads = 0;
    int connect_batch = 0;
    int connect_delay_ms = 0;
    bool keep_alive = true;
    std::vector<std::string> paths = {
        "/",
        "/assets/app.js",
        "/assets/dynamic-view.js?v=20260516",
        "/assets/data.json?cache_bust=20260516",
        "/assets/site.css",
        "/classroom/session/42",
    };
};

struct Stats {
    std::atomic<std::uint64_t> total_requests{0};
    std::atomic<std::uint64_t> ok_responses{0};
    std::atomic<std::uint64_t> failed_requests{0};
    std::atomic<std::uint64_t> non_ok_responses{0};
    std::atomic<std::uint64_t> exceptions{0};
    std::atomic<std::uint64_t> bytes_received{0};
    std::atomic<std::uint64_t> completed_sessions{0};
    std::mutex mutex;
    std::condition_variable done_cv;
    std::vector<std::uint64_t> latencies_us;
    std::unordered_map<std::string, std::uint64_t> exception_messages;
};

int ParsePositiveInt(const std::string& value, std::string_view name) {
    const auto parsed = std::stoi(value);
    if (parsed <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return parsed;
}

int ParseNonNegativeInt(const std::string& value, std::string_view name) {
    const auto parsed = std::stoi(value);
    if (parsed < 0) {
        throw std::invalid_argument(std::string(name) + " must be non-negative");
    }
    return parsed;
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--host" || arg == "-h") && i + 1 < argc) {
            options.host = argv[++i];
        } else if (arg.rfind("--host=", 0) == 0) {
            options.host = arg.substr(7);
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            options.port = argv[++i];
        } else if (arg.rfind("--port=", 0) == 0) {
            options.port = arg.substr(7);
        } else if ((arg == "--concurrency" || arg == "-c") && i + 1 < argc) {
            options.concurrency = ParsePositiveInt(argv[++i], "concurrency");
        } else if (arg.rfind("--concurrency=", 0) == 0) {
            options.concurrency = ParsePositiveInt(arg.substr(14), "concurrency");
        } else if ((arg == "--requests" || arg == "-n") && i + 1 < argc) {
            options.requests_per_session = ParsePositiveInt(argv[++i], "requests");
        } else if (arg.rfind("--requests=", 0) == 0) {
            options.requests_per_session = ParsePositiveInt(arg.substr(11), "requests");
        } else if ((arg == "--io-threads" || arg == "-t") && i + 1 < argc) {
            options.io_threads = ParsePositiveInt(argv[++i], "io-threads");
        } else if (arg.rfind("--io-threads=", 0) == 0) {
            options.io_threads = ParsePositiveInt(arg.substr(13), "io-threads");
        } else if (arg == "--connect-batch" && i + 1 < argc) {
            options.connect_batch = ParsePositiveInt(argv[++i], "connect-batch");
        } else if (arg.rfind("--connect-batch=", 0) == 0) {
            options.connect_batch = ParsePositiveInt(arg.substr(16), "connect-batch");
        } else if (arg == "--connect-delay-ms" && i + 1 < argc) {
            options.connect_delay_ms = ParseNonNegativeInt(argv[++i], "connect-delay-ms");
        } else if (arg.rfind("--connect-delay-ms=", 0) == 0) {
            options.connect_delay_ms = ParseNonNegativeInt(arg.substr(19), "connect-delay-ms");
        } else if (arg == "--no-keep-alive") {
            options.keep_alive = false;
        } else if (arg == "--help") {
            std::cout << "Usage: static_site_concurrency_bench [--host HOST] [--port PORT] "
                         "[--concurrency N] [--requests N] [--io-threads N] "
                         "[--connect-batch N] [--connect-delay-ms N] [--no-keep-alive]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return options;
}

int ResolveIoThreads(int requested) {
    if (requested > 0) {
        return requested;
    }
    const auto hardware = std::thread::hardware_concurrency();
    return static_cast<int>(std::max(1u, hardware == 0 ? 1u : hardware));
}

std::uint64_t Percentile(std::vector<std::uint64_t> values, double p) {
    if (values.empty()) {
        return 0;
    }
    const auto index = static_cast<std::size_t>(
        std::clamp(p, 0.0, 1.0) * static_cast<double>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

void RecordFailure(Stats& stats, std::string message) {
    stats.failed_requests.fetch_add(1, std::memory_order_relaxed);
    stats.exceptions.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(stats.mutex);
    ++stats.exception_messages[std::move(message)];
}

std::string FormatError(std::string_view stage, beast::error_code ec) {
    std::ostringstream oss;
    oss << stage << ": [" << ec.category().name() << ":" << ec.value() << "] " << ec.message();
    return oss.str();
}

class Session : public std::enable_shared_from_this<Session> {
public:
    Session(asio::io_context& io,
            const tcp::resolver::results_type& endpoints,
            const Options& options,
            Stats& stats,
            int session_id)
        : stream_(asio::make_strand(io)),
          endpoints_(endpoints),
          options_(options),
          stats_(stats),
          session_id_(session_id) {
        latencies_us_.reserve(static_cast<std::size_t>(options_.requests_per_session));
    }

    void Start() {
        Connect();
    }

private:
    void Connect() {
        stream_.expires_after(std::chrono::seconds(10));
        stream_.async_connect(endpoints_, beast::bind_front_handler(&Session::OnConnect, shared_from_this()));
    }

    void OnConnect(beast::error_code ec, const tcp::endpoint&) {
        if (ec) {
            RecordFailure(stats_, FormatError("connect", ec));
            Complete();
            return;
        }
        SendNext();
    }

    void SendNext() {
        if (sent_requests_ >= options_.requests_per_session) {
            Close();
            return;
        }

        const auto& path = options_.paths[static_cast<std::size_t>(session_id_ + sent_requests_) % options_.paths.size()];
        request_ = {};
        request_.version(11);
        request_.method(http::verb::get);
        request_.target(path);
        request_.set(http::field::host, options_.host);
        request_.set(http::field::user_agent, "static_site_concurrency_bench_async");
        request_.keep_alive(options_.keep_alive);
        buffer_.consume(buffer_.size());
        response_ = {};
        request_start_ = std::chrono::steady_clock::now();
        stats_.total_requests.fetch_add(1, std::memory_order_relaxed);

        stream_.expires_after(std::chrono::seconds(10));
        http::async_write(stream_, request_, beast::bind_front_handler(&Session::OnWrite, shared_from_this()));
    }

    void OnWrite(beast::error_code ec, std::size_t) {
        if (ec) {
            RecordFailure(stats_, FormatError("write", ec));
            RecoverOrComplete();
            return;
        }
        http::async_read(stream_, buffer_, response_, beast::bind_front_handler(&Session::OnRead, shared_from_this()));
    }

    void OnRead(beast::error_code ec, std::size_t) {
        if (ec) {
            RecordFailure(stats_, FormatError("read", ec));
            RecoverOrComplete();
            return;
        }

        const auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - request_start_).count();
        latencies_us_.push_back(static_cast<std::uint64_t>(std::max<std::int64_t>(0, latency)));
        stats_.bytes_received.fetch_add(response_.body().size(), std::memory_order_relaxed);

        if (response_.result() == http::status::ok) {
            stats_.ok_responses.fetch_add(1, std::memory_order_relaxed);
        } else {
            stats_.non_ok_responses.fetch_add(1, std::memory_order_relaxed);
            stats_.failed_requests.fetch_add(1, std::memory_order_relaxed);
        }

        ++sent_requests_;
        if (options_.keep_alive) {
            SendNext();
            return;
        }

        beast::error_code ignored;
        stream_.socket().shutdown(tcp::socket::shutdown_both, ignored);
        stream_.socket().close(ignored);
        if (sent_requests_ < options_.requests_per_session) {
            Connect();
        } else {
            Complete();
        }
    }

    void RecoverOrComplete() {
        beast::error_code ignored;
        stream_.socket().close(ignored);
        ++sent_requests_;
        if (sent_requests_ < options_.requests_per_session) {
            Connect();
        } else {
            Complete();
        }
    }

    void Close() {
        beast::error_code ignored;
        stream_.socket().shutdown(tcp::socket::shutdown_both, ignored);
        stream_.socket().close(ignored);
        Complete();
    }

    void Complete() {
        const auto completed = stats_.completed_sessions.fetch_add(1, std::memory_order_acq_rel) + 1;
        const auto done = completed >= static_cast<std::uint64_t>(options_.concurrency);
        {
            std::lock_guard lock(stats_.mutex);
            stats_.latencies_us.insert(stats_.latencies_us.end(), latencies_us_.begin(), latencies_us_.end());
            if (!done) {
                return;
            }
            stats_.done_cv.notify_all();
        }
    }

    beast::tcp_stream stream_;
    tcp::resolver::results_type endpoints_;
    const Options& options_;
    Stats& stats_;
    int session_id_ = 0;
    int sent_requests_ = 0;
    beast::flat_buffer buffer_;
    http::request<http::empty_body> request_;
    http::response<http::string_body> response_;
    std::chrono::steady_clock::time_point request_start_;
    std::vector<std::uint64_t> latencies_us_;
};

class LaunchState : public std::enable_shared_from_this<LaunchState> {
public:
    LaunchState(asio::io_context& io,
                const tcp::resolver::results_type& endpoints,
                const Options& options,
                Stats& stats)
        : io_(io),
          timer_(io),
          endpoints_(endpoints),
          options_(options),
          stats_(stats) {}

    void Start() {
        LaunchBatch();
    }

private:
    void LaunchBatch() {
        const auto remaining = options_.concurrency - next_session_;
        if (remaining <= 0) {
            return;
        }

        const auto configured_batch = options_.connect_batch > 0 ? options_.connect_batch : options_.concurrency;
        const auto batch = std::min(configured_batch, remaining);
        for (int i = 0; i < batch; ++i) {
            const auto session_id = next_session_++;
            std::make_shared<Session>(io_, endpoints_, options_, stats_, session_id)->Start();
        }

        if (next_session_ >= options_.concurrency) {
            return;
        }

        auto self = shared_from_this();
        if (options_.connect_delay_ms == 0) {
            asio::post(io_, [self] {
                self->LaunchBatch();
            });
            return;
        }

        timer_.expires_after(std::chrono::milliseconds(options_.connect_delay_ms));
        timer_.async_wait([self](beast::error_code ec) {
            if (!ec) {
                self->LaunchBatch();
            }
        });
    }

    asio::io_context& io_;
    asio::steady_timer timer_;
    const tcp::resolver::results_type& endpoints_;
    const Options& options_;
    Stats& stats_;
    int next_session_ = 0;
};

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = ParseOptions(argc, argv);
        const auto io_thread_count = ResolveIoThreads(options.io_threads);
        asio::io_context io;
        tcp::resolver resolver(io);
        const auto endpoints = resolver.resolve(options.host, options.port);
        Stats stats;
        stats.latencies_us.reserve(static_cast<std::size_t>(options.concurrency * options.requests_per_session));

        auto work_guard = asio::make_work_guard(io);
        std::vector<std::jthread> io_threads;
        io_threads.reserve(static_cast<std::size_t>(io_thread_count));
        for (int i = 0; i < io_thread_count; ++i) {
            io_threads.emplace_back([&io](std::stop_token) {
                io.run();
            });
        }

        const auto start = std::chrono::steady_clock::now();
        auto launcher = std::make_shared<LaunchState>(io, endpoints, options, stats);
        asio::post(io, [launcher] {
            launcher->Start();
        });

        {
            std::unique_lock lock(stats.mutex);
            stats.done_cv.wait(lock, [&] {
                return stats.completed_sessions.load(std::memory_order_acquire) >=
                       static_cast<std::uint64_t>(options.concurrency);
            });
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);

        work_guard.reset();
        io.stop();
        io_threads.clear();

        std::vector<std::uint64_t> latencies;
        std::unordered_map<std::string, std::uint64_t> exception_messages;
        {
            std::lock_guard lock(stats.mutex);
            latencies = stats.latencies_us;
            exception_messages = stats.exception_messages;
        }

        const auto total_requests = stats.total_requests.load(std::memory_order_relaxed);
        const auto ok_responses = stats.ok_responses.load(std::memory_order_relaxed);
        const auto failed_requests = stats.failed_requests.load(std::memory_order_relaxed);
        const auto non_ok_responses = stats.non_ok_responses.load(std::memory_order_relaxed);
        const auto exceptions = stats.exceptions.load(std::memory_order_relaxed);
        const auto bytes_received = stats.bytes_received.load(std::memory_order_relaxed);
        const auto latency_sum = std::accumulate(latencies.begin(), latencies.end(), std::uint64_t{0});
        const auto latency_max = latencies.empty() ? 0 : *std::max_element(latencies.begin(), latencies.end());
        const auto p50 = Percentile(latencies, 0.50);
        const auto p95 = Percentile(latencies, 0.95);
        const auto p99 = Percentile(latencies, 0.99);
        const auto avg_us = latencies.empty() ? 0 : latency_sum / latencies.size();
        const auto elapsed_seconds = std::max(0.001, static_cast<double>(elapsed.count()) / 1000.0);
        const auto rps = static_cast<double>(total_requests) / elapsed_seconds;

        std::cout << "Static site concurrency benchmark\n";
        std::cout << "  target      : http://" << options.host << ":" << options.port << "\n";
        std::cout << "  concurrency : " << options.concurrency << "\n";
        std::cout << "  requests/session: " << options.requests_per_session << "\n";
        std::cout << "  io_threads  : " << io_thread_count << "\n";
        std::cout << "  connect_batch: " << (options.connect_batch > 0 ? options.connect_batch : options.concurrency) << "\n";
        std::cout << "  connect_delay_ms: " << options.connect_delay_ms << "\n";
        std::cout << "  keep_alive  : " << (options.keep_alive ? "true" : "false") << "\n";
        std::cout << "  requests    : " << total_requests << "\n";
        std::cout << "  ok          : " << ok_responses << "\n";
        std::cout << "  failed      : " << failed_requests << "\n";
        std::cout << "  non_ok      : " << non_ok_responses << "\n";
        std::cout << "  exceptions  : " << exceptions << "\n";
        std::cout << "  bytes       : " << bytes_received << "\n";
        std::cout << "  elapsed_ms  : " << elapsed.count() << "\n";
        std::cout << "  rps         : " << rps << "\n";
        std::cout << "  avg_ms      : " << static_cast<double>(avg_us) / 1000.0 << "\n";
        std::cout << "  p50_ms      : " << static_cast<double>(p50) / 1000.0 << "\n";
        std::cout << "  p95_ms      : " << static_cast<double>(p95) / 1000.0 << "\n";
        std::cout << "  p99_ms      : " << static_cast<double>(p99) / 1000.0 << "\n";
        std::cout << "  max_ms      : " << static_cast<double>(latency_max) / 1000.0 << "\n";

        if (!exception_messages.empty()) {
            std::vector<std::pair<std::string, std::uint64_t>> exception_list(
                exception_messages.begin(), exception_messages.end());
            std::sort(exception_list.begin(), exception_list.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.second > rhs.second;
            });
            std::cout << "  exception_types:\n";
            for (std::size_t i = 0; i < exception_list.size() && i < 5; ++i) {
                std::cout << "    " << exception_list[i].second << " x " << exception_list[i].first << "\n";
            }
        }

        return failed_requests == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << "\n";
        return 2;
    }
}
