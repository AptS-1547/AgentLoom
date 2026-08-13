// 用于出站 HTTP 并发压测的 OpenAI-compatible 配额 Mock。
// 固定 Provider 延迟由 Asio 定时器驱动，不阻塞 HttpServer IO 线程。

#include "http_server.h"

#include <boost/asio.hpp>
#include <boost/beast/http.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace asio = boost::asio;
namespace http = boost::beast::http;

std::atomic<bool> g_stop_requested{false};

void OnSignal(int) {
    g_stop_requested.store(true, std::memory_order_release);
}

struct MockSnapshot {
    std::size_t delay_ms = 0;
    std::size_t max_inflight = 0;
    std::size_t inflight = 0;
    std::size_t peak_inflight = 0;
    std::size_t queued = 0;
    std::size_t peak_queued = 0;
    std::size_t completed = 0;
};

class QuotaMockState final : public std::enable_shared_from_this<QuotaMockState> {
public:
    QuotaMockState(std::chrono::milliseconds delay, std::size_t max_inflight)
        : delay_(delay),
          max_inflight_(std::max<std::size_t>(1, max_inflight)),
          work_guard_(asio::make_work_guard(io_context_)) {}

    ~QuotaMockState() {
        Shutdown();
    }

    void Start() {
        worker_ = std::jthread([this] { io_context_.run(); });
    }

    void Shutdown() noexcept {
        if (stopped_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        work_guard_.reset();
        io_context_.stop();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void Submit(std::shared_ptr<::net::IHttpRequest> request) {
        asio::post(io_context_, [self = shared_from_this(), request = std::move(request)]() mutable {
            if (self->stopped_.load(std::memory_order_acquire)) {
                self->RespondUnavailable(std::move(request));
                return;
            }
            self->pending_.push_back(std::move(request));
            self->queued_.store(self->pending_.size(), std::memory_order_relaxed);
            self->peak_queued_.store(
                std::max(self->peak_queued_.load(std::memory_order_relaxed),
                         self->pending_.size()),
                std::memory_order_relaxed);
            self->StartReady();
        });
    }

    MockSnapshot Snapshot() const {
        return {
            .delay_ms = static_cast<std::size_t>(delay_.count()),
            .max_inflight = max_inflight_,
            .inflight = inflight_.load(std::memory_order_relaxed),
            .peak_inflight = peak_inflight_.load(std::memory_order_relaxed),
            .queued = queued_.load(std::memory_order_relaxed),
            .peak_queued = peak_queued_.load(std::memory_order_relaxed),
            .completed = completed_.load(std::memory_order_relaxed),
        };
    }

private:
    struct PendingOperation final {
        PendingOperation(asio::io_context& context, std::shared_ptr<::net::IHttpRequest> request)
            : timer(context), request(std::move(request)) {}

        asio::steady_timer timer;
        std::shared_ptr<::net::IHttpRequest> request;
    };

    void StartReady() {
        while (!pending_.empty() &&
               inflight_.load(std::memory_order_relaxed) < max_inflight_) {
            auto operation = std::make_shared<PendingOperation>(
                io_context_, std::move(pending_.front()));
            pending_.pop_front();
            queued_.store(pending_.size(), std::memory_order_relaxed);
            const auto inflight = inflight_.fetch_add(1, std::memory_order_relaxed) + 1;
            peak_inflight_.store(
                std::max(peak_inflight_.load(std::memory_order_relaxed), inflight),
                std::memory_order_relaxed);
            operation->timer.expires_after(delay_);
            operation->timer.async_wait(
                [self = shared_from_this(), operation](const boost::system::error_code& error) {
                    self->Complete(operation, error);
                });
        }
    }

    void Complete(const std::shared_ptr<PendingOperation>& operation,
                  const boost::system::error_code& error) {
        if (inflight_.load(std::memory_order_relaxed) > 0) {
            inflight_.fetch_sub(1, std::memory_order_relaxed);
        }
        if (!error) {
            completed_.fetch_add(1, std::memory_order_relaxed);
            http::response<http::string_body> response{http::status::ok, 11};
            response.set(http::field::content_type, "application/json; charset=utf-8");
            response.keep_alive(operation->request->message().keep_alive());
            response.body() =
                R"({"id":"cpp-quota-mock","object":"chat.completion","created":0,"model":"mock-enterprise-chat","choices":[{"index":0,"message":{"role":"assistant","content":"C++ quota mock response."},"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":8,"total_tokens":24}})";
            response.prepare_payload();
            static_cast<void>(operation->request->Respond(http::message_generator(std::move(response))));
        } else {
            RespondUnavailable(operation->request);
        }
        StartReady();
    }

    static void RespondUnavailable(std::shared_ptr<::net::IHttpRequest> request) {
        http::response<http::string_body> response{http::status::service_unavailable, 11};
        response.set(http::field::content_type, "application/json; charset=utf-8");
        response.keep_alive(request->message().keep_alive());
        response.body() = R"({"error":"quota mock is stopping"})";
        response.prepare_payload();
        static_cast<void>(request->Respond(http::message_generator(std::move(response))));
    }

    const std::chrono::milliseconds delay_;
    const std::size_t max_inflight_;
    asio::io_context io_context_;
    asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
    std::jthread worker_;
    std::atomic<bool> stopped_{false};
    std::deque<std::shared_ptr<::net::IHttpRequest>> pending_;
    std::atomic<std::size_t> queued_{0};
    std::atomic<std::size_t> inflight_{0};
    std::atomic<std::size_t> peak_inflight_{0};
    std::atomic<std::size_t> peak_queued_{0};
    std::atomic<std::size_t> completed_{0};
};

std::string MetricsJson(const MockSnapshot& snapshot,
                        const ::net::ConnectionPoolStats& connections) {
    return "{\"ok\":true,\"implementation\":\"agentloom_cpp_http_server\",\"delayMs\":" +
           std::to_string(snapshot.delay_ms) +
           ",\"maxInflight\":" + std::to_string(snapshot.max_inflight) +
           ",\"inflight\":" + std::to_string(snapshot.inflight) +
           ",\"peakInflight\":" + std::to_string(snapshot.peak_inflight) +
           ",\"queued\":" + std::to_string(snapshot.queued) +
           ",\"peakQueued\":" + std::to_string(snapshot.peak_queued) +
           ",\"completed\":" + std::to_string(snapshot.completed) +
           ",\"acceptedConnections\":" + std::to_string(connections.accepted_connections) +
           ",\"activeConnections\":" + std::to_string(connections.active_connections) +
           ",\"closedConnections\":" + std::to_string(connections.closed_connections) +
           ",\"rejected\":0}";
}

void RespondJson(std::shared_ptr<::net::IHttpRequest> request,
                 http::status status,
                 std::string body) {
    http::response<http::string_body> response{status, 11};
    response.set(http::field::content_type, "application/json; charset=utf-8");
    response.keep_alive(request->message().keep_alive());
    response.body() = std::move(body);
    response.prepare_payload();
    static_cast<void>(request->Respond(http::message_generator(std::move(response))));
}

}

int main(int argc, char** argv) {
    const auto port = argc > 1 ? std::max(1, std::atoi(argv[1])) : 18081;
    const auto delay_ms = argc > 2 ? std::max(0, std::atoi(argv[2])) : 2000;
    const auto max_inflight = argc > 3 ? std::max(1, std::atoi(argv[3])) : 500;
    const auto io_threads = argc > 4 ? std::max(1, std::atoi(argv[4])) : 8;

    std::signal(SIGINT, OnSignal);
    std::signal(SIGTERM, OnSignal);
    auto state = std::make_shared<QuotaMockState>(
        std::chrono::milliseconds(delay_ms), static_cast<std::size_t>(max_inflight));
    state->Start();

    ::net::HttpServer server({
        .address = "127.0.0.1",
        .port = static_cast<unsigned short>(port),
        .io_threads = static_cast<std::size_t>(io_threads),
        .request_timeout = std::chrono::seconds(120),
        .connection_pool = {.max_connections = 2048, .max_http_connections = 2048},
    });
    server.SetHttpRequestHandler([state, &server](std::shared_ptr<::net::IHttpRequest> request) {
        const auto& message = request->message();
        const std::string_view target(message.target().data(), message.target().size());
        if (message.method() == http::verb::get && (target == "/health" || target == "/metrics")) {
            RespondJson(
                std::move(request), http::status::ok,
                MetricsJson(state->Snapshot(), server.ConnectionStats()));
            return;
        }
        if (message.method() == http::verb::post && target == "/v1/chat/completions") {
            state->Submit(std::move(request));
            return;
        }
        RespondJson(std::move(request), http::status::not_found, R"({"error":"not found"})");
    });

    const auto start = server.Start();
    if (!start.ok()) {
        std::cerr << "failed to start C++ quota mock: " << start.message() << '\n';
        return 1;
    }
    std::cout << "[cpp-quota-mock] listening on http://127.0.0.1:" << server.port()
              << " delay_ms=" << delay_ms << " max_inflight=" << max_inflight
              << " http_io_threads=" << io_threads << '\n';
    while (!g_stop_requested.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    server.Stop();
    state->Shutdown();
    return 0;
}
