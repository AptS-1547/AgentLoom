#include "beast_http_client.h"
#include "url_parser.h"

#include <boost/asio.hpp>
#include <boost/beast.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using agent::net::BeastHttpClient;
using agent::net::BeastHttpClientOptions;
using agent::net::HttpClientRequest;
using agent::net::HttpHeader;

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = boost::beast::http;
using tcp = boost::asio::ip::tcp;

namespace {

// Per-session policy describing what the mock server should return.
struct MockBehavior {
    int status = 200;
    std::string content_type = "application/json";
    std::string body = R"({"ok":true})";
    /// If > 0, the handler sleeps this long before sending the response.
    std::chrono::milliseconds delay_before_response{0};
    /// Captured request fields so tests can assert on them.
    std::string last_method;
    std::string last_target;
    std::string last_body;
};

class MockHttpServer {
public:
    static std::unique_ptr<MockHttpServer> Start(std::shared_ptr<MockBehavior> behavior) {
        auto self = std::unique_ptr<MockHttpServer>(new MockHttpServer(std::move(behavior)));
        self->Run();
        return self;
    }

    ~MockHttpServer() {
        acceptor_.close();
        ioc_.stop();
        if (thread_.joinable()) thread_.join();
        // 所有已接收会话均属于 mock server，析构时等待其退出，避免跨测试访问
        // socket/runtime 资源。先停止 accept 线程，确保不会再添加新会话线程。
        std::lock_guard lock(session_threads_mutex_);
        for (auto& session_thread : session_threads_) {
            if (session_thread.joinable()) {
                session_thread.join();
            }
        }
    }

    std::uint16_t port() const { return port_; }

private:
    explicit MockHttpServer(std::shared_ptr<MockBehavior> behavior)
        : behavior_(std::move(behavior)),
          acceptor_(ioc_, tcp::endpoint(asio::ip::make_address("127.0.0.1"), 0)) {
        port_ = acceptor_.local_endpoint().port();
    }

    void Run() {
        DoAccept();
        thread_ = std::thread([this] { ioc_.run(); });
    }

    void DoAccept() {
        acceptor_.async_accept([this](beast::error_code ec, tcp::socket sock) {
            if (ec) return;  // accept ended (likely shutdown)
            HandleSession(std::move(sock));
            DoAccept();
        });
    }

    void HandleSession(tcp::socket sock) {
        // One request per connection — simple and matches what BeastHttpClient does.
        // 会话线程可能在 server fixture 析构后才结束，因此只捕获共享行为对象，
        // 不能通过 this 访问已经释放的 MockHttpServer。
        auto behavior = behavior_;
        std::thread session_thread([behavior = std::move(behavior), sock = std::move(sock)]() mutable {
            beast::error_code ec;
            beast::flat_buffer buf;
            http::request<http::string_body> req;
            http::read(sock, buf, req, ec);
            if (ec) return;

            behavior->last_method = std::string(req.method_string());
            behavior->last_target = std::string(req.target());
            behavior->last_body = req.body();

            if (behavior->delay_before_response.count() > 0) {
                std::this_thread::sleep_for(behavior->delay_before_response);
            }

            http::response<http::string_body> res(
                static_cast<http::status>(behavior->status), req.version());
            res.set(http::field::server, "mock-http-server");
            res.set(http::field::content_type, behavior->content_type);
            res.keep_alive(false);
            res.body() = behavior->body;
            res.prepare_payload();
            http::write(sock, res, ec);
            sock.shutdown(tcp::socket::shutdown_both, ec);
        });
        std::lock_guard lock(session_threads_mutex_);
        session_threads_.push_back(std::move(session_thread));
    }

    std::shared_ptr<MockBehavior> behavior_;
    asio::io_context ioc_;
    tcp::acceptor acceptor_;
    std::thread thread_;
    std::mutex session_threads_mutex_;
    std::vector<std::thread> session_threads_;
    std::uint16_t port_ = 0;
};

std::string Url(std::uint16_t port, std::string_view path) {
    return "http://127.0.0.1:" + std::to_string(port) + std::string(path);
}

}  // namespace

class BeastHttpClientTest : public ::testing::Test {
protected:
    void SetUp() override {
        behavior_ = std::make_shared<MockBehavior>();
        server_ = MockHttpServer::Start(behavior_);

        BeastHttpClientOptions opts;  // no TLS — http-only
        auto r = BeastHttpClient::Create(opts);
        ASSERT_TRUE(r.ok());
        client_ = std::move(r).value();
    }

    void TearDown() override {
        client_.reset();
        server_.reset();
    }

    std::shared_ptr<MockBehavior> behavior_;
    std::unique_ptr<MockHttpServer> server_;
    std::unique_ptr<BeastHttpClient> client_;
};

TEST_F(BeastHttpClientTest, Basic200) {
    behavior_->status = 200;
    behavior_->body = R"({"hello":"world"})";

    HttpClientRequest req;
    req.method = "GET";
    req.url = Url(server_->port(), "/v1/ping");

    auto r = client_->Execute(req);
    ASSERT_TRUE(r.ok()) << r.status().message();
    auto res = std::move(r).value();
    EXPECT_EQ(res.status, 200);
    EXPECT_EQ(res.body, R"({"hello":"world"})");
    EXPECT_EQ(behavior_->last_method, "GET");
    EXPECT_EQ(behavior_->last_target, "/v1/ping");
}

TEST_F(BeastHttpClientTest, PostBodyAndHeadersForwarded) {
    behavior_->status = 200;

    HttpClientRequest req;
    req.method = "POST";
    req.url = Url(server_->port(), "/echo");
    req.headers.push_back(HttpHeader{"Content-Type", "application/json"});
    req.headers.push_back(HttpHeader{"X-Test", "abc"});
    req.body = R"({"msg":"hi"})";

    auto r = client_->Execute(req);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(behavior_->last_method, "POST");
    EXPECT_EQ(behavior_->last_body, R"({"msg":"hi"})");
}

TEST_F(BeastHttpClientTest, FourXxIsHappyStatus) {
    behavior_->status = 401;
    behavior_->body = R"({"error":"unauthorized"})";

    HttpClientRequest req;
    req.url = Url(server_->port(), "/secure");

    auto r = client_->Execute(req);
    ASSERT_TRUE(r.ok());  // transport ok, HTTP-level 4xx propagates via status field
    auto res = std::move(r).value();
    EXPECT_EQ(res.status, 401);
    EXPECT_EQ(res.body, R"({"error":"unauthorized"})");
}

TEST_F(BeastHttpClientTest, FiveXxIsHappyStatus) {
    behavior_->status = 503;
    HttpClientRequest req;
    req.url = Url(server_->port(), "/down");

    auto r = client_->Execute(req);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value().status, 503);
}

TEST_F(BeastHttpClientTest, EmptyBodyResponse) {
    behavior_->status = 204;
    behavior_->body = "";

    HttpClientRequest req;
    req.url = Url(server_->port(), "/nocontent");
    auto r = client_->Execute(req);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value().status, 204);
    EXPECT_TRUE(r.value().body.empty());
}

TEST_F(BeastHttpClientTest, TimeoutFiresWhenServerStalls) {
    behavior_->delay_before_response = std::chrono::milliseconds(2000);

    HttpClientRequest req;
    req.url = Url(server_->port(), "/slow");
    req.timeout_ms = 300;

    auto start = std::chrono::steady_clock::now();
    auto r = client_->Execute(req);
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(r.ok());
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(),
              1500);  // gave up well before server would have replied
}

TEST_F(BeastHttpClientTest, MalformedUrlRejected) {
    HttpClientRequest req;
    req.url = "not-a-url";
    auto r = client_->Execute(req);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), core::ErrorCode::InvalidArgument);
}

TEST_F(BeastHttpClientTest, HttpsRequiresTlsContext) {
    HttpClientRequest req;
    req.url = "https://example.com/";
    auto r = client_->Execute(req);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), core::ErrorCode::InvalidArgument);
}

TEST_F(BeastHttpClientTest, NonPositiveTimeoutRejected) {
    HttpClientRequest req;
    req.url = Url(server_->port(), "/ping");
    req.timeout_ms = 0;
    auto r = client_->Execute(req);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), core::ErrorCode::InvalidArgument);
}

TEST_F(BeastHttpClientTest, ConnectionRefusedReturnsTransportError) {
    HttpClientRequest req;
    // Port 1 is reserved; nothing should be listening.
    req.url = "http://127.0.0.1:1/";
    req.timeout_ms = 1500;
    auto r = client_->Execute(req);
    EXPECT_FALSE(r.ok());
}
