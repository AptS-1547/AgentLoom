#include "../../src/llm/openai_llm_client.h"
#include "../../src/llm/local_llm_client.h"
#include "../../src/net/http_client/beast_http_client.h"
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <gtest/gtest.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <memory>
#include <string>
#include <thread>

using agent::llm::ChatCompletionRequest;
using agent::llm::ChatCompletionResponse;
using agent::llm::ChatRole;
using agent::llm::FallbackLlmClient;
using agent::llm::FallbackLlmClientOptions;
using agent::llm::ILlmClient;
using agent::llm::ILocalLlm;
using agent::llm::LlmPromptStore;
using agent::llm::LocalLlmChatClient;
using agent::llm::LocalLlmRequest;
using agent::llm::LocalLlmResponse;
using agent::llm::OpenAiLlmClient;
using agent::llm::OpenAiLlmClientOptions;
using agent::net::BeastHttpClient;
using agent::net::BeastHttpClientOptions;

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = boost::beast::http;
using tcp = boost::asio::ip::tcp;

namespace {

struct MockBehavior {
    int status = 200;
    std::string body;
    std::chrono::milliseconds delay{0};
    std::atomic<int> call_count{0};
    std::string last_auth_header;
    std::string last_body;
};

class MockLlmServer {
public:
    static std::unique_ptr<MockLlmServer> Start(std::shared_ptr<MockBehavior> behavior) {
        auto self = std::unique_ptr<MockLlmServer>(new MockLlmServer(std::move(behavior)));
        self->Run();
        return self;
    }

    ~MockLlmServer() {
        acceptor_.close();
        ioc_.stop();
        if (thread_.joinable()) thread_.join();
    }

    std::uint16_t port() const { return port_; }

private:
    explicit MockLlmServer(std::shared_ptr<MockBehavior> behavior)
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
            if (ec) return;
            HandleSession(std::move(sock));
            DoAccept();
        });
    }

    void HandleSession(tcp::socket sock) {
        std::thread([this, sock = std::move(sock)]() mutable {
            beast::error_code ec;
            beast::flat_buffer buf;
            http::request<http::string_body> req;
            http::read(sock, buf, req, ec);
            if (ec) return;

            behavior_->call_count.fetch_add(1);
            behavior_->last_body = req.body();
            if (req.count(http::field::authorization)) {
                behavior_->last_auth_header = std::string(req[http::field::authorization]);
            }

            if (behavior_->delay.count() > 0) {
                std::this_thread::sleep_for(behavior_->delay);
            }

            http::response<http::string_body> res(
                static_cast<http::status>(behavior_->status), req.version());
            res.set(http::field::content_type, "application/json");
            res.keep_alive(false);
            res.body() = behavior_->body;
            res.prepare_payload();
            http::write(sock, res, ec);
            sock.shutdown(tcp::socket::shutdown_both, ec);
        }).detach();
    }

    std::shared_ptr<MockBehavior> behavior_;
    asio::io_context ioc_;
    tcp::acceptor acceptor_;
    std::thread thread_;
    std::uint16_t port_ = 0;
};

const std::string kValidResponse = R"({
    "id": "cmpl-123",
    "model": "deepseek-chat",
    "choices": [{
        "message": {"role": "assistant", "content": "42"},
        "finish_reason": "stop",
        "index": 0
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
})";

const std::string kServerErrorResponse = R"({"error":"internal server error"})";

std::string MakeBaseUrl(std::uint16_t port) {
    return "http://127.0.0.1:" + std::to_string(port) + "/v1";
}

}  // namespace

class ScriptedLlmClient final : public ILlmClient {
public:
    explicit ScriptedLlmClient(std::string name, core::Status status = core::Status::Ok())
        : name_(std::move(name)),
          status_(std::move(status)) {}

    core::Result<ChatCompletionResponse> Complete(const ChatCompletionRequest&) override {
        ++call_count;
        if (!status_.ok()) {
            return status_;
        }
        ChatCompletionResponse response;
        response.model = name_;
        response.content = name_ + "-response";
        return response;
    }

    int call_count = 0;

private:
    std::string name_;
    core::Status status_;
};

class FakeLocalLlm final : public ILocalLlm {
public:
    core::Result<LocalLlmResponse> Generate(const LocalLlmRequest& request) override {
        ++call_count;
        last_request = request;
        LocalLlmResponse response;
        response.text = "local grpc reply";
        response.prompt_tokens = 12;
        response.generated_tokens = 5;
        response.result_source = "grpc_local";
        return response;
    }

    int call_count = 0;
    LocalLlmRequest last_request;
};

class OpenAiLlmClientTest : public ::testing::Test {
protected:
    void SetUp() override {
        behavior_ = std::make_shared<MockBehavior>();
        behavior_->body = kValidResponse;
        server_ = MockLlmServer::Start(behavior_);

        BeastHttpClientOptions http_opts;
        auto r = BeastHttpClient::Create(http_opts);
        ASSERT_TRUE(r.ok());
        http_client_ = std::move(r).value();

        OpenAiLlmClientOptions opts;
        opts.base_url = MakeBaseUrl(server_->port());
        opts.api_key = "test-key";
        opts.default_model = "deepseek-chat";
        opts.timeout_ms = 5000;
        opts.retry_policy.max_retries = 0;

        auto cr = OpenAiLlmClient::Create(std::move(opts), *http_client_);
        ASSERT_TRUE(cr.ok());
        client_ = std::move(cr).value();
    }

    ChatCompletionRequest SimpleRequest() {
        ChatCompletionRequest req;
        req.messages.push_back({ChatRole::User, "hello"});
        return req;
    }

    std::shared_ptr<MockBehavior> behavior_;
    std::unique_ptr<MockLlmServer> server_;
    std::unique_ptr<BeastHttpClient> http_client_;
    std::unique_ptr<OpenAiLlmClient> client_;
};

// ── unit: Create validation ──────────────────────────────────────────────────

TEST(OpenAiLlmClientCreateTest, RejectsEmptyBaseUrl) {
    BeastHttpClientOptions http_opts;
    auto http = BeastHttpClient::Create(http_opts).value();

    OpenAiLlmClientOptions opts;
    opts.base_url = "";
    opts.api_key = "key";
    auto r = OpenAiLlmClient::Create(opts, *http);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), core::ErrorCode::InvalidArgument);
}

TEST(OpenAiLlmClientCreateTest, RejectsEmptyApiKey) {
    BeastHttpClientOptions http_opts;
    auto http = BeastHttpClient::Create(http_opts).value();

    OpenAiLlmClientOptions opts;
    opts.base_url = "http://127.0.0.1:8080/v1";
    opts.api_key = "";
    auto r = OpenAiLlmClient::Create(opts, *http);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), core::ErrorCode::InvalidArgument);
}

// ── unit: successful completion ──────────────────────────────────────────────

TEST_F(OpenAiLlmClientTest, BasicCompletion) {
    auto r = client_->Complete(SimpleRequest());
    ASSERT_TRUE(r.ok()) << r.status().message();

    auto& resp = r.value();
    EXPECT_EQ(resp.content, "42");
    EXPECT_EQ(resp.id, "cmpl-123");
    EXPECT_EQ(resp.model, "deepseek-chat");
    EXPECT_EQ(resp.prompt_tokens, 10);
    EXPECT_EQ(resp.completion_tokens, 5);
    EXPECT_EQ(resp.total_tokens, 15);
}

TEST_F(OpenAiLlmClientTest, AuthHeaderForwarded) {
    client_->Complete(SimpleRequest());
    EXPECT_EQ(behavior_->last_auth_header, "Bearer test-key");
}

TEST_F(OpenAiLlmClientTest, RequestBodyContainsModel) {
    auto req = SimpleRequest();
    req.model = "custom-model";
    client_->Complete(req);
    EXPECT_NE(behavior_->last_body.find("custom-model"), std::string::npos);
}

TEST_F(OpenAiLlmClientTest, DefaultModelUsedWhenEmpty) {
    client_->Complete(SimpleRequest());
    EXPECT_NE(behavior_->last_body.find("deepseek-chat"), std::string::npos);
}

TEST_F(OpenAiLlmClientTest, SystemMessageIncluded) {
    ChatCompletionRequest req;
    req.messages.push_back({ChatRole::System, "you are a tutor"});
    req.messages.push_back({ChatRole::User, "hello"});
    client_->Complete(req);
    EXPECT_NE(behavior_->last_body.find("system"), std::string::npos);
    EXPECT_NE(behavior_->last_body.find("you are a tutor"), std::string::npos);
}

// ── unit: HTTP-level error responses ────────────────────────────────────────

TEST_F(OpenAiLlmClientTest, FourXxReturnsError) {
    behavior_->status = 401;
    behavior_->body = R"({"error":"unauthorized"})";
    auto r = client_->Complete(SimpleRequest());
    EXPECT_FALSE(r.ok());
    EXPECT_NE(r.status().message().find("401"), std::string::npos);
}

TEST_F(OpenAiLlmClientTest, FiveXxReturnsError) {
    behavior_->status = 503;
    behavior_->body = kServerErrorResponse;
    auto r = client_->Complete(SimpleRequest());
    EXPECT_FALSE(r.ok());
}

TEST_F(OpenAiLlmClientTest, MalformedJsonReturnsError) {
    behavior_->body = "not-json{{{";
    auto r = client_->Complete(SimpleRequest());
    EXPECT_FALSE(r.ok());
}

TEST_F(OpenAiLlmClientTest, EmptyChoicesReturnsError) {
    behavior_->body = R"({"id":"x","model":"m","choices":[],"usage":{}})";
    auto r = client_->Complete(SimpleRequest());
    EXPECT_FALSE(r.ok());
}

TEST_F(OpenAiLlmClientTest, MissingContentFieldReturnsError) {
    behavior_->body = R"({"choices":[{"message":{"role":"assistant"},"index":0}]})";
    auto r = client_->Complete(SimpleRequest());
    EXPECT_FALSE(r.ok());
}

// ── unit: retry behavior ─────────────────────────────────────────────────────

TEST(OpenAiLlmClientRetryTest, FiveXxIsRetriedUntilSuccess) {
    auto behavior = std::make_shared<MockBehavior>();
    behavior->status = 503;
    behavior->body = kServerErrorResponse;
    auto server = MockLlmServer::Start(behavior);

    BeastHttpClientOptions http_opts;
    auto http = BeastHttpClient::Create(http_opts).value();

    OpenAiLlmClientOptions opts;
    opts.base_url = MakeBaseUrl(server->port());
    opts.api_key = "key";
    opts.timeout_ms = 5000;
    opts.retry_policy.max_retries = 2;
    opts.retry_policy.initial_delay = std::chrono::milliseconds(10);

    auto client = OpenAiLlmClient::Create(opts, *http).value();
    client->Complete({});
    // 1 initial + 2 retries = 3 calls
    EXPECT_EQ(behavior->call_count.load(), 3);
}

TEST(OpenAiLlmClientRetryTest, FourXxIsNotRetried) {
    auto behavior = std::make_shared<MockBehavior>();
    behavior->status = 400;
    behavior->body = R"({"error":"bad request"})";
    auto server = MockLlmServer::Start(behavior);

    BeastHttpClientOptions http_opts;
    auto http = BeastHttpClient::Create(http_opts).value();

    OpenAiLlmClientOptions opts;
    opts.base_url = MakeBaseUrl(server->port());
    opts.api_key = "key";
    opts.timeout_ms = 5000;
    opts.retry_policy.max_retries = 3;
    opts.retry_policy.initial_delay = std::chrono::milliseconds(10);

    auto client = OpenAiLlmClient::Create(opts, *http).value();
    client->Complete({});
    EXPECT_EQ(behavior->call_count.load(), 1);
}

// ── unit: LlmPromptStore ─────────────────────────────────────────────────────

TEST(FallbackLlmClientTest, MissingPrimaryFallsBackToLocalClient) {
    auto local = std::make_shared<ScriptedLlmClient>("local");
    FallbackLlmClient client(nullptr, local);

    auto result = client.Complete({});
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_EQ(result.value().content, "local-response");
    EXPECT_EQ(local->call_count, 1);
}

TEST(FallbackLlmClientTest, AuthFailureFallsBackToLocalClient) {
    auto cloud = std::make_shared<ScriptedLlmClient>(
        "cloud",
        core::Status::Error(core::ErrorCode::PermissionDenied, "missing or invalid api key"));
    auto local = std::make_shared<ScriptedLlmClient>("local");
    FallbackLlmClient client(cloud, local);

    auto result = client.Complete({});
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_EQ(result.value().content, "local-response");
    EXPECT_EQ(cloud->call_count, 1);
    EXPECT_EQ(local->call_count, 1);
}

TEST(FallbackLlmClientTest, ConsecutiveCloudFailuresOpenCircuitUntilReconnectProbe) {
    auto cloud = std::make_shared<ScriptedLlmClient>(
        "cloud",
        core::Status::Error(core::ErrorCode::Unavailable, "cloud unavailable"));
    auto local = std::make_shared<ScriptedLlmClient>("local");

    FallbackLlmClientOptions options;
    options.failure_threshold = 2;
    options.primary_reconnect_interval = std::chrono::milliseconds(1000);
    FallbackLlmClient client(cloud, local, options);

    auto first = client.Complete({});
    auto second = client.Complete({});
    auto third = client.Complete({});
    ASSERT_TRUE(first.ok()) << first.status().message();
    ASSERT_TRUE(second.ok()) << second.status().message();
    ASSERT_TRUE(third.ok()) << third.status().message();

    EXPECT_EQ(cloud->call_count, 2);
    EXPECT_EQ(local->call_count, 3);
}

TEST(OpenAiLlmClientCreateTest, AllowsLocalOpenAiCompatibleEndpointWithoutApiKey) {
    BeastHttpClientOptions http_opts;
    auto http = BeastHttpClient::Create(http_opts).value();

    OpenAiLlmClientOptions opts;
    opts.base_url = "http://127.0.0.1:8080/v1";
    opts.api_key = "";
    opts.require_api_key = false;
    auto r = OpenAiLlmClient::Create(opts, *http);
    EXPECT_TRUE(r.ok()) << r.status().message();
}

TEST(LocalLlmChatClientTest, AdaptsChatCompletionToLocalGrpcContract) {
    auto local = std::make_shared<FakeLocalLlm>();
    LocalLlmChatClient client(local, {.default_model = "local-model", .task_type = "persona_chat"});

    ChatCompletionRequest req;
    req.messages.push_back({ChatRole::System, "you are a tutor"});
    req.messages.push_back({ChatRole::User, "hello"});
    req.max_tokens = 128;
    req.temperature = 0.5f;
    req.top_p = 0.8f;

    auto result = client.Complete(req);
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_EQ(result.value().content, "local grpc reply");
    EXPECT_EQ(result.value().model, "local-model");
    EXPECT_EQ(result.value().prompt_tokens, 12);
    EXPECT_EQ(result.value().completion_tokens, 5);
    EXPECT_EQ(result.value().total_tokens, 17);

    EXPECT_EQ(local->call_count, 1);
    EXPECT_EQ(local->last_request.task_type, "persona_chat");
    EXPECT_EQ(local->last_request.max_tokens, 128);
    EXPECT_FLOAT_EQ(local->last_request.temperature, 0.5f);
    EXPECT_FLOAT_EQ(local->last_request.top_p, 0.8f);
    EXPECT_NE(local->last_request.prompt.find("<system>"), std::string::npos);
    EXPECT_NE(local->last_request.prompt.find("you are a tutor"), std::string::npos);
    EXPECT_NE(local->last_request.prompt.find("<user>"), std::string::npos);
    EXPECT_NE(local->last_request.prompt.find("hello"), std::string::npos);
}

TEST(LlmPromptStoreTest, LoadAndGet) {
    // Write a temp file
    auto tmp = std::filesystem::temp_directory_path() / "test_prompt.txt";
    { std::ofstream f(tmp); f << "extract memory from: {conversation}"; }

    LlmPromptStore store;
    std::unordered_map<std::string, std::filesystem::path> paths;
    paths["memory_extraction"] = tmp.filename();
    auto st = store.Load(paths, tmp.parent_path());
    ASSERT_TRUE(st.ok()) << st.message();

    auto r = store.Get("memory_extraction");
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value(), "extract memory from: {conversation}");

    std::filesystem::remove(tmp);
}

TEST(LlmPromptStoreTest, MissingFileReturnsNotFound) {
    LlmPromptStore store;
    std::unordered_map<std::string, std::filesystem::path> paths;
    paths["p"] = "no_such_file_xyz.txt";
    auto st = store.Load(paths, std::filesystem::temp_directory_path());
    EXPECT_FALSE(st.ok());
    EXPECT_EQ(st.code(), core::ErrorCode::NotFound);
}

TEST(LlmPromptStoreTest, GetUnknownNameReturnsNotFound) {
    LlmPromptStore store;
    auto r = store.Get("nonexistent");
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), core::ErrorCode::NotFound);
}

TEST(LlmPromptStoreTest, HasReturnsTrueAfterLoad) {
    auto tmp = std::filesystem::temp_directory_path() / "test_has_prompt.txt";
    { std::ofstream f(tmp); f << "hello"; }

    LlmPromptStore store;
    std::unordered_map<std::string, std::filesystem::path> paths;
    paths["greet"] = tmp.filename();
    store.Load(paths, tmp.parent_path());

    EXPECT_TRUE(store.Has("greet"));
    EXPECT_FALSE(store.Has("unknown"));

    std::filesystem::remove(tmp);
}

TEST(LlmPromptStoreTest, ClearRemovesAll) {
    auto tmp = std::filesystem::temp_directory_path() / "test_clear_prompt.txt";
    { std::ofstream f(tmp); f << "x"; }
     
    LlmPromptStore store;
    std::unordered_map<std::string, std::filesystem::path> paths;
    paths["p"] = tmp.filename();
    store.Load(paths, tmp.parent_path());
    EXPECT_TRUE(store.Has("p"));

    store.Clear();
    EXPECT_FALSE(store.Has("p"));

    std::filesystem::remove(tmp);
}
