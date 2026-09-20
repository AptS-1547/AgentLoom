#include "../../src/llm/openai_llm_client.h"
#if defined(AGENTLOOM_TEST_LOCAL_LLM)
#include "../../src/llm/local_llm_client.h"
#endif
#include "../../src/net/http_client/beast_http_client.h"
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <gtest/gtest.h>
#include <atomic>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <thread>

using agent::llm::ChatCompletionRequest;
using agent::llm::ChatCompletionResponse;
using agent::llm::ChatContentPart;
using agent::llm::ChatRole;
using agent::llm::FallbackLlmClient;
using agent::llm::FallbackLlmClientOptions;
using agent::llm::ILlmClient;
#if defined(AGENTLOOM_TEST_LOCAL_LLM)
using agent::llm::ILocalLlm;
using agent::llm::LlmPromptStore;
using agent::llm::LocalLlmChatClient;
using agent::llm::LocalLlmRequest;
using agent::llm::LocalLlmResponse;
#else
using agent::llm::LlmPromptStore;
#endif
using agent::llm::OpenAiLlmClient;
using agent::llm::OpenAiLlmClientOptions;
using agent::llm::OpenAiAsyncLlmClient;
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

class ScriptedAsyncHttpOperation final : public agent::net::IAsyncHttpOperation {
public:
    explicit ScriptedAsyncHttpOperation(agent::net::IAsyncHttpClient::Callback callback = {})
        : callback_(std::move(callback)) {}

    void Cancel() noexcept override {
        if (completed_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        if (callback_) {
            callback_(core::Status::Error(core::ErrorCode::Cancelled,
                                          "scripted HTTP operation cancelled"));
        }
    }

private:
    agent::net::IAsyncHttpClient::Callback callback_;
    std::atomic<bool> completed_{false};
};

class ScriptedAsyncHttpClient final : public agent::net::IAsyncHttpClient {
public:
    core::Result<std::shared_ptr<agent::net::IAsyncHttpOperation>> ExecuteAsync(
        agent::net::HttpClientRequest request,
        Callback callback) override {
        ++call_count;
        last_request = std::move(request);
        if (hold_response) {
            auto operation = std::make_shared<ScriptedAsyncHttpOperation>(std::move(callback));
            last_operation = operation;
            return std::static_pointer_cast<agent::net::IAsyncHttpOperation>(operation);
        }
        if (responses.empty()) {
            return core::Status::Error(core::ErrorCode::InternalError,
                                       "scripted HTTP response is missing");
        }
        auto response = std::move(responses.front());
        responses.pop_front();
        callback(std::move(response));
        return std::static_pointer_cast<agent::net::IAsyncHttpOperation>(
            std::make_shared<ScriptedAsyncHttpOperation>());
    }

    std::deque<core::Result<agent::net::HttpClientResponse>> responses;
    agent::net::HttpClientRequest last_request;
    std::shared_ptr<ScriptedAsyncHttpOperation> last_operation;
    int call_count = 0;
    bool hold_response = false;
};

agent::net::HttpClientResponse MakeHttpResponse(int status, std::string body) {
    agent::net::HttpClientResponse response;
    response.status = status;
    response.body = std::move(body);
    return response;
}

#if defined(AGENTLOOM_TEST_LOCAL_LLM)
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
#endif

class FixedTokenCounter final : public agent::llm::ICompletionTokenCounter {
public:
    explicit FixedTokenCounter(std::size_t count) : count_(count) {}

    core::Result<std::size_t> CountTokens(
        std::string_view,
        const ChatCompletionResponse&) const override {
        return count_;
    }

private:
    std::size_t count_;
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

TEST_F(OpenAiLlmClientTest, VisionContentPartsUseOpenAiCompatibleImageUrlFormat) {
    ChatCompletionRequest req;
    req.model = "glm-4.6v";
    req.messages.push_back({
        ChatRole::User,
        "",
        {
            ChatContentPart::ImageData("image/jpeg", "abc123"),
            ChatContentPart::Text("请描述画面"),
        },
    });

    auto result = client_->Complete(req);

    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_NE(behavior_->last_body.find("\"model\":\"glm-4.6v\""), std::string::npos);
    EXPECT_NE(behavior_->last_body.find("\"content\":[{"), std::string::npos);
    EXPECT_NE(behavior_->last_body.find("\"type\":\"image_url\""), std::string::npos);
    EXPECT_NE(behavior_->last_body.find("data:image/jpeg;base64,abc123"), std::string::npos);
    EXPECT_NE(behavior_->last_body.find("\"type\":\"text\""), std::string::npos);
    EXPECT_NE(behavior_->last_body.find("请描述画面"), std::string::npos);
}

TEST_F(OpenAiLlmClientTest, PlainTextMessageKeepsStringContentFormat) {
    auto req = SimpleRequest();
    auto result = client_->Complete(req);

    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_NE(behavior_->last_body.find("\"content\":\"hello\""), std::string::npos);
    EXPECT_EQ(behavior_->last_body.find("\"content\":[{"), std::string::npos);
}

TEST_F(OpenAiLlmClientTest, SerializesToolsAndParsesToolCalls) {
    behavior_->body = R"({"id":"cmpl-tool","model":"deepseek-chat","choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"vision.observe","arguments":"{\"reason\":\"look\"}"}}]}}]})";
    ChatCompletionRequest request;
    request.tools.push_back({"vision.observe", "observe", R"({"type":"object"})"});
    request.tool_choice = "auto";
    auto result = client_->Complete(request);
    ASSERT_TRUE(result.ok()) << result.status().message();
    ASSERT_EQ(result.value().tool_calls.size(), 1u);
    EXPECT_EQ(result.value().tool_calls[0].name, "vision.observe");
    EXPECT_NE(behavior_->last_body.find("\"tools\""), std::string::npos);
    EXPECT_NE(behavior_->last_body.find("vision.observe"), std::string::npos);
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

TEST_F(OpenAiLlmClientTest, EmptyContentWithoutToolCallsReturnsUnavailable) {
    behavior_->body = R"({"choices":[{"message":{"role":"assistant","content":""},"finish_reason":"stop"}]})";
    auto result = client_->Complete(SimpleRequest());
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::Unavailable);
}

TEST_F(OpenAiLlmClientTest, LengthFinishReasonReturnsResourceExhausted) {
    behavior_->body = R"({"choices":[{"message":{"role":"assistant","content":"partial"},"finish_reason":"length"}],"usage":{"completion_tokens":128,"total_tokens":128}})";
    auto result = client_->Complete(SimpleRequest());
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::ResourceExhausted);
}

TEST_F(OpenAiLlmClientTest, ParsesReasoningAlongsideVisibleContent) {
    behavior_->body = R"({"model":"deepseek-chat","choices":[{"message":{"role":"assistant","reasoning_content":"reason","content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}})";
    auto result = client_->Complete(SimpleRequest());
    ASSERT_TRUE(result.ok()) << result.status().message();
    ASSERT_TRUE(result.value().reasoning_content.has_value());
    EXPECT_EQ(*result.value().reasoning_content, "reason");
    EXPECT_EQ(result.value().content, "answer");
}

TEST_F(OpenAiLlmClientTest, RejectsInconsistentUsage) {
    behavior_->body = R"({"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":12}})";
    auto result = client_->Complete(SimpleRequest());
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::DataLoss);
}

TEST_F(OpenAiLlmClientTest, RejectsMalformedUsageType) {
    behavior_->body = R"({"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"completion_tokens":"5"}})";
    auto result = client_->Complete(SimpleRequest());
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::DataLoss);
}

TEST_F(OpenAiLlmClientTest, StrictTokenCountRejectsMismatch) {
    OpenAiLlmClientOptions options;
    options.base_url = MakeBaseUrl(server_->port());
    options.api_key = "test-key";
    options.retry_policy.max_retries = 0;
    options.response_validation.token_count_mode =
        agent::llm::CompletionTokenValidationMode::Strict;
    options.response_validation.max_token_difference = 0;
    options.response_validation.token_counter = std::make_shared<FixedTokenCounter>(2);
    auto client = OpenAiLlmClient::Create(std::move(options), *http_client_).value();

    auto result = client->Complete(SimpleRequest());
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::DataLoss);
}

TEST_F(OpenAiLlmClientTest, AuditTokenCountKeepsValidResponse) {
    OpenAiLlmClientOptions options;
    options.base_url = MakeBaseUrl(server_->port());
    options.api_key = "test-key";
    options.retry_policy.max_retries = 0;
    options.response_validation.token_count_mode =
        agent::llm::CompletionTokenValidationMode::Audit;
    options.response_validation.max_token_difference = 0;
    options.response_validation.token_counter = std::make_shared<FixedTokenCounter>(2);
    auto client = OpenAiLlmClient::Create(std::move(options), *http_client_).value();

    auto result = client->Complete(SimpleRequest());
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_EQ(result.value().content, "42");
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

TEST(OpenAiAsyncLlmClientTest, ImmediateTransportCallbackCompletesExactlyOnce) {
    ScriptedAsyncHttpClient http;
    http.responses.emplace_back(MakeHttpResponse(200, kValidResponse));
    OpenAiLlmClientOptions options;
    options.base_url = "http://127.0.0.1:8080/v1";
    options.api_key = "key";
    options.retry_policy.max_retries = 0;
    auto created = OpenAiAsyncLlmClient::Create(options, http);
    ASSERT_TRUE(created.ok()) << created.status().message();
    auto client = std::move(created).value();

    std::promise<core::Result<ChatCompletionResponse>> completed;
    auto future = completed.get_future();
    std::atomic<int> callback_count = 0;
    auto submitted = client->CompleteAsync({}, [&](auto result) {
        ++callback_count;
        completed.set_value(std::move(result));
    });

    ASSERT_TRUE(submitted.ok()) << submitted.status().message();
    auto result = future.get();
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_EQ(result.value().content, "42");
    EXPECT_EQ(callback_count.load(), 1);
    EXPECT_EQ(http.call_count, 1);
    EXPECT_NE(http.last_request.body.find("deepseek-chat"), std::string::npos);
}

TEST(OpenAiAsyncLlmClientTest, FiveXxRetryUsesTimerAndDoesNotBlockSubmission) {
    ScriptedAsyncHttpClient http;
    http.responses.emplace_back(MakeHttpResponse(503, kServerErrorResponse));
    http.responses.emplace_back(MakeHttpResponse(200, kValidResponse));
    OpenAiLlmClientOptions options;
    options.base_url = "http://127.0.0.1:8080/v1";
    options.api_key = "key";
    options.retry_policy.max_retries = 1;
    options.retry_policy.initial_delay = std::chrono::milliseconds(100);
    auto client = OpenAiAsyncLlmClient::Create(options, http).value();

    std::promise<core::Result<ChatCompletionResponse>> completed;
    auto future = completed.get_future();
    const auto started = std::chrono::steady_clock::now();
    auto submitted = client->CompleteAsync({}, [&completed](auto result) {
        completed.set_value(std::move(result));
    });
    const auto submit_elapsed = std::chrono::steady_clock::now() - started;

    ASSERT_TRUE(submitted.ok());
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(submit_elapsed).count(), 50);
    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    auto result = future.get();
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_EQ(http.call_count, 2);
}

TEST(OpenAiAsyncLlmClientTest, CancelPropagatesAndCompletesExactlyOnce) {
    ScriptedAsyncHttpClient http;
    http.hold_response = true;
    OpenAiLlmClientOptions options;
    options.base_url = "http://127.0.0.1:8080/v1";
    options.api_key = "key";
    auto client = OpenAiAsyncLlmClient::Create(options, http).value();

    std::promise<core::Status> completed;
    auto future = completed.get_future();
    std::atomic<int> callback_count = 0;
    auto submitted = client->CompleteAsync({}, [&](auto result) {
        ++callback_count;
        completed.set_value(result.ok() ? core::Status::Ok() : result.status());
    });
    ASSERT_TRUE(submitted.ok());
    auto operation = std::move(submitted).value();
    operation->Cancel();
    operation->Cancel();

    EXPECT_EQ(future.get().code(), core::ErrorCode::Cancelled);
    EXPECT_EQ(callback_count.load(), 1);
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

#if defined(AGENTLOOM_TEST_LOCAL_LLM)
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
#endif

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
