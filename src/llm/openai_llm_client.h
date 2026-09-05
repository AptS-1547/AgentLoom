#pragma once

#include "../net/http_client/http_client.h"
#include "../net/http_client/retry_policy.h"
#include <unordered_map>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <filesystem>
#include <vector>
#include <optional>
#include <../core/result.h>
namespace agent::llm {

struct OpenAiLlmClientOptions {
    /// Base URL for the OpenAI-compatible API (e.g., "https://api.deepseek.com/v1").
    std::string base_url;
    /// API key for authentication.
    std::string api_key;
    /// Default model name if not specified in the request.
    std::string default_model = "deepseek-chat";
    /// Request timeout in milliseconds.
    int timeout_ms = 30000;
    /// Retry policy for transient failures.
    net::RetryPolicy retry_policy;
    /// Local OpenAI-compatible endpoints normally do not require a bearer key.
    bool require_api_key = true;
};

enum class ChatRole {
    System,
    User,
    Assistant,
    Tool,
};

enum class ChatContentPartType {
    Text,
    ImageUrl,
};

struct ChatContentPart {
    ChatContentPartType type = ChatContentPartType::Text;
    std::string text;
    std::string image_url;

    static ChatContentPart Text(std::string text);
    static ChatContentPart ImageUrl(std::string image_url);
    static ChatContentPart ImageData(std::string_view media_type, std::string_view base64_data);
};

// 工具调用标识必须原样回传；arguments 在协议层保留为 JSON 文本。
struct ChatToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::string content;
    std::vector<ChatContentPart> parts;
    std::vector<ChatToolCall> tool_calls;
    std::string tool_call_id;
    std::optional<std::string> reasoning_content;
};

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    float temperature = 0.7f;
    int max_tokens = 0;
    float top_p = 1.0f;
    int n = 1;
    bool stream = false;
    // schema 文本保持接口轻量；发送前必须校验，不能静默降级为空对象。
    struct Tool {
        std::string name;
        std::string description;
        std::string parameters_json = R"({"type":"object"})";
    };
    std::vector<Tool> tools;
    std::string tool_choice; // 空串遵循 Provider 默认，另支持 none/auto/required。
    std::optional<bool> parallel_tool_calls;
};

struct ChatCompletionResponse {
    std::string id;
    std::string model;
    std::string content;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
    std::vector<ChatToolCall> tool_calls;
    std::string finish_reason;
    std::optional<std::string> reasoning_content;
};

// 同步和异步传输共用协议校验，错误请求不访问 Provider。
core::Status ValidateChatCompletionRequest(const ChatCompletionRequest& request);

class ILlmClient {
public:
    virtual ~ILlmClient() = default;

    /// 同步执行一次 chat completion。
    /// @param req 模型、消息和生成参数；调用期间只读。
    /// @return 传输失败、API 错误或响应格式错误时返回失败 Status。
    virtual core::Result<ChatCompletionResponse> Complete(
        const ChatCompletionRequest& req) = 0;
};

class IAsyncLlmOperation {
public:
    virtual ~IAsyncLlmOperation() = default;
    /// 幂等取消；完成 callback 仍恰好调用一次并返回 Cancelled。
    virtual void Cancel() noexcept = 0;
};

class IAsyncLlmClient {
public:
    using Callback = std::function<void(core::Result<ChatCompletionResponse>)>;

    virtual ~IAsyncLlmClient() = default;
    /// 异步执行 completion；返回句柄只用于取消，丢弃句柄不取消请求。
    virtual core::Result<std::shared_ptr<IAsyncLlmOperation>> CompleteAsync(
        ChatCompletionRequest request,
        Callback callback) = 0;
};

/// OpenAI-compatible LLM client.
///
/// Implements the OpenAI chat completions API format, compatible with
/// DeepSeek, vLLM, and other OpenAI-compatible proxies.
///
/// Thread-safe: the underlying `IHttpClient` is stateless and can be called
/// concurrently from multiple threads.
class OpenAiLlmClient : public ILlmClient {
public:
    /// 创建 OpenAI-compatible 客户端。
    /// @param options endpoint、认证、超时和重试配置。
    /// @param http_client 借用的 HTTP 客户端，生命周期必须长于返回对象。
    static core::Result<std::unique_ptr<OpenAiLlmClient>> Create(
        OpenAiLlmClientOptions options,
        net::IHttpClient& http_client);

    ~OpenAiLlmClient() override;

    core::Result<ChatCompletionResponse> Complete(
        const ChatCompletionRequest& req) override;

private:
    OpenAiLlmClient(OpenAiLlmClientOptions options, net::IHttpClient& http_client);

    core::Result<ChatCompletionResponse> ExecuteWithRetry(
        const ChatCompletionRequest& req);

    OpenAiLlmClientOptions options_;
    net::IHttpClient& http_client_;
};

/// OpenAI-compatible 真异步客户端；HTTP 等待和 retry backoff 均不占用业务线程池 worker。
class OpenAiAsyncLlmClient final : public IAsyncLlmClient {
public:
    struct Impl;

    static core::Result<std::unique_ptr<OpenAiAsyncLlmClient>> Create(
        OpenAiLlmClientOptions options,
        net::IAsyncHttpClient& http_client);
    ~OpenAiAsyncLlmClient() override;

    core::Result<std::shared_ptr<IAsyncLlmOperation>> CompleteAsync(
        ChatCompletionRequest request,
        Callback callback) override;

    /// 幂等关闭：取消在途 completion，等待 retry runtime 收口。
    void Shutdown() noexcept;

private:
    explicit OpenAiAsyncLlmClient(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;
};

struct FallbackLlmClientOptions {
    int failure_threshold = 3;
    std::chrono::milliseconds primary_reconnect_interval{30000};
    bool fallback_on_primary_missing = true;
    bool fallback_on_auth_failure = true;
    bool fallback_on_unavailable = true;
};

class FallbackLlmClient final : public ILlmClient {
public:
    /// @param primary 首选客户端，可为空并按 options 决定是否直接降级。
    /// @param fallback 降级客户端；需要降级但为空时返回原始失败。
    /// @param options 连续失败阈值、探测间隔和允许降级的错误类型。
    FallbackLlmClient(std::shared_ptr<ILlmClient> primary,
                      std::shared_ptr<ILlmClient> fallback,
                      FallbackLlmClientOptions options = {});

    core::Result<ChatCompletionResponse> Complete(const ChatCompletionRequest& req) override;

private:
    bool ShouldTryPrimary(std::chrono::steady_clock::time_point now) const;
    bool ShouldFallback(const core::Status& status) const;
    void RecordPrimarySuccess();
    void RecordPrimaryFailure();

    std::shared_ptr<ILlmClient> primary_;
    std::shared_ptr<ILlmClient> fallback_;
    FallbackLlmClientOptions options_;
    mutable std::mutex mutex_;
    int consecutive_failures_ = 0;
    std::chrono::steady_clock::time_point next_primary_probe_{};
};



class LlmPromptStore {
public:
    /// Load all prompts from the given map of name → relative path.
    /// Paths are resolved relative to `config_dir`.
    /// Returns `NotFound` if any file is missing, `InternalError` on read failure.
    core::Status Load(const std::unordered_map<std::string, std::filesystem::path>& prompt_paths,
                      const std::filesystem::path& config_dir);

    /// Retrieve a loaded prompt by name.
    /// Returns `NotFound` if the prompt was not loaded.
    core::Result<std::string> Get(const std::string& name) const;

    /// Check if a prompt exists.
    bool Has(const std::string& name) const;

    /// Clear all loaded prompts.
    void Clear();

private:
    std::unordered_map<std::string, std::string> prompts_;
};

};  // namespace agent::llm
