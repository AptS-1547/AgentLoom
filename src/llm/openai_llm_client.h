#pragma once

#include "../net/http_client/http_client.h"
#include "../net/http_client/retry_policy.h"
#include <unordered_map>
#include <memory>
#include <string>
#include <filesystem>

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
};

enum class ChatRole {
    System,
    User,
    Assistant,
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::string content;
};

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    float temperature = 0.7f;
    int max_tokens = 0;
    float top_p = 1.0f;
    int n = 1;
    bool stream = false;
};

struct ChatCompletionResponse {
    std::string id;
    std::string model;
    std::string content;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
};

class ILlmClient {
public:
    virtual ~ILlmClient() = default;

    /// Synchronous chat completion.  Returns a non-ok Status for transport
    /// failures, API errors, or malformed responses.  HTTP 4xx/5xx are
    /// reported as errors with the status code in the message.
    virtual core::Result<ChatCompletionResponse> Complete(
        const ChatCompletionRequest& req) = 0;
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
    /// `http_client` must outlive this object.
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
