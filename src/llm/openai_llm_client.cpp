#include "openai_llm_client.h"
#include <nlohmann/json.hpp>
#include <exception>
#include <fstream>
#include <thread>

namespace agent::llm {

using Json = nlohmann::json;

namespace {

std::string RoleToString(ChatRole role) {
    switch (role) {
        case ChatRole::System: return "system";
        case ChatRole::User: return "user";
        case ChatRole::Assistant: return "assistant";
    }
    return "user";
}

Json BuildRequestJson(const ChatCompletionRequest& req, const std::string& default_model) {
    Json j;
    j["model"] = req.model.empty() ? default_model : req.model;
    j["messages"] = Json::array();
    for (const auto& msg : req.messages) {
        Json msg_obj;
        msg_obj["role"] = RoleToString(msg.role);
        msg_obj["content"] = msg.content;
        j["messages"].push_back(msg_obj);
    }
    j["temperature"] = req.temperature;
    if (req.max_tokens > 0) {
        j["max_tokens"] = req.max_tokens;
    }
    j["top_p"] = req.top_p;
    j["n"] = req.n;
    j["stream"] = req.stream;
    return j;
}

core::Result<ChatCompletionResponse> ParseResponse(const std::string& body, int status_code) {
    if (status_code != 200) {
        core::ErrorCode code = core::ErrorCode::InternalError;
        if (status_code == 401 || status_code == 403) {
            code = core::ErrorCode::PermissionDenied;
        } else if (status_code == 429) {
            code = core::ErrorCode::ResourceExhausted;
        } else if (status_code >= 500 && status_code < 600) {
            code = core::ErrorCode::Unavailable;
        }
        return core::Status(code,
            "LLM API returned status " + std::to_string(status_code) + ": " + body);
    }

    Json j;
    try {
        j = Json::parse(body);
    } catch (const std::exception& e) {
        return core::Status(core::ErrorCode::InternalError,
            std::string("Failed to parse LLM response: ") + e.what());
    }

    ChatCompletionResponse resp;
    if (j.contains("id") && j["id"].is_string()) {
        resp.id = j["id"].get<std::string>();
    }
    if (j.contains("model") && j["model"].is_string()) {
        resp.model = j["model"].get<std::string>();
    }

    if (!j.contains("choices") || !j["choices"].is_array() || j["choices"].empty()) {
        return core::Status(core::ErrorCode::InternalError,
            "LLM response missing choices array");
    }

    const auto& choice = j["choices"][0];
    if (!choice.contains("message") || !choice["message"].is_object()) {
        return core::Status(core::ErrorCode::InternalError,
            "LLM response missing message object");
    }

    const auto& message = choice["message"];
    if (!message.contains("content") || !message["content"].is_string()) {
        return core::Status(core::ErrorCode::InternalError,
            "LLM response missing content field");
    }
    resp.content = message["content"].get<std::string>();

    if (j.contains("usage") && j["usage"].is_object()) {
        const auto& usage = j["usage"];
        if (usage.contains("prompt_tokens") && usage["prompt_tokens"].is_number()) {
            resp.prompt_tokens = usage["prompt_tokens"].get<int>();
        }
        if (usage.contains("completion_tokens") && usage["completion_tokens"].is_number()) {
            resp.completion_tokens = usage["completion_tokens"].get<int>();
        }
        if (usage.contains("total_tokens") && usage["total_tokens"].is_number()) {
            resp.total_tokens = usage["total_tokens"].get<int>();
        }
    }

    return resp;
}

}  // namespace

core::Status LlmPromptStore::Load(
    const std::unordered_map<std::string, std::filesystem::path>& prompt_paths,
    const std::filesystem::path& config_dir) {

    prompts_.clear();

    for (const auto& [name, relative_path] : prompt_paths) {
        std::filesystem::path full_path = config_dir / relative_path;

        if (!std::filesystem::exists(full_path)) {
            return core::Status(core::ErrorCode::NotFound,
                "Prompt file not found: " + full_path.string());
        }

        std::ifstream file(full_path, std::ios::in | std::ios::binary);
        if (!file) {
            return core::Status(core::ErrorCode::InternalError,
                "Failed to open prompt file: " + full_path.string());
        }

        std::ostringstream buffer;
        buffer << file.rdbuf();
        if (file.bad()) {
            return core::Status(core::ErrorCode::InternalError,
                "Failed to read prompt file: " + full_path.string());
        }

        prompts_[name] = buffer.str();
    }

    return core::Status::Ok();
}

core::Result<std::string> LlmPromptStore::Get(const std::string& name) const {
    auto it = prompts_.find(name);
    if (it == prompts_.end()) {
        return core::Status(core::ErrorCode::NotFound,
            "Prompt not found: " + name);
    }
    return it->second;
}

bool LlmPromptStore::Has(const std::string& name) const {
    return prompts_.find(name) != prompts_.end();
}

void LlmPromptStore::Clear() {
    prompts_.clear();
}

core::Result<std::unique_ptr<OpenAiLlmClient>> OpenAiLlmClient::Create(
    OpenAiLlmClientOptions options,
    net::IHttpClient& http_client) {

    if (options.base_url.empty()) {
        return core::Status(core::ErrorCode::InvalidArgument,
            "OpenAI LLM client requires base_url");
    }
    if (options.require_api_key && options.api_key.empty()) {
        return core::Status(core::ErrorCode::InvalidArgument,
            "OpenAI LLM client requires api_key");
    }

    auto client = std::unique_ptr<OpenAiLlmClient>(
        new OpenAiLlmClient(std::move(options), http_client));
    return client;
}

OpenAiLlmClient::OpenAiLlmClient(OpenAiLlmClientOptions options, net::IHttpClient& http_client)
    : options_(std::move(options)), http_client_(http_client) {}

OpenAiLlmClient::~OpenAiLlmClient() = default;

core::Result<ChatCompletionResponse> OpenAiLlmClient::Complete(
    const ChatCompletionRequest& req) {
    try {
        return ExecuteWithRetry(req);
    } catch (const std::exception& e) {
        return core::Status(core::ErrorCode::InternalError,
            std::string("LLM client exception: ") + e.what());
    } catch (...) {
        return core::Status(core::ErrorCode::InternalError,
            "LLM client exception: unknown error");
    }
}

core::Result<ChatCompletionResponse> OpenAiLlmClient::ExecuteWithRetry(
    const ChatCompletionRequest& req) {

    Json request_json = BuildRequestJson(req, options_.default_model);
    std::string request_body = request_json.dump(-1, ' ', false, Json::error_handler_t::replace);

    std::string url = options_.base_url;
    if (url.back() != '/') {
        url += '/';
    }
    url += "chat/completions";

    net::HttpClientRequest http_req;
    http_req.method = "POST";
    http_req.url = url;
    http_req.headers.push_back({"Content-Type", "application/json"});
    if (!options_.api_key.empty()) {
        http_req.headers.push_back({"Authorization", "Bearer " + options_.api_key});
    }
    http_req.body = request_body;
    http_req.timeout_ms = options_.timeout_ms;

    core::Status last_result_status;
    for (int attempt = 0; attempt <= options_.retry_policy.max_retries; ++attempt) {
        if (attempt > 0) {
            auto delay = options_.retry_policy.BackoffFor(attempt - 1);
            std::this_thread::sleep_for(delay);
        }

        auto last_result = http_client_.Execute(http_req);
        last_result_status = last_result.status();
        if (!last_result) {
            // Transport failure — retry if we have attempts left.
            if (attempt < options_.retry_policy.max_retries) {
                continue;
            }
            return last_result.status();
        }

        auto& http_resp = last_result.value();
        // 5xx = server error, retry.  4xx = client error, don't retry.
        if (http_resp.status >= 500 && http_resp.status < 600) {
            if (attempt < options_.retry_policy.max_retries) {
                continue;
            }
        }

        return ParseResponse(http_resp.body, http_resp.status);
    }

    return last_result_status;
}

FallbackLlmClient::FallbackLlmClient(std::shared_ptr<ILlmClient> primary,
                                     std::shared_ptr<ILlmClient> fallback,
                                     FallbackLlmClientOptions options)
    : primary_(std::move(primary)),
      fallback_(std::move(fallback)),
      options_(options) {}

core::Result<ChatCompletionResponse> FallbackLlmClient::Complete(const ChatCompletionRequest& req) {
    if (!fallback_) {
        return core::Status(core::ErrorCode::FailedPrecondition, "fallback llm client is required");
    }

    const auto now = std::chrono::steady_clock::now();
    if (!primary_) {
        if (options_.fallback_on_primary_missing) {
            return fallback_->Complete(req);
        }
        return core::Status(core::ErrorCode::FailedPrecondition, "primary llm client is missing");
    }

    if (!ShouldTryPrimary(now)) {
        return fallback_->Complete(req);
    }

    auto primary_result = primary_->Complete(req);
    if (primary_result.ok()) {
        RecordPrimarySuccess();
        return primary_result;
    }

    RecordPrimaryFailure();
    if (!ShouldFallback(primary_result.status())) {
        return primary_result.status();
    }

    auto fallback_result = fallback_->Complete(req);
    if (fallback_result.ok()) {
        return fallback_result;
    }

    return core::Status(
        fallback_result.status().code(),
        "primary llm failed: " + primary_result.status().message() +
            "; fallback llm failed: " + fallback_result.status().message());
}

bool FallbackLlmClient::ShouldTryPrimary(std::chrono::steady_clock::time_point now) const {
    std::lock_guard lock(mutex_);
    return consecutive_failures_ < options_.failure_threshold || now >= next_primary_probe_;
}

bool FallbackLlmClient::ShouldFallback(const core::Status& status) const {
    switch (status.code()) {
    case core::ErrorCode::InvalidArgument:
    case core::ErrorCode::FailedPrecondition:
        return options_.fallback_on_primary_missing;
    case core::ErrorCode::PermissionDenied:
        return options_.fallback_on_auth_failure;
    case core::ErrorCode::Unavailable:
    case core::ErrorCode::Timeout:
    case core::ErrorCode::ResourceExhausted:
        return options_.fallback_on_unavailable;
    default:
        return false;
    }
}

void FallbackLlmClient::RecordPrimarySuccess() {
    std::lock_guard lock(mutex_);
    consecutive_failures_ = 0;
    next_primary_probe_ = {};
}

void FallbackLlmClient::RecordPrimaryFailure() {
    std::lock_guard lock(mutex_);
    ++consecutive_failures_;
    if (consecutive_failures_ >= options_.failure_threshold) {
        next_primary_probe_ = std::chrono::steady_clock::now() + options_.primary_reconnect_interval;
    }
}

}  // namespace agent::llm
