#include "openai_llm_client.h"
#include "text_validation.h"
#include "logger_adapter.h"

#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include <atomic>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace agent::llm {

using Json = nlohmann::json;

namespace {

std::string RoleToString(ChatRole role) {
    switch (role) {
        case ChatRole::System: return "system";
        case ChatRole::User: return "user";
        case ChatRole::Assistant: return "assistant";
        case ChatRole::Tool: return "tool";
    }
    return "user";
}

Json ContentPartToJson(const ChatContentPart& part) {
    switch (part.type) {
        case ChatContentPartType::Text:
            return Json{{"type", "text"}, {"text", part.text}};
        case ChatContentPartType::ImageUrl:
            return Json{{"type", "image_url"}, {"image_url", {{"url", part.image_url}}}};
    }
    return Json{{"type", "text"}, {"text", part.text}};
}

Json BuildRequestJson(const ChatCompletionRequest& req, const std::string& default_model) {
    Json j;
    j["model"] = req.model.empty() ? default_model : req.model;
    j["messages"] = Json::array();
    for (const auto& msg : req.messages) {
        Json msg_obj;
        msg_obj["role"] = RoleToString(msg.role);
        if (msg.parts.empty()) {
            msg_obj["content"] = msg.content;
        } else {
            msg_obj["content"] = Json::array();
            for (const auto& part : msg.parts) {
                msg_obj["content"].push_back(ContentPartToJson(part));
            }
        }
        if (!msg.tool_calls.empty()) {
            if (msg.content.empty()) msg_obj["content"] = nullptr;
            msg_obj["tool_calls"] = Json::array();
            for (const auto& call : msg.tool_calls) {
                msg_obj["tool_calls"].push_back({{"id", call.id}, {"type", "function"},
                    {"function", {{"name", call.name}, {"arguments", call.arguments_json}}}});
            }
        }
        if (msg.role == ChatRole::Tool) msg_obj["tool_call_id"] = msg.tool_call_id;
        if (msg.reasoning_content) msg_obj["reasoning_content"] = *msg.reasoning_content;
        j["messages"].push_back(msg_obj);
    }
    j["temperature"] = req.temperature;
    if (req.max_tokens > 0) {
        j["max_tokens"] = req.max_tokens;
    }
    j["top_p"] = req.top_p;
    j["n"] = req.n;
    j["stream"] = req.stream;
    if (!req.tools.empty()) {
        j["tools"] = Json::array();
        for (const auto& tool : req.tools) {
            const auto parameters = Json::parse(tool.parameters_json);
            j["tools"].push_back({{"type", "function"}, {"function", {
                {"name", tool.name}, {"description", tool.description}, {"parameters", parameters}
            }}});
        }
    }
    if (!req.tool_choice.empty()) j["tool_choice"] = req.tool_choice;
    if (req.parallel_tool_calls) j["parallel_tool_calls"] = *req.parallel_tool_calls;
    return j;
}

core::Result<ChatCompletionResponse> ParseResponse(
    const std::string& body,
    int status_code,
    const ChatCompletionRequest& request,
    const OpenAiLlmClientOptions& options,
    core::LoggerAdapter& logger) {
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

    if (body.empty()) {
        return core::Status::Error(core::ErrorCode::DataLoss, "LLM response body is empty");
    }
    Json j;
    try {
        j = Json::parse(body);
    } catch (const std::exception& e) {
        return core::Status(core::ErrorCode::DataLoss,
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
        return core::Status(core::ErrorCode::DataLoss,
            "LLM response missing choices array");
    }

    const auto& choice = j["choices"][0];
    if (!choice.contains("message") || !choice["message"].is_object()) {
        return core::Status(core::ErrorCode::DataLoss,
            "LLM response missing message object");
    }

    const auto& message = choice["message"];
    if (message.contains("content") && !message["content"].is_null() && !message["content"].is_string()) {
        return core::Status::Error(core::ErrorCode::DataLoss, "LLM content must be text or null");
    }
    const bool has_content = message.contains("content") && message["content"].is_string();
    if (has_content) {
        resp.content = message["content"].get<std::string>();
    }
    if (choice.contains("finish_reason") && !choice["finish_reason"].is_null()) {
        if (!choice["finish_reason"].is_string()) {
            return core::Status::Error(core::ErrorCode::DataLoss,
                                       "LLM finish_reason must be text or null");
        }
        resp.finish_reason = choice["finish_reason"].get<std::string>();
    }
    if (message.contains("reasoning_content") && !message["reasoning_content"].is_null()) {
        if (!message["reasoning_content"].is_string()) {
            return core::Status::Error(core::ErrorCode::DataLoss,
                                       "LLM reasoning_content must be text or null");
        }
        resp.reasoning_content = message["reasoning_content"].get<std::string>();
    }
    if (message.contains("tool_calls") && !message["tool_calls"].is_null()) {
        if (!message["tool_calls"].is_array()) {
            return core::Status::Error(core::ErrorCode::DataLoss, "LLM tool_calls must be an array");
        }
        std::unordered_set<std::string> ids;
        for (const auto& call : message["tool_calls"]) {
            // 整批拒绝损坏调用，避免部分执行产生不可解释的副作用。
            if (!call.is_object() || !call.contains("id") || !call["id"].is_string() ||
                !call.contains("type") || call["type"] != "function" ||
                !call.contains("function") || !call["function"].is_object()) {
                return core::Status::Error(core::ErrorCode::DataLoss, "malformed LLM tool call");
            }
            const auto& fn = call["function"];
            if (!fn.contains("name") || !fn["name"].is_string() ||
                !fn.contains("arguments") ||
                (!fn["arguments"].is_string() && !fn["arguments"].is_object())) {
                return core::Status::Error(core::ErrorCode::DataLoss, "malformed LLM function fields");
            }
            const auto arguments = fn["arguments"].is_string()
                ? fn["arguments"].get<std::string>()
                : fn["arguments"].dump();
            ChatToolCall parsed{call["id"].get<std::string>(), fn["name"].get<std::string>(), arguments};
            if (parsed.id.empty() || parsed.name.empty() || !ids.insert(parsed.id).second) {
                return core::Status::Error(core::ErrorCode::DataLoss, "invalid or duplicate LLM tool call id");
            }
            resp.tool_calls.push_back(std::move(parsed));
        }
    }
    if (j.contains("usage") && !j["usage"].is_null() && !j["usage"].is_object()) {
        return core::Status::Error(core::ErrorCode::DataLoss,
                                   "LLM usage must be an object or null");
    }
    if (j.contains("usage") && j["usage"].is_object()) {
        const auto& usage = j["usage"];
        const auto read_tokens = [&usage](std::string_view name, int& target) -> core::Status {
            const auto found = usage.find(std::string(name));
            if (found == usage.end()) {
                return core::Status::Ok();
            }
            if (!found->is_number_integer() && !found->is_number_unsigned()) {
                return core::Status::Error(
                    core::ErrorCode::DataLoss,
                    "LLM usage." + std::string(name) + " must be a non-negative integer");
            }
            try {
                const auto value = found->get<std::int64_t>();
                if (value < 0 || value > std::numeric_limits<int>::max()) {
                    return core::Status::Error(
                        core::ErrorCode::DataLoss,
                        "LLM usage." + std::string(name) + " is out of range");
                }
                target = static_cast<int>(value);
                return core::Status::Ok();
            } catch (const std::exception&) {
                return core::Status::Error(
                    core::ErrorCode::DataLoss,
                    "LLM usage." + std::string(name) + " is out of range");
            }
        };
        if (auto status = read_tokens("prompt_tokens", resp.prompt_tokens); !status.ok()) {
            return status;
        }
        if (auto status = read_tokens("completion_tokens", resp.completion_tokens); !status.ok()) {
            return status;
        }
        if (auto status = read_tokens("total_tokens", resp.total_tokens); !status.ok()) {
            return status;
        }
    }

    if (resp.prompt_tokens < 0 || resp.completion_tokens < 0 || resp.total_tokens < 0 ||
        (resp.total_tokens > 0 &&
         resp.total_tokens < resp.prompt_tokens + resp.completion_tokens)) {
        return core::Status::Error(core::ErrorCode::DataLoss,
                                   "LLM response contains inconsistent token usage");
    }
    if (options.response_validation.reject_length_finish && resp.finish_reason == "length") {
        return core::Status::Error(
            core::ErrorCode::ResourceExhausted,
            "LLM generation was truncated because the token budget was exhausted");
    }
    if (has_content) {
        if (!core::IsValidUtf8(resp.content) || core::HasInvalidTextControl(resp.content)) {
            return core::Status::Error(core::ErrorCode::DataLoss,
                                       "LLM content is not valid UTF-8 text");
        }
    }
    if (resp.reasoning_content &&
        (!core::IsValidUtf8(*resp.reasoning_content) ||
         core::HasInvalidTextControl(*resp.reasoning_content))) {
        return core::Status::Error(core::ErrorCode::DataLoss,
                                   "LLM reasoning_content is not valid UTF-8 text");
    }
    if (resp.tool_calls.empty() &&
        (!has_content ||
         (options.response_validation.reject_empty_content && core::IsBlankAscii(resp.content)))) {
        return core::Status::Error(core::ErrorCode::Unavailable,
                                   "LLM response contains no usable content or tool calls");
    }

    const auto& validation = options.response_validation;
    if (validation.token_count_mode != CompletionTokenValidationMode::Off &&
        validation.token_counter && resp.completion_tokens > 0) {
        const auto model = resp.model.empty()
            ? (request.model.empty() ? options.default_model : request.model)
            : resp.model;
        auto counted = validation.token_counter->CountTokens(model, resp);
        if (!counted.ok()) {
            if (validation.token_count_mode == CompletionTokenValidationMode::Strict) {
                return counted.status();
            }
            logger.warn("LLM token count audit skipped model={} reason={}",
                        model, counted.status().message());
        } else {
            const auto reported = static_cast<std::size_t>(resp.completion_tokens);
            const auto actual = counted.value();
            const auto difference = actual > reported ? actual - reported : reported - actual;
            if (difference > validation.max_token_difference) {
                if (validation.token_count_mode == CompletionTokenValidationMode::Strict) {
                    return core::Status::Error(
                        core::ErrorCode::DataLoss,
                        "LLM completion token count differs from provider usage");
                }
                logger.warn(
                    "LLM token count audit mismatch model={} reported={} actual={} difference={}",
                    model, reported, actual, difference);
            }
        }
    }

    return resp;
}

}  // namespace

core::Status ValidateChatCompletionRequest(const ChatCompletionRequest& request) {
    const auto invalid = [](const char* reason) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, reason);
    };
    if (request.stream) return invalid("streaming completions are not implemented by this client");
    if (request.tool_choice != "" && request.tool_choice != "none" &&
        request.tool_choice != "auto" && request.tool_choice != "required") {
        return invalid("unsupported tool_choice");
    }
    if ((request.tool_choice == "auto" || request.tool_choice == "required" ||
         request.parallel_tool_calls.has_value()) && request.tools.empty()) {
        return invalid("tool selection requires tool definitions");
    }
    std::unordered_set<std::string> names;
    for (const auto& tool : request.tools) {
        const auto schema = Json::parse(tool.parameters_json, nullptr, false);
        if (tool.name.empty() || !names.insert(tool.name).second || !schema.is_object() ||
            !schema.contains("type") || schema["type"] != "object") {
            return invalid("tools require unique names and an object parameter schema");
        }
    }
    // 一次 assistant 调用批次必须收齐结果，才能继续用户/助手消息。
    std::unordered_set<std::string> pending;
    std::unordered_set<std::string> used;
    for (const auto& message : request.messages) {
        if (message.role == ChatRole::Tool) {
            if (message.tool_call_id.empty() || pending.erase(message.tool_call_id) != 1 ||
                !message.tool_calls.empty() || !message.parts.empty() || message.reasoning_content) {
                return invalid("tool result does not match a pending assistant call");
            }
            continue;
        }
        if (!pending.empty()) return invalid("assistant tool calls are missing results");
        if (!message.tool_call_id.empty() ||
            (message.role != ChatRole::Assistant && (!message.tool_calls.empty() || message.reasoning_content))) {
            return invalid("tool metadata belongs to assistant or tool messages only");
        }
        for (const auto& call : message.tool_calls) {
            if (call.id.empty() || call.name.empty() || !used.insert(call.id).second) {
                return invalid("assistant tool call id is empty or duplicated");
            }
            pending.insert(call.id);
        }
    }
    if (!pending.empty()) return invalid("assistant tool calls are missing results");
    return core::Status::Ok();
}

ChatContentPart ChatContentPart::Text(std::string text) {
    ChatContentPart part;
    part.type = ChatContentPartType::Text;
    part.text = std::move(text);
    return part;
}

ChatContentPart ChatContentPart::ImageUrl(std::string image_url) {
    ChatContentPart part;
    part.type = ChatContentPartType::ImageUrl;
    part.image_url = std::move(image_url);
    return part;
}

ChatContentPart ChatContentPart::ImageData(std::string_view media_type, std::string_view base64_data) {
    ChatContentPart part;
    part.type = ChatContentPartType::ImageUrl;
    part.image_url = "data:";
    part.image_url.append(media_type);
    part.image_url.append(";base64,");
    part.image_url.append(base64_data);
    return part;
}

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
        if (auto status = ValidateChatCompletionRequest(req); !status.ok()) {
            core::LoggerAdapter::ForModule("llm-client").warn("LLM request rejected: {}", status.message());
            return status;
        }
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

        auto logger = core::LoggerAdapter::ForModule("llm-client");
        return ParseResponse(http_resp.body, http_resp.status, req, options_, logger);
    }

    return last_result_status;
}

namespace {

net::HttpClientRequest BuildHttpRequest(const OpenAiLlmClientOptions& options,
                                        const ChatCompletionRequest& request) {
    auto url = options.base_url;
    if (url.back() != '/') {
        url.push_back('/');
    }
    url += "chat/completions";

    net::HttpClientRequest http_request;
    http_request.method = "POST";
    http_request.url = std::move(url);
    http_request.headers.push_back({"Content-Type", "application/json"});
    if (!options.api_key.empty()) {
        http_request.headers.push_back({"Authorization", "Bearer " + options.api_key});
    }
    http_request.body = BuildRequestJson(request, options.default_model)
        .dump(-1, ' ', false, Json::error_handler_t::replace);
    http_request.timeout_ms = options.timeout_ms;
    return http_request;
}

class AsyncOpenAiOperation;

} // namespace

struct OpenAiAsyncLlmClient::Impl final
    : public std::enable_shared_from_this<OpenAiAsyncLlmClient::Impl> {
    using WorkGuard = boost::asio::executor_work_guard<boost::asio::io_context::executor_type>;

    Impl(OpenAiLlmClientOptions client_options, net::IAsyncHttpClient& client)
        : options(std::move(client_options)),
          http_client(client),
          retry_guard(boost::asio::make_work_guard(retry_context)),
          logger(core::LoggerAdapter::ForModule("async-llm-client")) {}

    core::Status Start() {
        try {
            auto self = shared_from_this();
            retry_thread = std::thread([self] {
                try {
                    self->retry_context.run();
                } catch (const std::exception& error) {
                    self->logger.error("异步 LLM retry runtime 异常退出: {}", error.what());
                } catch (...) {
                    self->logger.error("异步 LLM retry runtime 发生未知异常");
                }
            });
            return core::Status::Ok();
        } catch (const std::exception& error) {
            retry_guard.reset();
            retry_context.stop();
            return core::Status::Error(
                core::ErrorCode::InternalError,
                std::string("failed to start async LLM retry runtime: ") + error.what());
        }
    }

    core::Status Register(const std::shared_ptr<AsyncOpenAiOperation>& operation) {
        std::lock_guard lock(mutex);
        if (stopping) {
            return core::Status::Error(core::ErrorCode::Cancelled,
                                       "async LLM client is shutting down");
        }
        operations.emplace(operation.get(), operation);
        return core::Status::Ok();
    }

    void Unregister(AsyncOpenAiOperation* operation) noexcept {
        std::lock_guard lock(mutex);
        operations.erase(operation);
    }

    void Shutdown() noexcept;

    OpenAiLlmClientOptions options;
    net::IAsyncHttpClient& http_client;
    boost::asio::io_context retry_context;
    WorkGuard retry_guard;
    core::LoggerAdapter logger;
    std::thread retry_thread;
    std::mutex mutex;
    bool stopping = false;
    std::unordered_map<AsyncOpenAiOperation*, std::shared_ptr<AsyncOpenAiOperation>> operations;
};

namespace {

class AsyncOpenAiOperation final : public IAsyncLlmOperation,
                                   public std::enable_shared_from_this<AsyncOpenAiOperation> {
public:
    AsyncOpenAiOperation(std::shared_ptr<OpenAiAsyncLlmClient::Impl> owner,
                         ChatCompletionRequest request,
                         IAsyncLlmClient::Callback callback)
        : owner_(std::move(owner)),
          request_(std::move(request)),
          http_request_(BuildHttpRequest(owner_->options, request_)),
          retry_timer_(owner_->retry_context),
          callback_(std::move(callback)) {}

    void Start() noexcept {
        BeginAttempt();
    }

    void Cancel() noexcept override {
        if (cancel_requested_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        std::shared_ptr<net::IAsyncHttpOperation> http_operation;
        {
            std::lock_guard lock(mutex_);
            http_operation = http_operation_;
        }
        if (http_operation) {
            http_operation->Cancel();
        }
        auto self = shared_from_this();
        boost::asio::post(owner_->retry_context, [self] {
            boost::system::error_code ignored;
            self->retry_timer_.cancel(ignored);
        });
        Finish(core::Status::Error(core::ErrorCode::Cancelled,
                                   "async LLM completion cancelled"));
    }

private:
    void BeginAttempt() noexcept {
        if (completed_.load(std::memory_order_acquire)) {
            return;
        }
        auto self = shared_from_this();
        auto submitted = owner_->http_client.ExecuteAsync(
            http_request_,
            [self](core::Result<net::HttpClientResponse> response) {
                self->OnHttpComplete(std::move(response));
            });
        if (!submitted.ok()) {
            OnAttemptFailure(submitted.status());
            return;
        }
        std::lock_guard lock(mutex_);
        if (completed_.load(std::memory_order_acquire)) {
            submitted.value()->Cancel();
        } else {
            http_operation_ = std::move(submitted).value();
        }
    }

    void OnHttpComplete(core::Result<net::HttpClientResponse> result) noexcept {
        {
            std::lock_guard lock(mutex_);
            http_operation_.reset();
        }
        if (completed_.load(std::memory_order_acquire)) {
            return;
        }
        if (!result.ok()) {
            OnAttemptFailure(result.status());
            return;
        }
        auto response = std::move(result).value();
        if (response.status >= 500 && response.status < 600 && CanRetry()) {
            ScheduleRetry();
            return;
        }
        Finish(ParseResponse(
            response.body, response.status, request_, owner_->options, owner_->logger));
    }

    void OnAttemptFailure(const core::Status& status) noexcept {
        if (completed_.load(std::memory_order_acquire)) {
            return;
        }
        if (cancel_requested_.load(std::memory_order_acquire)) {
            Finish(core::Status::Error(core::ErrorCode::Cancelled,
                                       "async LLM completion cancelled"));
            return;
        }
        if (CanRetry()) {
            ScheduleRetry();
            return;
        }
        Finish(status);
    }

    bool CanRetry() const noexcept {
        return attempt_ < owner_->options.retry_policy.max_retries;
    }

    void ScheduleRetry() noexcept {
        const auto delay = owner_->options.retry_policy.BackoffFor(attempt_);
        ++attempt_;
        auto self = shared_from_this();
        boost::asio::post(owner_->retry_context, [self, delay] {
            if (self->completed_.load(std::memory_order_acquire)) {
                return;
            }
            self->retry_timer_.expires_after(delay);
            self->retry_timer_.async_wait([self](const boost::system::error_code& error) {
                if (error == boost::asio::error::operation_aborted ||
                    self->completed_.load(std::memory_order_acquire)) {
                    return;
                }
                self->BeginAttempt();
            });
        });
    }

    void Finish(core::Result<ChatCompletionResponse> result) noexcept {
        if (completed_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        boost::asio::post(owner_->retry_context, [self = shared_from_this()] {
            boost::system::error_code ignored;
            self->retry_timer_.cancel(ignored);
        });
        owner_->Unregister(this);
        auto callback = std::move(callback_);
        try {
            callback(std::move(result));
        } catch (const std::exception& error) {
            owner_->logger.error("异步 LLM callback 抛出异常: {}", error.what());
        } catch (...) {
            owner_->logger.error("异步 LLM callback 抛出未知异常");
        }
    }

    std::shared_ptr<OpenAiAsyncLlmClient::Impl> owner_;
    ChatCompletionRequest request_;
    net::HttpClientRequest http_request_;
    boost::asio::steady_timer retry_timer_;
    IAsyncLlmClient::Callback callback_;
    std::mutex mutex_;
    std::shared_ptr<net::IAsyncHttpOperation> http_operation_;
    std::atomic<bool> cancel_requested_{false};
    std::atomic<bool> completed_{false};
    std::int32_t attempt_ = 0;
};

} // namespace

void OpenAiAsyncLlmClient::Impl::Shutdown() noexcept {
    std::vector<std::shared_ptr<AsyncOpenAiOperation>> pending;
    {
        std::lock_guard lock(mutex);
        if (stopping) {
            return;
        }
        stopping = true;
        pending.reserve(operations.size());
        for (const auto& [_, operation] : operations) {
            pending.push_back(operation);
        }
    }
    for (const auto& operation : pending) {
        operation->Cancel();
    }
    pending.clear();
    retry_guard.reset();
    if (retry_thread.joinable()) {
        if (retry_thread.get_id() == std::this_thread::get_id()) {
            retry_thread.detach();
        } else {
            retry_thread.join();
        }
    }
}

core::Result<std::unique_ptr<OpenAiAsyncLlmClient>> OpenAiAsyncLlmClient::Create(
    OpenAiLlmClientOptions options,
    net::IAsyncHttpClient& http_client) {
    if (options.base_url.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "OpenAI async LLM client requires base_url");
    }
    if (options.require_api_key && options.api_key.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "OpenAI async LLM client requires api_key");
    }
    if (options.timeout_ms <= 0 || options.retry_policy.max_retries < 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "OpenAI async LLM timeout and retry options are invalid");
    }
    try {
        auto impl = std::make_shared<Impl>(std::move(options), http_client);
        if (auto status = impl->Start(); !status.ok()) {
            return status;
        }
        return std::unique_ptr<OpenAiAsyncLlmClient>(new OpenAiAsyncLlmClient(std::move(impl)));
    } catch (const std::exception& error) {
        return core::Status::Error(
            core::ErrorCode::InternalError,
            std::string("failed to create async LLM client: ") + error.what());
    }
}

OpenAiAsyncLlmClient::OpenAiAsyncLlmClient(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

OpenAiAsyncLlmClient::~OpenAiAsyncLlmClient() {
    Shutdown();
}

core::Result<std::shared_ptr<IAsyncLlmOperation>> OpenAiAsyncLlmClient::CompleteAsync(
    ChatCompletionRequest request,
    Callback callback) {
    if (!callback) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "async LLM callback is required");
    }
    if (auto status = ValidateChatCompletionRequest(request); !status.ok()) {
        impl_->logger.warn("LLM request rejected: {}", status.message());
        return status;
    }
    try {
        auto operation = std::make_shared<AsyncOpenAiOperation>(
            impl_, std::move(request), std::move(callback));
        if (auto status = impl_->Register(operation); !status.ok()) {
            return status;
        }
        operation->Start();
        return std::static_pointer_cast<IAsyncLlmOperation>(std::move(operation));
    } catch (const std::exception& error) {
        return core::Status::Error(
            core::ErrorCode::InternalError,
            std::string("failed to submit async LLM completion: ") + error.what());
    }
}

void OpenAiAsyncLlmClient::Shutdown() noexcept {
    if (impl_) {
        impl_->Shutdown();
    }
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
