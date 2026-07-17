#include "grpc_vlm_vision_client.h"

#include "multimodal_inference.grpc.pb.h"

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <string_view>
#include <utility>

namespace media {
namespace {

constexpr std::string_view kDefaultPrompt =
    "请只输出一个 JSON 对象，字段为 scene_hint、action_hint、object_hint、facts、"
    "weak_interpretations、memory_candidate、confidence。只描述画面中直接可见的场景、动作和物体；"
    "不要推断人物身份、性别、情绪、关系或不可见事实。facts 和 weak_interpretations 是字符串数组，"
    "confidence 是 0 到 1 的数字。无法确认的内容放入 weak_interpretations。不要输出 Markdown。";

core::ErrorCode MapGrpcCode(grpc::StatusCode code) noexcept {
    switch (code) {
    case grpc::StatusCode::OK: return core::ErrorCode::Ok;
    case grpc::StatusCode::CANCELLED: return core::ErrorCode::Cancelled;
    case grpc::StatusCode::INVALID_ARGUMENT: return core::ErrorCode::InvalidArgument;
    case grpc::StatusCode::DEADLINE_EXCEEDED: return core::ErrorCode::Timeout;
    case grpc::StatusCode::NOT_FOUND: return core::ErrorCode::NotFound;
    case grpc::StatusCode::ALREADY_EXISTS: return core::ErrorCode::AlreadyExists;
    case grpc::StatusCode::PERMISSION_DENIED:
    case grpc::StatusCode::UNAUTHENTICATED:
        return core::ErrorCode::PermissionDenied;
    case grpc::StatusCode::RESOURCE_EXHAUSTED: return core::ErrorCode::ResourceExhausted;
    case grpc::StatusCode::FAILED_PRECONDITION:
    case grpc::StatusCode::ABORTED:
    case grpc::StatusCode::OUT_OF_RANGE:
        return core::ErrorCode::FailedPrecondition;
    case grpc::StatusCode::UNIMPLEMENTED: return core::ErrorCode::Unimplemented;
    case grpc::StatusCode::UNAVAILABLE: return core::ErrorCode::Unavailable;
    case grpc::StatusCode::INTERNAL:
    case grpc::StatusCode::DATA_LOSS:
        return core::ErrorCode::InternalError;
    case grpc::StatusCode::UNKNOWN:
    default:
        return core::ErrorCode::Unknown;
    }
}

void UpdateMaximum(std::atomic<std::uint64_t>& target, std::uint64_t value) noexcept {
    auto current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
    }
}

std::string StripJsonFence(std::string text) {
    const auto first = text.find('{');
    const auto last = text.rfind('}');
    if (first != std::string::npos && last != std::string::npos && last >= first) {
        return text.substr(first, last - first + 1);
    }
    return text;
}

std::string JsonString(const nlohmann::json& value, std::string_view key) {
    const auto it = value.find(std::string(key));
    return it != value.end() && it->is_string() ? it->get<std::string>() : std::string{};
}

std::vector<std::string> JsonStrings(const nlohmann::json& value, std::string_view key) {
    std::vector<std::string> result;
    const auto it = value.find(std::string(key));
    if (it == value.end() || !it->is_array()) {
        return result;
    }
    for (const auto& item : *it) {
        if (item.is_string()) {
            result.push_back(item.get<std::string>());
        }
    }
    return result;
}

VisionInferenceResult ParseResponse(const multimodal_inference::VLMResponse& response) {
    VisionInferenceResult result;
    result.raw_text = response.text();
    result.image_encode_ms = response.image_encode_ms();
    result.prompt_eval_ms = response.prompt_eval_ms();
    result.eval_ms = response.eval_ms();
    result.prompt_tokens = response.prompt_tokens();
    result.generated_tokens = response.generated_tokens();
    result.cache_hit = response.cache_hit();
    result.cache_stale = response.cache_stale();
    result.prompt_kv_cache_hit = response.prompt_kv_cache_hit();
    result.result_source = response.result_source();
    result.prompt_kv_near_candidate = response.prompt_kv_near_candidate();
    result.prompt_kv_near_accepted = response.prompt_kv_near_accepted();
    result.prompt_kv_near_same_session = response.prompt_kv_near_same_session();
    result.prompt_kv_global_cosine = response.prompt_kv_global_cosine();
    result.prompt_kv_mean_token_cosine = response.prompt_kv_mean_token_cosine();
    result.prompt_kv_p05_token_cosine = response.prompt_kv_p05_token_cosine();
    result.prompt_kv_min_token_cosine = response.prompt_kv_min_token_cosine();
    result.prompt_kv_relative_l2 = response.prompt_kv_relative_l2();
    result.prompt_kv_max_abs_error = response.prompt_kv_max_abs_error();

    const auto parsed = nlohmann::json::parse(StripJsonFence(response.text()), nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        result.scene_hint = response.text();
        result.weak_interpretations.push_back("VLM 返回了非结构化观察文本");
        result.confidence = response.text().empty() ? 0.0 : 0.5;
        return result;
    }

    result.scene_hint = JsonString(parsed, "scene_hint");
    result.action_hint = JsonString(parsed, "action_hint");
    result.object_hint = JsonString(parsed, "object_hint");
    result.agent_hint = JsonString(parsed, "agent_hint");
    result.memory_candidate = JsonString(parsed, "memory_candidate");
    result.facts = JsonStrings(parsed, "facts");
    result.weak_interpretations = JsonStrings(parsed, "weak_interpretations");
    if (const auto it = parsed.find("confidence"); it != parsed.end() && it->is_number()) {
        result.confidence = std::clamp(it->get<double>(), 0.0, 1.0);
    }
    if (result.scene_hint.empty() && !result.facts.empty()) {
        result.scene_hint = result.facts.front();
    }
    return result;
}

} // namespace

class GrpcVlmVisionClient::Impl {
public:
    Impl(GrpcVlmVisionClientOptions options, core::LoggerAdapter logger)
        : options_(std::move(options)),
          logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("grpc-vlm-client")) {
        grpc::ChannelArguments arguments;
        arguments.SetMaxReceiveMessageSize(static_cast<int>(std::min<std::size_t>(
            options_.max_receive_message_bytes,
            static_cast<std::size_t>(std::numeric_limits<int>::max()))));
        arguments.SetMaxSendMessageSize(static_cast<int>(std::min<std::size_t>(
            options_.max_send_message_bytes,
            static_cast<std::size_t>(std::numeric_limits<int>::max()))));
        channel_ = grpc::CreateCustomChannel(
            options_.target,
            grpc::InsecureChannelCredentials(),
            arguments);
        stub_ = multimodal_inference::MultimodalInference::NewStub(channel_);
    }

    core::Result<VisionInferenceResult> Analyze(const VisionInferenceRequest& input) {
        requests_.fetch_add(1, std::memory_order_relaxed);
        const auto started = std::chrono::steady_clock::now();
        const auto finish = [&](bool successful, bool cache_hit) {
            const auto latency_us = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - started).count());
            total_latency_us_.fetch_add(latency_us, std::memory_order_relaxed);
            UpdateMaximum(max_latency_us_, latency_us);
            if (successful) {
                successful_requests_.fetch_add(1, std::memory_order_relaxed);
            } else {
                failed_requests_.fetch_add(1, std::memory_order_relaxed);
            }
            if (cache_hit) {
                cache_hits_.fetch_add(1, std::memory_order_relaxed);
            }
        };

        if (input.session_id.empty() || input.encoded_image.empty()) {
            finish(false, false);
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "VLM request requires session_id and encoded image bytes");
        }
        if (input.mime_type != "image/jpeg" && input.mime_type != "image/png") {
            finish(false, false);
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "VLM gRPC adapter only accepts JPEG or PNG input");
        }

        multimodal_inference::VLMRequest request;
        // protobuf bytes 会立即复制这段连续缓冲区，span 生命周期不会越过 RPC 构造阶段。
        request.set_image_data(input.encoded_image.data(), input.encoded_image.size());
        request.set_prompt(input.prompt_hint.value_or(
            options_.prompt.empty() ? std::string(kDefaultPrompt) : options_.prompt));
        request.set_max_tokens(options_.max_tokens);
        request.set_context_size(options_.context_size);
        request.set_temperature(options_.temperature);
        request.set_top_p(options_.top_p);
        request.set_top_k(options_.top_k);
        request.set_session_id(input.session_id);
        request.set_request_id(input.session_id + "-frame-" + std::to_string(input.frame_id));
        request.set_task_type(options_.task_type);
        request.set_allow_cache(options_.allow_cache);
        request.set_force_refresh(options_.force_refresh);
        request.set_allow_stale_cache(options_.allow_stale_cache);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + options_.timeout);
        context.AddMetadata("x-trace-id", request.request_id());
        if (!options_.auth_token.empty()) {
            context.AddMetadata(options_.auth_metadata_key, options_.auth_token);
        }

        multimodal_inference::VLMResponse response;
        const auto status = stub_->GenerateVLMSync(&context, request, &response);
        if (!status.ok()) {
            finish(false, false);
            logger_.warn(
                "[grpc-vlm-client] inference failed session={} frame={} grpc_code={} message={}",
                input.session_id,
                input.frame_id,
                static_cast<int>(status.error_code()),
                status.error_message());
            return core::Status::Error(
                MapGrpcCode(status.error_code()),
                status.error_message().empty() ? "VLM gRPC request failed" : status.error_message());
        }
        if (!response.error().empty()) {
            finish(false, response.cache_hit());
            logger_.warn(
                "[grpc-vlm-client] inference returned business error session={} frame={} message={}",
                input.session_id,
                input.frame_id,
                response.error());
            return core::Status::Error(core::ErrorCode::InternalError, response.error());
        }

        auto result = ParseResponse(response);
        finish(true, result.cache_hit);
        return result;
    }

    GrpcVlmVisionClientSnapshot Snapshot() const {
        return {
            .requests = requests_.load(std::memory_order_relaxed),
            .successful_requests = successful_requests_.load(std::memory_order_relaxed),
            .failed_requests = failed_requests_.load(std::memory_order_relaxed),
            .cache_hits = cache_hits_.load(std::memory_order_relaxed),
            .total_latency_us = total_latency_us_.load(std::memory_order_relaxed),
            .max_latency_us = max_latency_us_.load(std::memory_order_relaxed),
        };
    }

private:
    GrpcVlmVisionClientOptions options_;
    core::LoggerAdapter logger_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<multimodal_inference::MultimodalInference::Stub> stub_;
    std::atomic<std::size_t> requests_{0};
    std::atomic<std::size_t> successful_requests_{0};
    std::atomic<std::size_t> failed_requests_{0};
    std::atomic<std::size_t> cache_hits_{0};
    std::atomic<std::uint64_t> total_latency_us_{0};
    std::atomic<std::uint64_t> max_latency_us_{0};
};

core::Result<std::unique_ptr<GrpcVlmVisionClient>> GrpcVlmVisionClient::Create(
    GrpcVlmVisionClientOptions options,
    core::LoggerAdapter logger) {
    if (options.target.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "VLM gRPC target must not be empty");
    }
    if (options.timeout <= std::chrono::milliseconds::zero() || options.max_tokens <= 0 ||
        options.context_size <= 0 || options.max_receive_message_bytes == 0 ||
        options.max_send_message_bytes == 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "invalid VLM gRPC client limits");
    }
    if (!options.auth_token.empty() && options.auth_metadata_key.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "VLM auth metadata key must not be empty");
    }
    return std::unique_ptr<GrpcVlmVisionClient>(
        new GrpcVlmVisionClient(std::make_unique<Impl>(std::move(options), std::move(logger))));
}

GrpcVlmVisionClient::GrpcVlmVisionClient(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

GrpcVlmVisionClient::~GrpcVlmVisionClient() = default;

core::Result<VisionInferenceResult> GrpcVlmVisionClient::Analyze(
    const VisionInferenceRequest& request) {
    return impl_->Analyze(request);
}

GrpcVlmVisionClientSnapshot GrpcVlmVisionClient::Snapshot() const {
    return impl_->Snapshot();
}

} // namespace media
