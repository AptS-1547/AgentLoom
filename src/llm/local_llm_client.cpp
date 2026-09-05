#include "local_llm_client.h"

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <sstream>
#include <utility>

namespace agent::llm {

namespace {

core::ErrorCode FromGrpcCode(grpc::StatusCode code) noexcept {
    switch (code) {
    case grpc::StatusCode::OK: return core::ErrorCode::Ok;
    case grpc::StatusCode::INVALID_ARGUMENT: return core::ErrorCode::InvalidArgument;
    case grpc::StatusCode::NOT_FOUND: return core::ErrorCode::NotFound;
    case grpc::StatusCode::DEADLINE_EXCEEDED: return core::ErrorCode::Timeout;
    case grpc::StatusCode::CANCELLED: return core::ErrorCode::Cancelled;
    case grpc::StatusCode::ALREADY_EXISTS: return core::ErrorCode::AlreadyExists;
    case grpc::StatusCode::PERMISSION_DENIED:
    case grpc::StatusCode::UNAUTHENTICATED:
        return core::ErrorCode::PermissionDenied;
    case grpc::StatusCode::FAILED_PRECONDITION: return core::ErrorCode::FailedPrecondition;
    case grpc::StatusCode::UNIMPLEMENTED: return core::ErrorCode::Unimplemented;
    case grpc::StatusCode::RESOURCE_EXHAUSTED: return core::ErrorCode::ResourceExhausted;
    case grpc::StatusCode::UNAVAILABLE: return core::ErrorCode::Unavailable;
    default: return core::ErrorCode::InternalError;
    }
}

std::string RoleLabel(ChatRole role) {
    switch (role) {
    case ChatRole::System: return "system";
    case ChatRole::User: return "user";
    case ChatRole::Assistant: return "assistant";
    case ChatRole::Tool: return "tool";
    }
    return "user";
}

} // namespace

GrpcLocalLlmClient::GrpcLocalLlmClient(GrpcLocalLlmClientOptions options)
    : GrpcLocalLlmClient(
          options,
          grpc::CreateChannel(options.target, grpc::InsecureChannelCredentials())) {}

GrpcLocalLlmClient::GrpcLocalLlmClient(GrpcLocalLlmClientOptions options,
                                       std::shared_ptr<grpc::Channel> channel)
    : options_(std::move(options)),
      stub_(multimodal_inference::MultimodalInference::NewStub(std::move(channel))) {}

core::Result<LocalLlmResponse> GrpcLocalLlmClient::Generate(const LocalLlmRequest& request) {
    if (!stub_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "local llm grpc stub is not initialized");
    }
    if (request.prompt.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "local llm prompt is required");
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + options_.deadline);
    if (!options_.auth_token.empty()) {
        context.AddMetadata(options_.auth_metadata_key, options_.auth_token);
    }

    auto grpc_request = BuildGrpcRequest(request);
    multimodal_inference::VLMResponse grpc_response;
    auto status = stub_->GenerateVLMSync(&context, grpc_request, &grpc_response);
    if (!status.ok()) {
        return FromGrpcStatus(status);
    }
    if (!grpc_response.error().empty()) {
        return core::Status::Error(core::ErrorCode::InternalError, grpc_response.error());
    }

    LocalLlmResponse response;
    response.text = grpc_response.text();
    response.prompt_tokens = grpc_response.prompt_tokens();
    response.generated_tokens = grpc_response.generated_tokens();
    response.prompt_eval_ms = grpc_response.prompt_eval_ms();
    response.eval_ms = grpc_response.eval_ms();
    response.result_source = grpc_response.result_source();
    return response;
}

core::Status GrpcLocalLlmClient::FromGrpcStatus(const grpc::Status& status) {
    if (status.ok()) {
        return core::Status::Ok();
    }
    return core::Status::Error(FromGrpcCode(status.error_code()), status.error_message());
}

multimodal_inference::VLMRequest GrpcLocalLlmClient::BuildGrpcRequest(const LocalLlmRequest& request) const {
    multimodal_inference::VLMRequest grpc_request;
    grpc_request.set_prompt(request.prompt);
    grpc_request.set_max_tokens(request.max_tokens);
    grpc_request.set_temperature(request.temperature);
    grpc_request.set_top_p(request.top_p);
    grpc_request.set_top_k(request.top_k);
    grpc_request.set_context_size(request.context_size);
    grpc_request.set_session_id(request.session_id);
    grpc_request.set_request_id(request.request_id);
    grpc_request.set_task_type(request.task_type);
    grpc_request.set_allow_cache(false);
    grpc_request.set_force_refresh(false);
    grpc_request.set_allow_stale_cache(false);
    return grpc_request;
}

LocalLlmChatClient::LocalLlmChatClient(std::shared_ptr<ILocalLlm> local_llm,
                                       LocalLlmChatClientOptions options)
    : local_llm_(std::move(local_llm)),
      options_(std::move(options)) {}

core::Result<ChatCompletionResponse> LocalLlmChatClient::Complete(const ChatCompletionRequest& req) {
    if (!local_llm_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "local llm backend is required");
    }

    LocalLlmRequest local_request;
    local_request.prompt = BuildPrompt(req);
    local_request.task_type = options_.task_type;
    local_request.max_tokens = req.max_tokens;
    local_request.temperature = req.temperature;
    local_request.top_p = req.top_p;

    auto generated = local_llm_->Generate(local_request);
    if (!generated.ok()) {
        return generated.status();
    }

    ChatCompletionResponse response;
    response.model = req.model.empty() ? options_.default_model : req.model;
    response.content = generated.value().text;
    response.prompt_tokens = generated.value().prompt_tokens;
    response.completion_tokens = generated.value().generated_tokens;
    response.total_tokens = response.prompt_tokens + response.completion_tokens;
    return response;
}

std::string LocalLlmChatClient::BuildPrompt(const ChatCompletionRequest& req) {
    std::ostringstream out;
    for (const auto& message : req.messages) {
        out << '<' << RoleLabel(message.role) << ">\n";
        out << message.content << "\n";
        out << "</" << RoleLabel(message.role) << ">\n";
    }
    out << "<assistant>\n";
    return out.str();
}

} // namespace agent::llm
