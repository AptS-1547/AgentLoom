#include "emotion_grpc_service.h"

#include "grpc_error.h"

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/status_code_enum.h>

#include <cstddef>
#include <string>
#include <utility>

namespace server::grpc_service {

namespace {

core::Status InvalidRpcArguments() {
    return core::Status::Error(core::ErrorCode::InvalidArgument, "invalid gRPC request arguments");
}

core::Status UnimplementedStatus(std::string method) {
    return core::Status::Error(
        core::ErrorCode::Unimplemented,
        std::move(method) + " is not available in emotion_inference_server");
}

template <typename Response>
void SetError(Response& response, const core::Status& status) {
    if (!status.ok() && response.error().empty()) {
        response.set_error(status.message());
    }
}

} // namespace

EmotionGrpcService::EmotionGrpcService(const MultimodalServerOptions& options,
                                       server_common::RuntimeStats& stats,
                                       service::IEmotionInferenceService& service)
    : stats_(stats),
      service_(service),
      slow_request_ms_(options.grpc.slow_request_ms),
      auth_options_(options.auth),
      request_limits_(options.limits) {}

grpc::Status EmotionGrpcService::PredictEmotion(
    grpc::ServerContext* context,
    const multimodal_inference::EmotionRequest* request,
    multimodal_inference::EmotionResponse* response) {
    grpc_error::RpcCall rpc(*context, stats_, "PredictEmotion", 1, false, slow_request_ms_);
    return rpc.Run([&] {
        if (!request || !response) {
            return rpc.Failure(InvalidRpcArguments());
        }
        if (auto status = CheckAuth(*context); !status.ok()) {
            SetError(*response, status);
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        if (auto validation = request_validation::ValidateEmotionRequest(*request, request_limits_); !validation.ok) {
            const auto status = core::Status::Error(core::ErrorCode::InvalidArgument, validation.error);
            SetError(*response, status);
            return rpc.Failure(status);
        }
        if (auto status = rpc.CancellationStatus(); !status.ok()) {
            return rpc.Failure(status);
        }

        auto status = service_.PredictEmotion(*request, *response);
        if (!status.ok()) {
            SetError(*response, status);
            return rpc.Failure(status);
        }
        return rpc.Success();
    });
}

grpc::Status EmotionGrpcService::PredictEmotionBatch(
    grpc::ServerContext* context,
    const multimodal_inference::EmotionBatchRequest* request,
    multimodal_inference::EmotionBatchResponse* response) {
    const std::size_t reported_batch_size = request && request->batch_size() > 0
        ? static_cast<std::size_t>(request->batch_size())
        : 0;
    grpc_error::RpcCall rpc(
        *context,
        stats_,
        "PredictEmotionBatch",
        reported_batch_size,
        true,
        slow_request_ms_);
    return rpc.Run([&] {
        if (!request || !response) {
            return rpc.Failure(InvalidRpcArguments());
        }
        if (auto status = CheckAuth(*context); !status.ok()) {
            SetError(*response, status);
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        if (auto validation = request_validation::ValidateEmotionBatchRequest(*request, request_limits_); !validation.ok) {
            const auto status = core::Status::Error(core::ErrorCode::InvalidArgument, validation.error);
            SetError(*response, status);
            return rpc.Failure(status);
        }
        if (auto status = rpc.CancellationStatus(); !status.ok()) {
            return rpc.Failure(status);
        }

        auto status = service_.PredictEmotionBatch(*request, *response);
        if (!status.ok()) {
            SetError(*response, status);
            return rpc.Failure(status);
        }
        return rpc.Success();
    });
}

grpc::Status EmotionGrpcService::DetectSaliency(
    grpc::ServerContext* context,
    const multimodal_inference::SaliencyRequest*,
    multimodal_inference::SaliencyResponse*) {
    grpc_error::RpcCall rpc(*context, stats_, "DetectSaliency", 1, false, slow_request_ms_);
    return rpc.Run([&] {
        if (auto status = CheckAuth(*context); !status.ok()) {
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        return rpc.Failure(UnimplementedStatus("DetectSaliency"));
    });
}

grpc::Status EmotionGrpcService::GenerateVLM(
    grpc::ServerContext* context,
    const multimodal_inference::VLMRequest* request,
    grpc::ServerWriter<multimodal_inference::VLMToken>*) {
    grpc_error::RpcLogContext log_context;
    if (request) {
        log_context.request_id = request->request_id();
        log_context.session_id = request->session_id();
        log_context.task_type = request->task_type();
    }
    grpc_error::RpcCall rpc(
        *context,
        stats_,
        "GenerateVLM",
        1,
        false,
        slow_request_ms_,
        std::move(log_context));
    return rpc.Run([&] {
        if (auto status = CheckAuth(*context); !status.ok()) {
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        return rpc.Failure(UnimplementedStatus("GenerateVLM"));
    });
}

grpc::Status EmotionGrpcService::GenerateVLMSync(
    grpc::ServerContext* context,
    const multimodal_inference::VLMRequest* request,
    multimodal_inference::VLMResponse* response) {
    grpc_error::RpcLogContext log_context;
    if (request) {
        log_context.request_id = request->request_id();
        log_context.session_id = request->session_id();
        log_context.task_type = request->task_type();
    }
    grpc_error::RpcCall rpc(
        *context,
        stats_,
        "GenerateVLMSync",
        1,
        false,
        slow_request_ms_,
        std::move(log_context));
    return rpc.Run([&] {
        if (!response) {
            return rpc.Failure(InvalidRpcArguments());
        }
        if (auto status = CheckAuth(*context); !status.ok()) {
            SetError(*response, status);
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        const auto status = UnimplementedStatus("GenerateVLMSync");
        SetError(*response, status);
        return rpc.Failure(status);
    });
}

core::Status EmotionGrpcService::CheckAuth(const grpc::ServerContext& context) const {
    auto validation = request_validation::ValidateAuth(context, auth_options_);
    if (!validation.ok) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, validation.error);
    }
    return core::Status::Ok();
}

} // namespace server::grpc_service
