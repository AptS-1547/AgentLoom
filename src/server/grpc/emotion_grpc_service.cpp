#include "emotion_grpc_service.h"

#include "grpc_error.h"

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/status_code_enum.h>

#include <cstddef>
#include <utility>

namespace server::grpc_service {

namespace {

grpc::Status ToGrpc(const core::Status& status) {
    return grpc_error::ToGrpcStatus(status);
}

void SetError(multimodal_inference::EmotionResponse& response, const core::Status& status) {
    if (!status.ok() && response.error().empty()) {
        response.set_error(status.message());
    }
}

void SetError(multimodal_inference::EmotionBatchResponse& response, const core::Status& status) {
    if (!status.ok() && response.error().empty()) {
        response.set_error(status.message());
    }
}

grpc::Status Unimplemented(std::string method) {
    return grpc::Status(
        grpc::StatusCode::UNIMPLEMENTED,
        std::move(method) + " is not available in emotion_inference_server");
}

} // namespace

EmotionGrpcService::EmotionGrpcService(const MultimodalServerOptions& options,
                                       server_common::RuntimeStats& stats,
                                       service::EmotionInferenceService& service)
    : stats_(stats),
      service_(service),
      slow_request_ms_(options.grpc.slow_request_ms),
      auth_options_(options.auth),
      request_limits_(options.limits) {}

grpc::Status EmotionGrpcService::PredictEmotion(
    grpc::ServerContext* context,
    const multimodal_inference::EmotionRequest* request,
    multimodal_inference::EmotionResponse* response) {
    return grpc_error::GuardRpc([&] {
        server_common::ScopedRequestStats request_stats(stats_, "PredictEmotion", 1, false, slow_request_ms_);

        if (auto auth_status = CheckAuth(*context, request_stats); !auth_status.ok()) {
            response->set_error(auth_status.error_message());
            return auth_status;
        }
        if (auto validation = request_validation::ValidateEmotionRequest(*request, request_limits_); !validation.ok) {
            response->set_error(validation.error);
            request_stats.MarkFailure(validation.error);
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, validation.error);
        }

        auto status = service_.PredictEmotion(*request, *response);
        if (!status.ok()) {
            SetError(*response, status);
            request_stats.MarkFailure(status.message());
            return ToGrpc(status);
        }

        request_stats.MarkSuccess();
        return grpc::Status::OK;
    });
}

grpc::Status EmotionGrpcService::PredictEmotionBatch(
    grpc::ServerContext* context,
    const multimodal_inference::EmotionBatchRequest* request,
    multimodal_inference::EmotionBatchResponse* response) {
    return grpc_error::GuardRpc([&] {
        const std::size_t reported_batch_size =
            request->batch_size() > 0 ? static_cast<std::size_t>(request->batch_size()) : 0;
        server_common::ScopedRequestStats request_stats(
            stats_, "PredictEmotionBatch", reported_batch_size, true, slow_request_ms_);

        if (auto auth_status = CheckAuth(*context, request_stats); !auth_status.ok()) {
            response->set_error(auth_status.error_message());
            return auth_status;
        }
        if (auto validation = request_validation::ValidateEmotionBatchRequest(*request, request_limits_); !validation.ok) {
            response->set_error(validation.error);
            request_stats.MarkFailure(validation.error);
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, validation.error);
        }

        auto status = service_.PredictEmotionBatch(*request, *response);
        if (!status.ok()) {
            SetError(*response, status);
            request_stats.MarkFailure(status.message());
            return ToGrpc(status);
        }

        request_stats.MarkSuccess();
        return grpc::Status::OK;
    });
}

grpc::Status EmotionGrpcService::DetectSaliency(
    grpc::ServerContext*,
    const multimodal_inference::SaliencyRequest*,
    multimodal_inference::SaliencyResponse*) {
    return Unimplemented("DetectSaliency");
}

grpc::Status EmotionGrpcService::GenerateVLM(
    grpc::ServerContext*,
    const multimodal_inference::VLMRequest*,
    grpc::ServerWriter<multimodal_inference::VLMToken>*) {
    return Unimplemented("GenerateVLM");
}

grpc::Status EmotionGrpcService::GenerateVLMSync(
    grpc::ServerContext*,
    const multimodal_inference::VLMRequest*,
    multimodal_inference::VLMResponse* response) {
    if (response) {
        response->set_error("GenerateVLMSync is not available in emotion_inference_server");
    }
    return Unimplemented("GenerateVLMSync");
}

grpc::Status EmotionGrpcService::CheckAuth(
    const grpc::ServerContext& context,
    server_common::ScopedRequestStats& request_stats) const {
    auto validation = request_validation::ValidateAuth(context, auth_options_);
    if (!validation.ok) {
        request_stats.MarkFailure(validation.error);
        return grpc::Status(grpc::StatusCode::UNAUTHENTICATED, validation.error);
    }
    return grpc::Status::OK;
}

} // namespace server::grpc_service
