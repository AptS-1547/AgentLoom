#include "async_emotion_inference_handler.h"

#include <chrono>
#include <utility>
#include <vector>

namespace server::grpc_service {
namespace {

template <typename Response>
void SetError(Response& response, const core::Status& status) {
    if (!status.ok() && response.error().empty()) {
        response.set_error(status.message());
    }
}

std::vector<std::pair<std::string, std::string>> CopyMetadata(
    const grpc_runtime::AsyncGrpcCallContext& context) {
    std::vector<std::pair<std::string, std::string>> metadata;
    metadata.reserve(context.metadata.size());
    for (const auto& entry : context.metadata) {
        metadata.emplace_back(entry.key, entry.value);
    }
    return metadata;
}

}

AsyncEmotionInferenceHandler::AsyncEmotionInferenceHandler(
    const MultimodalServerOptions& options,
    server_common::RuntimeStats& stats,
    std::shared_ptr<service::IEmotionInferenceService> service)
    : stats_(stats),
      service_(std::move(service)),
      slow_request_ms_(options.grpc.slow_request_ms),
      auth_options_(options.auth),
      request_limits_(options.limits) {}

core::Status AsyncEmotionInferenceHandler::Handle(
    const grpc_runtime::AsyncGrpcCallContext& context,
    const multimodal_inference::EmotionRequest& request,
    multimodal_inference::EmotionResponse& response) {
    server_common::ScopedRequestStats stats(
        stats_, context.method_name, 1, false, slow_request_ms_, context.trace_id);
    if (!service_) {
        const auto status = core::Status::Error(core::ErrorCode::FailedPrecondition,
                                                 "emotion inference service is unavailable");
        SetError(response, status);
        stats.MarkFailure(status.message());
        return status;
    }
    if (const auto status = CheckAuth(context); !status.ok()) {
        SetError(response, status);
        stats.MarkFailure(status.message());
        return status;
    }
    if (const auto validation = request_validation::ValidateEmotionRequest(request, request_limits_);
        !validation.ok) {
        const auto status = core::Status::Error(core::ErrorCode::InvalidArgument, validation.error);
        SetError(response, status);
        stats.MarkFailure(status.message());
        return status;
    }
    if (const auto status = CancellationStatus(context); !status.ok()) {
        SetError(response, status);
        stats.MarkFailure(status.message());
        return status;
    }
    const auto status = service_->PredictEmotion(request, response);
    if (!status.ok()) {
        SetError(response, status);
        stats.MarkFailure(status.message());
        return status;
    }
    if (const auto cancellation = CancellationStatus(context); !cancellation.ok()) {
        SetError(response, cancellation);
        stats.MarkFailure(cancellation.message());
        return cancellation;
    }
    stats.MarkSuccess();
    return core::Status::Ok();
}

core::Status AsyncEmotionInferenceHandler::Handle(
    const grpc_runtime::AsyncGrpcCallContext& context,
    const multimodal_inference::EmotionBatchRequest& request,
    multimodal_inference::EmotionBatchResponse& response) {
    const auto sample_count = request.batch_size() > 0
        ? static_cast<std::size_t>(request.batch_size())
        : 0;
    server_common::ScopedRequestStats stats(
        stats_, context.method_name, sample_count, true, slow_request_ms_, context.trace_id);
    if (!service_) {
        const auto status = core::Status::Error(core::ErrorCode::FailedPrecondition,
                                                 "emotion inference service is unavailable");
        SetError(response, status);
        stats.MarkFailure(status.message());
        return status;
    }
    if (const auto status = CheckAuth(context); !status.ok()) {
        SetError(response, status);
        stats.MarkFailure(status.message());
        return status;
    }
    if (const auto validation = request_validation::ValidateEmotionBatchRequest(request, request_limits_);
        !validation.ok) {
        const auto status = core::Status::Error(core::ErrorCode::InvalidArgument, validation.error);
        SetError(response, status);
        stats.MarkFailure(status.message());
        return status;
    }
    if (const auto status = CancellationStatus(context); !status.ok()) {
        SetError(response, status);
        stats.MarkFailure(status.message());
        return status;
    }
    const auto status = service_->PredictEmotionBatch(request, response);
    if (!status.ok()) {
        SetError(response, status);
        stats.MarkFailure(status.message());
        return status;
    }
    if (const auto cancellation = CancellationStatus(context); !cancellation.ok()) {
        SetError(response, cancellation);
        stats.MarkFailure(cancellation.message());
        return cancellation;
    }
    stats.MarkSuccess();
    return core::Status::Ok();
}

core::Status AsyncEmotionInferenceHandler::CheckAuth(
    const grpc_runtime::AsyncGrpcCallContext& context) const {
    const auto validation = request_validation::ValidateAuthMetadata(CopyMetadata(context), auth_options_);
    if (!validation.ok) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, validation.error);
    }
    return core::Status::Ok();
}

core::Status AsyncEmotionInferenceHandler::CancellationStatus(
    const grpc_runtime::AsyncGrpcCallContext& context) {
    if (context.stop_token.stop_requested()) {
        return core::Status::Error(core::ErrorCode::Cancelled, "async gRPC call cancelled");
    }
    if (std::chrono::system_clock::now() >= context.deadline) {
        return core::Status::Error(core::ErrorCode::Timeout, "async gRPC deadline exceeded");
    }
    return core::Status::Ok();
}

}
