#include "async_emotion_grpc_service.h"

namespace server::grpc_service {

AsyncEmotionGrpcService::AsyncEmotionGrpcService(
    std::shared_ptr<grpc_runtime::AsyncGrpcRuntime> runtime,
    std::shared_ptr<grpc_runtime::IAsyncUnaryRpcHandler<
        multimodal_inference::EmotionRequest,
        multimodal_inference::EmotionResponse>> emotion_handler,
    std::shared_ptr<grpc_runtime::IAsyncUnaryRpcHandler<
        multimodal_inference::EmotionBatchRequest,
        multimodal_inference::EmotionBatchResponse>> batch_handler)
    : runtime_(std::move(runtime)),
      emotion_handler_(std::move(emotion_handler)),
      batch_handler_(std::move(batch_handler)) {
    if (!runtime_) {
        throw core::AppException(
            core::ErrorCode::InvalidArgument,
            "async emotion gRPC service requires a runtime");
    }
    if (!emotion_handler_) {
        throw core::AppException(
            core::ErrorCode::InvalidArgument,
            "async emotion gRPC service requires an emotion handler");
    }
}

grpc::ServerUnaryReactor* AsyncEmotionGrpcService::PredictEmotion(
    grpc::CallbackServerContext* context,
    const multimodal_inference::EmotionRequest* request,
    multimodal_inference::EmotionResponse* response) {
    if (context == nullptr || request == nullptr || response == nullptr) {
        return nullptr;
    }
    if (!batch_handler_) {
        response->set_error("async emotion batch handler is not configured");
    }
    return runtime_->StartUnary(
        *context,
        *request,
        *response,
        emotion_handler_,
        "PredictEmotion");
}

grpc::ServerUnaryReactor* AsyncEmotionGrpcService::PredictEmotionBatch(
    grpc::CallbackServerContext* context,
    const multimodal_inference::EmotionBatchRequest* request,
    multimodal_inference::EmotionBatchResponse* response) {
    if (context == nullptr || request == nullptr || response == nullptr) {
        return nullptr;
    }
    return runtime_->StartUnary(
        *context,
        *request,
        *response,
        batch_handler_,
        "PredictEmotionBatch");
}

} // namespace server::grpc_service
