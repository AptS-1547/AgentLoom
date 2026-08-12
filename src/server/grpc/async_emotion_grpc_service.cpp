#include "async_emotion_grpc_service.h"

namespace server::grpc_service {

AsyncEmotionGrpcService::AsyncEmotionGrpcService(
    std::shared_ptr<grpc_runtime::AsyncGrpcRuntime> runtime,
    std::shared_ptr<grpc_runtime::IAsyncUnaryRpcHandler<
        multimodal_inference::EmotionRequest,
        multimodal_inference::EmotionResponse>> handler)
    : runtime_(std::move(runtime)),
      handler_(std::move(handler)) {
    if (!runtime_) {
        throw core::AppException(
            core::ErrorCode::InvalidArgument,
            "async emotion gRPC service requires a runtime");
    }
    if (!handler_) {
        throw core::AppException(
            core::ErrorCode::InvalidArgument,
            "async emotion gRPC service requires a handler");
    }
}

grpc::ServerUnaryReactor* AsyncEmotionGrpcService::PredictEmotion(
    grpc::CallbackServerContext* context,
    const multimodal_inference::EmotionRequest* request,
    multimodal_inference::EmotionResponse* response) {
    if (context == nullptr || request == nullptr || response == nullptr) {
        return nullptr;
    }
    return runtime_->StartUnary(
        *context,
        *request,
        *response,
        handler_,
        "PredictEmotion");
}

} // namespace server::grpc_service
