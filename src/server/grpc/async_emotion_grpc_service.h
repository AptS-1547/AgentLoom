#pragma once

#include "async_grpc_runtime.h"

#include "multimodal_inference.grpc.pb.h"

#include <memory>

namespace server::grpc_service {

using AsyncEmotionGrpcServiceBase =
    multimodal_inference::MultimodalInference::WithCallbackMethod_PredictEmotion<
        multimodal_inference::MultimodalInference::WithCallbackMethod_PredictEmotionBatch<
            multimodal_inference::MultimodalInference::Service>>;

/// 通用异步运行时在 Emotion RPC 上的接入示例。
class AsyncEmotionGrpcService final : public AsyncEmotionGrpcServiceBase {
public:
    AsyncEmotionGrpcService(
        std::shared_ptr<grpc_runtime::AsyncGrpcRuntime> runtime,
        std::shared_ptr<grpc_runtime::IAsyncUnaryRpcHandler<
            multimodal_inference::EmotionRequest,
            multimodal_inference::EmotionResponse>> emotion_handler,
        std::shared_ptr<grpc_runtime::IAsyncUnaryRpcHandler<
            multimodal_inference::EmotionBatchRequest,
            multimodal_inference::EmotionBatchResponse>> batch_handler = nullptr);

    grpc::ServerUnaryReactor* PredictEmotion(
        grpc::CallbackServerContext* context,
        const multimodal_inference::EmotionRequest* request,
        multimodal_inference::EmotionResponse* response) override;

    grpc::ServerUnaryReactor* PredictEmotionBatch(
        grpc::CallbackServerContext* context,
        const multimodal_inference::EmotionBatchRequest* request,
        multimodal_inference::EmotionBatchResponse* response) override;

private:
    std::shared_ptr<grpc_runtime::AsyncGrpcRuntime> runtime_;
    std::shared_ptr<grpc_runtime::IAsyncUnaryRpcHandler<
        multimodal_inference::EmotionRequest,
        multimodal_inference::EmotionResponse>> emotion_handler_;
    std::shared_ptr<grpc_runtime::IAsyncUnaryRpcHandler<
        multimodal_inference::EmotionBatchRequest,
        multimodal_inference::EmotionBatchResponse>> batch_handler_;
};

} // namespace server::grpc_service
