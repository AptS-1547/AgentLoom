#pragma once

#include "async_grpc_runtime.h"
#include "emotion_inference_service.h"
#include "request_options.h"
#include "request_validation.h"
#include "server_common.h"

#include "multimodal_inference.pb.h"

#include <memory>

namespace server::grpc_service {

/// 将情绪推理业务接入 callback gRPC runtime，并保留同步服务已有的校验、认证与统计语义。
class AsyncEmotionInferenceHandler final
    : public grpc_runtime::IAsyncUnaryRpcHandler<
          multimodal_inference::EmotionRequest,
          multimodal_inference::EmotionResponse>,
      public grpc_runtime::IAsyncUnaryRpcHandler<
          multimodal_inference::EmotionBatchRequest,
          multimodal_inference::EmotionBatchResponse> {
public:
    AsyncEmotionInferenceHandler(const MultimodalServerOptions& options,
                                 server_common::RuntimeStats& stats,
                                 std::shared_ptr<service::IEmotionInferenceService> service);

    core::Status Handle(const grpc_runtime::AsyncGrpcCallContext& context,
                        const multimodal_inference::EmotionRequest& request,
                        multimodal_inference::EmotionResponse& response) override;

    core::Status Handle(const grpc_runtime::AsyncGrpcCallContext& context,
                        const multimodal_inference::EmotionBatchRequest& request,
                        multimodal_inference::EmotionBatchResponse& response) override;

private:
    core::Status CheckAuth(const grpc_runtime::AsyncGrpcCallContext& context) const;
    static core::Status CancellationStatus(const grpc_runtime::AsyncGrpcCallContext& context);

    server_common::RuntimeStats& stats_;
    std::shared_ptr<service::IEmotionInferenceService> service_;
    int slow_request_ms_ = 250;
    request_validation::AuthOptions auth_options_;
    request_validation::RequestLimits request_limits_;
};

}
