#pragma once

#include "multimodal_service.h"
#include "request_validation.h"
#include "server_common.h"
#include "server_options.h"
#include "inference_frame_ipc_control.h"
#include "multimodal_inference.grpc.pb.h"

namespace server::grpc_service {

class MultimodalGrpcService final : public multimodal_inference::MultimodalInference::Service {
public:
    MultimodalGrpcService(const MultimodalServerOptions& options,
                          server_common::RuntimeStats& stats,
                          service::IMultimodalService& service,
                          ipc::media::IInferenceFrameIpcGrantReceiver* ipc_control = nullptr);

    grpc::Status PredictEmotion(grpc::ServerContext* context,
                                const multimodal_inference::EmotionRequest* request,
                                multimodal_inference::EmotionResponse* response) override;

    grpc::Status PredictEmotionBatch(grpc::ServerContext* context,
                                     const multimodal_inference::EmotionBatchRequest* request,
                                     multimodal_inference::EmotionBatchResponse* response) override;

    grpc::Status DetectSaliency(grpc::ServerContext* context,
                                const multimodal_inference::SaliencyRequest* request,
                                multimodal_inference::SaliencyResponse* response) override;

    grpc::Status GenerateVLM(grpc::ServerContext* context,
                             const multimodal_inference::VLMRequest* request,
                             grpc::ServerWriter<multimodal_inference::VLMToken>* writer) override;

    grpc::Status GenerateVLMSync(grpc::ServerContext* context,
                                 const multimodal_inference::VLMRequest* request,
                                 multimodal_inference::VLMResponse* response) override;

    grpc::Status ApplyInferenceFrameIpcGrant(
        grpc::ServerContext* context,
        const multimodal_inference::InferenceFrameIpcGrantRequest* request,
        multimodal_inference::InferenceFrameIpcControlResponse* response) override;

    grpc::Status RevokeInferenceFrameIpcGrant(
        grpc::ServerContext* context,
        const multimodal_inference::InferenceFrameIpcRevokeRequest* request,
        multimodal_inference::InferenceFrameIpcControlResponse* response) override;

    grpc::Status GetInferenceFrameIpcStatus(
        grpc::ServerContext* context,
        const multimodal_inference::InferenceFrameIpcStatusRequest* request,
        multimodal_inference::InferenceFrameIpcControlResponse* response) override;

private:
    core::Status CheckAuth(const grpc::ServerContext& context) const;

    server_common::RuntimeStats& stats_;
    service::IMultimodalService& service_;
    ipc::media::IInferenceFrameIpcGrantReceiver* ipc_control_ = nullptr;
    int slow_request_ms_ = 250;
    request_validation::AuthOptions auth_options_;
    request_validation::RequestLimits request_limits_;
};

} // namespace server::grpc_service
