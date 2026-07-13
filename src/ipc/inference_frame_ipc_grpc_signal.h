#pragma once

#include "inference_frame_ipc_control.h"
#include "multimodal_inference.grpc.pb.h"

#include <chrono>
#include <memory>
#include <string>

namespace ipc::media {

struct GrpcInferenceFrameIpcSignalOptions {
    std::chrono::milliseconds deadline{3000};
    std::string auth_metadata_key = "authorization";
    std::string auth_token;
};

class GrpcInferenceFrameIpcSignal final : public IInferenceFrameIpcControlSignal {
public:
    explicit GrpcInferenceFrameIpcSignal(
        std::shared_ptr<grpc::ChannelInterface> channel,
        GrpcInferenceFrameIpcSignalOptions options = {},
        core::LoggerAdapter logger = {});

    core::Status ApplyGrant(const InferenceFrameIpcGrant& grant) override;
    core::Status Revoke(std::uint64_t epoch, std::string_view reason) override;
    core::Status Probe(std::uint64_t expected_epoch) override;

private:
    void PrepareContext(grpc::ClientContext& context) const;
    core::Status RpcStatus(std::string_view operation, const grpc::Status& status) const;

    GrpcInferenceFrameIpcSignalOptions options_;
    std::unique_ptr<multimodal_inference::MultimodalInference::Stub> stub_;
    core::LoggerAdapter logger_;
};

} // namespace ipc::media
