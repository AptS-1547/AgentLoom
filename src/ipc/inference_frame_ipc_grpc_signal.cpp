#include "inference_frame_ipc_grpc_signal.h"

#include <grpcpp/client_context.h>

#include <chrono>
#include <utility>

namespace ipc::media {
namespace {

core::ErrorCode ToCoreErrorCode(grpc::StatusCode code) noexcept {
    switch (code) {
    case grpc::StatusCode::INVALID_ARGUMENT: return core::ErrorCode::InvalidArgument;
    case grpc::StatusCode::NOT_FOUND: return core::ErrorCode::NotFound;
    case grpc::StatusCode::DEADLINE_EXCEEDED: return core::ErrorCode::Timeout;
    case grpc::StatusCode::CANCELLED: return core::ErrorCode::Cancelled;
    case grpc::StatusCode::ALREADY_EXISTS: return core::ErrorCode::AlreadyExists;
    case grpc::StatusCode::PERMISSION_DENIED:
    case grpc::StatusCode::UNAUTHENTICATED: return core::ErrorCode::PermissionDenied;
    case grpc::StatusCode::FAILED_PRECONDITION: return core::ErrorCode::FailedPrecondition;
    case grpc::StatusCode::UNIMPLEMENTED: return core::ErrorCode::Unimplemented;
    case grpc::StatusCode::RESOURCE_EXHAUSTED: return core::ErrorCode::ResourceExhausted;
    case grpc::StatusCode::UNAVAILABLE: return core::ErrorCode::Unavailable;
    default: return core::ErrorCode::InternalError;
    }
}

} // namespace

GrpcInferenceFrameIpcSignal::GrpcInferenceFrameIpcSignal(
    std::shared_ptr<grpc::ChannelInterface> channel,
    GrpcInferenceFrameIpcSignalOptions options,
    core::LoggerAdapter logger)
    : options_(std::move(options)),
      stub_(multimodal_inference::MultimodalInference::NewStub(channel)),
      logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("frame-ipc-grpc")) {}

core::Status GrpcInferenceFrameIpcSignal::ApplyGrant(const InferenceFrameIpcGrant& grant) {
    grpc::ClientContext context;
    PrepareContext(context);
    multimodal_inference::InferenceFrameIpcGrantRequest request;
    request.set_channel_name(grant.channel_name);
    request.set_epoch(grant.epoch);
    request.set_slot_count(grant.slot_count);
    request.set_payload_capacity(grant.payload_capacity);
    multimodal_inference::InferenceFrameIpcControlResponse response;
    const auto status = stub_->ApplyInferenceFrameIpcGrant(&context, request, &response);
    if (!status.ok()) {
        return RpcStatus("apply IPC grant", status);
    }
    if (response.state() != "granted" || response.epoch() != grant.epoch) {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "inference peer did not accept the granted IPC epoch");
    }
    return core::Status::Ok();
}

core::Status GrpcInferenceFrameIpcSignal::Revoke(
    std::uint64_t epoch,
    std::string_view reason) {
    grpc::ClientContext context;
    PrepareContext(context);
    multimodal_inference::InferenceFrameIpcRevokeRequest request;
    request.set_epoch(epoch);
    request.set_reason(std::string(reason));
    multimodal_inference::InferenceFrameIpcControlResponse response;
    const auto status = stub_->RevokeInferenceFrameIpcGrant(&context, request, &response);
    if (!status.ok()) {
        return RpcStatus("revoke IPC grant", status);
    }
    if (response.state() != "fenced") {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "inference peer did not revoke the IPC epoch");
    }
    return core::Status::Ok();
}

core::Status GrpcInferenceFrameIpcSignal::Probe(std::uint64_t expected_epoch) {
    grpc::ClientContext context;
    PrepareContext(context);
    multimodal_inference::InferenceFrameIpcStatusRequest request;
    multimodal_inference::InferenceFrameIpcControlResponse response;
    const auto status = stub_->GetInferenceFrameIpcStatus(&context, request, &response);
    if (!status.ok()) {
        return RpcStatus("probe IPC grant", status);
    }
    if (response.state() != "granted" || response.epoch() != expected_epoch) {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "inference peer IPC epoch is not active");
    }
    return core::Status::Ok();
}

void GrpcInferenceFrameIpcSignal::PrepareContext(grpc::ClientContext& context) const {
    context.set_deadline(std::chrono::system_clock::now() + options_.deadline);
    if (!options_.auth_token.empty()) {
        context.AddMetadata(options_.auth_metadata_key, options_.auth_token);
    }
}

core::Status GrpcInferenceFrameIpcSignal::RpcStatus(
    std::string_view operation,
    const grpc::Status& status) const {
    logger_.warn(
        "[frame-ipc-grpc] {} failed grpc_code={} message={}",
        operation,
        static_cast<int>(status.error_code()),
        status.error_message());
    return core::Status::Error(
        ToCoreErrorCode(status.error_code()),
        std::string(operation) + " failed: " + status.error_message());
}

} // namespace ipc::media
