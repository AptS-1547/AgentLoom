#include "multimodal_grpc_service.h"

#include "grpc_error.h"

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/status_code_enum.h>

#include <cstddef>
#include <limits>
#include <utility>

namespace server::grpc_service {

namespace {

core::Status InvalidRpcArguments() {
    return core::Status::Error(core::ErrorCode::InvalidArgument, "invalid gRPC request arguments");
}

template <typename Response>
void SetError(Response& response, const core::Status& status) {
    if (!status.ok() && response.error().empty()) {
        response.set_error(status.message());
    }
}

grpc_error::RpcLogContext VlmLogContext(const multimodal_inference::VLMRequest* request) {
    if (!request) {
        return {};
    }
    return grpc_error::RpcLogContext{
        .request_id = request->request_id(),
        .session_id = request->session_id(),
        .task_type = request->task_type(),
    };
}

const char* IpcControlStateName(ipc::media::InferenceFrameIpcControlState state) noexcept {
    using State = ipc::media::InferenceFrameIpcControlState;
    switch (state) {
    case State::Idle: return "idle";
    case State::Granted: return "granted";
    case State::Fenced: return "fenced";
    case State::Recovering: return "recovering";
    case State::Failed: return "failed";
    case State::Shutdown: return "shutdown";
    }
    return "unknown";
}

void FillIpcControlResponse(
    const ipc::media::InferenceFrameIpcControlSnapshot& snapshot,
    multimodal_inference::InferenceFrameIpcControlResponse& response) {
    response.set_state(IpcControlStateName(snapshot.state));
    response.set_channel_name(snapshot.grant.channel_name);
    response.set_epoch(snapshot.grant.epoch);
    response.set_slot_count(snapshot.grant.slot_count);
    response.set_payload_capacity(snapshot.grant.payload_capacity);
    response.set_grants_applied(snapshot.grants_applied);
    response.set_revocations(snapshot.revocations);
    response.set_recoveries(snapshot.recoveries);
    response.set_failures(snapshot.failures);
    if (!snapshot.last_status.ok()) {
        response.set_error(snapshot.last_status.message());
    }
}

} // namespace

MultimodalGrpcService::MultimodalGrpcService(const MultimodalServerOptions& options,
                                             server_common::RuntimeStats& stats,
                                             service::IMultimodalService& service,
                                             ipc::media::IInferenceFrameIpcGrantReceiver* ipc_control)
    : stats_(stats),
      service_(service),
      ipc_control_(ipc_control),
      slow_request_ms_(options.grpc.slow_request_ms),
      auth_options_(options.auth),
      request_limits_(options.limits) {}

grpc::Status MultimodalGrpcService::PredictEmotion(
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

grpc::Status MultimodalGrpcService::PredictEmotionBatch(
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

grpc::Status MultimodalGrpcService::DetectSaliency(
    grpc::ServerContext* context,
    const multimodal_inference::SaliencyRequest* request,
    multimodal_inference::SaliencyResponse* response) {
    grpc_error::RpcCall rpc(*context, stats_, "DetectSaliency", 1, false, slow_request_ms_);
    return rpc.Run([&] {
        if (!request || !response) {
            return rpc.Failure(InvalidRpcArguments());
        }
        if (auto status = CheckAuth(*context); !status.ok()) {
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        if (auto validation = request_validation::ValidateSaliencyRequest(*request, request_limits_); !validation.ok) {
            return rpc.Failure(core::Status::Error(core::ErrorCode::InvalidArgument, validation.error));
        }
        if (auto status = rpc.CancellationStatus(); !status.ok()) {
            return rpc.Failure(status);
        }

        auto status = service_.DetectSaliency(*request, *response);
        if (!status.ok()) {
            return rpc.Failure(status);
        }
        return rpc.Success();
    });
}

grpc::Status MultimodalGrpcService::GenerateVLM(
    grpc::ServerContext* context,
    const multimodal_inference::VLMRequest* request,
    grpc::ServerWriter<multimodal_inference::VLMToken>* writer) {
    grpc_error::RpcCall rpc(
        *context,
        stats_,
        "GenerateVLM",
        1,
        false,
        slow_request_ms_,
        VlmLogContext(request));
    return rpc.Run([&] {
        if (!request || !writer) {
            return rpc.Failure(InvalidRpcArguments());
        }
        if (auto status = CheckAuth(*context); !status.ok()) {
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        if (auto validation = request_validation::ValidateVLMRequest(*request, request_limits_); !validation.ok) {
            return rpc.Failure(core::Status::Error(core::ErrorCode::InvalidArgument, validation.error));
        }
        if (auto status = rpc.CancellationStatus(); !status.ok()) {
            return rpc.Failure(status);
        }

        auto write_status = core::Status::Ok();
        auto status = service_.GenerateVLM(*request, [&](multimodal_inference::VLMToken token) {
            if (!write_status.ok()) {
                return;
            }
            if (auto cancelled = rpc.CancellationStatus(); !cancelled.ok()) {
                write_status = std::move(cancelled);
                return;
            }
            if (!writer->Write(token)) {
                auto cancellation = rpc.CancellationStatus();
                write_status = cancellation.ok()
                    ? core::Status::Error(
                          core::ErrorCode::Cancelled,
                          "failed to write VLM token to gRPC stream")
                    : std::move(cancellation);
            }
        });
        if (!write_status.ok()) {
            return rpc.Failure(write_status);
        }
        if (!status.ok()) {
            return rpc.Failure(status);
        }
        return rpc.Success();
    });
}

grpc::Status MultimodalGrpcService::GenerateVLMSync(
    grpc::ServerContext* context,
    const multimodal_inference::VLMRequest* request,
    multimodal_inference::VLMResponse* response) {
    grpc_error::RpcCall rpc(
        *context,
        stats_,
        "GenerateVLMSync",
        1,
        false,
        slow_request_ms_,
        VlmLogContext(request));
    return rpc.Run([&] {
        if (!request || !response) {
            return rpc.Failure(InvalidRpcArguments());
        }
        if (auto status = CheckAuth(*context); !status.ok()) {
            SetError(*response, status);
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        if (auto validation = request_validation::ValidateVLMRequest(*request, request_limits_); !validation.ok) {
            const auto status = core::Status::Error(core::ErrorCode::InvalidArgument, validation.error);
            SetError(*response, status);
            return rpc.Failure(status);
        }
        if (auto status = rpc.CancellationStatus(); !status.ok()) {
            return rpc.Failure(status);
        }

        auto status = service_.GenerateVLMSync(*request, *response);
        if (!status.ok()) {
            SetError(*response, status);
            return rpc.Failure(status);
        }
        return rpc.Success();
    });
}

grpc::Status MultimodalGrpcService::ApplyInferenceFrameIpcGrant(
    grpc::ServerContext* context,
    const multimodal_inference::InferenceFrameIpcGrantRequest* request,
    multimodal_inference::InferenceFrameIpcControlResponse* response) {
    grpc_error::RpcCall rpc(*context, stats_, "ApplyInferenceFrameIpcGrant", 1, false, slow_request_ms_);
    return rpc.Run([&] {
        if (!request || !response) {
            return rpc.Failure(InvalidRpcArguments());
        }
        if (auto status = CheckAuth(*context); !status.ok()) {
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        if (!ipc_control_) {
            return rpc.Failure(core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "inference frame IPC control is not configured"));
        }
        if (request->slot_count() > std::numeric_limits<std::size_t>::max() ||
            request->payload_capacity() > std::numeric_limits<std::size_t>::max()) {
            return rpc.Failure(core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "inference frame IPC grant exceeds platform limits"));
        }
        const ipc::media::InferenceFrameIpcGrant grant{
            .channel_name = request->channel_name(),
            .epoch = request->epoch(),
            .slot_count = static_cast<std::size_t>(request->slot_count()),
            .payload_capacity = static_cast<std::size_t>(request->payload_capacity()),
        };
        const auto status = ipc_control_->ApplyGrant(grant);
        FillIpcControlResponse(ipc_control_->ControlSnapshot(), *response);
        if (!status.ok()) {
            response->set_error(status.message());
            return rpc.Failure(status);
        }
        return rpc.Success();
    });
}

grpc::Status MultimodalGrpcService::RevokeInferenceFrameIpcGrant(
    grpc::ServerContext* context,
    const multimodal_inference::InferenceFrameIpcRevokeRequest* request,
    multimodal_inference::InferenceFrameIpcControlResponse* response) {
    grpc_error::RpcCall rpc(*context, stats_, "RevokeInferenceFrameIpcGrant", 1, false, slow_request_ms_);
    return rpc.Run([&] {
        if (!request || !response) {
            return rpc.Failure(InvalidRpcArguments());
        }
        if (auto status = CheckAuth(*context); !status.ok()) {
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        if (!ipc_control_) {
            return rpc.Failure(core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "inference frame IPC control is not configured"));
        }
        const auto status = ipc_control_->Revoke(request->epoch(), request->reason());
        FillIpcControlResponse(ipc_control_->ControlSnapshot(), *response);
        if (!status.ok()) {
            response->set_error(status.message());
            return rpc.Failure(status);
        }
        return rpc.Success();
    });
}

grpc::Status MultimodalGrpcService::GetInferenceFrameIpcStatus(
    grpc::ServerContext* context,
    const multimodal_inference::InferenceFrameIpcStatusRequest* request,
    multimodal_inference::InferenceFrameIpcControlResponse* response) {
    grpc_error::RpcCall rpc(*context, stats_, "GetInferenceFrameIpcStatus", 1, false, slow_request_ms_);
    return rpc.Run([&] {
        if (!request || !response) {
            return rpc.Failure(InvalidRpcArguments());
        }
        if (auto status = CheckAuth(*context); !status.ok()) {
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        if (!ipc_control_) {
            return rpc.Failure(core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "inference frame IPC control is not configured"));
        }
        FillIpcControlResponse(ipc_control_->ControlSnapshot(), *response);
        return rpc.Success();
    });
}

core::Status MultimodalGrpcService::CheckAuth(const grpc::ServerContext& context) const {
    auto validation = request_validation::ValidateAuth(context, auth_options_);
    if (!validation.ok) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, validation.error);
    }
    return core::Status::Ok();
}

} // namespace server::grpc_service
