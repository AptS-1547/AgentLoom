#include "multimodal_grpc_service.h"

#include "grpc_error.h"

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/status_code_enum.h>

#include <cstddef>
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

} // namespace

MultimodalGrpcService::MultimodalGrpcService(const MultimodalServerOptions& options,
                                             server_common::RuntimeStats& stats,
                                             service::IMultimodalService& service)
    : stats_(stats),
      service_(service),
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

core::Status MultimodalGrpcService::CheckAuth(const grpc::ServerContext& context) const {
    auto validation = request_validation::ValidateAuth(context, auth_options_);
    if (!validation.ok) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, validation.error);
    }
    return core::Status::Ok();
}

} // namespace server::grpc_service
