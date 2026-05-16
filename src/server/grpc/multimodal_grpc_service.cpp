#include "multimodal_grpc_service.h"

#include "grpc_error.h"

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/status_code_enum.h>

#include <cstddef>
#include <utility>

namespace server::grpc_service {

namespace {

grpc::Status ToGrpc(const core::Status& status) {
    return grpc_error::ToGrpcStatus(status);
}

void SetError(multimodal_inference::EmotionResponse& response, const core::Status& status) {
    if (!status.ok() && response.error().empty()) {
        response.set_error(status.message());
    }
}

void SetError(multimodal_inference::EmotionBatchResponse& response, const core::Status& status) {
    if (!status.ok() && response.error().empty()) {
        response.set_error(status.message());
    }
}

void SetError(multimodal_inference::VLMResponse& response, const core::Status& status) {
    if (!status.ok() && response.error().empty()) {
        response.set_error(status.message());
    }
}

} // namespace

MultimodalGrpcService::MultimodalGrpcService(const MultimodalServerOptions& options,
                                             server_common::RuntimeStats& stats,
                                             service::MultimodalService& service)
    : stats_(stats),
      service_(service),
      slow_request_ms_(options.grpc.slow_request_ms),
      auth_options_(options.auth),
      request_limits_(options.limits) {}

grpc::Status MultimodalGrpcService::PredictEmotion(
    grpc::ServerContext* context,
    const multimodal_inference::EmotionRequest* request,
    multimodal_inference::EmotionResponse* response) {
    return grpc_error::GuardRpc([&] {
        server_common::ScopedRequestStats request_stats(stats_, "PredictEmotion", 1, false, slow_request_ms_);

        if (auto auth_status = CheckAuth(*context, request_stats); !auth_status.ok()) {
            response->set_error(auth_status.error_message());
            return auth_status;
        }

        if (auto validation = request_validation::ValidateEmotionRequest(*request, request_limits_); !validation.ok) {
            response->set_error(validation.error);
            request_stats.MarkFailure(validation.error);
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, validation.error);
        }

        auto status = service_.PredictEmotion(*request, *response);
        if (!status.ok()) {
            SetError(*response, status);
            request_stats.MarkFailure(status.message());
            return ToGrpc(status);
        }

        request_stats.MarkSuccess();
        return grpc::Status::OK;
    });
}

grpc::Status MultimodalGrpcService::PredictEmotionBatch(
    grpc::ServerContext* context,
    const multimodal_inference::EmotionBatchRequest* request,
    multimodal_inference::EmotionBatchResponse* response) {
    return grpc_error::GuardRpc([&] {
        const std::size_t reported_batch_size =
            request->batch_size() > 0 ? static_cast<std::size_t>(request->batch_size()) : 0;
        server_common::ScopedRequestStats request_stats(
            stats_, "PredictEmotionBatch", reported_batch_size, true, slow_request_ms_);

        if (auto auth_status = CheckAuth(*context, request_stats); !auth_status.ok()) {
            response->set_error(auth_status.error_message());
            return auth_status;
        }

        if (auto validation = request_validation::ValidateEmotionBatchRequest(*request, request_limits_); !validation.ok) {
            response->set_error(validation.error);
            request_stats.MarkFailure(validation.error);
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, validation.error);
        }

        auto status = service_.PredictEmotionBatch(*request, *response);
        if (!status.ok()) {
            SetError(*response, status);
            request_stats.MarkFailure(status.message());
            return ToGrpc(status);
        }

        request_stats.MarkSuccess();
        return grpc::Status::OK;
    });
}

grpc::Status MultimodalGrpcService::DetectSaliency(
    grpc::ServerContext* context,
    const multimodal_inference::SaliencyRequest* request,
    multimodal_inference::SaliencyResponse* response) {
    return grpc_error::GuardRpc([&] {
        server_common::ScopedRequestStats request_stats(stats_, "DetectSaliency", 1, false, slow_request_ms_);

        if (auto auth_status = CheckAuth(*context, request_stats); !auth_status.ok()) {
            return auth_status;
        }

        if (auto validation = request_validation::ValidateSaliencyRequest(*request, request_limits_); !validation.ok) {
            request_stats.MarkFailure(validation.error);
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, validation.error);
        }

        auto status = service_.DetectSaliency(*request, *response);
        if (!status.ok()) {
            request_stats.MarkFailure(status.message());
            return ToGrpc(status);
        }

        request_stats.MarkSuccess();
        return grpc::Status::OK;
    });
}

grpc::Status MultimodalGrpcService::GenerateVLM(
    grpc::ServerContext* context,
    const multimodal_inference::VLMRequest* request,
    grpc::ServerWriter<multimodal_inference::VLMToken>* writer) {
    return grpc_error::GuardRpc([&] {
        server_common::ScopedRequestStats request_stats(stats_, "GenerateVLM", 1, false, slow_request_ms_);

        if (auto auth_status = CheckAuth(*context, request_stats); !auth_status.ok()) {
            return auth_status;
        }

        if (auto validation = request_validation::ValidateVLMRequest(*request, request_limits_); !validation.ok) {
            request_stats.MarkFailure(validation.error);
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, validation.error);
        }

        auto status = service_.GenerateVLM(*request, [writer](multimodal_inference::VLMToken token) {
            writer->Write(token);
        });
        if (!status.ok()) {
            request_stats.MarkFailure(status.message());
            return ToGrpc(status);
        }

        request_stats.MarkSuccess();
        return grpc::Status::OK;
    });
}

grpc::Status MultimodalGrpcService::GenerateVLMSync(
    grpc::ServerContext* context,
    const multimodal_inference::VLMRequest* request,
    multimodal_inference::VLMResponse* response) {
    return grpc_error::GuardRpc([&] {
        server_common::ScopedRequestStats request_stats(stats_, "GenerateVLMSync", 1, false, slow_request_ms_);

        if (auto auth_status = CheckAuth(*context, request_stats); !auth_status.ok()) {
            response->set_error(auth_status.error_message());
            return auth_status;
        }

        if (auto validation = request_validation::ValidateVLMRequest(*request, request_limits_); !validation.ok) {
            response->set_error(validation.error);
            request_stats.MarkFailure(validation.error);
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, validation.error);
        }

        auto status = service_.GenerateVLMSync(*request, *response);
        if (!status.ok()) {
            SetError(*response, status);
            request_stats.MarkFailure(status.message());
            return ToGrpc(status);
        }

        request_stats.MarkSuccess();
        return grpc::Status::OK;
    });
}

grpc::Status MultimodalGrpcService::CheckAuth(
    const grpc::ServerContext& context,
    server_common::ScopedRequestStats& request_stats) const {
    auto validation = request_validation::ValidateAuth(context, auth_options_);
    if (!validation.ok) {
        request_stats.MarkFailure(validation.error);
        return grpc::Status(grpc::StatusCode::UNAUTHENTICATED, validation.error);
    }
    return grpc::Status::OK;
}

} // namespace server::grpc_service
