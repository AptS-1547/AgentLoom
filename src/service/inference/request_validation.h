#pragma once

#include "request_options.h"

#include <string>
#include <vector>

#include <grpcpp/server_context.h>

#include "multimodal_inference.pb.h"

namespace request_validation {

struct ValidationResult {
    bool ok = true;
    std::string error;
};

ValidationResult Ok();
ValidationResult Fail(std::string error);

ValidationResult ValidateAuth(
    const grpc::ServerContext& context,
    const AuthOptions& options);

/// callback API 在请求离开 gRPC 线程后使用已复制的 metadata 完成认证校验。
ValidationResult ValidateAuthMetadata(
    const std::vector<std::pair<std::string, std::string>>& metadata,
    const AuthOptions& options);

ValidationResult ValidateEmotionRequest(
    const multimodal_inference::EmotionRequest& request,
    const RequestLimits& limits);

ValidationResult ValidateEmotionBatchRequest(
    const multimodal_inference::EmotionBatchRequest& request,
    const RequestLimits& limits);

ValidationResult ValidateVLMRequest(
    const multimodal_inference::VLMRequest& request,
    const RequestLimits& limits);

ValidationResult ValidateSaliencyRequest(
    const multimodal_inference::SaliencyRequest& request,
    const RequestLimits& limits);

}
