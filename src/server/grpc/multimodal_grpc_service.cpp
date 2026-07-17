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

void FillSkillMediaExecutionResponse(
    const service::SharedMediaExecutionSnapshot& snapshot,
    multimodal_inference::SkillMediaExecutionResponse& response) {
    response.set_execution_id(snapshot.execution_id);
    response.set_session_id(snapshot.session_id);
    response.set_state(snapshot.state);
    response.set_expected_selected_frames(snapshot.expected_selected_frames);
    response.set_selected_frames(snapshot.selected_frames);
    response.set_committed_frames(snapshot.committed_frames);
    response.set_hot_frames(snapshot.hot_frames);
    response.set_spooled_frames(snapshot.spooled_frames);
    response.set_terminal_frames(snapshot.terminal_frames);
    response.set_failed_inference_frames(snapshot.failed_inference_frames);
    response.set_seal_requested(snapshot.seal_requested);
    response.set_replay_complete(snapshot.replay_complete);
    response.set_complete(snapshot.complete);
    if (!snapshot.status.ok()) response.set_error(snapshot.status.message());
    for (const auto& record : snapshot.results) {
        auto* result = response.add_results();
        result->set_selected_sequence(record.frame.selected_sequence);
        result->set_transport_sequence(record.frame.transport_sequence);
        result->set_frame_id(record.frame.frame_id);
        result->set_timestamp_us(record.frame.timestamp_us);
        result->set_status_code(static_cast<int>(record.status.code()));
        if (!record.status.ok()) result->set_error(record.status.message());
        const auto& timing = record.frame.timing;
        result->set_publish_to_receive_us(media::inference::InferenceFrameDurationUs(
            timing.published_at_unix_us, timing.received_at_unix_us));
        result->set_receive_to_admit_us(media::inference::InferenceFrameDurationUs(
            timing.received_at_unix_us, timing.admitted_at_unix_us));
        const auto queue_started_at = timing.replayed_at_unix_us > 0
            ? timing.replayed_at_unix_us
            : timing.admitted_at_unix_us;
        result->set_queue_wait_us(media::inference::InferenceFrameDurationUs(
            queue_started_at, timing.inference_started_at_unix_us));
        result->set_spool_wait_us(media::inference::InferenceFrameDurationUs(
            timing.spooled_at_unix_us, timing.replayed_at_unix_us));
        const auto stream_wait_us = timing.spooled_at_unix_us > 0
            ? media::inference::InferenceFrameDurationUs(
                  timing.spooled_at_unix_us, snapshot.input_sealed_at_unix_us)
            : 0;
        const auto replay_wait_us = timing.spooled_at_unix_us > 0
            ? media::inference::InferenceFrameDurationUs(
                  snapshot.input_sealed_at_unix_us, timing.replayed_at_unix_us)
            : 0;
        result->set_stream_wait_us(stream_wait_us);
        result->set_replay_wait_us(replay_wait_us);
        result->set_inference_us(media::inference::InferenceFrameDurationUs(
            timing.inference_started_at_unix_us, timing.terminal_at_unix_us));
        result->set_publish_to_terminal_us(media::inference::InferenceFrameDurationUs(
            timing.published_at_unix_us, timing.terminal_at_unix_us));
        const auto total_us = media::inference::InferenceFrameDurationUs(
            timing.published_at_unix_us, timing.terminal_at_unix_us);
        result->set_publish_to_terminal_excluding_stream_us(
            total_us >= stream_wait_us ? total_us - stream_wait_us : 0);
        result->set_spooled(timing.spooled_at_unix_us > 0);
        if (!record.result.has_value()) continue;
        const auto& value = record.result.value();
        result->set_scene_hint(value.scene_hint);
        result->set_action_hint(value.action_hint);
        result->set_object_hint(value.object_hint);
        result->set_agent_hint(value.agent_hint);
        result->set_memory_candidate(value.memory_candidate);
        for (const auto& fact : value.facts) result->add_facts(fact);
        for (const auto& item : value.weak_interpretations) result->add_weak_interpretations(item);
        result->set_raw_text(value.raw_text);
        result->set_confidence(value.confidence);
        result->set_image_encode_ms(value.image_encode_ms);
        result->set_prompt_eval_ms(value.prompt_eval_ms);
        result->set_eval_ms(value.eval_ms);
        result->set_prompt_tokens(value.prompt_tokens);
        result->set_generated_tokens(value.generated_tokens);
        result->set_cache_hit(value.cache_hit);
        result->set_cache_stale(value.cache_stale);
        result->set_prompt_kv_cache_hit(value.prompt_kv_cache_hit);
        result->set_result_source(value.result_source);
        result->set_prompt_kv_near_candidate(value.prompt_kv_near_candidate);
        result->set_prompt_kv_near_accepted(value.prompt_kv_near_accepted);
        result->set_prompt_kv_near_same_session(value.prompt_kv_near_same_session);
        result->set_prompt_kv_global_cosine(value.prompt_kv_global_cosine);
        result->set_prompt_kv_mean_token_cosine(value.prompt_kv_mean_token_cosine);
        result->set_prompt_kv_p05_token_cosine(value.prompt_kv_p05_token_cosine);
        result->set_prompt_kv_min_token_cosine(value.prompt_kv_min_token_cosine);
        result->set_prompt_kv_relative_l2(value.prompt_kv_relative_l2);
        result->set_prompt_kv_max_abs_error(value.prompt_kv_max_abs_error);
    }
}

} // namespace

MultimodalGrpcService::MultimodalGrpcService(const MultimodalServerOptions& options,
                                             server_common::RuntimeStats& stats,
                                             service::IMultimodalService& service,
                                             ipc::media::IInferenceFrameIpcGrantReceiver* ipc_control,
                                             service::ISharedMemoryMediaRuntime* media_runtime)
    : stats_(stats),
      service_(service),
      ipc_control_(ipc_control),
      media_runtime_(media_runtime),
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

grpc::Status MultimodalGrpcService::OpenSkillMediaExecution(
    grpc::ServerContext* context,
    const multimodal_inference::OpenSkillMediaExecutionRequest* request,
    multimodal_inference::SkillMediaExecutionResponse* response) {
    grpc_error::RpcCall rpc(*context, stats_, "OpenSkillMediaExecution", 1, false, slow_request_ms_);
    return rpc.Run([&] {
        if (!request || !response) return rpc.Failure(InvalidRpcArguments());
        if (auto status = CheckAuth(*context); !status.ok()) {
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        if (!media_runtime_) {
            return rpc.Failure(core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "shared memory media runtime is not configured"));
        }
        if (request->execution_id().empty() || request->session_id().empty() ||
            request->execution_id().size() >= ipc::media::kSharedFrameExecutionIdCapacity ||
            request->session_id().size() >= ipc::media::kSharedFrameSessionIdCapacity ||
            request->trace_id().size() >= ipc::media::kSharedFrameTraceIdCapacity) {
            return rpc.Failure(core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "shared media execution identity is invalid"));
        }
        auto opened = media_runtime_->Open({
            .execution_id = request->execution_id(),
            .session_id = request->session_id(),
            .skill_id = request->skill_id().empty() ? "vision.observe" : request->skill_id(),
            .trace_id = request->trace_id(),
        });
        if (!opened.ok()) return rpc.Failure(opened.status());
        FillSkillMediaExecutionResponse(opened.value(), *response);
        return rpc.Success();
    });
}

grpc::Status MultimodalGrpcService::SealSkillMediaInput(
    grpc::ServerContext* context,
    const multimodal_inference::SealSkillMediaInputRequest* request,
    multimodal_inference::SkillMediaExecutionResponse* response) {
    grpc_error::RpcCall rpc(*context, stats_, "SealSkillMediaInput", 1, false, slow_request_ms_);
    return rpc.Run([&] {
        if (!request || !response) return rpc.Failure(InvalidRpcArguments());
        if (auto status = CheckAuth(*context); !status.ok()) {
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        if (!media_runtime_) {
            return rpc.Failure(core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "shared memory media runtime is not configured"));
        }
        if (request->expected_selected_frames() > std::numeric_limits<std::size_t>::max()) {
            return rpc.Failure(core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "shared media expected frame count exceeds platform limits"));
        }
        auto sealed = media_runtime_->Seal({
            .execution_id = request->execution_id(),
            .session_id = request->session_id(),
            .expected_selected_frames = static_cast<std::size_t>(request->expected_selected_frames()),
            .final_transport_sequence = request->final_transport_sequence(),
            .reason = request->reason(),
        });
        if (!sealed.ok()) return rpc.Failure(sealed.status());
        FillSkillMediaExecutionResponse(sealed.value(), *response);
        return rpc.Success();
    });
}

grpc::Status MultimodalGrpcService::GetSkillMediaExecutionStatus(
    grpc::ServerContext* context,
    const multimodal_inference::GetSkillMediaExecutionStatusRequest* request,
    multimodal_inference::SkillMediaExecutionResponse* response) {
    grpc_error::RpcCall rpc(*context, stats_, "GetSkillMediaExecutionStatus", 1, false, slow_request_ms_);
    return rpc.Run([&] {
        if (!request || !response) return rpc.Failure(InvalidRpcArguments());
        if (auto status = CheckAuth(*context); !status.ok()) {
            return rpc.Failure(status, grpc::StatusCode::UNAUTHENTICATED);
        }
        if (!media_runtime_) {
            return rpc.Failure(core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "shared memory media runtime is not configured"));
        }
        auto snapshot = media_runtime_->Get(
            request->session_id(),
            request->execution_id(),
            request->include_results());
        if (!snapshot.ok()) return rpc.Failure(snapshot.status());
        FillSkillMediaExecutionResponse(snapshot.value(), *response);
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
