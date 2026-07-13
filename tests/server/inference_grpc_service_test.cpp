#include "emotion_grpc_service.h"
#include "multimodal_grpc_service.h"
#include "inference_frame_ipc_grpc_signal.h"

#include "exception.h"

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include <gtest/gtest.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;

std::string UniqueIpcChannelName() {
    static std::atomic<std::uint64_t> sequence{0};
    return "agent_grpc_ipc_control_" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

class IpcChannelCleanup {
public:
    explicit IpcChannelCleanup(std::string name) : name_(std::move(name)) {
        ipc::media::SharedMemoryInferenceFrameChannel::Remove(name_);
    }

    ~IpcChannelCleanup() {
        ipc::media::SharedMemoryInferenceFrameChannel::Remove(name_);
    }

    const std::string& name() const noexcept { return name_; }

private:
    std::string name_;
};

class GrpcServerHarness {
public:
    explicit GrpcServerHarness(grpc::Service& service) {
        grpc::ServerBuilder builder;
        int selected_port = 0;
        builder.AddListeningPort(
            "127.0.0.1:0",
            grpc::InsecureServerCredentials(),
            &selected_port);
        builder.RegisterService(&service);
        server_ = builder.BuildAndStart();
        if (!server_ || selected_port <= 0) {
            throw std::runtime_error("failed to start inference gRPC test server");
        }
        channel_ = grpc::CreateChannel(
            "127.0.0.1:" + std::to_string(selected_port),
            grpc::InsecureChannelCredentials());
    }

    ~GrpcServerHarness() {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
    }

    std::shared_ptr<grpc::Channel> channel() const {
        return channel_;
    }

private:
    std::unique_ptr<grpc::Server> server_;
    std::shared_ptr<grpc::Channel> channel_;
};

class FakeEmotionService final : public service::IEmotionInferenceService {
public:
    core::Status PredictEmotion(
        const multimodal_inference::EmotionRequest& request,
        multimodal_inference::EmotionResponse& response) override {
        ++predict_calls;
        return predict ? predict(request, response) : core::Status::Ok();
    }

    core::Status PredictEmotionBatch(
        const multimodal_inference::EmotionBatchRequest& request,
        multimodal_inference::EmotionBatchResponse& response) override {
        ++batch_calls;
        return predict_batch ? predict_batch(request, response) : core::Status::Ok();
    }

    std::function<core::Status(
        const multimodal_inference::EmotionRequest&,
        multimodal_inference::EmotionResponse&)> predict;
    std::function<core::Status(
        const multimodal_inference::EmotionBatchRequest&,
        multimodal_inference::EmotionBatchResponse&)> predict_batch;
    std::atomic<int> predict_calls{0};
    std::atomic<int> batch_calls{0};
};

class FakeMultimodalService final : public service::IMultimodalService {
public:
    core::Status PredictEmotion(
        const multimodal_inference::EmotionRequest&,
        multimodal_inference::EmotionResponse&) override {
        return core::Status::Ok();
    }

    core::Status PredictEmotionBatch(
        const multimodal_inference::EmotionBatchRequest&,
        multimodal_inference::EmotionBatchResponse&) override {
        return core::Status::Ok();
    }

    core::Status DetectSaliency(
        const multimodal_inference::SaliencyRequest&,
        multimodal_inference::SaliencyResponse&) override {
        return core::Status::Ok();
    }

    core::Status GenerateVLM(
        const multimodal_inference::VLMRequest& request,
        service::VlmTokenEmitter emit) override {
        return generate_vlm ? generate_vlm(request, std::move(emit)) : core::Status::Ok();
    }

    core::Status GenerateVLMSync(
        const multimodal_inference::VLMRequest& request,
        multimodal_inference::VLMResponse& response) override {
        return generate_vlm_sync ? generate_vlm_sync(request, response) : core::Status::Ok();
    }

    std::function<core::Status(
        const multimodal_inference::VLMRequest&,
        service::VlmTokenEmitter)> generate_vlm;
    std::function<core::Status(
        const multimodal_inference::VLMRequest&,
        multimodal_inference::VLMResponse&)> generate_vlm_sync;
};

multimodal_inference::EmotionRequest ValidEmotionRequest() {
    multimodal_inference::EmotionRequest request;
    request.add_input_ids(1);
    request.add_attention_mask(1);
    for (int index = 0; index < 11; ++index) {
        request.add_personality(0.5f);
    }
    return request;
}

multimodal_inference::VLMRequest ValidVlmRequest() {
    multimodal_inference::VLMRequest request;
    request.set_prompt("sensitive prompt must not be logged");
    request.set_request_id("request-42");
    request.set_session_id("session-7");
    request.set_task_type("scene-description");
    request.set_max_tokens(8);
    request.set_context_size(128);
    request.set_top_p(0.9f);
    return request;
}

std::string MetadataValue(
    const std::multimap<grpc::string_ref, grpc::string_ref>& metadata,
    std::string_view key) {
    for (const auto& entry : metadata) {
        if (std::string_view(entry.first.data(), entry.first.length()) == key) {
            return std::string(entry.second.data(), entry.second.length());
        }
    }
    return {};
}

class InferenceGrpcTest : public ::testing::Test {
protected:
    void SetUp() override {
        spdlog::drop("inference-grpc");
        log_stream_.str({});
        log_stream_.clear();
        auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(log_stream_);
        logger_ = std::make_shared<spdlog::logger>("inference-grpc", std::move(sink));
        logger_->set_level(spdlog::level::debug);
        logger_->set_pattern("%v");
        spdlog::register_logger(logger_);
    }

    void TearDown() override {
        spdlog::drop("inference-grpc");
        logger_.reset();
    }

    std::string Logs() {
        logger_->flush();
        return log_stream_.str();
    }

    std::ostringstream log_stream_;
    std::shared_ptr<spdlog::logger> logger_;
};

TEST_F(InferenceGrpcTest, SuccessPropagatesTraceAndUpdatesStats) {
    MultimodalServerOptions options;
    options.grpc.slow_request_ms = 0;
    server_common::RuntimeStats stats;
    FakeEmotionService backend;
    server::grpc_service::EmotionGrpcService service(options, stats, backend);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext context;
    context.AddMetadata("x-trace-id", "trace-success-1");
    auto request = ValidEmotionRequest();
    multimodal_inference::EmotionResponse response;
    const auto status = stub->PredictEmotion(&context, request, &response);

    ASSERT_TRUE(status.ok()) << status.error_message();
    EXPECT_EQ(backend.predict_calls.load(), 1);
    EXPECT_EQ(MetadataValue(context.GetServerInitialMetadata(), "x-trace-id"), "trace-success-1");
    EXPECT_EQ(MetadataValue(context.GetServerTrailingMetadata(), "x-trace-id"), "trace-success-1");
    const auto snapshot = stats.Snapshot();
    EXPECT_EQ(snapshot.total_rpc_requests, 1);
    EXPECT_EQ(snapshot.total_errors, 0);
    EXPECT_EQ(snapshot.inflight_requests, 0);
}

TEST_F(InferenceGrpcTest, ValidationFailureSkipsBackendAndUpdatesStats) {
    MultimodalServerOptions options;
    options.grpc.slow_request_ms = 0;
    server_common::RuntimeStats stats;
    FakeEmotionService backend;
    server::grpc_service::EmotionGrpcService service(options, stats, backend);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext context;
    multimodal_inference::EmotionRequest request;
    multimodal_inference::EmotionResponse response;
    const auto status = stub->PredictEmotion(&context, request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.error_message(), "Empty input");
    EXPECT_EQ(backend.predict_calls.load(), 0);
    const auto snapshot = stats.Snapshot();
    EXPECT_EQ(snapshot.total_errors, 1);
    EXPECT_EQ(snapshot.last_error, "Empty input");
}

TEST_F(InferenceGrpcTest, AuthFailureUsesUnauthenticatedAndDoesNotLogToken) {
    MultimodalServerOptions options;
    options.grpc.slow_request_ms = 0;
    options.auth.token = "super-secret-auth-token";
    server_common::RuntimeStats stats;
    FakeEmotionService backend;
    server::grpc_service::EmotionGrpcService service(options, stats, backend);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext context;
    auto request = ValidEmotionRequest();
    multimodal_inference::EmotionResponse response;
    const auto status = stub->PredictEmotion(&context, request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAUTHENTICATED);
    EXPECT_EQ(backend.predict_calls.load(), 0);
    EXPECT_EQ(Logs().find("super-secret-auth-token"), std::string::npos);
}

TEST_F(InferenceGrpcTest, CoreStatusMapsToGrpcStatusAndFailureStats) {
    MultimodalServerOptions options;
    options.grpc.slow_request_ms = 0;
    server_common::RuntimeStats stats;
    FakeEmotionService backend;
    backend.predict = [](const auto&, auto&) {
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "inference queue is full");
    };
    server::grpc_service::EmotionGrpcService service(options, stats, backend);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext context;
    auto request = ValidEmotionRequest();
    multimodal_inference::EmotionResponse response;
    const auto status = stub->PredictEmotion(&context, request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(status.error_message(), "inference queue is full");
    EXPECT_EQ(stats.Snapshot().total_errors, 1);
    EXPECT_NE(Logs().find("core_code=RESOURCE_EXHAUSTED"), std::string::npos);
}

TEST_F(InferenceGrpcTest, AppExceptionPreservesStructuredStatus) {
    MultimodalServerOptions options;
    options.grpc.slow_request_ms = 0;
    server_common::RuntimeStats stats;
    FakeEmotionService backend;
    backend.predict = [](const auto&, auto&) -> core::Status {
        throw core::AppException(core::ErrorCode::FailedPrecondition, "model is warming up");
    };
    server::grpc_service::EmotionGrpcService service(options, stats, backend);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext context;
    auto request = ValidEmotionRequest();
    multimodal_inference::EmotionResponse response;
    const auto status = stub->PredictEmotion(&context, request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
    EXPECT_EQ(status.error_message(), "model is warming up");
    EXPECT_EQ(stats.Snapshot().last_error, "model is warming up");
}

TEST_F(InferenceGrpcTest, UnexpectedExceptionsAreSanitizedAndTracked) {
    MultimodalServerOptions options;
    options.grpc.slow_request_ms = 0;
    server_common::RuntimeStats stats;
    FakeEmotionService backend;
    backend.predict = [](const auto&, auto&) -> core::Status {
        throw std::runtime_error("raw prompt and credential details");
    };
    server::grpc_service::EmotionGrpcService service(options, stats, backend);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext context;
    context.AddMetadata("x-trace-id", "trace-exception-9");
    auto request = ValidEmotionRequest();
    multimodal_inference::EmotionResponse response;
    const auto status = stub->PredictEmotion(&context, request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(status.error_message(), "Unhandled inference RPC exception");
    EXPECT_EQ(stats.Snapshot().last_error, "Unhandled inference RPC exception");
    const auto logs = Logs();
    EXPECT_NE(logs.find("trace_id=trace-exception-9"), std::string::npos);
    EXPECT_EQ(logs.find("raw prompt and credential details"), std::string::npos);
}

TEST_F(InferenceGrpcTest, UnknownExceptionsAreSanitizedAndTracked) {
    MultimodalServerOptions options;
    options.grpc.slow_request_ms = 0;
    server_common::RuntimeStats stats;
    FakeEmotionService backend;
    backend.predict = [](const auto&, auto&) -> core::Status {
        throw 7;
    };
    server::grpc_service::EmotionGrpcService service(options, stats, backend);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext context;
    auto request = ValidEmotionRequest();
    multimodal_inference::EmotionResponse response;
    const auto status = stub->PredictEmotion(&context, request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(status.error_message(), "Unknown inference RPC exception");
    EXPECT_EQ(stats.Snapshot().total_errors, 1);
}

TEST_F(InferenceGrpcTest, StreamingFailureReturnsFinalGrpcStatusAndTrace) {
    MultimodalServerOptions options;
    options.grpc.slow_request_ms = 0;
    server_common::RuntimeStats stats;
    FakeMultimodalService backend;
    backend.generate_vlm = [](const auto&, service::VlmTokenEmitter emit) {
        multimodal_inference::VLMToken token;
        token.set_token("partial");
        emit(std::move(token));
        return core::Status::Error(core::ErrorCode::Unavailable, "runner stopped");
    };
    server::grpc_service::MultimodalGrpcService service(options, stats, backend);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext context;
    context.AddMetadata("x-trace-id", "trace-stream-4");
    auto request = ValidVlmRequest();
    auto reader = stub->GenerateVLM(&context, request);
    multimodal_inference::VLMToken token;
    ASSERT_TRUE(reader->Read(&token));
    EXPECT_EQ(token.token(), "partial");
    while (reader->Read(&token)) {
    }
    const auto status = reader->Finish();

    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(status.error_message(), "runner stopped");
    EXPECT_EQ(MetadataValue(context.GetServerTrailingMetadata(), "x-trace-id"), "trace-stream-4");
    EXPECT_EQ(stats.Snapshot().total_errors, 1);
}

TEST_F(InferenceGrpcTest, MultimodalExceptionLogsOnlySafeRequestIdentifiers) {
    MultimodalServerOptions options;
    options.grpc.slow_request_ms = 0;
    server_common::RuntimeStats stats;
    FakeMultimodalService backend;
    backend.generate_vlm_sync = [](const auto&, auto&) -> core::Status {
        throw std::runtime_error("sensitive prompt must not be logged");
    };
    server::grpc_service::MultimodalGrpcService service(options, stats, backend);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext context;
    auto request = ValidVlmRequest();
    multimodal_inference::VLMResponse response;
    const auto status = stub->GenerateVLMSync(&context, request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    const auto logs = Logs();
    EXPECT_NE(logs.find("request_id=request-42"), std::string::npos);
    EXPECT_NE(logs.find("session_id=session-7"), std::string::npos);
    EXPECT_NE(logs.find("task_type=scene-description"), std::string::npos);
    EXPECT_EQ(logs.find("sensitive prompt must not be logged"), std::string::npos);
}

TEST_F(InferenceGrpcTest, GrpcPeerLossFencesGatewaySharedMemoryLease) {
    IpcChannelCleanup cleanup(UniqueIpcChannelName());
    auto created_sink = ipc::media::RecoverableInferenceFrameIpcSink::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(created_sink.ok()) << created_sink.status().message();
    std::shared_ptr<ipc::media::IRecoverableInferenceFrameIpcSink> sink(
        std::move(created_sink).value());
    auto receiver = std::make_shared<ipc::media::InferenceFrameIpcGrantReceiver>();
    std::shared_ptr<ipc::media::GrpcInferenceFrameIpcSignal> signal;
    std::unique_ptr<ipc::media::InferenceFrameIpcLeaseCoordinator> coordinator;

    MultimodalServerOptions options;
    options.grpc.slow_request_ms = 0;
    server_common::RuntimeStats stats;
    FakeMultimodalService backend;
    server::grpc_service::MultimodalGrpcService grpc_service(
        options,
        stats,
        backend,
        receiver.get());
    {
        GrpcServerHarness server(grpc_service);
        signal = std::make_shared<ipc::media::GrpcInferenceFrameIpcSignal>(
            server.channel(),
            ipc::media::GrpcInferenceFrameIpcSignalOptions{.deadline = 200ms});
        coordinator = std::make_unique<ipc::media::InferenceFrameIpcLeaseCoordinator>(
            sink,
            signal);
        ASSERT_TRUE(coordinator->Start().ok());
        ASSERT_TRUE(coordinator->CheckPeer().ok());
        EXPECT_EQ(receiver->ControlSnapshot().state, ipc::media::InferenceFrameIpcControlState::Granted);
    }

    const auto probe = coordinator->CheckPeer();
    ASSERT_FALSE(probe.ok());
    EXPECT_TRUE(
        probe.code() == core::ErrorCode::Unavailable ||
        probe.code() == core::ErrorCode::Timeout);
    EXPECT_EQ(coordinator->Snapshot().state, ipc::media::InferenceFrameIpcControlState::Fenced);
    EXPECT_TRUE(sink->Snapshot().fenced);

    std::vector<std::byte> payload(8, std::byte{0x45});
    const ipc::media::SharedFramePublishRequest request{
        .execution_id = "execution-grpc-crash",
        .session_id = "session-grpc-crash",
        .trace_id = "trace-grpc-crash",
        .selected_sequence = 1,
        .transport_sequence = 1,
        .frame_id = 1,
        .payload = payload,
    };
    EXPECT_EQ(sink->Publish(request).code(), core::ErrorCode::Cancelled);
}

TEST_F(InferenceGrpcTest, GrpcControlRevokeAndRecoverRotateInferenceEpoch) {
    IpcChannelCleanup cleanup(UniqueIpcChannelName());
    auto created_sink = ipc::media::RecoverableInferenceFrameIpcSink::Create({
        .name = cleanup.name(),
        .slot_count = 4,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(created_sink.ok()) << created_sink.status().message();
    std::shared_ptr<ipc::media::IRecoverableInferenceFrameIpcSink> sink(
        std::move(created_sink).value());
    auto receiver = std::make_shared<ipc::media::InferenceFrameIpcGrantReceiver>();

    MultimodalServerOptions options;
    options.grpc.slow_request_ms = 0;
    server_common::RuntimeStats stats;
    FakeMultimodalService backend;
    server::grpc_service::MultimodalGrpcService grpc_service(
        options,
        stats,
        backend,
        receiver.get());
    GrpcServerHarness server(grpc_service);
    auto signal = std::make_shared<ipc::media::GrpcInferenceFrameIpcSignal>(
        server.channel(),
        ipc::media::GrpcInferenceFrameIpcSignalOptions{.deadline = 500ms});
    ipc::media::InferenceFrameIpcLeaseCoordinator coordinator(sink, signal);

    ASSERT_TRUE(coordinator.Start().ok());
    const auto old_epoch = coordinator.Snapshot().grant.epoch;
    ASSERT_TRUE(coordinator.Revoke("planned inference restart").ok());
    EXPECT_EQ(receiver->ControlSnapshot().state, ipc::media::InferenceFrameIpcControlState::Fenced);
    EXPECT_EQ(receiver->TryClaim().status().code(), core::ErrorCode::Cancelled);

    ASSERT_TRUE(coordinator.Recover().ok());
    const auto recovered = coordinator.Snapshot();
    EXPECT_NE(recovered.grant.epoch, old_epoch);
    EXPECT_EQ(receiver->ControlSnapshot().grant.epoch, recovered.grant.epoch);
    EXPECT_TRUE(coordinator.CheckPeer().ok());
    const auto stale_revoke = signal->Revoke(old_epoch, "delayed stale revoke");
    EXPECT_EQ(stale_revoke.code(), core::ErrorCode::FailedPrecondition);
    EXPECT_EQ(receiver->ControlSnapshot().state, ipc::media::InferenceFrameIpcControlState::Granted);
    EXPECT_EQ(receiver->ControlSnapshot().grant.epoch, recovered.grant.epoch);
}

} // namespace
