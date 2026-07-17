#include "grpc_vlm_vision_client.h"

#include "multimodal_inference.grpc.pb.h"

#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace std::chrono_literals;

class FakeVlmService final : public multimodal_inference::MultimodalInference::Service {
public:
    grpc::Status GenerateVLMSync(
        grpc::ServerContext*,
        const multimodal_inference::VLMRequest* request,
        multimodal_inference::VLMResponse* response) override {
        last_request = *request;
        return handler ? handler(*request, *response) : grpc::Status::OK;
    }

    std::function<grpc::Status(
        const multimodal_inference::VLMRequest&,
        multimodal_inference::VLMResponse&)> handler;
    multimodal_inference::VLMRequest last_request;
};

class GrpcHarness final {
public:
    explicit GrpcHarness(grpc::Service& service) {
        grpc::ServerBuilder builder;
        int selected_port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
        builder.RegisterService(&service);
        server_ = builder.BuildAndStart();
        if (!server_ || selected_port <= 0) {
            throw std::runtime_error("failed to start VLM adapter test server");
        }
        target_ = "127.0.0.1:" + std::to_string(selected_port);
    }

    ~GrpcHarness() {
        if (server_) {
            server_->Shutdown();
        }
    }

    const std::string& target() const noexcept { return target_; }

private:
    std::unique_ptr<grpc::Server> server_;
    std::string target_;
};

media::VisionInferenceRequest Request(std::vector<std::byte>& image) {
    return {
        .session_id = "session-1",
        .frame_id = 42,
        .encoded_image = image,
        .mime_type = "image/jpeg",
    };
}

TEST(GrpcVlmVisionClientTest, MapsStructuredResponseAndMetrics) {
    FakeVlmService service;
    service.handler = [](const auto&, auto& response) {
        response.set_text(R"({"scene_hint":"桌面场景","action_hint":"手拿杯子","object_hint":"杯子","facts":["桌上有杯子"],"weak_interpretations":["人物可能正在喝水"],"memory_candidate":"","confidence":0.82})");
        response.set_image_encode_ms(1.25F);
        response.set_prompt_eval_ms(12.5F);
        response.set_eval_ms(31.0F);
        response.set_prompt_tokens(64);
        response.set_generated_tokens(28);
        response.set_cache_hit(true);
        response.set_prompt_kv_cache_hit(true);
        response.set_result_source("exact_cache");
        response.set_prompt_kv_near_candidate(true);
        response.set_prompt_kv_near_accepted(false);
        response.set_prompt_kv_near_same_session(true);
        response.set_prompt_kv_global_cosine(0.998F);
        response.set_prompt_kv_mean_token_cosine(0.999F);
        response.set_prompt_kv_p05_token_cosine(0.991F);
        response.set_prompt_kv_min_token_cosine(0.95F);
        response.set_prompt_kv_relative_l2(0.04F);
        response.set_prompt_kv_max_abs_error(0.12F);
        return grpc::Status::OK;
    };
    GrpcHarness server(service);
    auto created = media::GrpcVlmVisionClient::Create({
        .target = server.target(),
        .timeout = 2s,
        .allow_cache = true,
        .force_refresh = false,
    });
    ASSERT_TRUE(created.ok()) << created.status().message();

    std::vector<std::byte> image{std::byte{0xFF}, std::byte{0xD8}, std::byte{0xFF}};
    auto result = created.value()->Analyze(Request(image));

    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_EQ(result.value().scene_hint, "桌面场景");
    EXPECT_EQ(result.value().action_hint, "手拿杯子");
    ASSERT_EQ(result.value().facts.size(), 1);
    EXPECT_DOUBLE_EQ(result.value().confidence, 0.82);
    EXPECT_EQ(result.value().generated_tokens, 28);
    EXPECT_TRUE(result.value().cache_hit);
    EXPECT_TRUE(result.value().prompt_kv_cache_hit);
    EXPECT_EQ(result.value().result_source, "exact_cache");
    EXPECT_TRUE(result.value().prompt_kv_near_candidate);
    EXPECT_FALSE(result.value().prompt_kv_near_accepted);
    EXPECT_TRUE(result.value().prompt_kv_near_same_session);
    EXPECT_FLOAT_EQ(result.value().prompt_kv_global_cosine, 0.998F);
    EXPECT_FLOAT_EQ(result.value().prompt_kv_mean_token_cosine, 0.999F);
    EXPECT_FLOAT_EQ(result.value().prompt_kv_p05_token_cosine, 0.991F);
    EXPECT_FLOAT_EQ(result.value().prompt_kv_min_token_cosine, 0.95F);
    EXPECT_FLOAT_EQ(result.value().prompt_kv_relative_l2, 0.04F);
    EXPECT_FLOAT_EQ(result.value().prompt_kv_max_abs_error, 0.12F);
    EXPECT_EQ(service.last_request.session_id(), "session-1");
    EXPECT_EQ(service.last_request.request_id(), "session-1-frame-42");
    EXPECT_EQ(service.last_request.image_data().size(), image.size());

    const auto snapshot = created.value()->Snapshot();
    EXPECT_EQ(snapshot.requests, 1);
    EXPECT_EQ(snapshot.successful_requests, 1);
    EXPECT_EQ(snapshot.failed_requests, 0);
    EXPECT_EQ(snapshot.cache_hits, 1);
}

TEST(GrpcVlmVisionClientTest, PreservesUnstructuredTextAsControlledFallback) {
    FakeVlmService service;
    service.handler = [](const auto&, auto& response) {
        response.set_text("画面中有一张桌子和一个杯子。");
        return grpc::Status::OK;
    };
    GrpcHarness server(service);
    auto created = media::GrpcVlmVisionClient::Create({.target = server.target(), .timeout = 2s});
    ASSERT_TRUE(created.ok());

    std::vector<std::byte> image{std::byte{0x01}};
    auto result = created.value()->Analyze(Request(image));

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().scene_hint, "画面中有一张桌子和一个杯子。");
    EXPECT_EQ(result.value().raw_text, result.value().scene_hint);
    EXPECT_EQ(result.value().confidence, 0.5);
    ASSERT_EQ(result.value().weak_interpretations.size(), 1);
}

TEST(GrpcVlmVisionClientTest, MapsGrpcFailureToCoreStatus) {
    FakeVlmService service;
    service.handler = [](const auto&, auto&) {
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "runner queue is full");
    };
    GrpcHarness server(service);
    auto created = media::GrpcVlmVisionClient::Create({.target = server.target(), .timeout = 2s});
    ASSERT_TRUE(created.ok());

    std::vector<std::byte> image{std::byte{0x01}};
    auto result = created.value()->Analyze(Request(image));

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::ResourceExhausted);
    EXPECT_EQ(created.value()->Snapshot().failed_requests, 1);
}

TEST(GrpcVlmVisionClientTest, RejectsInvalidRequestBeforeRpc) {
    auto created = media::GrpcVlmVisionClient::Create({.target = "127.0.0.1:1", .timeout = 100ms});
    ASSERT_TRUE(created.ok());
    media::VisionInferenceRequest request;
    request.session_id = "session";

    auto result = created.value()->Analyze(request);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::InvalidArgument);
}

} // namespace
