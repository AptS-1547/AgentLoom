#include "shared_memory_media_runtime.h"

#include "inference_frame_ipc_control.h"
#include "inference_frame_ipc_lifecycle.h"
#include "multimodal_service.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

std::string UniqueName(std::string_view prefix) {
    static std::atomic<std::uint64_t> sequence{0};
    return std::string(prefix) + "_" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

class FakeMultimodalService final : public service::IMultimodalService {
public:
    core::Status PredictEmotion(
        const multimodal_inference::EmotionRequest&,
        multimodal_inference::EmotionResponse&) override {
        return core::Status::Error(core::ErrorCode::Unimplemented, "not used");
    }

    core::Status PredictEmotionBatch(
        const multimodal_inference::EmotionBatchRequest&,
        multimodal_inference::EmotionBatchResponse&) override {
        return core::Status::Error(core::ErrorCode::Unimplemented, "not used");
    }

    core::Status DetectSaliency(
        const multimodal_inference::SaliencyRequest&,
        multimodal_inference::SaliencyResponse&) override {
        return core::Status::Error(core::ErrorCode::Unimplemented, "not used");
    }

    core::Status GenerateVLM(
        const multimodal_inference::VLMRequest&,
        service::VlmTokenEmitter) override {
        return core::Status::Error(core::ErrorCode::Unimplemented, "not used");
    }

    core::Status GenerateVLMSync(
        const multimodal_inference::VLMRequest& request,
        multimodal_inference::VLMResponse& response) override {
        std::this_thread::sleep_for(10ms);
        response.set_text(
            R"({"scene_hint":"共享内存场景","action_hint":"帧处理中","object_hint":"测试帧","facts":["已收到编码帧"],"weak_interpretations":[],"memory_candidate":"","confidence":0.9})");
        response.set_prompt_eval_ms(2.0F);
        response.set_eval_ms(8.0F);
        response.set_generated_tokens(16);
        response.set_result_source("fake-local-vlm");
        static_cast<void>(request);
        return core::Status::Ok();
    }
};

ipc::media::SharedFramePublishRequest Frame(
    std::string_view execution_id,
    std::string_view session_id,
    std::uint64_t sequence,
    std::span<const std::byte> payload) {
    return {
        .execution_id = execution_id,
        .session_id = session_id,
        .trace_id = "shared-runtime-test",
        .selected_sequence = sequence,
        .transport_sequence = sequence,
        .frame_id = sequence,
        .timestamp_us = static_cast<std::int64_t>(sequence * 1000),
        .published_at_unix_us = media::inference::InferenceFrameNowUnixUs(),
        .width = 16,
        .height = 16,
        .format = static_cast<std::uint32_t>(ipc::media::SharedFrameFormat::Jpeg),
        .saliency = 0.8,
        .payload = payload,
    };
}

core::Status PublishWithBackpressure(
    ipc::media::IInferenceFrameIpcSink& sink,
    const ipc::media::SharedFramePublishRequest& request) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
        auto status = sink.Publish(request);
        if (status.ok()) return status;
        if (status.code() != core::ErrorCode::ResourceExhausted) return status;
        std::this_thread::yield();
    }
    return core::Status::Error(core::ErrorCode::Timeout, "test publish timed out");
}

TEST(SharedMemoryMediaRuntimeTest, RejectsInvalidGlobalSpoolByteLimit) {
    FakeMultimodalService backend;
    ipc::media::InferenceFrameIpcGrantReceiver grant_receiver;

    auto smaller_than_segment = service::SharedMemoryMediaRuntime::Create(
        backend,
        grant_receiver,
        {
            .spool_segment_bytes = 1024,
            .max_spool_bytes_per_execution = 4096,
            .max_spool_bytes_total = 512,
        });
    ASSERT_FALSE(smaller_than_segment.ok());
    EXPECT_EQ(smaller_than_segment.status().code(), core::ErrorCode::InvalidArgument);

    auto zero = service::SharedMemoryMediaRuntime::Create(
        backend,
        grant_receiver,
        {
            .spool_segment_bytes = 1024,
            .max_spool_bytes_per_execution = 4096,
            .max_spool_bytes_total = 0,
        });
    ASSERT_FALSE(zero.ok());
    EXPECT_EQ(zero.status().code(), core::ErrorCode::InvalidArgument);
}

TEST(SharedMemoryMediaRuntimeTest, RoutesMultipleExecutionsAndDrainsExactlyOnce) {
    const auto channel_name = UniqueName("shared_media_runtime");
    const auto spool_root = std::filesystem::path("build") / "test-shared-media-runtime" / channel_name;
    std::error_code cleanup_error;
    std::filesystem::remove_all(spool_root, cleanup_error);

    auto created_sink = ipc::media::RecoverableInferenceFrameIpcSink::Create({
        .name = channel_name,
        .slot_count = 8,
        .payload_capacity = 1024,
        .remove_existing = true,
        .remove_on_destroy = true,
    });
    ASSERT_TRUE(created_sink.ok()) << created_sink.status().message();
    std::shared_ptr<ipc::media::IRecoverableInferenceFrameIpcSink> sink(
        std::move(created_sink).value());
    auto grant_receiver = std::make_shared<ipc::media::InferenceFrameIpcGrantReceiver>();
    ipc::media::InferenceFrameIpcLeaseCoordinator lease(sink, grant_receiver);
    ASSERT_TRUE(lease.Start().ok());

    FakeMultimodalService backend;
    auto runtime = service::SharedMemoryMediaRuntime::Create(
        backend,
        *grant_receiver,
        {
            .spool_root = spool_root,
            .max_executions = 4,
            .receiver_workers = 4,
            .vlm_workers = 2,
            .backlog_segments_per_session = 1,
            .backlog_slots_per_segment = 1,
            .max_results_per_execution = 16,
            .spool_segment_bytes = 1024 * 1024,
            .max_spool_bytes_per_execution = 16 * 1024 * 1024,
            .max_spool_bytes_total = 32 * 1024 * 1024,
            .seal_wait_timeout = 5s,
        });
    ASSERT_TRUE(runtime.ok()) << runtime.status().message();

    ASSERT_TRUE(runtime.value()->Open({
        .execution_id = "execution-a",
        .session_id = "session-a",
        .trace_id = "trace-a",
    }).ok());
    ASSERT_TRUE(runtime.value()->Open({
        .execution_id = "execution-b",
        .session_id = "session-b",
        .trace_id = "trace-b",
    }).ok());

    std::vector<std::byte> payload(128, std::byte{0x42});
    std::atomic<std::size_t> publish_failures{0};
    std::jthread producer_a([&] {
        for (std::uint64_t sequence = 1; sequence <= 4; ++sequence) {
            if (!PublishWithBackpressure(*sink, Frame("execution-a", "session-a", sequence, payload)).ok()) {
                publish_failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });
    std::jthread producer_b([&] {
        for (std::uint64_t sequence = 1; sequence <= 4; ++sequence) {
            if (!PublishWithBackpressure(*sink, Frame("execution-b", "session-b", sequence, payload)).ok()) {
                publish_failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });
    producer_a.join();
    producer_b.join();
    ASSERT_EQ(publish_failures.load(std::memory_order_relaxed), 0u);

    ASSERT_TRUE(runtime.value()->Seal({
        .execution_id = "execution-a",
        .session_id = "session-a",
        .expected_selected_frames = 4,
        .final_transport_sequence = 4,
        .reason = "test input complete",
    }).ok());
    ASSERT_TRUE(runtime.value()->Seal({
        .execution_id = "execution-b",
        .session_id = "session-b",
        .expected_selected_frames = 4,
        .final_transport_sequence = 4,
        .reason = "test input complete",
    }).ok());

    const auto deadline = std::chrono::steady_clock::now() + 10s;
    service::SharedMediaExecutionSnapshot result_a;
    service::SharedMediaExecutionSnapshot result_b;
    while (std::chrono::steady_clock::now() < deadline) {
        auto a = runtime.value()->Get("session-a", "execution-a", true);
        auto b = runtime.value()->Get("session-b", "execution-b", true);
        ASSERT_TRUE(a.ok()) << a.status().message();
        ASSERT_TRUE(b.ok()) << b.status().message();
        result_a = std::move(a).value();
        result_b = std::move(b).value();
        if (result_a.complete && result_b.complete) break;
        std::this_thread::sleep_for(2ms);
    }

    ASSERT_TRUE(result_a.complete) << result_a.status.message();
    ASSERT_TRUE(result_b.complete) << result_b.status.message();
    EXPECT_EQ(result_a.state, "closed");
    EXPECT_EQ(result_b.state, "closed");
    ASSERT_EQ(result_a.results.size(), 4u);
    ASSERT_EQ(result_b.results.size(), 4u);
    for (std::size_t index = 0; index < 4; ++index) {
        EXPECT_EQ(result_a.results[index].frame.selected_sequence, index + 1);
        EXPECT_EQ(result_b.results[index].frame.selected_sequence, index + 1);
        ASSERT_TRUE(result_a.results[index].result.has_value());
        EXPECT_EQ(result_a.results[index].result->scene_hint, "共享内存场景");
        const auto& timing = result_a.results[index].frame.timing;
        EXPECT_GT(timing.published_at_unix_us, 0);
        EXPECT_GE(timing.received_at_unix_us, timing.published_at_unix_us);
        EXPECT_GE(timing.admitted_at_unix_us, timing.received_at_unix_us);
        EXPECT_GE(timing.inference_started_at_unix_us, timing.admitted_at_unix_us);
        EXPECT_GE(timing.terminal_at_unix_us, timing.inference_started_at_unix_us);
        EXPECT_GT(media::inference::InferenceFrameDurationUs(
            timing.published_at_unix_us, timing.terminal_at_unix_us), 0u);
    }
    const auto channel = sink->Snapshot();
    EXPECT_EQ(channel.published_frames, 8u);
    EXPECT_EQ(channel.acknowledged_frames, 8u);
    EXPECT_TRUE(std::filesystem::exists(spool_root));
    EXPECT_TRUE(std::filesystem::is_empty(spool_root));

    runtime.value()->Shutdown();
    lease.Shutdown();
    std::filesystem::remove_all(spool_root, cleanup_error);
}

} // namespace
