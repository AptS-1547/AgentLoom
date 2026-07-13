#include "inference_frame_coordinator.h"
#include "inference_frame_ipc_receiver.h"
#include "inference_frame_shared_memory.h"
#include "inference_frame_spool.h"
#include "inference_frame_spool_replayer.h"

#include "memory_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

std::string UniqueName(std::string_view prefix) {
    static std::atomic<std::uint64_t> next{0};
    return std::string(prefix) + "-" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
           std::to_string(next.fetch_add(1, std::memory_order_relaxed));
}

class SpoolE2eCleanup {
public:
    SpoolE2eCleanup()
        : channel_name_(UniqueName("agent-frame-spool-e2e")),
          spool_root_(std::filesystem::current_path() / "build" / "test-spool-e2e" / channel_name_) {
        ipc::media::SharedMemoryInferenceFrameChannel::Remove(channel_name_);
        std::filesystem::create_directories(spool_root_);
    }

    ~SpoolE2eCleanup() {
        ipc::media::SharedMemoryInferenceFrameChannel::Remove(channel_name_);
        std::error_code ec;
        std::filesystem::remove_all(spool_root_, ec);
    }

    const std::string& channel_name() const noexcept {
        return channel_name_;
    }

    const std::filesystem::path& spool_root() const noexcept {
        return spool_root_;
    }

private:
    std::string channel_name_;
    std::filesystem::path spool_root_;
};

class SuccessfulFakeVlm final : public media::IVlmVisionClient {
public:
    core::Result<media::VisionInferenceResult> Analyze(
        const media::VisionInferenceRequest& request) override {
        if (request.encoded_image.empty()) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "encoded image is empty");
        }
        media::VisionInferenceResult result;
        result.scene_hint = "frame-" + std::to_string(request.frame_id);
        result.confidence = 0.9;
        return result;
    }
};

TEST(InferenceFrameSpoolE2eTest, DrainsHotAndSpilledFramesIntoOrderedResults) {
    constexpr std::size_t kFrameCount = 10;
    const std::string execution_id = "execution-spool-e2e";
    const std::string session_id = "session-spool-e2e";
    SpoolE2eCleanup cleanup;

    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.channel_name(),
        .slot_count = 4,
        .payload_capacity = 256,
    });
    ASSERT_TRUE(producer.ok()) << producer.status().message();
    auto source = ipc::media::SharedMemoryInferenceFrameChannel::Open({
        .name = cleanup.channel_name(),
        .slot_count = 4,
        .payload_capacity = 256,
    });
    ASSERT_TRUE(source.ok()) << source.status().message();
    auto spool = media::inference::MappedInferenceFrameSpool::Create({
        .root_directory = cleanup.spool_root(),
        .execution_id = execution_id,
        .segment_bytes = 4096,
        .max_spool_bytes = 64 * 1024,
        .flush_on_append = true,
        .remove_on_destroy = true,
    });
    ASSERT_TRUE(spool.ok()) << spool.status().message();
    std::shared_ptr<media::inference::IInferenceFrameSpool> shared_spool(
        std::move(spool).value());

    core::BucketMemoryPool memory_pool;
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 1,
        .segments_per_session = 1,
        .slots_per_segment = 2,
    });
    media::inference::InferenceFrameIpcReceiver receiver(
        *source.value(),
        memory_pool,
        backlog,
        {},
        {},
        {.overflow_spool = shared_spool});

    std::vector<std::byte> payload(128, std::byte{0x7A});
    for (std::uint64_t sequence = 1; sequence <= kFrameCount; ++sequence) {
        ipc::media::SharedFramePublishRequest request{
            .execution_id = execution_id,
            .session_id = session_id,
            .trace_id = "trace-spool-e2e",
            .selected_sequence = sequence,
            .frame_id = sequence,
            .timestamp_us = static_cast<std::int64_t>(sequence * 1'000),
            .width = 320,
            .height = 180,
            .format = static_cast<std::uint32_t>(ipc::media::SharedFrameFormat::Jpeg),
            .saliency = 0.8,
            .payload = payload,
        };
        ASSERT_TRUE(producer.value()->Publish(request).ok());
        ASSERT_TRUE(receiver.PollOnce().ok());
    }
    EXPECT_EQ(receiver.Snapshot().submitted_frames, 2u);
    EXPECT_EQ(receiver.Snapshot().spooled_frames, kFrameCount - 2);
    EXPECT_EQ(receiver.Snapshot().rejected_frames, 0u);
    ASSERT_TRUE(shared_spool->Seal().ok());

    media::inference::SessionInferenceFrameResultTable results({
        .max_sessions = 1,
        .max_results_per_session = kFrameCount,
    });
    SuccessfulFakeVlm vlm;
    media::inference::InferenceFrameCoordinator coordinator(
        backlog,
        vlm,
        results,
        {
            .worker_count = 2,
            .wait_timeout = 5ms,
            .shutdown_backlog = true,
        });
    media::inference::InferenceFrameSpoolReplayer replayer(
        *shared_spool,
        memory_pool,
        backlog,
        {.max_records_per_pump = 2});
    ASSERT_TRUE(coordinator.Start().ok());

    const auto deadline = std::chrono::steady_clock::now() + 5s;
    bool drained = false;
    while (std::chrono::steady_clock::now() < deadline) {
        auto replay = replayer.Pump();
        ASSERT_TRUE(replay.ok()) << replay.status().message();
        const auto coordinator_snapshot = coordinator.Snapshot();
        if (replay.value().complete &&
            coordinator_snapshot.processed_frames == kFrameCount &&
            backlog.Snapshot().queued_frames == 0) {
            drained = true;
            break;
        }
        std::this_thread::sleep_for(1ms);
    }
    EXPECT_TRUE(drained);
    coordinator.Shutdown();

    auto finalized = results.FinalizeSession(session_id);
    ASSERT_TRUE(finalized.ok()) << finalized.status().message();
    ASSERT_EQ(finalized.value().size(), kFrameCount);
    for (std::size_t index = 0; index < finalized.value().size(); ++index) {
        const auto& record = finalized.value()[index];
        EXPECT_TRUE(record.status.ok());
        ASSERT_TRUE(record.result.has_value());
        EXPECT_EQ(record.frame.execution_id, execution_id);
        EXPECT_EQ(record.frame.selected_sequence, index + 1);
        EXPECT_EQ(record.frame.frame_id, index + 1);
    }
    EXPECT_EQ(replayer.Snapshot().replayed_to_backlog, kFrameCount - 2);
    EXPECT_TRUE(replayer.Snapshot().complete);
    EXPECT_EQ(coordinator.Snapshot().successful_frames, kFrameCount);
}

} // namespace
