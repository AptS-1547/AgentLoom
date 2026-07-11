#include "inference_frame_coordinator.h"

#include "memory_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;

core::Result<media::inference::OwnedInferenceFrame> MakeEncodedFrame(
    core::RawMemoryPool& pool,
    std::string session_id,
    std::uint64_t frame_id,
    media::inference::InferenceFrameFormat format = media::inference::InferenceFrameFormat::Jpeg) {
    std::vector<std::byte> payload(64, static_cast<std::byte>(frame_id & 0xFF));
    media::inference::InferenceFrameMetadata metadata;
    metadata.session_id = std::move(session_id);
    metadata.trace_id = "trace-" + std::to_string(frame_id);
    metadata.frame_id = frame_id;
    metadata.timestamp_us = static_cast<std::int64_t>(frame_id * 1'000);
    metadata.width = 8;
    metadata.height = 8;
    metadata.format = format;
    metadata.saliency = 0.8;
    return media::inference::CopyInferenceFrame(pool, std::move(metadata), payload);
}

bool WaitForProcessed(
    const media::inference::InferenceFrameCoordinator& coordinator,
    std::size_t expected,
    std::chrono::milliseconds timeout = 2s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (coordinator.Snapshot().processed_frames >= expected) {
            return true;
        }
        std::this_thread::sleep_for(2ms);
    }
    return coordinator.Snapshot().processed_frames >= expected;
}

class FakeVlmClient final : public media::IVlmVisionClient {
public:
    enum class Behavior {
        Success,
        Failure,
        ThrowStandard,
        ThrowUnknown
    };

    explicit FakeVlmClient(Behavior behavior = Behavior::Success)
        : behavior_(behavior) {}

    core::Result<media::VisionInferenceResult> Analyze(
        const media::VisionInferenceRequest& request) override {
        calls_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard lock(mutex_);
            frame_ids_.insert(request.frame_id);
            if (!request.encoded_image.empty()) {
                observed_bytes_.push_back(request.encoded_image.front());
            }
            mime_types_.push_back(request.mime_type);
        }

        switch (behavior_) {
        case Behavior::Success: {
            media::VisionInferenceResult result;
            result.scene_hint = "scene-" + std::to_string(request.frame_id);
            result.confidence = 0.9;
            return result;
        }
        case Behavior::Failure:
            return core::Status::Error(core::ErrorCode::Unavailable, "fake VLM unavailable");
        case Behavior::ThrowStandard:
            throw std::runtime_error("fake VLM exception");
        case Behavior::ThrowUnknown:
            throw 42;
        }
        return core::Status::Error(core::ErrorCode::Unknown, "unreachable fake VLM behavior");
    }

    std::size_t calls() const noexcept {
        return calls_.load(std::memory_order_relaxed);
    }

    std::unordered_set<std::uint64_t> frame_ids() const {
        std::lock_guard lock(mutex_);
        return frame_ids_;
    }

    std::vector<std::byte> observed_bytes() const {
        std::lock_guard lock(mutex_);
        return observed_bytes_;
    }

    std::vector<std::string> mime_types() const {
        std::lock_guard lock(mutex_);
        return mime_types_;
    }

private:
    Behavior behavior_;
    std::atomic<std::size_t> calls_{0};
    mutable std::mutex mutex_;
    std::unordered_set<std::uint64_t> frame_ids_;
    std::vector<std::byte> observed_bytes_;
    std::vector<std::string> mime_types_;
};

class RejectingResultTable final : public media::inference::IInferenceFrameResultTable {
public:
    core::Status Publish(media::inference::InferenceFrameResultRecord) override {
        publish_calls_.fetch_add(1, std::memory_order_relaxed);
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "fake result table full");
    }

    core::Result<std::vector<media::inference::InferenceFrameResultRecord>> FinalizeSession(
        std::string_view) override {
        return core::Status::Error(core::ErrorCode::NotFound, "no fake results");
    }

    void Shutdown() override {}

    media::inference::InferenceFrameResultTableSnapshot Snapshot() const override {
        return {
            .rejected_results = publish_calls_.load(std::memory_order_relaxed),
        };
    }

    std::size_t publish_calls() const noexcept {
        return publish_calls_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<std::size_t> publish_calls_{0};
};

TEST(InferenceFrameCoordinatorTest, PublishesSuccessfulJpegResultAndKeepsPayloadAlive) {
    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog;
    media::inference::SessionInferenceFrameResultTable results;
    FakeVlmClient vlm;
    media::inference::InferenceFrameCoordinator coordinator(
        backlog,
        vlm,
        results,
        {.worker_count = 1, .wait_timeout = 20ms});

    auto frame = MakeEncodedFrame(pool, "session-success", 7);
    ASSERT_TRUE(frame.ok()) << frame.status().message();
    ASSERT_TRUE(backlog.Submit(std::move(frame).value()).ok());
    ASSERT_TRUE(coordinator.Start().ok());
    ASSERT_TRUE(WaitForProcessed(coordinator, 1));
    coordinator.Shutdown();

    auto finalized = results.FinalizeSession("session-success");
    ASSERT_TRUE(finalized.ok()) << finalized.status().message();
    ASSERT_EQ(finalized.value().size(), 1u);
    ASSERT_TRUE(finalized.value().front().status.ok());
    ASSERT_TRUE(finalized.value().front().result.has_value());
    EXPECT_EQ(finalized.value().front().result->scene_hint, "scene-7");
    EXPECT_EQ(vlm.mime_types(), std::vector<std::string>{"image/jpeg"});
    EXPECT_EQ(vlm.observed_bytes(), std::vector<std::byte>{std::byte{0x07}});
    EXPECT_EQ(coordinator.Snapshot().successful_frames, 1u);
}

TEST(InferenceFrameCoordinatorTest, PublishesVlmFailureAsTerminalRecord) {
    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog;
    media::inference::SessionInferenceFrameResultTable results;
    FakeVlmClient vlm(FakeVlmClient::Behavior::Failure);
    media::inference::InferenceFrameCoordinator coordinator(backlog, vlm, results);

    auto frame = MakeEncodedFrame(pool, "session-failure", 1);
    ASSERT_TRUE(frame.ok());
    ASSERT_TRUE(backlog.Submit(std::move(frame).value()).ok());
    ASSERT_TRUE(coordinator.Start().ok());
    ASSERT_TRUE(WaitForProcessed(coordinator, 1));
    coordinator.Shutdown();

    auto finalized = results.FinalizeSession("session-failure");
    ASSERT_TRUE(finalized.ok());
    ASSERT_EQ(finalized.value().size(), 1u);
    EXPECT_EQ(finalized.value().front().status.code(), core::ErrorCode::Unavailable);
    EXPECT_FALSE(finalized.value().front().result.has_value());
    EXPECT_EQ(coordinator.Snapshot().failed_frames, 1u);
}

TEST(InferenceFrameCoordinatorTest, SanitizesVlmExceptions) {
    for (const auto behavior : {FakeVlmClient::Behavior::ThrowStandard, FakeVlmClient::Behavior::ThrowUnknown}) {
        core::BucketMemoryPool pool;
        media::inference::SegmentedInferenceFrameBacklog backlog;
        media::inference::SessionInferenceFrameResultTable results;
        FakeVlmClient vlm(behavior);
        media::inference::InferenceFrameCoordinator coordinator(
            backlog,
            vlm,
            results,
            {.worker_count = 1, .wait_timeout = 20ms});

        auto frame = MakeEncodedFrame(pool, "session-exception", 1);
        ASSERT_TRUE(frame.ok());
        ASSERT_TRUE(backlog.Submit(std::move(frame).value()).ok());
        ASSERT_TRUE(coordinator.Start().ok());
        ASSERT_TRUE(WaitForProcessed(coordinator, 1));
        coordinator.Shutdown();

        auto finalized = results.FinalizeSession("session-exception");
        ASSERT_TRUE(finalized.ok());
        ASSERT_EQ(finalized.value().size(), 1u);
        EXPECT_NE(finalized.value().front().status.code(), core::ErrorCode::Ok);
        EXPECT_EQ(finalized.value().front().status.message(), "VLM client failed unexpectedly");
    }
}

TEST(InferenceFrameCoordinatorTest, MultipleWorkersProcessEachFrameExactlyOnce) {
    constexpr std::size_t kFrameCount = 128;
    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 4,
        .segments_per_session = 8,
        .slots_per_segment = 32,
    });
    media::inference::SessionInferenceFrameResultTable results;
    FakeVlmClient vlm;
    media::inference::InferenceFrameCoordinator coordinator(
        backlog,
        vlm,
        results,
        {.worker_count = 4, .wait_timeout = 20ms});

    for (std::uint64_t frame_id = 1; frame_id <= kFrameCount; ++frame_id) {
        auto frame = MakeEncodedFrame(pool, "session-concurrent", frame_id);
        ASSERT_TRUE(frame.ok());
        ASSERT_TRUE(backlog.Submit(std::move(frame).value()).ok());
    }
    ASSERT_TRUE(coordinator.Start().ok());
    ASSERT_TRUE(WaitForProcessed(coordinator, kFrameCount));
    coordinator.Shutdown();

    EXPECT_EQ(vlm.calls(), kFrameCount);
    EXPECT_EQ(vlm.frame_ids().size(), kFrameCount);
    EXPECT_EQ(coordinator.Snapshot().successful_frames, kFrameCount);
    auto finalized = results.FinalizeSession("session-concurrent");
    ASSERT_TRUE(finalized.ok());
    EXPECT_EQ(finalized.value().size(), kFrameCount);
}

TEST(InferenceFrameCoordinatorTest, CountsResultPublishFailures) {
    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog;
    RejectingResultTable results;
    FakeVlmClient vlm;
    media::inference::InferenceFrameCoordinator coordinator(backlog, vlm, results);

    auto frame = MakeEncodedFrame(pool, "session-result-full", 1);
    ASSERT_TRUE(frame.ok());
    ASSERT_TRUE(backlog.Submit(std::move(frame).value()).ok());
    ASSERT_TRUE(coordinator.Start().ok());
    ASSERT_TRUE(WaitForProcessed(coordinator, 1));
    coordinator.Shutdown();

    EXPECT_EQ(results.publish_calls(), 1u);
    EXPECT_EQ(coordinator.Snapshot().result_publish_failures, 1u);
}

TEST(InferenceFrameCoordinatorTest, RejectsUnencodedFramesWithoutCallingVlm) {
    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog;
    media::inference::SessionInferenceFrameResultTable results;
    FakeVlmClient vlm;
    media::inference::InferenceFrameCoordinator coordinator(backlog, vlm, results);

    auto frame = MakeEncodedFrame(
        pool,
        "session-rgb",
        1,
        media::inference::InferenceFrameFormat::Rgb);
    ASSERT_TRUE(frame.ok());
    ASSERT_TRUE(backlog.Submit(std::move(frame).value()).ok());
    ASSERT_TRUE(coordinator.Start().ok());
    ASSERT_TRUE(WaitForProcessed(coordinator, 1));
    coordinator.Shutdown();

    EXPECT_EQ(vlm.calls(), 0u);
    auto finalized = results.FinalizeSession("session-rgb");
    ASSERT_TRUE(finalized.ok());
    ASSERT_EQ(finalized.value().size(), 1u);
    EXPECT_EQ(finalized.value().front().status.code(), core::ErrorCode::InvalidArgument);
}

TEST(InferenceFrameCoordinatorTest, ShutdownWakesIdleWorkersPromptly) {
    media::inference::SegmentedInferenceFrameBacklog backlog;
    media::inference::SessionInferenceFrameResultTable results;
    FakeVlmClient vlm;
    media::inference::InferenceFrameCoordinator coordinator(
        backlog,
        vlm,
        results,
        {.worker_count = 2, .wait_timeout = 5s});

    ASSERT_TRUE(coordinator.Start().ok());
    std::this_thread::sleep_for(20ms);
    const auto started = std::chrono::steady_clock::now();
    coordinator.Shutdown();
    const auto elapsed = std::chrono::steady_clock::now() - started;

    EXPECT_LT(elapsed, 500ms);
    EXPECT_FALSE(coordinator.Snapshot().running);
}

} // namespace
