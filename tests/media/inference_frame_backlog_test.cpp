#include "inference_frame_backlog.h"

#include "memory_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;

core::Result<media::inference::OwnedInferenceFrame> MakeFrame(
    core::RawMemoryPool& pool,
    std::string session_id,
    std::uint64_t frame_id,
    std::byte value = std::byte{0x2A}) {
    std::vector<std::byte> payload(64, value);
    media::inference::InferenceFrameMetadata metadata;
    metadata.session_id = std::move(session_id);
    metadata.trace_id = "trace-" + std::to_string(frame_id);
    metadata.frame_id = frame_id;
    metadata.timestamp_us = static_cast<std::int64_t>(frame_id * 1'000);
    metadata.width = 8;
    metadata.height = 8;
    metadata.format = media::inference::InferenceFrameFormat::Rgb;
    return media::inference::CopyInferenceFrame(pool, std::move(metadata), payload);
}

media::inference::InferenceFrameResultRecord MakeResult(
    std::string session_id,
    std::uint64_t frame_id,
    std::int64_t timestamp_us) {
    media::inference::InferenceFrameResultRecord record;
    record.frame.session_id = std::move(session_id);
    record.frame.frame_id = frame_id;
    record.frame.timestamp_us = timestamp_us;
    media::VisionInferenceResult result;
    result.scene_hint = "scene-" + std::to_string(frame_id);
    result.confidence = 0.9;
    record.result = std::move(result);
    return record;
}

TEST(InferenceFrameCopyTest, CreatesIndependentPooledPrivateFrame) {
    core::BucketMemoryPool pool;
    std::vector<std::byte> source(32, std::byte{0x11});
    media::inference::InferenceFrameMetadata metadata;
    metadata.session_id = "session-copy";
    metadata.frame_id = 7;
    metadata.timestamp_us = 42'000;

    auto copied = media::inference::CopyInferenceFrame(pool, std::move(metadata), source);
    ASSERT_TRUE(copied.ok()) << copied.status().message();
    source.assign(source.size(), std::byte{0x7F});

    ASSERT_TRUE(copied.value().valid());
    ASSERT_EQ(copied.value().bytes().size(), 32u);
    EXPECT_EQ(copied.value().bytes().front(), std::byte{0x11});
    EXPECT_EQ(copied.value().metadata().session_id, "session-copy");
    EXPECT_EQ(pool.stats().allocated_bytes > 0, true);
}

TEST(SegmentedInferenceFrameBacklogTest, SubmitsAndTakesFramesAcrossSegments) {
    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 4,
        .segments_per_session = 4,
        .slots_per_segment = 4,
    });

    for (std::uint64_t frame_id = 1; frame_id <= 12; ++frame_id) {
        auto frame = MakeFrame(pool, "session-a", frame_id);
        ASSERT_TRUE(frame.ok()) << frame.status().message();
        ASSERT_TRUE(backlog.Submit(std::move(frame).value()).ok());
    }

    std::unordered_set<std::uint64_t> frame_ids;
    for (int index = 0; index < 12; ++index) {
        auto frame = backlog.WaitTake(100ms);
        ASSERT_TRUE(frame.ok()) << frame.status().message();
        frame_ids.insert(frame.value().metadata().frame_id);
    }

    EXPECT_EQ(frame_ids.size(), 12u);
    const auto snapshot = backlog.Snapshot();
    EXPECT_EQ(snapshot.queued_frames, 0u);
    EXPECT_EQ(snapshot.submitted_frames, 12u);
    EXPECT_EQ(snapshot.taken_frames, 12u);
    EXPECT_EQ(snapshot.rejected_frames, 0u);
}

TEST(SegmentedInferenceFrameBacklogTest, RejectsWhenSessionRegionIsFull) {
    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 1,
        .segments_per_session = 1,
        .slots_per_segment = 2,
    });

    for (std::uint64_t frame_id = 1; frame_id <= 2; ++frame_id) {
        auto frame = MakeFrame(pool, "session-full", frame_id);
        ASSERT_TRUE(frame.ok());
        ASSERT_TRUE(backlog.Submit(std::move(frame).value()).ok());
    }
    auto overflow = MakeFrame(pool, "session-full", 3);
    ASSERT_TRUE(overflow.ok());
    const auto status = backlog.Submit(std::move(overflow).value());

    EXPECT_EQ(status.code(), core::ErrorCode::ResourceExhausted);
    EXPECT_EQ(backlog.Snapshot().rejected_frames, 1u);
}

TEST(SegmentedInferenceFrameBacklogTest, CloseSessionDiscardsOnlyThatSession) {
    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 4,
        .segments_per_session = 2,
        .slots_per_segment = 4,
    });

    auto a1 = MakeFrame(pool, "session-a", 1);
    auto a2 = MakeFrame(pool, "session-a", 2);
    auto b1 = MakeFrame(pool, "session-b", 3);
    ASSERT_TRUE(a1.ok() && a2.ok() && b1.ok());
    ASSERT_TRUE(backlog.Submit(std::move(a1).value()).ok());
    ASSERT_TRUE(backlog.Submit(std::move(a2).value()).ok());
    ASSERT_TRUE(backlog.Submit(std::move(b1).value()).ok());

    ASSERT_TRUE(backlog.CloseSession("session-a").ok());
    auto remaining = backlog.WaitTake(100ms);
    ASSERT_TRUE(remaining.ok()) << remaining.status().message();
    EXPECT_EQ(remaining.value().metadata().session_id, "session-b");

    const auto snapshot = backlog.Snapshot();
    EXPECT_EQ(snapshot.session_count, 1u);
    EXPECT_EQ(snapshot.discarded_frames, 2u);
}

TEST(SegmentedInferenceFrameBacklogTest, WaitTakeWakesWhenIngressSubmits) {
    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog;

    auto future = std::async(std::launch::async, [&] {
        return backlog.WaitTake(1s);
    });
    std::this_thread::sleep_for(20ms);
    auto frame = MakeFrame(pool, "session-wake", 1);
    ASSERT_TRUE(frame.ok());
    ASSERT_TRUE(backlog.Submit(std::move(frame).value()).ok());

    auto taken = future.get();
    ASSERT_TRUE(taken.ok()) << taken.status().message();
    EXPECT_EQ(taken.value().metadata().frame_id, 1u);
}

TEST(SegmentedInferenceFrameBacklogTest, ConcurrentWorkersTakeEveryFrameExactlyOnce) {
    constexpr std::size_t kSessionCount = 4;
    constexpr std::size_t kFramesPerSession = 128;
    constexpr std::size_t kWorkerCount = 8;
    constexpr std::size_t kTotalFrames = kSessionCount * kFramesPerSession;

    core::BucketMemoryPool pool(0, 128);
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = kSessionCount,
        .segments_per_session = 8,
        .slots_per_segment = 16,
        .default_wait_timeout = 1s,
    });

    for (std::size_t session = 0; session < kSessionCount; ++session) {
        for (std::size_t frame_index = 0; frame_index < kFramesPerSession; ++frame_index) {
            const auto frame_id = static_cast<std::uint64_t>(session * kFramesPerSession + frame_index + 1);
            auto frame = MakeFrame(pool, "session-" + std::to_string(session), frame_id);
            ASSERT_TRUE(frame.ok()) << frame.status().message();
            ASSERT_TRUE(backlog.Submit(std::move(frame).value()).ok());
        }
    }

    std::mutex seen_mutex;
    std::unordered_set<std::uint64_t> seen;
    std::atomic<std::size_t> taken_count{0};
    std::atomic<std::size_t> failures{0};
    std::vector<std::thread> workers;
    workers.reserve(kWorkerCount);
    for (std::size_t worker = 0; worker < kWorkerCount; ++worker) {
        workers.emplace_back([&] {
            for (;;) {
                const auto index = taken_count.fetch_add(1, std::memory_order_relaxed);
                if (index >= kTotalFrames) {
                    return;
                }
                auto frame = backlog.WaitTake(2s);
                if (!frame.ok()) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                std::lock_guard lock(seen_mutex);
                if (!seen.insert(frame.value().metadata().frame_id).second) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    EXPECT_EQ(failures.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(seen.size(), kTotalFrames);
    EXPECT_EQ(backlog.Snapshot().queued_frames, 0u);
}

TEST(SegmentedInferenceFrameBacklogTest, EnforcesSessionLimit) {
    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 1,
        .segments_per_session = 1,
        .slots_per_segment = 2,
    });

    auto first = MakeFrame(pool, "session-a", 1);
    auto second = MakeFrame(pool, "session-b", 2);
    ASSERT_TRUE(first.ok() && second.ok());
    ASSERT_TRUE(backlog.Submit(std::move(first).value()).ok());
    const auto status = backlog.Submit(std::move(second).value());

    EXPECT_EQ(status.code(), core::ErrorCode::ResourceExhausted);
}

TEST(SegmentedInferenceFrameBacklogTest, ShutdownRejectsNewFramesAndDiscardsQueuedPayloads) {
    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog;
    auto queued = MakeFrame(pool, "session-shutdown", 1);
    ASSERT_TRUE(queued.ok());
    ASSERT_TRUE(backlog.Submit(std::move(queued).value()).ok());

    backlog.Shutdown();
    auto rejected = MakeFrame(pool, "session-shutdown", 2);
    ASSERT_TRUE(rejected.ok());
    const auto status = backlog.Submit(std::move(rejected).value());

    EXPECT_EQ(status.code(), core::ErrorCode::Cancelled);
    EXPECT_TRUE(backlog.Snapshot().shutdown);
    EXPECT_EQ(backlog.Snapshot().queued_frames, 0u);
}

TEST(SegmentedInferenceFrameBacklogTest, ConcurrentCloseDoesNotLeaveFramesInClosedRegion) {
    constexpr std::size_t kIterations = 200;
    core::BucketMemoryPool pool(0, 64);

    for (std::size_t iteration = 0; iteration < kIterations; ++iteration) {
        media::inference::SegmentedInferenceFrameBacklog backlog({
            .max_sessions = 1,
            .segments_per_session = 4,
            .slots_per_segment = 8,
        });
        auto initial = MakeFrame(pool, "session-close-race", 1);
        ASSERT_TRUE(initial.ok());
        ASSERT_TRUE(backlog.Submit(std::move(initial).value()).ok());

        std::atomic<bool> start{false};
        std::thread submitter([&] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            auto frame = MakeFrame(pool, "session-close-race", 2);
            if (frame.ok()) {
                static_cast<void>(backlog.Submit(std::move(frame).value()));
            }
        });
        std::thread closer([&] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            static_cast<void>(backlog.CloseSession("session-close-race"));
        });
        start.store(true, std::memory_order_release);
        submitter.join();
        closer.join();

        const auto snapshot = backlog.Snapshot();
        EXPECT_LE(snapshot.session_count, 1u);
        if (snapshot.session_count == 0) {
            EXPECT_EQ(snapshot.queued_frames, 0u);
        }
    }
    EXPECT_EQ(pool.stats().allocated_bytes, 0u);
}

TEST(InferenceFrameResultTableTest, FinalizesResultsInSessionTimestampOrder) {
    media::inference::SessionInferenceFrameResultTable table;
    ASSERT_TRUE(table.Publish(MakeResult("session-a", 3, 30'000)).ok());
    ASSERT_TRUE(table.Publish(MakeResult("session-a", 1, 10'000)).ok());
    ASSERT_TRUE(table.Publish(MakeResult("session-a", 2, 20'000)).ok());

    auto finalized = table.FinalizeSession("session-a");
    ASSERT_TRUE(finalized.ok()) << finalized.status().message();
    ASSERT_EQ(finalized.value().size(), 3u);
    EXPECT_EQ(finalized.value()[0].frame.frame_id, 1u);
    EXPECT_EQ(finalized.value()[1].frame.frame_id, 2u);
    EXPECT_EQ(finalized.value()[2].frame.frame_id, 3u);
    EXPECT_EQ(table.Snapshot().pending_results, 0u);
}

TEST(InferenceFrameResultTableTest, IsolatesSessionsWithOverlappingTimestamps) {
    media::inference::SessionInferenceFrameResultTable table;
    ASSERT_TRUE(table.Publish(MakeResult("session-a", 1, 10'000)).ok());
    ASSERT_TRUE(table.Publish(MakeResult("session-b", 2, 10'000)).ok());

    auto session_a = table.FinalizeSession("session-a");
    ASSERT_TRUE(session_a.ok());
    ASSERT_EQ(session_a.value().size(), 1u);
    EXPECT_EQ(session_a.value().front().frame.session_id, "session-a");
    EXPECT_EQ(table.Snapshot().session_count, 1u);

    auto session_b = table.FinalizeSession("session-b");
    ASSERT_TRUE(session_b.ok());
    ASSERT_EQ(session_b.value().size(), 1u);
    EXPECT_EQ(session_b.value().front().frame.session_id, "session-b");
}

TEST(InferenceFrameResultTableTest, UsesFrameIdToDisambiguateEqualTimestamps) {
    media::inference::SessionInferenceFrameResultTable table;
    ASSERT_TRUE(table.Publish(MakeResult("session-a", 2, 10'000)).ok());
    ASSERT_TRUE(table.Publish(MakeResult("session-a", 1, 10'000)).ok());

    auto finalized = table.FinalizeSession("session-a");
    ASSERT_TRUE(finalized.ok());
    ASSERT_EQ(finalized.value().size(), 2u);
    EXPECT_EQ(finalized.value()[0].frame.frame_id, 1u);
    EXPECT_EQ(finalized.value()[1].frame.frame_id, 2u);
}

TEST(InferenceFrameResultTableTest, AcceptsFailedTerminalResultsWithoutPayload) {
    media::inference::SessionInferenceFrameResultTable table;
    media::inference::InferenceFrameResultRecord failed;
    failed.frame.session_id = "session-failure";
    failed.frame.frame_id = 1;
    failed.frame.timestamp_us = 5'000;
    failed.status = core::Status::Error(core::ErrorCode::Unavailable, "VLM unavailable");

    ASSERT_TRUE(table.Publish(std::move(failed)).ok());
    auto finalized = table.FinalizeSession("session-failure");
    ASSERT_TRUE(finalized.ok());
    ASSERT_EQ(finalized.value().size(), 1u);
    EXPECT_EQ(finalized.value().front().status.code(), core::ErrorCode::Unavailable);
    EXPECT_FALSE(finalized.value().front().result.has_value());
}

TEST(InferenceFrameResultTableTest, ConcurrentPublishProducesStableOrderedResult) {
    constexpr std::size_t kThreadCount = 8;
    constexpr std::size_t kResultsPerThread = 64;
    constexpr std::size_t kTotalResults = kThreadCount * kResultsPerThread;

    media::inference::SessionInferenceFrameResultTable table({
        .max_sessions = 4,
        .max_results_per_session = kTotalResults,
    });
    std::atomic<std::size_t> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (std::size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
        threads.emplace_back([&, thread_index] {
            for (std::size_t index = 0; index < kResultsPerThread; ++index) {
                const auto frame_id = static_cast<std::uint64_t>(
                    thread_index * kResultsPerThread + index + 1);
                auto status = table.Publish(MakeResult(
                    "session-concurrent",
                    frame_id,
                    static_cast<std::int64_t>(frame_id * 1'000)));
                if (!status.ok()) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    ASSERT_EQ(failures.load(std::memory_order_relaxed), 0u);
    auto finalized = table.FinalizeSession("session-concurrent");
    ASSERT_TRUE(finalized.ok()) << finalized.status().message();
    ASSERT_EQ(finalized.value().size(), kTotalResults);
    for (std::size_t index = 0; index < finalized.value().size(); ++index) {
        EXPECT_EQ(finalized.value()[index].frame.frame_id, index + 1);
    }
}

TEST(InferenceFrameResultTableTest, ShutdownRejectsNewResults) {
    media::inference::SessionInferenceFrameResultTable table;
    ASSERT_TRUE(table.Publish(MakeResult("session-shutdown", 1, 1'000)).ok());
    table.Shutdown();

    const auto status = table.Publish(MakeResult("session-shutdown", 2, 2'000));
    EXPECT_EQ(status.code(), core::ErrorCode::Cancelled);
    EXPECT_EQ(table.Snapshot().session_count, 0u);
    EXPECT_EQ(table.Snapshot().pending_results, 0u);
}

} // namespace
