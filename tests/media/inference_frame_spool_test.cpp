#include "inference_frame_spool.h"
#include "inference_frame_spool_replayer.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic<std::uint64_t> next{0};
        const auto suffix = next.fetch_add(1, std::memory_order_relaxed);
        path_ = std::filesystem::current_path() / "build" / "test-media-spool" /
                ("agentloom-spool-test-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 "-" + std::to_string(suffix));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

media::inference::SpoolFrameMetadata MakeMetadata(
    std::string execution_id,
    std::uint64_t sequence,
    std::uint64_t frame_id) {
    media::inference::SpoolFrameMetadata metadata;
    metadata.execution_id = std::move(execution_id);
    metadata.selected_sequence = sequence;
    metadata.frame.session_id = "session-spool";
    metadata.frame.trace_id = "trace-" + std::to_string(frame_id);
    metadata.frame.frame_id = frame_id;
    metadata.frame.timestamp_us = static_cast<std::int64_t>(frame_id * 1'000);
    metadata.frame.width = 320;
    metadata.frame.height = 180;
    metadata.frame.format = media::inference::InferenceFrameFormat::Jpeg;
    metadata.frame.saliency = 0.75;
    return metadata;
}

std::vector<std::byte> MakePayload(std::size_t size, std::uint8_t value) {
    return std::vector<std::byte>(size, static_cast<std::byte>(value));
}

TEST(MappedInferenceFrameSpoolTest, AppendsSealsAndReplaysAcrossSegments) {
    TemporaryDirectory temporary;
    const std::string execution_id = "execution-spool-basic";
    auto spool = media::inference::MappedInferenceFrameSpool::Create({
        .root_directory = temporary.path() / std::filesystem::path(u8"媒体缓存"),
        .execution_id = execution_id,
        .segment_bytes = 512,
        .max_spool_bytes = 2048,
        .flush_on_append = true,
        .remove_on_destroy = true,
    });
    ASSERT_TRUE(spool.ok()) << spool.status().message();

    const auto first_payload = MakePayload(180, 0x11);
    const auto second_payload = MakePayload(180, 0x22);
    const auto third_payload = MakePayload(180, 0x33);
    ASSERT_TRUE(spool.value()->Append(MakeMetadata(execution_id, 2, 20), first_payload).ok());
    ASSERT_TRUE(spool.value()->Append(MakeMetadata(execution_id, 1, 10), second_payload).ok());
    ASSERT_TRUE(spool.value()->Append(MakeMetadata(execution_id, 3, 30), third_payload).ok());

    auto before_seal = spool.value()->ReplayNext();
    ASSERT_FALSE(before_seal.ok());
    EXPECT_EQ(before_seal.status().code(), core::ErrorCode::FailedPrecondition);

    ASSERT_TRUE(spool.value()->Seal().ok());
    ASSERT_TRUE(spool.value()->Seal().ok());
    auto append_after_seal = spool.value()->Append(MakeMetadata(execution_id, 4, 40), first_payload);
    ASSERT_FALSE(append_after_seal.ok());
    EXPECT_EQ(append_after_seal.status().code(), core::ErrorCode::FailedPrecondition);

    const std::vector<std::uint64_t> expected_sequences{2, 1, 3};
    const std::vector<std::uint8_t> expected_values{0x11, 0x22, 0x33};
    for (std::size_t index = 0; index < expected_sequences.size(); ++index) {
        auto replayed = spool.value()->ReplayNext();
        ASSERT_TRUE(replayed.ok()) << replayed.status().message();
        ASSERT_TRUE(replayed.value().has_value());
        auto lease = std::move(replayed).value().value();
        ASSERT_TRUE(lease.valid());
        EXPECT_EQ(lease.metadata().execution_id, execution_id);
        EXPECT_EQ(lease.metadata().selected_sequence, expected_sequences[index]);
        ASSERT_FALSE(lease.bytes().empty());
        EXPECT_EQ(static_cast<std::uint8_t>(lease.bytes().front()), expected_values[index]);
    }

    auto complete = spool.value()->ReplayNext();
    ASSERT_TRUE(complete.ok()) << complete.status().message();
    EXPECT_FALSE(complete.value().has_value());

    const auto snapshot = spool.value()->Snapshot();
    EXPECT_EQ(snapshot.admitted_records, 3u);
    EXPECT_EQ(snapshot.replayed_records, 3u);
    EXPECT_GE(snapshot.segment_count, 2u);
    EXPECT_TRUE(snapshot.sealed);
}

TEST(MappedInferenceFrameSpoolTest, EmptySpoolDoesNotAllocateSegment) {
    TemporaryDirectory temporary;
    auto spool = media::inference::MappedInferenceFrameSpool::Create({
        .root_directory = temporary.path(),
        .execution_id = "execution-spool-empty",
        .segment_bytes = 1024,
        .max_spool_bytes = 1024,
        .flush_on_append = true,
        .remove_on_destroy = true,
    });
    ASSERT_TRUE(spool.ok()) << spool.status().message();
    EXPECT_EQ(spool.value()->Snapshot().segment_count, 0u);
    ASSERT_TRUE(spool.value()->Seal().ok());
    auto replayed = spool.value()->ReplayNext();
    ASSERT_TRUE(replayed.ok()) << replayed.status().message();
    EXPECT_FALSE(replayed.value().has_value());
}

TEST(MappedInferenceFrameSpoolTest, RejectsDuplicateSequenceAndCapacityOverflow) {
    TemporaryDirectory temporary;
    const std::string execution_id = "execution-spool-limit";
    auto spool = media::inference::MappedInferenceFrameSpool::Create({
        .root_directory = temporary.path(),
        .execution_id = execution_id,
        .segment_bytes = 512,
        .max_spool_bytes = 512,
        .flush_on_append = false,
        .remove_on_destroy = true,
    });
    ASSERT_TRUE(spool.ok()) << spool.status().message();

    const auto payload = MakePayload(180, 0x44);
    ASSERT_TRUE(spool.value()->Append(MakeMetadata(execution_id, 1, 1), payload).ok());
    auto duplicate = spool.value()->Append(MakeMetadata(execution_id, 1, 2), payload);
    ASSERT_FALSE(duplicate.ok());
    EXPECT_EQ(duplicate.status().code(), core::ErrorCode::AlreadyExists);

    auto overflow = spool.value()->Append(MakeMetadata(execution_id, 2, 2), payload);
    ASSERT_FALSE(overflow.ok());
    EXPECT_EQ(overflow.status().code(), core::ErrorCode::ResourceExhausted);
}

TEST(MappedInferenceFrameSpoolTest, CleanupWaitsForMappedReadLease) {
    TemporaryDirectory temporary;
    const std::string execution_id = "execution-spool-lease";
    auto spool = media::inference::MappedInferenceFrameSpool::Create({
        .root_directory = temporary.path(),
        .execution_id = execution_id,
        .segment_bytes = 1024,
        .max_spool_bytes = 1024,
        .flush_on_append = true,
        .remove_on_destroy = false,
    });
    ASSERT_TRUE(spool.ok()) << spool.status().message();

    const auto payload = MakePayload(64, 0x55);
    ASSERT_TRUE(spool.value()->Append(MakeMetadata(execution_id, 1, 1), payload).ok());
    ASSERT_TRUE(spool.value()->Seal().ok());
    auto replayed = spool.value()->ReplayNext();
    ASSERT_TRUE(replayed.ok()) << replayed.status().message();
    ASSERT_TRUE(replayed.value().has_value());
    auto lease = std::move(replayed).value().value();

    auto active_cleanup = spool.value()->Cleanup();
    EXPECT_EQ(active_cleanup.code(), core::ErrorCode::FailedPrecondition);
    lease = {};
    ASSERT_TRUE(spool.value()->Cleanup().ok());
    EXPECT_TRUE(spool.value()->Snapshot().cleaned);
}

TEST(MappedInferenceFrameSpoolTest, SerializesConcurrentOutOfOrderAppends) {
    TemporaryDirectory temporary;
    const std::string execution_id = "execution-spool-concurrent";
    auto spool = media::inference::MappedInferenceFrameSpool::Create({
        .root_directory = temporary.path(),
        .execution_id = execution_id,
        .segment_bytes = 4096,
        .max_spool_bytes = 1024 * 1024,
        .flush_on_append = false,
        .remove_on_destroy = true,
    });
    ASSERT_TRUE(spool.ok()) << spool.status().message();

    constexpr std::size_t kThreadCount = 4;
    constexpr std::size_t kRecordsPerThread = 25;
    std::atomic<std::size_t> failures{0};
    std::vector<std::thread> writers;
    for (std::size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
        writers.emplace_back([&, thread_index] {
            for (std::size_t record_index = 0; record_index < kRecordsPerThread; ++record_index) {
                const auto sequence = static_cast<std::uint64_t>(
                    thread_index * kRecordsPerThread + record_index + 1);
                const auto payload = MakePayload(32, static_cast<std::uint8_t>(sequence));
                if (!spool.value()->Append(
                        MakeMetadata(execution_id, sequence, sequence),
                        payload).ok()) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& writer : writers) {
        writer.join();
    }
    ASSERT_EQ(failures.load(std::memory_order_relaxed), 0u);
    ASSERT_TRUE(spool.value()->Seal().ok());

    std::unordered_set<std::uint64_t> observed;
    for (;;) {
        auto replayed = spool.value()->ReplayNext();
        ASSERT_TRUE(replayed.ok()) << replayed.status().message();
        if (!replayed.value().has_value()) {
            break;
        }
        observed.insert(replayed.value()->metadata().selected_sequence);
    }
    EXPECT_EQ(observed.size(), kThreadCount * kRecordsPerThread);
    EXPECT_EQ(spool.value()->Snapshot().replayed_records, kThreadCount * kRecordsPerThread);
}

TEST(InferenceFrameSpoolReplayerTest, RetainsPendingFrameAcrossBackpressure) {
    TemporaryDirectory temporary;
    const std::string execution_id = "execution-spool-replayer";
    auto spool = media::inference::MappedInferenceFrameSpool::Create({
        .root_directory = temporary.path(),
        .execution_id = execution_id,
        .segment_bytes = 4096,
        .max_spool_bytes = 4096,
        .flush_on_append = false,
        .remove_on_destroy = true,
    });
    ASSERT_TRUE(spool.ok()) << spool.status().message();
    for (std::uint64_t sequence = 1; sequence <= 3; ++sequence) {
        const auto payload = MakePayload(64, static_cast<std::uint8_t>(sequence));
        ASSERT_TRUE(spool.value()->Append(
            MakeMetadata(execution_id, sequence, sequence),
            payload).ok());
    }
    ASSERT_TRUE(spool.value()->Seal().ok());

    core::BucketMemoryPool memory_pool;
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 1,
        .segments_per_session = 1,
        .slots_per_segment = 1,
    });
    media::inference::InferenceFrameSpoolReplayer replayer(
        *spool.value(),
        memory_pool,
        backlog,
        {.max_records_per_pump = 8});

    auto first_pump = replayer.Pump();
    ASSERT_TRUE(first_pump.ok()) << first_pump.status().message();
    EXPECT_EQ(first_pump.value().replayed_to_backlog, 1u);
    EXPECT_TRUE(first_pump.value().pending_frame);
    EXPECT_EQ(first_pump.value().backpressure_events, 1u);
    auto first = backlog.WaitTake(std::chrono::milliseconds(100));
    ASSERT_TRUE(first.ok()) << first.status().message();
    EXPECT_EQ(first.value().metadata().selected_sequence, 1u);

    auto second_pump = replayer.Pump();
    ASSERT_TRUE(second_pump.ok()) << second_pump.status().message();
    EXPECT_EQ(second_pump.value().replayed_to_backlog, 2u);
    EXPECT_TRUE(second_pump.value().pending_frame);
    auto second = backlog.WaitTake(std::chrono::milliseconds(100));
    ASSERT_TRUE(second.ok()) << second.status().message();
    EXPECT_EQ(second.value().metadata().selected_sequence, 2u);

    auto final_pump = replayer.Pump();
    ASSERT_TRUE(final_pump.ok()) << final_pump.status().message();
    EXPECT_EQ(final_pump.value().replayed_to_backlog, 3u);
    EXPECT_TRUE(final_pump.value().spool_exhausted);
    EXPECT_TRUE(final_pump.value().complete);
    auto third = backlog.WaitTake(std::chrono::milliseconds(100));
    ASSERT_TRUE(third.ok()) << third.status().message();
    EXPECT_EQ(third.value().metadata().selected_sequence, 3u);
    EXPECT_EQ(third.value().bytes().front(), std::byte{0x03});
}

} // namespace
