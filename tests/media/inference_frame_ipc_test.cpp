#include "inference_frame_ipc_receiver.h"
#include "inference_frame_ipc_lifecycle.h"
#include "inference_frame_shared_memory.h"

#include "memory_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;

std::string UniqueChannelName(std::string_view suffix) {
    static std::atomic<std::uint64_t> sequence{0};
    return "agent_frame_ipc_" + std::string(suffix) + "_" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

class ChannelCleanup {
public:
    explicit ChannelCleanup(std::string name)
        : name_(std::move(name)) {
        ipc::media::SharedMemoryInferenceFrameChannel::Remove(name_);
    }

    ~ChannelCleanup() {
        ipc::media::SharedMemoryInferenceFrameChannel::Remove(name_);
    }

    const std::string& name() const noexcept {
        return name_;
    }

private:
    std::string name_;
};

ipc::media::SharedFramePublishRequest MakeRequest(
    std::string_view session_id,
    std::uint64_t frame_id,
    std::span<const std::byte> payload,
    media::inference::InferenceFrameFormat format = media::inference::InferenceFrameFormat::Jpeg) {
    return {
        .session_id = session_id,
        .trace_id = "trace-ipc",
        .frame_id = frame_id,
        .timestamp_us = static_cast<std::int64_t>(frame_id * 1'000),
        .width = 320,
        .height = 180,
        .format = static_cast<std::uint32_t>(format),
        .saliency = 0.75,
        .payload = payload,
    };
}

TEST(InferenceFrameSharedMemoryTest, PublishesClaimsAndReusesAcknowledgedSlot) {
    ChannelCleanup cleanup(UniqueChannelName("reuse"));
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(producer.ok()) << producer.status().message();
    auto consumer = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = cleanup.name()});
    ASSERT_TRUE(consumer.ok()) << consumer.status().message();

    std::vector<std::byte> payload(32, std::byte{0x2A});
    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-a", 1, payload)).ok());
    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-a", 2, payload)).ok());
    EXPECT_EQ(
        producer.value()->Publish(MakeRequest("session-a", 3, payload)).code(),
        core::ErrorCode::ResourceExhausted);

    auto claimed = consumer.value()->TryClaim();
    ASSERT_TRUE(claimed.ok()) << claimed.status().message();
    EXPECT_EQ(claimed.value().metadata().session_id, "session-a");
    EXPECT_EQ(claimed.value().metadata().frame_id, 1u);
    ASSERT_EQ(claimed.value().payload().size(), payload.size());
    EXPECT_EQ(claimed.value().payload().front(), std::byte{0x2A});
    ASSERT_TRUE(claimed.value().Acknowledge().ok());

    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-a", 3, payload)).ok());
    const auto snapshot = producer.value()->Snapshot();
    EXPECT_EQ(snapshot.published_frames, 3u);
    EXPECT_EQ(snapshot.claimed_frames, 1u);
    EXPECT_EQ(snapshot.acknowledged_frames, 1u);
}

TEST(InferenceFrameSharedMemoryTest, ClaimDestructorAcknowledgesSlot) {
    ChannelCleanup cleanup(UniqueChannelName("raii"));
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(producer.ok());
    auto consumer = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = cleanup.name()});
    ASSERT_TRUE(consumer.ok());
    std::vector<std::byte> payload(16, std::byte{0x11});

    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-raii", 1, payload)).ok());
    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-raii", 2, payload)).ok());
    {
        auto claimed = consumer.value()->TryClaim();
        ASSERT_TRUE(claimed.ok());
        ASSERT_TRUE(claimed.value().valid());
    }
    EXPECT_EQ(producer.value()->Snapshot().acknowledged_frames, 1u);
    EXPECT_TRUE(producer.value()->Publish(MakeRequest("session-raii", 3, payload)).ok());
}

TEST(InferenceFrameSharedMemoryTest, RejectsSingleSlotSequenceLayout) {
    ChannelCleanup cleanup(UniqueChannelName("single-slot"));
    auto channel = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 1,
        .payload_capacity = 64,
    });

    ASSERT_FALSE(channel.ok());
    EXPECT_EQ(channel.status().code(), core::ErrorCode::InvalidArgument);
}

TEST(InferenceFrameSharedMemoryTest, ConcurrentPublishAndClaimIsExactlyOnce) {
    constexpr std::size_t kProducerCount = 4;
    constexpr std::size_t kConsumerCount = 4;
    constexpr std::size_t kFramesPerProducer = 128;
    constexpr std::size_t kFrameCount = kProducerCount * kFramesPerProducer;

    ChannelCleanup cleanup(UniqueChannelName("concurrent"));
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 64,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(producer.ok()) << producer.status().message();
    auto consumer = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = cleanup.name()});
    ASSERT_TRUE(consumer.ok()) << consumer.status().message();

    std::atomic<std::size_t> producers_done{0};
    std::atomic<std::size_t> consumed{0};
    std::atomic<std::size_t> failures{0};
    std::mutex seen_mutex;
    std::unordered_set<std::uint64_t> seen;
    std::vector<std::thread> threads;

    for (std::size_t producer_index = 0; producer_index < kProducerCount; ++producer_index) {
        threads.emplace_back([&, producer_index] {
            std::vector<std::byte> payload(32);
            for (std::size_t index = 0; index < kFramesPerProducer; ++index) {
                const auto frame_id = static_cast<std::uint64_t>(
                    producer_index * kFramesPerProducer + index + 1);
                std::memcpy(payload.data(), &frame_id, sizeof(frame_id));
                for (;;) {
                    const auto status = producer.value()->Publish(
                        MakeRequest("session-concurrent", frame_id, payload));
                    if (status.ok()) {
                        break;
                    }
                    if (status.code() != core::ErrorCode::ResourceExhausted) {
                        failures.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                    std::this_thread::yield();
                }
            }
            producers_done.fetch_add(1, std::memory_order_release);
        });
    }

    for (std::size_t consumer_index = 0; consumer_index < kConsumerCount; ++consumer_index) {
        threads.emplace_back([&] {
            for (;;) {
                if (consumed.load(std::memory_order_acquire) >= kFrameCount) {
                    return;
                }
                auto claimed = consumer.value()->TryClaim();
                if (!claimed.ok()) {
                    if (claimed.status().code() == core::ErrorCode::NotFound) {
                        if (producers_done.load(std::memory_order_acquire) == kProducerCount &&
                            consumed.load(std::memory_order_acquire) >= kFrameCount) {
                            return;
                        }
                        std::this_thread::yield();
                        continue;
                    }
                    failures.fetch_add(1, std::memory_order_relaxed);
                    return;
                }

                std::uint64_t payload_frame_id = 0;
                std::memcpy(
                    &payload_frame_id,
                    claimed.value().payload().data(),
                    sizeof(payload_frame_id));
                if (payload_frame_id != claimed.value().metadata().frame_id) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
                {
                    std::lock_guard lock(seen_mutex);
                    if (!seen.insert(payload_frame_id).second) {
                        failures.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                claimed.value().Acknowledge();
                consumed.fetch_add(1, std::memory_order_release);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(failures.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(consumed.load(std::memory_order_relaxed), kFrameCount);
    EXPECT_EQ(seen.size(), kFrameCount);
    EXPECT_EQ(producer.value()->Snapshot().acknowledged_frames, kFrameCount);
}

TEST(InferenceFrameIpcLifecycleTest, ConsumerReconnectsAfterProducerRecreatesEpoch) {
    ChannelCleanup cleanup(UniqueChannelName("reconnect"));
    auto producer = ipc::media::RecoverableInferenceFrameIpcSink::Create({
        .name = cleanup.name(),
        .slot_count = 4,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(producer.ok()) << producer.status().message();
    auto consumer = ipc::media::ReconnectableInferenceFrameIpcSource::Open({.name = cleanup.name()});
    ASSERT_TRUE(consumer.ok()) << consumer.status().message();
    std::vector<std::byte> payload(16, std::byte{0x44});

    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-reconnect", 1, payload)).ok());
    auto before = consumer.value()->TryClaim();
    ASSERT_TRUE(before.ok()) << before.status().message();
    ASSERT_TRUE(before.value().Acknowledge().ok());
    const auto old_epoch = producer.value()->Snapshot().epoch;

    ASSERT_TRUE(producer.value()->Recreate().ok());
    ASSERT_TRUE(consumer.value()->Reconnect().ok());
    EXPECT_NE(producer.value()->Snapshot().epoch, old_epoch);
    EXPECT_EQ(consumer.value()->LifecycleSnapshot().recoveries, 1u);

    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-reconnect", 2, payload)).ok());
    auto after = consumer.value()->TryClaim();
    ASSERT_TRUE(after.ok()) << after.status().message();
    EXPECT_EQ(after.value().metadata().frame_id, 2u);
}

TEST(InferenceFrameIpcReceiverTest, CopiesPrivatePayloadAndSubmitsBacklogFrame) {
    ChannelCleanup cleanup(UniqueChannelName("receiver"));
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 128,
    });
    ASSERT_TRUE(producer.ok());
    auto source = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = cleanup.name()});
    ASSERT_TRUE(source.ok());

    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 2,
        .segments_per_session = 1,
        .slots_per_segment = 2,
    });
    media::inference::InferenceFrameIpcReceiver receiver(*source.value(), pool, backlog);
    std::vector<std::byte> payload(80, std::byte{0x4C});

    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-rx", 9, payload)).ok());
    ASSERT_TRUE(receiver.PollOnce().ok());
    ASSERT_EQ(producer.value()->Snapshot().acknowledged_frames, 1u);

    auto frame = backlog.WaitTake(100ms);
    ASSERT_TRUE(frame.ok()) << frame.status().message();
    EXPECT_EQ(frame.value().metadata().session_id, "session-rx");
    EXPECT_EQ(frame.value().metadata().frame_id, 9u);
    ASSERT_EQ(frame.value().bytes().size(), payload.size());
    EXPECT_EQ(frame.value().bytes().front(), std::byte{0x4C});
    const auto snapshot = receiver.Snapshot();
    EXPECT_EQ(snapshot.received_frames, 1u);
    EXPECT_EQ(snapshot.copied_frames, 1u);
    EXPECT_EQ(snapshot.submitted_frames, 1u);
}

TEST(InferenceFrameIpcReceiverTest, ObserverExceptionDoesNotRejectAdmittedFrame) {
    ChannelCleanup cleanup(UniqueChannelName("receiver-observer"));
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 128,
    });
    ASSERT_TRUE(producer.ok());
    auto source = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = cleanup.name()});
    ASSERT_TRUE(source.ok());

    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog;
    media::inference::InferenceFrameIpcReceiver receiver(
        *source.value(),
        pool,
        backlog,
        {},
        [](const ipc::media::SharedFrameMetadata&, const core::Status&) {
            throw std::runtime_error("observer failure");
        });
    std::vector<std::byte> payload(32, std::byte{0x55});

    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-observer", 1, payload)).ok());
    EXPECT_TRUE(receiver.PollOnce().ok());
    auto frame = backlog.WaitTake(100ms);
    ASSERT_TRUE(frame.ok()) << frame.status().message();
    EXPECT_EQ(frame.value().metadata().frame_id, 1u);
    EXPECT_EQ(receiver.Snapshot().submitted_frames, 1u);
    EXPECT_EQ(receiver.Snapshot().rejected_frames, 0u);
}

TEST(InferenceFrameIpcReceiverTest, RejectsInvalidFormatAndReleasesSharedSlot) {
    ChannelCleanup cleanup(UniqueChannelName("invalid-format"));
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(producer.ok());
    auto source = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = cleanup.name()});
    ASSERT_TRUE(source.ok());

    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog;
    media::inference::InferenceFrameIpcReceiver receiver(*source.value(), pool, backlog);
    std::vector<std::byte> payload(32, std::byte{0x33});
    auto invalid = MakeRequest("session-invalid", 1, payload);
    invalid.format = 999;

    ASSERT_TRUE(producer.value()->Publish(invalid).ok());
    EXPECT_EQ(receiver.PollOnce().code(), core::ErrorCode::InvalidArgument);
    EXPECT_EQ(producer.value()->Snapshot().acknowledged_frames, 1u);
    EXPECT_TRUE(producer.value()->Publish(MakeRequest("session-invalid", 2, payload)).ok());
    EXPECT_EQ(receiver.Snapshot().rejected_frames, 1u);
}

TEST(InferenceFrameIpcReceiverTest, BacklogRejectionDoesNotRetainSharedSlot) {
    ChannelCleanup cleanup(UniqueChannelName("backlog-full"));
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(producer.ok());
    auto source = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = cleanup.name()});
    ASSERT_TRUE(source.ok());

    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 1,
        .segments_per_session = 1,
        .slots_per_segment = 1,
    });
    media::inference::InferenceFrameIpcReceiver receiver(*source.value(), pool, backlog);
    std::vector<std::byte> payload(32, std::byte{0x55});

    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-full", 1, payload)).ok());
    ASSERT_TRUE(receiver.PollOnce().ok());
    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-full", 2, payload)).ok());
    EXPECT_EQ(receiver.PollOnce().code(), core::ErrorCode::ResourceExhausted);
    EXPECT_EQ(producer.value()->Snapshot().acknowledged_frames, 2u);
    EXPECT_TRUE(producer.value()->Publish(MakeRequest("session-full", 3, payload)).ok());
}

} // namespace
