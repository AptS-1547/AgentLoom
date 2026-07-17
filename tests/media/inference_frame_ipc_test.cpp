#include "inference_frame_ipc_receiver.h"
#include "inference_frame_ipc_control.h"
#include "inference_frame_ipc_lifecycle.h"
#include "inference_frame_shared_memory.h"

#include "memory_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
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

class SpoolDirectoryCleanup {
public:
    explicit SpoolDirectoryCleanup(std::string_view suffix)
        : path_(std::filesystem::current_path() / "build" / "test-receiver-spool" /
                (std::string(suffix) + "-" + UniqueChannelName("directory"))) {
        std::filesystem::create_directories(path_);
    }

    ~SpoolDirectoryCleanup() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

ipc::media::SharedFramePublishRequest MakeRequest(
    std::string_view session_id,
    std::uint64_t frame_id,
    std::span<const std::byte> payload,
    media::inference::InferenceFrameFormat format = media::inference::InferenceFrameFormat::Jpeg) {
    return {
        .execution_id = "execution-ipc",
        .session_id = session_id,
        .trace_id = "trace-ipc",
        .selected_sequence = frame_id,
        .transport_sequence = frame_id + 100,
        .frame_id = frame_id,
        .timestamp_us = static_cast<std::int64_t>(frame_id * 1'000),
        .published_at_unix_us = 1'700'000'000'000'000LL + static_cast<std::int64_t>(frame_id),
        .width = 320,
        .height = 180,
        .format = static_cast<std::uint32_t>(format),
        .saliency = 0.75,
        .payload = payload,
    };
}

class CapturingAdmissionSink final : public media::inference::IInferenceFrameAdmissionSink {
public:
    core::Status AdmitFrame(media::inference::OwnedInferenceFrame frame) override {
        std::lock_guard lock(mutex_);
        frame_.emplace(std::move(frame));
        return core::Status::Ok();
    }

    std::optional<media::inference::OwnedInferenceFrame> Take() {
        std::lock_guard lock(mutex_);
        auto frame = std::move(frame_);
        frame_.reset();
        return frame;
    }

private:
    std::mutex mutex_;
    std::optional<media::inference::OwnedInferenceFrame> frame_;
};

class RejectingControlSignal final : public ipc::media::IInferenceFrameIpcControlSignal {
public:
    core::Status ApplyGrant(const ipc::media::InferenceFrameIpcGrant&) override {
        return core::Status::Error(core::ErrorCode::Unavailable, "fake control signal unavailable");
    }

    core::Status Revoke(std::uint64_t, std::string_view) override {
        return core::Status::Error(core::ErrorCode::Unavailable, "fake control signal unavailable");
    }

    core::Status Probe(std::uint64_t) override {
        return core::Status::Error(core::ErrorCode::Unavailable, "fake control signal unavailable");
    }
};

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
    EXPECT_EQ(claimed.value().metadata().execution_id, "execution-ipc");
    EXPECT_EQ(claimed.value().metadata().session_id, "session-a");
    EXPECT_EQ(claimed.value().metadata().selected_sequence, 1u);
    EXPECT_EQ(claimed.value().metadata().transport_sequence, 101u);
    EXPECT_EQ(claimed.value().metadata().frame_id, 1u);
    EXPECT_EQ(claimed.value().metadata().published_at_unix_us, 1'700'000'000'000'001LL);
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

TEST(InferenceFrameSharedMemoryTest, FenceInvalidatesOldReaderAndOutstandingClaim) {
    ChannelCleanup cleanup(UniqueChannelName("fence"));
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(producer.ok()) << producer.status().message();
    const auto epoch = producer.value()->Snapshot().epoch;
    auto consumer = ipc::media::SharedMemoryInferenceFrameChannel::Open({
        .name = cleanup.name(),
        .expected_epoch = epoch,
    });
    ASSERT_TRUE(consumer.ok()) << consumer.status().message();
    std::vector<std::byte> payload(16, std::byte{0x31});

    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-fence", 1, payload)).ok());
    auto claimed = consumer.value()->TryClaim();
    ASSERT_TRUE(claimed.ok()) << claimed.status().message();
    ASSERT_TRUE(claimed.value().valid());

    ASSERT_TRUE(producer.value()->Fence().ok());
    EXPECT_TRUE(producer.value()->Snapshot().fenced);
    EXPECT_FALSE(claimed.value().valid());
    EXPECT_TRUE(claimed.value().payload().empty());
    EXPECT_TRUE(claimed.value().metadata().session_id.empty());
    EXPECT_EQ(consumer.value()->TryClaim().status().code(), core::ErrorCode::Cancelled);
    EXPECT_EQ(
        producer.value()->Publish(MakeRequest("session-fence", 2, payload)).code(),
        core::ErrorCode::Cancelled);
    EXPECT_TRUE(claimed.value().Acknowledge().ok());
}

TEST(InferenceFrameSharedMemoryTest, OpenRejectsEpochNotGrantedByControlPlane) {
    ChannelCleanup cleanup(UniqueChannelName("epoch-grant"));
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(producer.ok()) << producer.status().message();
    const auto epoch = producer.value()->Snapshot().epoch;

    auto rejected = ipc::media::SharedMemoryInferenceFrameChannel::Open({
        .name = cleanup.name(),
        .expected_epoch = epoch + 1,
    });
    ASSERT_FALSE(rejected.ok());
    EXPECT_EQ(rejected.status().code(), core::ErrorCode::FailedPrecondition);

    auto accepted = ipc::media::SharedMemoryInferenceFrameChannel::Open({
        .name = cleanup.name(),
        .expected_epoch = epoch,
    });
    EXPECT_TRUE(accepted.ok()) << accepted.status().message();
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

TEST(InferenceFrameIpcLifecycleTest, ConsumerAppliesOnlyCurrentProducerGrant) {
    ChannelCleanup cleanup(UniqueChannelName("apply-grant"));
    auto producer = ipc::media::RecoverableInferenceFrameIpcSink::Create({
        .name = cleanup.name(),
        .slot_count = 4,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(producer.ok()) << producer.status().message();
    const auto old_grant = producer.value()->CurrentGrant();
    auto consumer = ipc::media::ReconnectableInferenceFrameIpcSource::Open({
        .name = old_grant.channel_name,
        .expected_epoch = old_grant.epoch,
    });
    ASSERT_TRUE(consumer.ok()) << consumer.status().message();

    ASSERT_TRUE(producer.value()->Recreate().ok());
    const auto new_grant = producer.value()->CurrentGrant();
    EXPECT_NE(new_grant.epoch, old_grant.epoch);
    EXPECT_EQ(consumer.value()->TryClaim().status().code(), core::ErrorCode::Cancelled);
    EXPECT_EQ(
        consumer.value()->ApplyGrant(old_grant).code(),
        core::ErrorCode::FailedPrecondition);
    auto invalid_layout_grant = new_grant;
    ++invalid_layout_grant.slot_count;
    EXPECT_EQ(
        consumer.value()->ApplyGrant(invalid_layout_grant).code(),
        core::ErrorCode::FailedPrecondition);
    ASSERT_TRUE(consumer.value()->ApplyGrant(new_grant).ok());

    std::vector<std::byte> payload(16, std::byte{0x52});
    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-grant", 2, payload)).ok());
    auto claimed = consumer.value()->TryClaim();
    ASSERT_TRUE(claimed.ok()) << claimed.status().message();
    EXPECT_EQ(claimed.value().metadata().frame_id, 2u);
}

TEST(InferenceFrameIpcControlTest, RevokeAndRecoverRotateGrantWithoutStaleAccess) {
    ChannelCleanup cleanup(UniqueChannelName("control-recover"));
    auto created_sink = ipc::media::RecoverableInferenceFrameIpcSink::Create({
        .name = cleanup.name(),
        .slot_count = 4,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(created_sink.ok()) << created_sink.status().message();
    std::shared_ptr<ipc::media::IRecoverableInferenceFrameIpcSink> sink(
        std::move(created_sink).value());
    auto receiver = std::make_shared<ipc::media::InferenceFrameIpcGrantReceiver>();
    ipc::media::InferenceFrameIpcLeaseCoordinator coordinator(sink, receiver);
    std::vector<std::byte> payload(16, std::byte{0x63});

    ASSERT_TRUE(coordinator.Start().ok());
    const auto old_epoch = coordinator.Snapshot().grant.epoch;
    ASSERT_TRUE(sink->Publish(MakeRequest("session-control", 1, payload)).ok());
    auto old_claim = receiver->TryClaim();
    ASSERT_TRUE(old_claim.ok()) << old_claim.status().message();
    ASSERT_TRUE(old_claim.value().valid());

    ASSERT_TRUE(coordinator.Revoke("inference peer lost").ok());
    EXPECT_FALSE(old_claim.value().valid());
    EXPECT_TRUE(old_claim.value().payload().empty());
    EXPECT_EQ(receiver->TryClaim().status().code(), core::ErrorCode::Cancelled);
    EXPECT_EQ(sink->Publish(MakeRequest("session-control", 2, payload)).code(), core::ErrorCode::Cancelled);

    ASSERT_TRUE(coordinator.Recover().ok());
    const auto recovered = coordinator.Snapshot();
    EXPECT_EQ(recovered.state, ipc::media::InferenceFrameIpcControlState::Granted);
    EXPECT_NE(recovered.grant.epoch, old_epoch);
    EXPECT_EQ(recovered.recoveries, 1u);
    ASSERT_TRUE(sink->Publish(MakeRequest("session-control", 3, payload)).ok());
    auto new_claim = receiver->TryClaim();
    ASSERT_TRUE(new_claim.ok()) << new_claim.status().message();
    EXPECT_EQ(new_claim.value().metadata().frame_id, 3u);
}

TEST(InferenceFrameIpcControlTest, SignalFailureFencesProducer) {
    ChannelCleanup cleanup(UniqueChannelName("control-failure"));
    auto created_sink = ipc::media::RecoverableInferenceFrameIpcSink::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(created_sink.ok()) << created_sink.status().message();
    std::shared_ptr<ipc::media::IRecoverableInferenceFrameIpcSink> sink(
        std::move(created_sink).value());
    auto signal = std::make_shared<RejectingControlSignal>();
    ipc::media::InferenceFrameIpcLeaseCoordinator coordinator(sink, signal);

    const auto status = coordinator.Start();
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), core::ErrorCode::Unavailable);
    EXPECT_TRUE(sink->Snapshot().fenced);
    std::vector<std::byte> payload(16, std::byte{0x64});
    EXPECT_EQ(
        sink->Publish(MakeRequest("session-control-failure", 1, payload)).code(),
        core::ErrorCode::Cancelled);
}

TEST(InferenceFrameIpcControlTest, FailedPeerProbeFencesActiveProducer) {
    ChannelCleanup cleanup(UniqueChannelName("control-probe"));
    auto created_sink = ipc::media::RecoverableInferenceFrameIpcSink::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(created_sink.ok()) << created_sink.status().message();
    std::shared_ptr<ipc::media::IRecoverableInferenceFrameIpcSink> sink(
        std::move(created_sink).value());
    auto receiver = std::make_shared<ipc::media::InferenceFrameIpcGrantReceiver>();
    ipc::media::InferenceFrameIpcLeaseCoordinator coordinator(sink, receiver);
    ASSERT_TRUE(coordinator.Start().ok());
    receiver->Shutdown();

    const auto status = coordinator.CheckPeer();
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), core::ErrorCode::Unavailable);
    EXPECT_EQ(coordinator.Snapshot().state, ipc::media::InferenceFrameIpcControlState::Fenced);
    EXPECT_TRUE(sink->Snapshot().fenced);
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
    EXPECT_EQ(frame.value().metadata().timing.published_at_unix_us, 1'700'000'000'000'009LL);
    EXPECT_GE(
        frame.value().metadata().timing.received_at_unix_us,
        frame.value().metadata().timing.published_at_unix_us);
    EXPECT_GE(
        frame.value().metadata().timing.admitted_at_unix_us,
        frame.value().metadata().timing.received_at_unix_us);
    ASSERT_EQ(frame.value().bytes().size(), payload.size());
    EXPECT_EQ(frame.value().bytes().front(), std::byte{0x4C});
    const auto snapshot = receiver.Snapshot();
    EXPECT_EQ(snapshot.received_frames, 1u);
    EXPECT_EQ(snapshot.copied_frames, 1u);
    EXPECT_EQ(snapshot.submitted_frames, 1u);
}

TEST(InferenceFrameIpcReceiverTest, RoutesOwnedFrameThroughExecutionAdmissionSink) {
    ChannelCleanup cleanup(UniqueChannelName("execution-sink"));
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 128,
    });
    ASSERT_TRUE(producer.ok());
    auto source = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = cleanup.name()});
    ASSERT_TRUE(source.ok());

    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog unused_backlog;
    auto sink = std::make_shared<CapturingAdmissionSink>();
    media::inference::InferenceFrameIpcReceiver receiver(
        *source.value(), pool, unused_backlog, {}, {}, {.admission_sink = sink});
    std::vector<std::byte> payload(80, std::byte{0x6D});
    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-execution-sink", 7, payload)).ok());
    ASSERT_TRUE(receiver.PollOnce().ok());

    auto admitted = sink->Take();
    ASSERT_TRUE(admitted.has_value());
    EXPECT_EQ(admitted->metadata().execution_id, "execution-ipc");
    EXPECT_EQ(admitted->metadata().selected_sequence, 7u);
    EXPECT_EQ(admitted->metadata().timing.published_at_unix_us, 1'700'000'000'000'007LL);
    EXPECT_GE(
        admitted->metadata().timing.received_at_unix_us,
        admitted->metadata().timing.published_at_unix_us);
    ASSERT_EQ(admitted->bytes().size(), payload.size());
    EXPECT_EQ(admitted->bytes().front(), std::byte{0x6D});
    EXPECT_EQ(unused_backlog.Snapshot().queued_frames, 0u);
    EXPECT_EQ(receiver.Snapshot().submitted_frames, 1u);
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

TEST(InferenceFrameIpcReceiverTest, SpoolsOverflowFrameWithoutRejectingAdmission) {
    ChannelCleanup cleanup(UniqueChannelName("overflow-spool"));
    SpoolDirectoryCleanup spool_directory("overflow");
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(producer.ok());
    auto source = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = cleanup.name()});
    ASSERT_TRUE(source.ok());
    auto spool = media::inference::MappedInferenceFrameSpool::Create({
        .root_directory = spool_directory.path(),
        .execution_id = "execution-ipc",
        .segment_bytes = 1024,
        .max_spool_bytes = 4096,
        .flush_on_append = true,
        .remove_on_destroy = true,
    });
    ASSERT_TRUE(spool.ok()) << spool.status().message();
    std::shared_ptr<media::inference::IInferenceFrameSpool> shared_spool(
        std::move(spool).value());
    std::weak_ptr<media::inference::IInferenceFrameSpool> weak_spool = shared_spool;

    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 1,
        .segments_per_session = 1,
        .slots_per_segment = 1,
    });
    media::inference::InferenceFrameIpcReceiver receiver(
        *source.value(),
        pool,
        backlog,
        {},
        {},
        {.overflow_spool = shared_spool});
    shared_spool.reset();
    EXPECT_FALSE(weak_spool.expired());
    std::vector<std::byte> payload(32, std::byte{0x66});

    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-overflow", 1, payload)).ok());
    ASSERT_TRUE(receiver.PollOnce().ok());
    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-overflow", 2, payload)).ok());
    ASSERT_TRUE(receiver.PollOnce().ok());

    const auto receiver_snapshot = receiver.Snapshot();
    EXPECT_EQ(receiver_snapshot.submitted_frames, 1u);
    EXPECT_EQ(receiver_snapshot.spooled_frames, 1u);
    EXPECT_EQ(receiver_snapshot.rejected_frames, 0u);
    auto retained_spool = weak_spool.lock();
    ASSERT_TRUE(retained_spool);
    EXPECT_EQ(retained_spool->Snapshot().admitted_records, 1u);
    ASSERT_TRUE(retained_spool->Seal().ok());
    auto replayed = retained_spool->ReplayNext();
    ASSERT_TRUE(replayed.ok()) << replayed.status().message();
    ASSERT_TRUE(replayed.value().has_value());
    EXPECT_EQ(replayed.value()->metadata().execution_id, "execution-ipc");
    EXPECT_EQ(replayed.value()->metadata().selected_sequence, 2u);
    EXPECT_EQ(replayed.value()->metadata().frame.frame_id, 2u);
    EXPECT_EQ(replayed.value()->metadata().frame.transport_sequence, 102u);
    EXPECT_GT(replayed.value()->metadata().frame.timing.spooled_at_unix_us, 0);
    EXPECT_EQ(replayed.value()->bytes().front(), std::byte{0x66});
}

TEST(InferenceFrameIpcReceiverTest, ReportsSpoolFailureAsRejectedAdmission) {
    ChannelCleanup cleanup(UniqueChannelName("overflow-failure"));
    SpoolDirectoryCleanup spool_directory("failure");
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 64,
    });
    ASSERT_TRUE(producer.ok());
    auto source = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = cleanup.name()});
    ASSERT_TRUE(source.ok());
    auto spool = media::inference::MappedInferenceFrameSpool::Create({
        .root_directory = spool_directory.path(),
        .execution_id = "execution-ipc",
        .segment_bytes = 1024,
        .max_spool_bytes = 1024,
        .flush_on_append = true,
        .remove_on_destroy = true,
    });
    ASSERT_TRUE(spool.ok()) << spool.status().message();
    std::shared_ptr<media::inference::IInferenceFrameSpool> shared_spool(
        std::move(spool).value());
    ASSERT_TRUE(shared_spool->Seal().ok());

    core::BucketMemoryPool pool;
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 1,
        .segments_per_session = 1,
        .slots_per_segment = 1,
    });
    media::inference::InferenceFrameIpcReceiver receiver(
        *source.value(),
        pool,
        backlog,
        {},
        {},
        {.overflow_spool = shared_spool});
    std::vector<std::byte> payload(32, std::byte{0x77});

    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-failure", 1, payload)).ok());
    ASSERT_TRUE(receiver.PollOnce().ok());
    ASSERT_TRUE(producer.value()->Publish(MakeRequest("session-failure", 2, payload)).ok());
    EXPECT_EQ(receiver.PollOnce().code(), core::ErrorCode::FailedPrecondition);
    EXPECT_EQ(receiver.Snapshot().spooled_frames, 0u);
    EXPECT_EQ(receiver.Snapshot().rejected_frames, 1u);
}

} // namespace
