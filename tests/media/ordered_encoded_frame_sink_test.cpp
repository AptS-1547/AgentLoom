#include "memory_pool.h"
#include "ordered_encoded_frame_sink.h"

#include <gtest/gtest.h>

#include <mutex>
#include <vector>

namespace {

media::EncodedVideoFrame MakeEncoded(
    core::RawMemoryPool& pool,
    std::uint64_t selected_sequence,
    std::int64_t timestamp_us) {
    auto block = pool.allocate(32, alignof(std::max_align_t));
    EXPECT_TRUE(block.ok());
    auto shared = core::SharedMemoryBlock::adopt(std::move(block).value());
    EXPECT_TRUE(shared.ok());
    media::EncodedVideoFrameMetadata metadata;
    metadata.execution_id = "execution-a";
    metadata.session_id = "session-a";
    metadata.selected_sequence = selected_sequence;
    metadata.frame_id = selected_sequence;
    metadata.timestamp_us = timestamp_us;
    metadata.width = 8;
    metadata.height = 8;
    return media::EncodedVideoFrame(std::move(metadata), std::move(shared).value(), 32);
}

class RecordingEncodedSink final : public media::IEncodedVideoFrameSink {
public:
    core::Status Publish(media::EncodedVideoFrame frame) override {
        return PublishBorrowed(frame);
    }

    core::Status PublishBorrowed(const media::EncodedVideoFrame& frame) override {
        std::lock_guard lock(mutex_);
        selected_.push_back(frame.metadata().selected_sequence);
        transport_.push_back(frame.metadata().transport_sequence);
        return core::Status::Ok();
    }

    std::vector<std::uint64_t> selected() const {
        std::lock_guard lock(mutex_);
        return selected_;
    }

    std::vector<std::uint64_t> transport() const {
        std::lock_guard lock(mutex_);
        return transport_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<std::uint64_t> selected_;
    std::vector<std::uint64_t> transport_;
};

} // namespace

TEST(OrderedEncodedFrameSinkTest, RestoresSelectedOrderAndAssignsContinuousTransportSequence) {
    core::BucketMemoryPool pool;
    auto downstream = std::make_shared<RecordingEncodedSink>();
    media::OrderedEncodedFrameSink orderer(downstream, {.window_capacity = 8});

    ASSERT_TRUE(orderer.Publish(MakeEncoded(pool, 3, 30'000)).ok());
    ASSERT_TRUE(orderer.Publish(MakeEncoded(pool, 1, 10'000)).ok());
    ASSERT_TRUE(orderer.Publish(MakeEncoded(pool, 2, 20'000)).ok());
    auto sealed = orderer.SealExecution("session-a", "execution-a", 3);
    ASSERT_TRUE(sealed.ok());
    EXPECT_EQ(sealed.value(), 3u);

    EXPECT_EQ(downstream->selected(), (std::vector<std::uint64_t>{1, 2, 3}));
    EXPECT_EQ(downstream->transport(), (std::vector<std::uint64_t>{1, 2, 3}));
}

TEST(OrderedEncodedFrameSinkTest, SkippedSelectedFrameDoesNotCreateTransportGap) {
    core::BucketMemoryPool pool;
    auto downstream = std::make_shared<RecordingEncodedSink>();
    media::OrderedEncodedFrameSink orderer(downstream, {.window_capacity = 8});

    ASSERT_TRUE(orderer.Publish(MakeEncoded(pool, 3, 30'000)).ok());
    ASSERT_TRUE(orderer.Publish(MakeEncoded(pool, 1, 10'000)).ok());
    ASSERT_TRUE(orderer.MarkSkipped(
        "session-a",
        "execution-a",
        2,
        core::Status::Error(core::ErrorCode::InvalidArgument, "bad frame")).ok());
    auto sealed = orderer.SealExecution("session-a", "execution-a", 3);
    ASSERT_TRUE(sealed.ok());
    EXPECT_EQ(sealed.value(), 2u);

    EXPECT_EQ(downstream->selected(), (std::vector<std::uint64_t>{1, 3}));
    EXPECT_EQ(downstream->transport(), (std::vector<std::uint64_t>{1, 2}));
}

TEST(OrderedEncodedFrameSinkTest, RejectsMediaTimestampRegressionWithinExecution) {
    core::BucketMemoryPool pool;
    auto downstream = std::make_shared<RecordingEncodedSink>();
    media::OrderedEncodedFrameSink orderer(downstream, {.window_capacity = 8});

    ASSERT_TRUE(orderer.Publish(MakeEncoded(pool, 1, 20'000)).ok());
    const auto status = orderer.Publish(MakeEncoded(pool, 2, 10'000));

    EXPECT_EQ(status.code(), core::ErrorCode::FailedPrecondition);
    EXPECT_EQ(downstream->selected(), (std::vector<std::uint64_t>{1}));
}
