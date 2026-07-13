#include "memory_pool.h"
#include "ordered_inference_frame_admission.h"

#include <gtest/gtest.h>

#include <mutex>
#include <vector>

namespace {

core::Result<media::inference::OwnedInferenceFrame> MakeTransportFrame(
    core::RawMemoryPool& pool,
    std::uint64_t selected_sequence,
    std::uint64_t transport_sequence) {
    std::vector<std::byte> payload(32, std::byte{0x22});
    media::inference::InferenceFrameMetadata metadata;
    metadata.execution_id = "execution-a";
    metadata.session_id = "session-a";
    metadata.selected_sequence = selected_sequence;
    metadata.transport_sequence = transport_sequence;
    metadata.frame_id = selected_sequence;
    metadata.timestamp_us = static_cast<std::int64_t>(selected_sequence * 1'000);
    return media::inference::CopyInferenceFrame(pool, std::move(metadata), payload);
}

class RecordingAdmission final : public media::inference::IInferenceFrameAdmissionSink {
public:
    core::Status AdmitFrame(media::inference::OwnedInferenceFrame frame) override {
        std::lock_guard lock(mutex_);
        transport_.push_back(frame.metadata().transport_sequence);
        selected_.push_back(frame.metadata().selected_sequence);
        return core::Status::Ok();
    }

    std::vector<std::uint64_t> transport() const {
        std::lock_guard lock(mutex_);
        return transport_;
    }

    std::vector<std::uint64_t> selected() const {
        std::lock_guard lock(mutex_);
        return selected_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<std::uint64_t> transport_;
    std::vector<std::uint64_t> selected_;
};

} // namespace

TEST(OrderedInferenceFrameAdmissionTest, RestoresReceiverWorkerCompletionOrder) {
    core::BucketMemoryPool pool;
    auto downstream = std::make_shared<RecordingAdmission>();
    media::inference::OrderedInferenceFrameAdmission orderer(
        downstream,
        {.window_capacity = 8});

    auto third = MakeTransportFrame(pool, 4, 3);
    auto first = MakeTransportFrame(pool, 1, 1);
    auto second = MakeTransportFrame(pool, 3, 2);
    ASSERT_TRUE(third.ok() && first.ok() && second.ok());
    ASSERT_TRUE(orderer.AdmitFrame(std::move(third).value()).ok());
    ASSERT_TRUE(orderer.AdmitFrame(std::move(first).value()).ok());
    ASSERT_TRUE(orderer.AdmitFrame(std::move(second).value()).ok());
    ASSERT_TRUE(orderer.SealExecution("session-a", "execution-a", 3).ok());

    EXPECT_EQ(downstream->transport(), (std::vector<std::uint64_t>{1, 2, 3}));
    EXPECT_EQ(downstream->selected(), (std::vector<std::uint64_t>{1, 3, 4}));
}

TEST(OrderedInferenceFrameAdmissionTest, RejectsExecutionIdentityCollision) {
    core::BucketMemoryPool pool;
    auto downstream = std::make_shared<RecordingAdmission>();
    media::inference::OrderedInferenceFrameAdmission orderer(downstream, {.window_capacity = 8});
    auto first = MakeTransportFrame(pool, 1, 1);
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(orderer.AdmitFrame(std::move(first).value()).ok());

    auto conflict = MakeTransportFrame(pool, 2, 2);
    ASSERT_TRUE(conflict.ok());
    conflict.value().metadata().session_id = "session-b";
    EXPECT_EQ(
        orderer.AdmitFrame(std::move(conflict).value()).code(),
        core::ErrorCode::FailedPrecondition);
}
