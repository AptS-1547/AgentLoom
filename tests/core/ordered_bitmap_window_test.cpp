#include "ordered_bitmap_window.h"

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>

using namespace std::chrono_literals;

TEST(OrderedBitmapWindowTest, ReleasesMoveOnlyValuesInSequenceOrder) {
    core::OrderedBitmapWindow<std::unique_ptr<int>> window({.capacity = 8});
    ASSERT_TRUE(window.Admit(3, std::make_unique<int>(30)).ok());
    ASSERT_TRUE(window.Admit(1, std::make_unique<int>(10)).ok());
    ASSERT_TRUE(window.Admit(2, std::make_unique<int>(20)).ok());

    for (int expected = 1; expected <= 3; ++expected) {
        auto item = window.TryTake();
        ASSERT_TRUE(item.ok()) << item.status().message();
        EXPECT_EQ(item.value().sequence, static_cast<std::uint64_t>(expected));
        ASSERT_TRUE(item.value().value.has_value());
        EXPECT_EQ(**item.value().value, expected * 10);
    }
}

TEST(OrderedBitmapWindowTest, SkippedSequenceUnblocksLaterReadyValue) {
    core::OrderedBitmapWindow<int> window({.capacity = 8});
    ASSERT_TRUE(window.Admit(2, 20).ok());
    EXPECT_EQ(window.TryTake().status().code(), core::ErrorCode::NotFound);
    ASSERT_TRUE(window.MarkSkipped(
        1,
        core::Status::Error(core::ErrorCode::InvalidArgument, "bad frame")).ok());

    auto skipped = window.TryTake();
    ASSERT_TRUE(skipped.ok());
    EXPECT_TRUE(skipped.value().skipped());
    EXPECT_EQ(skipped.value().terminal_status.code(), core::ErrorCode::InvalidArgument);
    auto ready = window.TryTake();
    ASSERT_TRUE(ready.ok());
    EXPECT_EQ(*ready.value().value, 20);
}

TEST(OrderedBitmapWindowTest, EnforcesCapacitySealAndDrain) {
    core::OrderedBitmapWindow<int> window({.capacity = 4});
    EXPECT_EQ(window.Admit(5, 50).code(), core::ErrorCode::ResourceExhausted);
    ASSERT_TRUE(window.Admit(1, 10).ok());
    ASSERT_TRUE(window.Seal(1).ok());
    EXPECT_EQ(window.Admit(2, 20).code(), core::ErrorCode::FailedPrecondition);
    ASSERT_TRUE(window.TryTake().ok());
    EXPECT_TRUE(window.Snapshot().drained);
    EXPECT_EQ(window.TryTake().status().code(), core::ErrorCode::Cancelled);
}

TEST(OrderedBitmapWindowTest, ConcurrentOutOfOrderAdmissionWakesContiguousDrain) {
    core::OrderedBitmapWindow<int> window({.capacity = 64});
    auto waiter = std::async(std::launch::async, [&] {
        int sum = 0;
        for (int sequence = 1; sequence <= 32; ++sequence) {
            auto item = window.WaitTake(2s);
            if (!item.ok() || !item.value().value.has_value()) {
                return -1;
            }
            sum += *item.value().value;
        }
        return sum;
    });

    for (int sequence = 32; sequence >= 1; --sequence) {
        ASSERT_TRUE(window.Admit(
            static_cast<std::uint64_t>(sequence),
            sequence).ok());
    }

    EXPECT_EQ(waiter.get(), 32 * 33 / 2);
}
