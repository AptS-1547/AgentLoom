#include "retry_policy.h"

#include <gtest/gtest.h>

using agent::net::RetryPolicy;
using namespace std::chrono_literals;

TEST(RetryPolicyTest, FirstAttemptUsesInitial) {
    RetryPolicy p;
    p.initial_delay = 1000ms;
    p.multiplier = 2.0f;
    p.max_delay = 10000ms;
    EXPECT_EQ(p.BackoffFor(0), 1000ms);
}

TEST(RetryPolicyTest, ExponentialGrowth) {
    RetryPolicy p;
    p.initial_delay = 100ms;
    p.multiplier = 2.0f;
    p.max_delay = 100000ms;
    EXPECT_EQ(p.BackoffFor(0), 100ms);
    EXPECT_EQ(p.BackoffFor(1), 200ms);
    EXPECT_EQ(p.BackoffFor(2), 400ms);
    EXPECT_EQ(p.BackoffFor(3), 800ms);
}

TEST(RetryPolicyTest, CappedAtMaxDelay) {
    RetryPolicy p;
    p.initial_delay = 1000ms;
    p.multiplier = 2.0f;
    p.max_delay = 3000ms;
    EXPECT_EQ(p.BackoffFor(0), 1000ms);
    EXPECT_EQ(p.BackoffFor(1), 2000ms);
    EXPECT_EQ(p.BackoffFor(2), 3000ms);  // would be 4000, capped
    EXPECT_EQ(p.BackoffFor(10), 3000ms);
}

TEST(RetryPolicyTest, NegativeAttemptTreatedAsZero) {
    RetryPolicy p;
    p.initial_delay = 500ms;
    EXPECT_EQ(p.BackoffFor(-1), 500ms);
}
