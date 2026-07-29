#include "optimizer.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>

TEST(AdamOptimizerTest, RejectsInvalidOptions) {
    core::optimization::AdamOptions options;
    options.beta1 = 1.0;

    auto optimizer = core::optimization::AdamOptimizer::Create(options);

    ASSERT_FALSE(optimizer.ok());
    EXPECT_EQ(optimizer.status().code(), core::ErrorCode::InvalidArgument);
}

TEST(AdamOptimizerTest, FirstStepMatchesBiasCorrectedAdam) {
    core::optimization::AdamOptions options;
    options.learning_rate = 0.1;
    auto created = core::optimization::AdamOptimizer::Create(options);
    ASSERT_TRUE(created.ok()) << created.status().message();
    auto optimizer = std::move(created).value();
    std::array<double, 1> parameters{0.0};
    const std::array<double, 1> gradients{1.0};

    const auto status = optimizer->Step(parameters, gradients);

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_NEAR(parameters[0], -0.1, 1e-7);
    EXPECT_EQ(optimizer->StepCount(), 1u);
}

TEST(AdamOptimizerTest, ConvergesOnQuadraticLoss) {
    core::optimization::AdamOptions options;
    options.learning_rate = 0.05;
    options.max_gradient_norm = 10.0;
    auto created = core::optimization::AdamOptimizer::Create(options);
    ASSERT_TRUE(created.ok()) << created.status().message();
    auto optimizer = std::move(created).value();
    std::array<double, 1> parameters{5.0};

    for (int step = 0; step < 500; ++step) {
        const std::array<double, 1> gradients{2.0 * (parameters[0] - 2.0)};
        const auto status = optimizer->Step(parameters, gradients);
        ASSERT_TRUE(status.ok()) << status.message();
    }

    EXPECT_NEAR(parameters[0], 2.0, 1e-5);
}

TEST(AdamOptimizerTest, ResetAllowsAnotherParameterShape) {
    auto created = core::optimization::AdamOptimizer::Create();
    ASSERT_TRUE(created.ok()) << created.status().message();
    auto optimizer = std::move(created).value();
    std::array<double, 1> first_parameters{1.0};
    const std::array<double, 1> first_gradients{0.5};
    ASSERT_TRUE(optimizer->Step(first_parameters, first_gradients).ok());

    std::array<double, 2> mismatched_parameters{1.0, 2.0};
    const std::array<double, 2> mismatched_gradients{0.5, 0.25};
    auto mismatched = optimizer->Step(mismatched_parameters, mismatched_gradients);
    ASSERT_FALSE(mismatched.ok());
    EXPECT_EQ(mismatched.code(), core::ErrorCode::InvalidArgument);

    optimizer->Reset();
    EXPECT_TRUE(optimizer->Step(mismatched_parameters, mismatched_gradients).ok());
    EXPECT_EQ(optimizer->StepCount(), 1u);
}
