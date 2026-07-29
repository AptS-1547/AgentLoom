#include "optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace core::optimization {
namespace {

core::Status ValidateOptions(const AdamOptions& options) {
    if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0.0 ||
        !std::isfinite(options.beta1) || options.beta1 < 0.0 || options.beta1 >= 1.0 ||
        !std::isfinite(options.beta2) || options.beta2 < 0.0 || options.beta2 >= 1.0 ||
        !std::isfinite(options.epsilon) || options.epsilon <= 0.0 ||
        !std::isfinite(options.weight_decay) || options.weight_decay < 0.0 ||
        !std::isfinite(options.max_gradient_norm) || options.max_gradient_norm < 0.0) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "Adam optimizer options are invalid");
    }
    return core::Status::Ok();
}

} // namespace

AdamOptimizer::AdamOptimizer(
    AdamOptions options,
    std::shared_ptr<core::RawMemoryPool> memory_pool) noexcept
    : options_(std::move(options)),
      memory_pool_(std::move(memory_pool)) {}

core::Result<std::unique_ptr<AdamOptimizer>> AdamOptimizer::Create(
    AdamOptions options,
    std::shared_ptr<core::RawMemoryPool> memory_pool) {
    auto status = ValidateOptions(options);
    if (!status.ok()) {
        return status;
    }
    try {
        if (!memory_pool) {
            memory_pool = std::make_shared<core::BucketMemoryPool>();
        }
        return std::unique_ptr<AdamOptimizer>(
            new AdamOptimizer(std::move(options), std::move(memory_pool)));
    } catch (const std::bad_alloc&) {
        return core::Status::Error(
            core::ErrorCode::OutOfMemory,
            "Adam optimizer allocation failed");
    }
}

core::Status AdamOptimizer::EnsureState(std::size_t parameter_count) {
    if (!state_.empty()) {
        if (parameter_count != parameter_count_) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "Adam parameter count changed without Reset");
        }
        return core::Status::Ok();
    }
    if (parameter_count > std::numeric_limits<std::size_t>::max() /
            (2 * sizeof(double))) {
        return core::Status::Error(
            core::ErrorCode::ResourceExhausted,
            "Adam optimizer state is too large");
    }
    const auto state_bytes = parameter_count * 2 * sizeof(double);
    auto allocated = memory_pool_->allocate(state_bytes, alignof(double));
    if (!allocated.ok()) {
        return allocated.status();
    }
    state_ = std::move(allocated).value();
    std::memset(state_.data(), 0, state_.size());
    parameter_count_ = parameter_count;
    return core::Status::Ok();
}

core::Status AdamOptimizer::Step(
    std::span<double> parameters,
    std::span<const double> gradients) {
    if (parameters.empty() || parameters.size() != gradients.size()) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "Adam parameters and gradients must have the same non-zero size");
    }
    if (step_count_ == std::numeric_limits<std::size_t>::max()) {
        return core::Status::Error(
            core::ErrorCode::ResourceExhausted,
            "Adam step counter exhausted");
    }
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        if (!std::isfinite(parameters[index]) || !std::isfinite(gradients[index])) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "Adam parameters and gradients must be finite");
        }
    }
    auto state_status = EnsureState(parameters.size());
    if (!state_status.ok()) {
        return state_status;
    }

    long double squared_gradient_norm = 0.0L;
    for (const auto gradient : gradients) {
        squared_gradient_norm += static_cast<long double>(gradient) * gradient;
    }
    double gradient_scale = 1.0;
    if (options_.max_gradient_norm > 0.0) {
        const auto norm = std::sqrt(squared_gradient_norm);
        if (norm > options_.max_gradient_norm) {
            gradient_scale = static_cast<double>(
                options_.max_gradient_norm / norm);
        }
    }

    ++step_count_;
    const auto bias1 = 1.0 - std::pow(options_.beta1, static_cast<double>(step_count_));
    const auto bias2 = 1.0 - std::pow(options_.beta2, static_cast<double>(step_count_));
    auto* first_moment = reinterpret_cast<double*>(state_.data());
    auto* second_moment = first_moment + parameter_count_;
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const auto gradient = gradients[index] * gradient_scale;
        first_moment[index] = options_.beta1 * first_moment[index] +
            (1.0 - options_.beta1) * gradient;
        second_moment[index] = options_.beta2 * second_moment[index] +
            (1.0 - options_.beta2) * gradient * gradient;
        const auto corrected_first = first_moment[index] / bias1;
        const auto corrected_second = second_moment[index] / bias2;
        if (options_.weight_decay > 0.0) {
            parameters[index] -= options_.learning_rate *
                options_.weight_decay * parameters[index];
        }
        parameters[index] -= options_.learning_rate * corrected_first /
            (std::sqrt(corrected_second) + options_.epsilon);
    }
    return core::Status::Ok();
}

void AdamOptimizer::Reset() noexcept {
    state_.reset();
    parameter_count_ = 0;
    step_count_ = 0;
}

std::size_t AdamOptimizer::StepCount() const noexcept {
    return step_count_;
}

} // namespace core::optimization
