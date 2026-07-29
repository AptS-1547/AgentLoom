#pragma once

#include "memory_pool.h"
#include "result.h"

#include <cstddef>
#include <memory>
#include <span>

namespace core::optimization {

struct AdamOptions {
    double learning_rate = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    double weight_decay = 0.0;
    double max_gradient_norm = 0.0;
};

class IOptimizer {
public:
    virtual ~IOptimizer() = default;

    virtual core::Status Step(
        std::span<double> parameters,
        std::span<const double> gradients) = 0;
    virtual void Reset() noexcept = 0;
    virtual std::size_t StepCount() const noexcept = 0;
};

class AdamOptimizer final : public IOptimizer {
public:
    static core::Result<std::unique_ptr<AdamOptimizer>> Create(
        AdamOptions options = {},
        std::shared_ptr<core::RawMemoryPool> memory_pool = {});

    AdamOptimizer(const AdamOptimizer&) = delete;
    AdamOptimizer& operator=(const AdamOptimizer&) = delete;
    AdamOptimizer(AdamOptimizer&&) = delete;
    AdamOptimizer& operator=(AdamOptimizer&&) = delete;
    ~AdamOptimizer() override = default;

    core::Status Step(
        std::span<double> parameters,
        std::span<const double> gradients) override;
    void Reset() noexcept override;
    std::size_t StepCount() const noexcept override;

private:
    AdamOptimizer(
        AdamOptions options,
        std::shared_ptr<core::RawMemoryPool> memory_pool) noexcept;

    core::Status EnsureState(std::size_t parameter_count);

    AdamOptions options_;
    std::shared_ptr<core::RawMemoryPool> memory_pool_;
    core::MemoryBlock state_;
    std::size_t parameter_count_ = 0;
    std::size_t step_count_ = 0;
};

} // namespace core::optimization
