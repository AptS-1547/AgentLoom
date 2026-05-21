#pragma once

#include <chrono>
#include <cstdint>

namespace agent::net {

/// Exponential backoff parameters.  Stateless — call `BackoffFor(attempt)`
/// to get the sleep duration for retry `attempt` (0-based).
struct RetryPolicy {
    std::int32_t max_retries = 3;
    std::chrono::milliseconds initial_delay{1000};
    float multiplier = 2.0f;
    std::chrono::milliseconds max_delay{10000};

    /// Delay before retry `attempt` (attempt 0 = first retry after the
    /// initial failure).  Capped at `max_delay`.
    std::chrono::milliseconds BackoffFor(std::int32_t attempt) const noexcept;
};

}  // namespace agent::net
