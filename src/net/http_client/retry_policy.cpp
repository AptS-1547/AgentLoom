#include "retry_policy.h"

#include <algorithm>
#include <cmath>

namespace agent::net {

std::chrono::milliseconds RetryPolicy::BackoffFor(std::int32_t attempt) const noexcept {
    if (attempt < 0) attempt = 0;
    double base = static_cast<double>(initial_delay.count());
    double mult = std::pow(static_cast<double>(multiplier), static_cast<double>(attempt));
    double ms = base * mult;
    double capped = std::min(ms, static_cast<double>(max_delay.count()));
    return std::chrono::milliseconds(static_cast<std::int64_t>(capped));
}

}  // namespace agent::net
