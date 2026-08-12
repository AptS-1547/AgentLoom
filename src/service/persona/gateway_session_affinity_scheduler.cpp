#include "gateway_session_affinity_scheduler.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace agent::service::persona {

struct GatewaySessionAffinityScheduler::Impl {
    struct Lane {
        std::string fairness_key;
        std::string tenant_key;
        std::deque<std::shared_ptr<core::ThreadPoolWorkItem>> tasks;
        std::size_t outstanding = 0;
        bool running = false;
        bool ready = false;
    };

    explicit Impl(GatewaySessionAffinitySchedulerOptions value)
        : options(std::move(value)) {}

    GatewaySessionAffinitySchedulerOptions options;
    mutable std::mutex mutex;
    std::condition_variable ready_cv;
    std::unordered_map<std::string, Lane> lanes;
    std::unordered_map<std::string, std::size_t> fairness_outstanding;
    std::unordered_map<std::string, std::size_t> tenant_outstanding;
    std::unordered_map<const core::ThreadPoolWorkItem*, std::string> task_lanes;
    std::deque<std::string> ready_keys;
    std::size_t queue_capacity = 0;
    std::size_t queued_tasks = 0;
    std::size_t running_tasks = 0;
    std::size_t rejected_tasks = 0;
    std::size_t rejected_global = 0;
    std::size_t rejected_per_key = 0;
    std::size_t rejected_per_fairness_key = 0;
    std::size_t rejected_per_tenant = 0;
    std::size_t max_lane_depth = 0;
    std::uint64_t anonymous_key_counter = 0;
    bool started = false;
    bool closed = false;
};

GatewaySessionAffinityScheduler::GatewaySessionAffinityScheduler(
    GatewaySessionAffinitySchedulerOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

GatewaySessionAffinityScheduler::~GatewaySessionAffinityScheduler() = default;

core::Status GatewaySessionAffinityScheduler::Start(const core::ThreadPoolSchedulerOptions& options) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->closed) {
        return core::Status::Error(core::ErrorCode::Unavailable, "session affinity scheduler has been shut down");
    }
    if (impl_->options.max_active_keys == 0 ||
        impl_->options.max_outstanding_per_key == 0 ||
        impl_->options.max_outstanding_per_fairness_key == 0 ||
        impl_->options.max_outstanding_per_tenant == 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "session affinity scheduler limits must be positive");
    }
    impl_->started = true;
    impl_->queue_capacity = options.queue_capacity;
    return core::Status::Ok();
}

core::Status GatewaySessionAffinityScheduler::TryEnqueue(
    const std::shared_ptr<core::ThreadPoolWorkItem>& item) {
    if (!item) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "thread pool work item is empty");
    }

    std::lock_guard lock(impl_->mutex);
    if (!impl_->started || impl_->closed) {
        ++impl_->rejected_tasks;
        return core::Status::Error(core::ErrorCode::Unavailable, "session affinity scheduler is not running");
    }
    if (impl_->queue_capacity > 0 && impl_->queued_tasks >= impl_->queue_capacity) {
        ++impl_->rejected_tasks;
        ++impl_->rejected_global;
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "thread pool queue is full");
    }

    auto key = item->metadata().concurrency_key;
    if (key.empty()) {
        key = "__anonymous-" + std::to_string(++impl_->anonymous_key_counter);
    }
    auto lane_it = impl_->lanes.find(key);
    if (lane_it == impl_->lanes.end()) {
        if (impl_->lanes.size() >= impl_->options.max_active_keys) {
            ++impl_->rejected_tasks;
            ++impl_->rejected_global;
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "session affinity key limit reached");
        }
        lane_it = impl_->lanes.emplace(
            key,
            Impl::Lane{
                .fairness_key = item->metadata().fairness_key,
                .tenant_key = item->metadata().tenant_key,
            }).first;
    }
    auto& lane = lane_it->second;
    if (lane.fairness_key != item->metadata().fairness_key ||
        lane.tenant_key != item->metadata().tenant_key) {
        ++impl_->rejected_tasks;
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                   "session affinity key owner changed");
    }
    if (lane.outstanding >= impl_->options.max_outstanding_per_key) {
        ++impl_->rejected_tasks;
        ++impl_->rejected_per_key;
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "session affinity key limit reached");
    }
    const auto fairness_it = impl_->fairness_outstanding.find(lane.fairness_key);
    const auto fairness_count = fairness_it == impl_->fairness_outstanding.end() ? 0 : fairness_it->second;
    if (!lane.fairness_key.empty() &&
        fairness_count >= impl_->options.max_outstanding_per_fairness_key) {
        ++impl_->rejected_tasks;
        ++impl_->rejected_per_fairness_key;
        if (lane.outstanding == 0) {
            impl_->lanes.erase(lane_it);
        }
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "session affinity fairness limit reached");
    }
    const auto tenant_it = impl_->tenant_outstanding.find(lane.tenant_key);
    const auto tenant_count = tenant_it == impl_->tenant_outstanding.end() ? 0 : tenant_it->second;
    if (!lane.tenant_key.empty() &&
        tenant_count >= impl_->options.max_outstanding_per_tenant) {
        ++impl_->rejected_tasks;
        ++impl_->rejected_per_tenant;
        if (lane.outstanding == 0) {
            impl_->lanes.erase(lane_it);
        }
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "session affinity tenant limit reached");
    }

    lane.tasks.push_back(item);
    ++lane.outstanding;
    ++impl_->queued_tasks;
    impl_->task_lanes[item.get()] = key;
    if (!lane.fairness_key.empty()) {
        ++impl_->fairness_outstanding[lane.fairness_key];
    }
    if (!lane.tenant_key.empty()) {
        ++impl_->tenant_outstanding[lane.tenant_key];
    }
    impl_->max_lane_depth = std::max(impl_->max_lane_depth, lane.outstanding);
    if (!lane.running && !lane.ready) {
        lane.ready = true;
        impl_->ready_keys.push_back(key);
        impl_->ready_cv.notify_one();
    }
    return core::Status::Ok();
}

core::Result<std::shared_ptr<core::ThreadPoolWorkItem>> GatewaySessionAffinityScheduler::WaitDequeue(
    std::size_t,
    std::stop_token) {
    std::unique_lock lock(impl_->mutex);
    impl_->ready_cv.wait(lock, [this] {
        return !impl_->ready_keys.empty() || (impl_->closed && impl_->running_tasks == 0);
    });
    if (impl_->ready_keys.empty()) {
        return core::Status::Error(core::ErrorCode::Cancelled, "session affinity scheduler is closed");
    }
    const auto key = std::move(impl_->ready_keys.front());
    impl_->ready_keys.pop_front();
    auto lane_it = impl_->lanes.find(key);
    if (lane_it == impl_->lanes.end() || lane_it->second.tasks.empty() || lane_it->second.running) {
        return core::Status::Error(core::ErrorCode::InternalError, "session affinity lane state is invalid");
    }
    auto& lane = lane_it->second;
    lane.ready = false;
    lane.running = true;
    auto item = std::move(lane.tasks.front());
    lane.tasks.pop_front();
    --impl_->queued_tasks;
    ++impl_->running_tasks;
    return item;
}

void GatewaySessionAffinityScheduler::Complete(const core::ThreadPoolWorkItem& item,
                                               std::size_t,
                                               const core::Status&) noexcept {
    std::lock_guard lock(impl_->mutex);
    const auto task_it = impl_->task_lanes.find(&item);
    if (task_it == impl_->task_lanes.end()) {
        return;
    }
    auto lane_it = impl_->lanes.find(task_it->second);
    impl_->task_lanes.erase(task_it);
    if (lane_it == impl_->lanes.end()) {
        return;
    }
    auto& lane = lane_it->second;
    lane.running = false;
    if (lane.outstanding > 0) {
        --lane.outstanding;
    }
    if (impl_->running_tasks > 0) {
        --impl_->running_tasks;
    }
    if (!lane.fairness_key.empty()) {
        auto fairness = impl_->fairness_outstanding.find(lane.fairness_key);
        if (fairness != impl_->fairness_outstanding.end() && fairness->second > 0) {
            if (--fairness->second == 0) {
                impl_->fairness_outstanding.erase(fairness);
            }
        }
    }
    if (!lane.tenant_key.empty()) {
        auto tenant = impl_->tenant_outstanding.find(lane.tenant_key);
        if (tenant != impl_->tenant_outstanding.end() && tenant->second > 0) {
            if (--tenant->second == 0) {
                impl_->tenant_outstanding.erase(tenant);
            }
        }
    }
    if (!lane.tasks.empty()) {
        lane.ready = true;
        impl_->ready_keys.push_back(lane_it->first);
        impl_->ready_cv.notify_one();
    } else {
        impl_->lanes.erase(lane_it);
        impl_->ready_cv.notify_all();
    }
}

void GatewaySessionAffinityScheduler::Close(bool discard) noexcept {
    std::vector<std::shared_ptr<core::ThreadPoolWorkItem>> discarded;
    {
        std::lock_guard lock(impl_->mutex);
        impl_->closed = true;
        if (discard) {
            for (auto& [key, lane] : impl_->lanes) {
                while (!lane.tasks.empty()) {
                    auto item = std::move(lane.tasks.front());
                    lane.tasks.pop_front();
                    impl_->task_lanes.erase(item.get());
                    discarded.push_back(std::move(item));
                    if (lane.outstanding > 0) {
                        --lane.outstanding;
                    }
                    if (!lane.fairness_key.empty()) {
                        auto fairness = impl_->fairness_outstanding.find(lane.fairness_key);
                        if (fairness != impl_->fairness_outstanding.end() && fairness->second > 0 && --fairness->second == 0) {
                            impl_->fairness_outstanding.erase(fairness);
                        }
                    }
                    if (!lane.tenant_key.empty()) {
                        auto tenant = impl_->tenant_outstanding.find(lane.tenant_key);
                        if (tenant != impl_->tenant_outstanding.end() && tenant->second > 0 && --tenant->second == 0) {
                            impl_->tenant_outstanding.erase(tenant);
                        }
                    }
                }
            }
            impl_->queued_tasks = 0;
            impl_->ready_keys.clear();
            for (auto lane_it = impl_->lanes.begin(); lane_it != impl_->lanes.end();) {
                if (!lane_it->second.running) {
                    lane_it = impl_->lanes.erase(lane_it);
                } else {
                    ++lane_it;
                }
            }
        }
        impl_->ready_cv.notify_all();
    }
    discarded.clear();
}

bool GatewaySessionAffinityScheduler::closed() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->closed;
}

std::size_t GatewaySessionAffinityScheduler::QueuedTaskCount() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->queued_tasks;
}

core::ThreadPoolConcurrencySnapshot GatewaySessionAffinityScheduler::Snapshot() const {
    std::lock_guard lock(impl_->mutex);
    return {
        .active_keys = impl_->lanes.size(),
        .ready_keys = impl_->ready_keys.size(),
        .queued_tasks = impl_->queued_tasks,
        .running_tasks = impl_->running_tasks,
        .rejected_tasks = impl_->rejected_tasks,
        .rejected_global = impl_->rejected_global,
        .rejected_per_key = impl_->rejected_per_key,
        .rejected_per_fairness_key = impl_->rejected_per_fairness_key,
        .rejected_per_tenant = impl_->rejected_per_tenant,
        .max_lane_depth = impl_->max_lane_depth,
    };
}

} // namespace agent::service::persona
