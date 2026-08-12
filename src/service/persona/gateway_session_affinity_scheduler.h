#pragma once

#include "thread_pool_scheduler.h"

#include <memory>

namespace agent::service::persona {

struct GatewaySessionAffinitySchedulerOptions {
    std::size_t max_active_keys = 1024;
    std::size_t max_outstanding_per_key = 8;
    std::size_t max_outstanding_per_fairness_key = 32;
    std::size_t max_outstanding_per_tenant = 256;
};

class GatewaySessionAffinityScheduler final : public core::IThreadPoolTaskScheduler {
public:
    explicit GatewaySessionAffinityScheduler(
        GatewaySessionAffinitySchedulerOptions options = {});
    ~GatewaySessionAffinityScheduler() override;

    core::Status Start(const core::ThreadPoolSchedulerOptions& options) override;
    core::Status TryEnqueue(const std::shared_ptr<core::ThreadPoolWorkItem>& item) override;
    core::Result<std::shared_ptr<core::ThreadPoolWorkItem>> WaitDequeue(
        std::size_t worker_index,
        std::stop_token stop_token) override;
    void Complete(const core::ThreadPoolWorkItem& item,
                  std::size_t worker_index,
                  const core::Status& status) noexcept override;
    void Close(bool discard) noexcept override;
    bool closed() const noexcept override;
    std::size_t QueuedTaskCount() const noexcept override;
    core::ThreadPoolConcurrencySnapshot Snapshot() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace agent::service::persona
