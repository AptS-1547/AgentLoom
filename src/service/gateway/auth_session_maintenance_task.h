#pragma once

#include "gateway_auth.h"
#include "gateway_maintenance.h"

#include <chrono>
#include <memory>

namespace agent::service::gateway {

// reference Gateway 私有认证会话清理；下游可信身份适配器不依赖该任务。
class AuthSessionMaintenanceTask final : public IRuntimeMaintenanceTask {
public:
    AuthSessionMaintenanceTask(
        std::shared_ptr<IAuthSessionStore> store,
        std::chrono::milliseconds interval,
        std::size_t batch_size = 256,
        core::LoggerAdapter logger = core::LoggerAdapter::ForModule("gateway"));

    std::string_view Name() const noexcept override;
    std::chrono::milliseconds Interval() const noexcept override;
    core::Status Tick(std::stop_token stop_token) override;

private:
    std::shared_ptr<IAuthSessionStore> store_;
    std::chrono::milliseconds interval_;
    std::size_t batch_size_;
    core::LoggerAdapter logger_;
};

}
