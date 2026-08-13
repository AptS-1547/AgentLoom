#pragma once

#include "logger_adapter.h"
#include "result.h"

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace agent::service::gateway {

enum class GatewayLifecycleState {
    Stopped,
    Starting,
    Running,
    Stopping,
};

class IGatewayLifecycleComponent {
public:
    virtual ~IGatewayLifecycleComponent() = default;

    virtual std::string_view Name() const noexcept = 0;
    virtual core::Status Start() = 0;
    virtual core::Status Stop(std::chrono::steady_clock::time_point deadline) = 0;
};

class CallbackGatewayLifecycleComponent final : public IGatewayLifecycleComponent {
public:
    using StartCallback = std::function<core::Status()>;
    using StopCallback = std::function<core::Status(std::chrono::steady_clock::time_point)>;

    CallbackGatewayLifecycleComponent(std::string name,
                                      StartCallback start,
                                      StopCallback stop);

    std::string_view Name() const noexcept override;
    core::Status Start() override;
    core::Status Stop(std::chrono::steady_clock::time_point deadline) override;

private:
    std::string name_;
    StartCallback start_;
    StopCallback stop_;
};

/// 按注册顺序启动组件，并在失败或停止时按相反顺序回滚。
class GatewayLifecycleCoordinator final {
public:
    explicit GatewayLifecycleCoordinator(
        core::LoggerAdapter logger = core::LoggerAdapter::ForModule("gateway-foundation"));
    ~GatewayLifecycleCoordinator();

    GatewayLifecycleCoordinator(const GatewayLifecycleCoordinator&) = delete;
    GatewayLifecycleCoordinator& operator=(const GatewayLifecycleCoordinator&) = delete;

    core::Status Register(std::shared_ptr<IGatewayLifecycleComponent> component);
    core::Status Start();
    core::Status Stop(std::chrono::steady_clock::time_point deadline);
    core::Status Stop(std::chrono::milliseconds timeout = std::chrono::seconds(30));

    GatewayLifecycleState state() const noexcept;
    std::size_t started_component_count() const noexcept;

private:
    core::Status StopStartedLocked(std::chrono::steady_clock::time_point deadline);

    core::LoggerAdapter logger_;
    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<IGatewayLifecycleComponent>> components_;
    std::size_t started_component_count_ = 0;
    GatewayLifecycleState state_ = GatewayLifecycleState::Stopped;
};

}
