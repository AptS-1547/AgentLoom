#include "gateway_lifecycle.h"

#include <exception>
#include <utility>

namespace agent::service::gateway {
namespace {

core::Status ExceptionStatus(std::string_view action,
                             std::string_view component,
                             const std::exception& error) {
    return core::Status::Error(
        core::ErrorCode::InternalError,
        std::string(action) + " gateway component '" + std::string(component) +
            "' failed: " + error.what());
}

}

CallbackGatewayLifecycleComponent::CallbackGatewayLifecycleComponent(
    std::string name,
    StartCallback start,
    StopCallback stop)
    : name_(std::move(name)), start_(std::move(start)), stop_(std::move(stop)) {}

std::string_view CallbackGatewayLifecycleComponent::Name() const noexcept {
    return name_;
}

core::Status CallbackGatewayLifecycleComponent::Start() {
    if (!start_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                   "gateway lifecycle start callback is not configured");
    }
    return start_();
}

core::Status CallbackGatewayLifecycleComponent::Stop(
    std::chrono::steady_clock::time_point deadline) {
    if (!stop_) {
        return core::Status::Ok();
    }
    return stop_(deadline);
}

GatewayLifecycleCoordinator::GatewayLifecycleCoordinator(core::LoggerAdapter logger)
    : logger_(std::move(logger)) {}

GatewayLifecycleCoordinator::~GatewayLifecycleCoordinator() {
    static_cast<void>(Stop());
}

core::Status GatewayLifecycleCoordinator::Register(
    std::shared_ptr<IGatewayLifecycleComponent> component) {
    if (!component) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "gateway lifecycle component is required");
    }
    std::lock_guard lock(mutex_);
    if (state_ != GatewayLifecycleState::Stopped || started_component_count_ != 0) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                   "gateway lifecycle components must be registered while stopped");
    }
    for (const auto& existing : components_) {
        if (existing->Name() == component->Name()) {
            return core::Status::Error(core::ErrorCode::AlreadyExists,
                                       "gateway lifecycle component already registered: " +
                                           std::string(component->Name()));
        }
    }
    components_.push_back(std::move(component));
    return core::Status::Ok();
}

core::Status GatewayLifecycleCoordinator::Start() {
    std::lock_guard lock(mutex_);
    if (state_ == GatewayLifecycleState::Running) {
        return core::Status::Ok();
    }
    if (state_ != GatewayLifecycleState::Stopped || started_component_count_ != 0) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                   "gateway lifecycle is already transitioning");
    }

    state_ = GatewayLifecycleState::Starting;
    for (const auto& component : components_) {
        core::Status status;
        try {
            status = component->Start();
        } catch (const std::exception& error) {
            status = ExceptionStatus("start", component->Name(), error);
        } catch (...) {
            status = core::Status::Error(
                core::ErrorCode::InternalError,
                "start gateway component '" + std::string(component->Name()) +
                    "' failed with unknown exception");
        }
        if (!status.ok()) {
            logger_.error("[gateway-lifecycle] start component={} failed: {}",
                          component->Name(), status.message());
            const auto rollback_deadline = std::chrono::steady_clock::now() +
                                           std::chrono::seconds(30);
            try {
                const auto partial_status = component->Stop(rollback_deadline);
                if (!partial_status.ok()) {
                    logger_.error("[gateway-lifecycle] partial component rollback={} failed: {}",
                                  component->Name(), partial_status.message());
                }
            } catch (const std::exception& error) {
                logger_.error("[gateway-lifecycle] partial component rollback={} threw: {}",
                              component->Name(), error.what());
            } catch (...) {
                logger_.error("[gateway-lifecycle] partial component rollback={} threw unknown exception",
                              component->Name());
            }
            const auto rollback_status = StopStartedLocked(rollback_deadline);
            if (!rollback_status.ok()) {
                logger_.error("[gateway-lifecycle] rollback failed: {}",
                              rollback_status.message());
            }
            return status;
        }
        ++started_component_count_;
    }
    state_ = GatewayLifecycleState::Running;
    return core::Status::Ok();
}

core::Status GatewayLifecycleCoordinator::Stop(
    std::chrono::steady_clock::time_point deadline) {
    std::lock_guard lock(mutex_);
    if (state_ == GatewayLifecycleState::Stopped && started_component_count_ == 0) {
        return core::Status::Ok();
    }
    if (state_ == GatewayLifecycleState::Starting ||
        state_ == GatewayLifecycleState::Stopping) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                   "gateway lifecycle is already transitioning");
    }
    state_ = GatewayLifecycleState::Stopping;
    return StopStartedLocked(deadline);
}

core::Status GatewayLifecycleCoordinator::Stop(std::chrono::milliseconds timeout) {
    return Stop(std::chrono::steady_clock::now() + timeout);
}

GatewayLifecycleState GatewayLifecycleCoordinator::state() const noexcept {
    std::lock_guard lock(mutex_);
    return state_;
}

std::size_t GatewayLifecycleCoordinator::started_component_count() const noexcept {
    std::lock_guard lock(mutex_);
    return started_component_count_;
}

core::Status GatewayLifecycleCoordinator::StopStartedLocked(
    std::chrono::steady_clock::time_point deadline) {
    state_ = GatewayLifecycleState::Stopping;
    core::Status first_failure = core::Status::Ok();
    while (started_component_count_ > 0) {
        auto& component = components_[started_component_count_ - 1];
        core::Status status;
        try {
            status = component->Stop(deadline);
        } catch (const std::exception& error) {
            status = ExceptionStatus("stop", component->Name(), error);
        } catch (...) {
            status = core::Status::Error(
                core::ErrorCode::InternalError,
                "stop gateway component '" + std::string(component->Name()) +
                    "' failed with unknown exception");
        }
        if (!status.ok()) {
            logger_.error("[gateway-lifecycle] stop component={} failed: {}",
                          component->Name(), status.message());
            if (first_failure.ok()) {
                first_failure = status;
            }
        }
        --started_component_count_;
    }
    state_ = GatewayLifecycleState::Stopped;
    return first_failure;
}

}
