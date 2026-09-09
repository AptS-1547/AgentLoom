#include "auth_session_maintenance_task.h"

#include <utility>

namespace agent::service::gateway {

AuthSessionMaintenanceTask::AuthSessionMaintenanceTask(
    std::shared_ptr<IAuthSessionStore> store,
    std::chrono::milliseconds interval,
    std::size_t batch_size,
    core::LoggerAdapter logger)
    : store_(std::move(store)),
      interval_(interval),
      batch_size_(batch_size),
      logger_(std::move(logger)) {}

std::string_view AuthSessionMaintenanceTask::Name() const noexcept {
    return "gateway_auth_session_cleanup";
}

std::chrono::milliseconds AuthSessionMaintenanceTask::Interval() const noexcept {
    return interval_;
}

core::Status AuthSessionMaintenanceTask::Tick(std::stop_token stop_token) {
    if (stop_token.stop_requested() || !store_) {
        return core::Status::Ok();
    }
    auto cleanup = store_->CleanupExpired(std::chrono::system_clock::now(), batch_size_);
    if (!cleanup.ok()) return cleanup.status();
    if (cleanup.value() > 0) {
        logger_.info(
            "[maintenance] gateway auth session cleanup removed_count={}", cleanup.value());
    }
    return core::Status::Ok();
}

}
