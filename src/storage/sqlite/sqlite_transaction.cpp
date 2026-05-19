#include "sqlite/sqlite_transaction.h"

#include "sqlite/sqlite_connection.h"

#include <utility>

namespace storage::sqlite {

SqliteTransaction::SqliteTransaction(SqliteConnection& connection)
    : connection_(&connection) {}

SqliteTransaction::~SqliteTransaction() {
    if (active_) {
        (void)Rollback();
    }
}

SqliteTransaction::SqliteTransaction(SqliteTransaction&& other) noexcept
    : connection_(std::exchange(other.connection_, nullptr)),
      active_(std::exchange(other.active_, false)) {}

SqliteTransaction& SqliteTransaction::operator=(SqliteTransaction&& other) noexcept {
    if (this != &other) {
        if (active_) {
            (void)Rollback();
        }
        connection_ = std::exchange(other.connection_, nullptr);
        active_ = std::exchange(other.active_, false);
    }
    return *this;
}

core::Status SqliteTransaction::Begin() {
    if (!connection_ || !*connection_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite connection is not open");
    }
    if (active_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite transaction is already active");
    }

    auto status = connection_->Execute("BEGIN IMMEDIATE TRANSACTION");
    if (status.ok()) {
        active_ = true;
    }
    return status;
}

core::Status SqliteTransaction::Commit() {
    if (!connection_ || !*connection_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite connection is not open");
    }
    if (!active_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite transaction is not active");
    }

    auto status = connection_->Execute("COMMIT");
    if (status.ok()) {
        active_ = false;
    }
    return status;
}

core::Status SqliteTransaction::Rollback() {
    if (!connection_ || !*connection_) {
        active_ = false;
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite connection is not open");
    }
    if (!active_) {
        return core::Status::Ok();
    }

    auto status = connection_->Execute("ROLLBACK");
    active_ = false;
    return status;
}

bool SqliteTransaction::active() const noexcept {
    return active_;
}

} // namespace storage::sqlite
