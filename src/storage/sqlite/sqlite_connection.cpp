#include "sqlite/sqlite_connection.h"

#include "sqlite/sqlite_internal.h"
#include "sqlite/sqlite_statement.h"

#include <sqlite3.h>

#include <utility>

namespace storage::sqlite {

class SqliteConnection::Impl {
public:
    explicit Impl(::sqlite3* db) noexcept : db_(db) {}

    ~Impl() noexcept {
        if (db_) {
            sqlite3_close_v2(db_);
        }
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    ::sqlite3* db() noexcept {
        return db_;
    }

    const ::sqlite3* db() const noexcept {
        return db_;
    }

private:
    ::sqlite3* db_ = nullptr;
};

SqliteConnection::SqliteConnection() noexcept = default;
SqliteConnection::~SqliteConnection() = default;
SqliteConnection::SqliteConnection(SqliteConnection&&) noexcept = default;
SqliteConnection& SqliteConnection::operator=(SqliteConnection&&) noexcept = default;

SqliteConnection::SqliteConnection(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

core::Result<SqliteConnection> SqliteConnection::Open(const std::string& path) {
    return Open(path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX);
}

core::Result<SqliteConnection> SqliteConnection::Open(const std::string& path, int open_flags) {
    ::sqlite3* db = nullptr;
    const int rc = sqlite3_open_v2(
        path.c_str(),
        &db,
        open_flags,
        nullptr);

    if (rc != SQLITE_OK) {
        auto status = internal::MakeSqliteStatus(rc, db, "open SQLite database");
        if (db) {
            sqlite3_close_v2(db);
        }
        return status;
    }

    return SqliteConnection(std::make_unique<Impl>(db));
}

core::Result<SqliteConnection> SqliteConnection::OpenMemory() {
    return OpenMemory(SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX);
}

core::Result<SqliteConnection> SqliteConnection::OpenMemory(int open_flags) {
    return Open(":memory:", open_flags);
}

bool SqliteConnection::valid() const noexcept {
    return impl_ && impl_->db() != nullptr;
}

SqliteConnection::operator bool() const noexcept {
    return valid();
}

core::Status SqliteConnection::Execute(const std::string& sql) {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite connection is not open");
    }

    char* error_message = nullptr;
    const int rc = sqlite3_exec(impl_->db(), sql.c_str(), nullptr, nullptr, &error_message);
    if (rc != SQLITE_OK) {
        std::string message = "execute SQL failed";
        if (error_message) {
            message += ": ";
            message += error_message;
            sqlite3_free(error_message);
        } else {
            message += ": ";
            message += sqlite3_errmsg(impl_->db());
        }
        message += " (sqlite code ";
        message += std::to_string(rc);
        message += ")";
        return core::Status::Error(core::ErrorCode::InvalidArgument, std::move(message));
    }

    return core::Status::Ok();
}

core::Result<SqliteStatement> SqliteConnection::Prepare(const std::string& sql) {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite connection is not open");
    }

    sqlite3_stmt* statement = nullptr;
    const int rc = sqlite3_prepare_v2(impl_->db(), sql.c_str(), -1, &statement, nullptr);
    if (rc != SQLITE_OK) {
        return internal::MakeSqliteStatus(rc, impl_->db(), "prepare SQLite statement");
    }

    return internal::AdoptSqliteStatement(impl_->db(), statement);
}

core::Status SqliteConnection::SetBusyTimeoutMs(int milliseconds) {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite connection is not open");
    }
    if (milliseconds < 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "busy timeout must be non-negative");
    }

    const int rc = sqlite3_busy_timeout(impl_->db(), milliseconds);
    if (rc != SQLITE_OK) {
        return internal::MakeSqliteStatus(rc, impl_->db(), "set SQLite busy timeout");
    }

    return core::Status::Ok();
}

core::Status SqliteConnection::EnableWal() {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite connection is not open");
    }

    auto status = Execute("PRAGMA journal_mode=WAL;");
    if (!status.ok()) {
        return status;
    }

    return core::Status::Ok();
}

core::Status SqliteConnection::InitializeForPool(int busy_timeout_ms, bool enable_wal) {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite connection is not open");
    }

    auto status = SetBusyTimeoutMs(busy_timeout_ms);
    if (!status.ok()) {
        return status;
    }

    if (enable_wal) {
        status = EnableWal();
    }
    return status;
}

std::int64_t SqliteConnection::LastInsertRowId() const noexcept {
    return valid() ? sqlite3_last_insert_rowid(const_cast<::sqlite3*>(impl_->db())) : 0;
}

int SqliteConnection::Changes() const noexcept {
    return valid() ? sqlite3_changes(const_cast<::sqlite3*>(impl_->db())) : 0;
}

SqliteConnection::Impl* SqliteConnection::impl() noexcept {
    return impl_.get();
}

const SqliteConnection::Impl* SqliteConnection::impl() const noexcept {
    return impl_.get();
}

} // namespace storage::sqlite
