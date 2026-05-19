#include "sqlite/sqlite_statement.h"

#include "sqlite/sqlite_internal.h"

#include <sqlite3.h>

#include <cstring>
#include <utility>

namespace storage::sqlite {

class SqliteStatement::Impl {
public:
    Impl(::sqlite3* db, ::sqlite3_stmt* statement) noexcept
        : db_(db), statement_(statement) {}

    ~Impl() noexcept {
        if (statement_) {
            sqlite3_finalize(statement_);
        }
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    ::sqlite3* db() noexcept {
        return db_;
    }

    ::sqlite3_stmt* statement() noexcept {
        return statement_;
    }

    const ::sqlite3_stmt* statement() const noexcept {
        return statement_;
    }

private:
    ::sqlite3* db_ = nullptr;
    ::sqlite3_stmt* statement_ = nullptr;
};

SqliteStatement::SqliteStatement() noexcept = default;
SqliteStatement::~SqliteStatement() = default;
SqliteStatement::SqliteStatement(SqliteStatement&&) noexcept = default;
SqliteStatement& SqliteStatement::operator=(SqliteStatement&&) noexcept = default;

SqliteStatement::SqliteStatement(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

SqliteStatement internal::AdoptSqliteStatement(::sqlite3* db, ::sqlite3_stmt* statement) {
    return SqliteStatement(std::make_unique<SqliteStatement::Impl>(db, statement));
}

bool SqliteStatement::valid() const noexcept {
    return impl_ && impl_->statement() != nullptr;
}

SqliteStatement::operator bool() const noexcept {
    return valid();
}

core::Status SqliteStatement::BindInt(int index, int value) {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite statement is not prepared");
    }

    const int rc = sqlite3_bind_int(impl_->statement(), index, value);
    return rc == SQLITE_OK ? core::Status::Ok() : internal::MakeSqliteStatus(rc, impl_->db(), "bind SQLite int");
}

core::Status SqliteStatement::BindInt64(int index, std::int64_t value) {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite statement is not prepared");
    }

    const int rc = sqlite3_bind_int64(impl_->statement(), index, value);
    return rc == SQLITE_OK ? core::Status::Ok() : internal::MakeSqliteStatus(rc, impl_->db(), "bind SQLite int64");
}

core::Status SqliteStatement::BindDouble(int index, double value) {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite statement is not prepared");
    }

    const int rc = sqlite3_bind_double(impl_->statement(), index, value);
    return rc == SQLITE_OK ? core::Status::Ok() : internal::MakeSqliteStatus(rc, impl_->db(), "bind SQLite double");
}

core::Status SqliteStatement::BindText(int index, const std::string& value) {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite statement is not prepared");
    }

    const int rc = sqlite3_bind_text(impl_->statement(), index, value.data(), static_cast<int>(value.size()), SQLITE_TRANSIENT);
    return rc == SQLITE_OK ? core::Status::Ok() : internal::MakeSqliteStatus(rc, impl_->db(), "bind SQLite text");
}

core::Status SqliteStatement::BindBlob(int index, std::span<const std::byte> value) {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite statement is not prepared");
    }

    const auto* data = reinterpret_cast<const void*>(value.data());
    const int rc = sqlite3_bind_blob(impl_->statement(), index, data, static_cast<int>(value.size()), SQLITE_TRANSIENT);
    return rc == SQLITE_OK ? core::Status::Ok() : internal::MakeSqliteStatus(rc, impl_->db(), "bind SQLite blob");
}

core::Status SqliteStatement::BindNull(int index) {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite statement is not prepared");
    }

    const int rc = sqlite3_bind_null(impl_->statement(), index);
    return rc == SQLITE_OK ? core::Status::Ok() : internal::MakeSqliteStatus(rc, impl_->db(), "bind SQLite null");
}

core::Result<SqliteStepResult> SqliteStatement::Step() {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite statement is not prepared");
    }

    const int rc = sqlite3_step(impl_->statement());
    if (rc == SQLITE_ROW) {
        return SqliteStepResult::Row;
    }
    if (rc == SQLITE_DONE) {
        return SqliteStepResult::Done;
    }

    return internal::MakeSqliteStatus(rc, impl_->db(), "step SQLite statement");
}

core::Status SqliteStatement::Reset() {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite statement is not prepared");
    }

    const int rc = sqlite3_reset(impl_->statement());
    return rc == SQLITE_OK ? core::Status::Ok() : internal::MakeSqliteStatus(rc, impl_->db(), "reset SQLite statement");
}

core::Status SqliteStatement::ClearBindings() {
    if (!valid()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite statement is not prepared");
    }

    const int rc = sqlite3_clear_bindings(impl_->statement());
    return rc == SQLITE_OK ? core::Status::Ok() : internal::MakeSqliteStatus(rc, impl_->db(), "clear SQLite bindings");
}

int SqliteStatement::ColumnCount() const noexcept {
    return valid() ? sqlite3_column_count(const_cast<::sqlite3_stmt*>(impl_->statement())) : 0;
}

int SqliteStatement::ColumnType(int index) const noexcept {
    return valid() ? sqlite3_column_type(const_cast<::sqlite3_stmt*>(impl_->statement()), index) : SQLITE_NULL;
}

bool SqliteStatement::ColumnIsNull(int index) const noexcept {
    return ColumnType(index) == SQLITE_NULL;
}

int SqliteStatement::ColumnInt(int index) const noexcept {
    return valid() ? sqlite3_column_int(const_cast<::sqlite3_stmt*>(impl_->statement()), index) : 0;
}

std::int64_t SqliteStatement::ColumnInt64(int index) const noexcept {
    return valid() ? sqlite3_column_int64(const_cast<::sqlite3_stmt*>(impl_->statement()), index) : 0;
}

double SqliteStatement::ColumnDouble(int index) const noexcept {
    return valid() ? sqlite3_column_double(const_cast<::sqlite3_stmt*>(impl_->statement()), index) : 0.0;
}

std::string SqliteStatement::ColumnText(int index) const {
    if (!valid()) {
        return {};
    }

    const auto* text = sqlite3_column_text(const_cast<::sqlite3_stmt*>(impl_->statement()), index);
    const int bytes = sqlite3_column_bytes(const_cast<::sqlite3_stmt*>(impl_->statement()), index);
    if (!text || bytes <= 0) {
        return {};
    }

    return std::string(reinterpret_cast<const char*>(text), static_cast<std::size_t>(bytes));
}

std::vector<std::byte> SqliteStatement::ColumnBlob(int index) const {
    if (!valid()) {
        return {};
    }

    const void* data = sqlite3_column_blob(const_cast<::sqlite3_stmt*>(impl_->statement()), index);
    const int bytes = sqlite3_column_bytes(const_cast<::sqlite3_stmt*>(impl_->statement()), index);
    if (!data || bytes <= 0) {
        return {};
    }

    std::vector<std::byte> blob(static_cast<std::size_t>(bytes));
    std::memcpy(blob.data(), data, static_cast<std::size_t>(bytes));
    return blob;
}

SqliteStatement::Impl* SqliteStatement::impl() noexcept {
    return impl_.get();
}

const SqliteStatement::Impl* SqliteStatement::impl() const noexcept {
    return impl_.get();
}

} // namespace storage::sqlite
