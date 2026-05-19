#pragma once

#include "result.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

struct sqlite3;
struct sqlite3_stmt;

namespace storage::sqlite {

class SqliteStatement;

namespace internal {
SqliteStatement AdoptSqliteStatement(::sqlite3* db, ::sqlite3_stmt* statement);
}

enum class SqliteStepResult {
    Row,
    Done
};

class SqliteStatement {
public:
    SqliteStatement() noexcept;
    ~SqliteStatement();

    SqliteStatement(const SqliteStatement&) = delete;
    SqliteStatement& operator=(const SqliteStatement&) = delete;
    SqliteStatement(SqliteStatement&&) noexcept;
    SqliteStatement& operator=(SqliteStatement&&) noexcept;

    bool valid() const noexcept;
    explicit operator bool() const noexcept;

    core::Status BindInt(int index, int value);
    core::Status BindInt64(int index, std::int64_t value);
    core::Status BindDouble(int index, double value);
    core::Status BindText(int index, const std::string& value);
    core::Status BindBlob(int index, std::span<const std::byte> value);
    core::Status BindNull(int index);

    core::Result<SqliteStepResult> Step();
    core::Status Reset();
    core::Status ClearBindings();

    int ColumnCount() const noexcept;
    int ColumnType(int index) const noexcept;
    bool ColumnIsNull(int index) const noexcept;
    int ColumnInt(int index) const noexcept;
    std::int64_t ColumnInt64(int index) const noexcept;
    double ColumnDouble(int index) const noexcept;
    std::string ColumnText(int index) const;
    std::vector<std::byte> ColumnBlob(int index) const;

private:
    class Impl;

    explicit SqliteStatement(std::unique_ptr<Impl> impl) noexcept;

    Impl* impl() noexcept;
    const Impl* impl() const noexcept;

    std::unique_ptr<Impl> impl_;

    friend SqliteStatement internal::AdoptSqliteStatement(::sqlite3* db, ::sqlite3_stmt* statement);
};

} // namespace storage::sqlite
