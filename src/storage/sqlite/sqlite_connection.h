#pragma once

#include "result.h"

#include <cstdint>
#include <memory>
#include <string>

namespace storage::sqlite {

class SqliteStatement;

class SqliteConnection {
public:
    SqliteConnection() noexcept;
    ~SqliteConnection();

    SqliteConnection(const SqliteConnection&) = delete;
    SqliteConnection& operator=(const SqliteConnection&) = delete;
    SqliteConnection(SqliteConnection&&) noexcept;
    SqliteConnection& operator=(SqliteConnection&&) noexcept;

    static core::Result<SqliteConnection> Open(const std::string& path);
    static core::Result<SqliteConnection> Open(const std::string& path, int open_flags);
    static core::Result<SqliteConnection> OpenMemory();
    static core::Result<SqliteConnection> OpenMemory(int open_flags);

    bool valid() const noexcept;
    explicit operator bool() const noexcept;

    core::Status Execute(const std::string& sql);
    core::Result<SqliteStatement> Prepare(const std::string& sql);
    core::Status SetBusyTimeoutMs(int milliseconds);
    core::Status EnableWal();
    core::Status InitializeForPool(int busy_timeout_ms, bool enable_wal);

    std::int64_t LastInsertRowId() const noexcept;
    int Changes() const noexcept;

private:
    class Impl;

    explicit SqliteConnection(std::unique_ptr<Impl> impl) noexcept;

    Impl* impl() noexcept;
    const Impl* impl() const noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace storage::sqlite
