#pragma once

#include "result.h"
#include "sqlite/sqlite_connection.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace storage::sqlite {

enum class SqliteConnectionKind {
    Read,
    Write
};

struct SqliteConnectionPoolOptions {
    std::string path;
    std::size_t read_connection_count = 4;
    std::size_t write_connection_count = 1;
    int busy_timeout_ms = 5000;
    bool enable_wal = true;
    bool full_mutex = false;
};

struct SqliteConnectionPoolStats {
    std::size_t total_read_connections = 0;
    std::size_t idle_read_connections = 0;
    std::size_t leased_read_connections = 0;
    std::size_t total_write_connections = 0;
    std::size_t idle_write_connections = 0;
    std::size_t leased_write_connections = 0;
    std::size_t acquired_connections = 0;
    std::size_t rejected_acquires = 0;
};

struct SqliteConnectionSnapshot {
    std::uint64_t connection_id = 0;
    SqliteConnectionKind kind = SqliteConnectionKind::Read;
    bool leased = false;
    std::chrono::steady_clock::time_point last_used;
};

namespace detail {
struct SqliteConnectionPoolEntry;
struct SqliteConnectionPoolState;
} // namespace detail

class SqliteConnectionLease {
public:
    SqliteConnectionLease() noexcept = default;
    SqliteConnectionLease(SqliteConnectionLease&& other) noexcept;
    SqliteConnectionLease& operator=(SqliteConnectionLease&& other) noexcept;
    ~SqliteConnectionLease();

    SqliteConnectionLease(const SqliteConnectionLease&) = delete;
    SqliteConnectionLease& operator=(const SqliteConnectionLease&) = delete;

    bool valid() const noexcept;
    explicit operator bool() const noexcept;

    std::uint64_t connection_id() const noexcept;
    SqliteConnectionKind kind() const noexcept;

    SqliteConnection& connection() noexcept;
    const SqliteConnection& connection() const noexcept;
    SqliteConnection* operator->() noexcept;
    const SqliteConnection* operator->() const noexcept;

    void Release();

private:
    friend class SqliteConnectionPool;
    friend struct detail::SqliteConnectionPoolState;

    SqliteConnectionLease(std::shared_ptr<detail::SqliteConnectionPoolState> owner,
                          std::shared_ptr<detail::SqliteConnectionPoolEntry> entry) noexcept;

    std::shared_ptr<detail::SqliteConnectionPoolState> owner_;
    std::shared_ptr<detail::SqliteConnectionPoolEntry> entry_;
};

class SqliteConnectionPool {
public:
    explicit SqliteConnectionPool(SqliteConnectionPoolOptions options);
    ~SqliteConnectionPool();

    SqliteConnectionPool(const SqliteConnectionPool&) = delete;
    SqliteConnectionPool& operator=(const SqliteConnectionPool&) = delete;

    core::Status Start();
    void Close();

    core::Result<SqliteConnectionLease> AcquireRead();
    core::Result<SqliteConnectionLease> AcquireWrite();
    core::Result<SqliteConnectionLease> WaitAcquireRead();
    core::Result<SqliteConnectionLease> WaitAcquireWrite();
    core::Result<SqliteConnectionLease> WaitAcquireReadFor(std::chrono::milliseconds timeout);
    core::Result<SqliteConnectionLease> WaitAcquireWriteFor(std::chrono::milliseconds timeout);

    SqliteConnectionPoolStats Stats() const;
    std::vector<SqliteConnectionSnapshot> Snapshots() const;

private:
    std::shared_ptr<detail::SqliteConnectionPoolState> state_;
};

} // namespace storage::sqlite
