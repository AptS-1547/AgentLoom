#include "sqlite/sqlite_connection_pool.h"

#include <sqlite3.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <new>
#include <utility>

namespace storage::sqlite {
namespace detail {

struct SqliteConnectionPoolEntry {
    SqliteConnectionPoolEntry(std::uint64_t connection_id_value,
                              SqliteConnectionKind connection_kind,
                              SqliteConnection sqlite_connection)
        : connection_id(connection_id_value),
          kind(connection_kind),
          connection(std::move(sqlite_connection)),
          last_used(std::chrono::steady_clock::now()) {}

    std::uint64_t connection_id = 0;
    SqliteConnectionKind kind = SqliteConnectionKind::Read;
    SqliteConnection connection;
    std::chrono::steady_clock::time_point last_used;
    bool leased = false;
};

struct SqliteConnectionPoolState {
    explicit SqliteConnectionPoolState(SqliteConnectionPoolOptions pool_options)
        : options(std::move(pool_options)) {}

    core::Status Start() {
        std::lock_guard lock(mutex);
        if (started) {
            return core::Status::Ok();
        }
        if (closed) {
            return core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "SQLite connection pool cannot be restarted after close");
        }
        if (options.path.empty()) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "SQLite database path is required");
        }
        if (options.read_connection_count == 0 && options.write_connection_count == 0) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "at least one SQLite connection is required");
        }
        if (options.write_connection_count > 1) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "SQLite connection pool supports at most one write connection");
        }

        // 启动阶段先创建数据库文件并统一设置 busy timeout 与 WAL，避免业务连接配置不一致。
        auto init_result = SqliteConnection::Open(options.path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                                                   (options.full_mutex ? SQLITE_OPEN_FULLMUTEX : SQLITE_OPEN_NOMUTEX));
        if (!init_result.ok()) {
            return init_result.status();
        }
        auto init_conn = std::move(init_result).value();
        auto init_status = init_conn.InitializeForPool(options.busy_timeout_ms, options.enable_wal);
        if (!init_status.ok()) {
            return init_status;
        }
        // Close init connection - it's just for file creation
        init_conn = SqliteConnection();

        auto status = CreateConnections(SqliteConnectionKind::Read, options.read_connection_count, read_idle);
        if (!status.ok()) {
            CloseLocked();
            return status;
        }

        status = CreateConnections(SqliteConnectionKind::Write, options.write_connection_count, write_idle);
        if (!status.ok()) {
            CloseLocked();
            return status;
        }

        started = true;
        return core::Status::Ok();
    }

    core::Result<SqliteConnectionLease> Acquire(SqliteConnectionKind kind) {
        std::lock_guard lock(mutex);
        if (!started || closed) {
            ++stats.rejected_acquires;
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "SQLite connection pool is not started");
        }

        auto& idle = kind == SqliteConnectionKind::Read ? read_idle : write_idle;
        if (idle.empty()) {
            ++stats.rejected_acquires;
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "SQLite connection pool has no idle connection");
        }

        auto entry = std::move(idle.front());
        idle.pop_front();
        entry->leased = true;
        entry->last_used = std::chrono::steady_clock::now();
        ++stats.acquired_connections;

        auto owner = self.lock();
        if (!owner) {
            entry->leased = false;
            idle.push_front(std::move(entry));
            ++stats.rejected_acquires;
            return core::Status::Error(core::ErrorCode::InternalError, "SQLite connection pool is not initialized");
        }

        return SqliteConnectionLease(std::move(owner), std::move(entry));
    }

    core::Result<SqliteConnectionLease> WaitAcquire(SqliteConnectionKind kind) {
        std::unique_lock lock(mutex);
        auto& idle = kind == SqliteConnectionKind::Read ? read_idle : write_idle;
        if (!started || closed || !idle.empty()) {
            return TakeIdleLocked(kind, idle);
        }

        auto& waiting = kind == SqliteConnectionKind::Read
            ? stats.waiting_read_acquires
            : stats.waiting_write_acquires;
        ++waiting;
        const auto wait_started = std::chrono::steady_clock::now();

        available.wait(lock, [&] {
            return closed || !started || !idle.empty();
        });
        --waiting;
        RecordWaitLocked(kind, wait_started);

        return TakeIdleLocked(kind, idle);
    }

    core::Result<SqliteConnectionLease> WaitAcquireFor(SqliteConnectionKind kind, std::chrono::milliseconds timeout) {
        if (timeout.count() < 0) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "SQLite acquire timeout must be non-negative");
        }

        std::unique_lock lock(mutex);
        auto& idle = kind == SqliteConnectionKind::Read ? read_idle : write_idle;
        if (!started || closed || !idle.empty()) {
            return TakeIdleLocked(kind, idle);
        }

        auto& waiting = kind == SqliteConnectionKind::Read
            ? stats.waiting_read_acquires
            : stats.waiting_write_acquires;
        ++waiting;
        const auto wait_started = std::chrono::steady_clock::now();

        const bool ready = available.wait_for(lock, timeout, [&] {
            return closed || !started || !idle.empty();
        });
        --waiting;
        RecordWaitLocked(kind, wait_started);

        if (!ready) {
            ++stats.rejected_acquires;
            return core::Status::Error(core::ErrorCode::Timeout, "SQLite connection acquire timed out");
        }

        return TakeIdleLocked(kind, idle);
    }

    void RecordWaitLocked(SqliteConnectionKind kind,
                          std::chrono::steady_clock::time_point started_at) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started_at);
        const auto elapsed_ns = static_cast<std::uint64_t>(std::max<std::int64_t>(0, elapsed.count()));
        if (kind == SqliteConnectionKind::Read) {
            ++stats.read_wait_count;
            stats.total_read_wait_ns += elapsed_ns;
            return;
        }
        ++stats.write_wait_count;
        stats.total_write_wait_ns += elapsed_ns;
    }

    core::Result<SqliteConnectionLease> TakeIdleLocked(SqliteConnectionKind,
                                                       std::deque<std::shared_ptr<SqliteConnectionPoolEntry>>& idle) {
        if (!started || closed) {
            ++stats.rejected_acquires;
            return core::Status::Error(core::ErrorCode::Cancelled, "SQLite connection pool is closed");
        }

        auto entry = std::move(idle.front());
        idle.pop_front();
        entry->leased = true;
        entry->last_used = std::chrono::steady_clock::now();
        ++stats.acquired_connections;

        auto owner = self.lock();
        if (!owner) {
            entry->leased = false;
            idle.push_front(std::move(entry));
            ++stats.rejected_acquires;
            available.notify_one();
            return core::Status::Error(core::ErrorCode::InternalError, "SQLite connection pool is not initialized");
        }

        return SqliteConnectionLease(std::move(owner), std::move(entry));
    }

    void Release(std::shared_ptr<SqliteConnectionPoolEntry> entry) {
        if (!entry) {
            return;
        }

        {
            std::lock_guard lock(mutex);
            if (closed) {
                entry->leased = false;
                return;
            }

            entry->leased = false;
            entry->last_used = std::chrono::steady_clock::now();
            auto& idle = entry->kind == SqliteConnectionKind::Read ? read_idle : write_idle;
            idle.push_back(std::move(entry));
        }

        available.notify_one();
    }

    void Close() {
        {
            std::lock_guard lock(mutex);
            CloseLocked();
        }
        available.notify_all();
    }

    SqliteConnectionPoolStats Stats() const {
        std::lock_guard lock(mutex);
        auto result = stats;
        result.total_read_connections = CountKind(SqliteConnectionKind::Read);
        result.idle_read_connections = read_idle.size();
        result.leased_read_connections = CountLeased(SqliteConnectionKind::Read);
        result.total_write_connections = CountKind(SqliteConnectionKind::Write);
        result.idle_write_connections = write_idle.size();
        result.leased_write_connections = CountLeased(SqliteConnectionKind::Write);
        return result;
    }

    std::vector<SqliteConnectionSnapshot> Snapshots() const {
        std::lock_guard lock(mutex);
        std::vector<SqliteConnectionSnapshot> snapshots;
        snapshots.reserve(entries.size());
        for (const auto& entry : entries) {
            snapshots.push_back(SqliteConnectionSnapshot{
                entry->connection_id,
                entry->kind,
                entry->leased,
                entry->last_used});
        }
        std::sort(snapshots.begin(), snapshots.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.connection_id < rhs.connection_id;
        });
        return snapshots;
    }

    core::Status CreateConnections(SqliteConnectionKind kind,
                                   std::size_t count,
                                   std::deque<std::shared_ptr<SqliteConnectionPoolEntry>>& idle) {
        for (std::size_t i = 0; i < count; ++i) {
            auto connection_result = SqliteConnection::Open(options.path, OpenFlags(kind));
            if (!connection_result.ok()) {
                return connection_result.status();
            }

            auto connection = std::move(connection_result).value();
            auto status = connection.InitializeForPool(options.busy_timeout_ms, options.enable_wal);
            if (!status.ok()) {
                return status;
            }

            std::shared_ptr<SqliteConnectionPoolEntry> entry;
            try {
                entry = std::make_shared<SqliteConnectionPoolEntry>(
                    next_connection_id++,
                    kind,
                    std::move(connection));
            } catch (const std::bad_alloc&) {
                return core::Status::Error(core::ErrorCode::OutOfMemory, "SQLite connection pool entry allocation failed");
            }

            idle.push_back(entry);
            entries.push_back(std::move(entry));
        }
        return core::Status::Ok();
    }

    int OpenFlags(SqliteConnectionKind kind) const noexcept {
        int flags = SQLITE_OPEN_NOMUTEX;
        if (options.full_mutex) {
            flags = SQLITE_OPEN_FULLMUTEX;
        }

        if (kind == SqliteConnectionKind::Read) {
            return SQLITE_OPEN_READONLY | flags;
        }
        return SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | flags;
    }

    void CloseLocked() {
        closed = true;
        started = false;
        read_idle.clear();
        write_idle.clear();
        entries.clear();
    }

    std::size_t CountKind(SqliteConnectionKind kind) const noexcept {
        return static_cast<std::size_t>(std::count_if(entries.begin(), entries.end(), [kind](const auto& entry) {
            return entry->kind == kind;
        }));
    }

    std::size_t CountLeased(SqliteConnectionKind kind) const noexcept {
        return static_cast<std::size_t>(std::count_if(entries.begin(), entries.end(), [kind](const auto& entry) {
            return entry->kind == kind && entry->leased;
        }));
    }

    SqliteConnectionPoolOptions options;
    mutable std::mutex mutex;
    std::condition_variable available;
    bool started = false;
    bool closed = false;
    std::uint64_t next_connection_id = 1;
    std::vector<std::shared_ptr<SqliteConnectionPoolEntry>> entries;
    std::deque<std::shared_ptr<SqliteConnectionPoolEntry>> read_idle;
    std::deque<std::shared_ptr<SqliteConnectionPoolEntry>> write_idle;
    SqliteConnectionPoolStats stats;
    std::weak_ptr<SqliteConnectionPoolState> self;
};

} // namespace detail

SqliteConnectionLease::SqliteConnectionLease(std::shared_ptr<detail::SqliteConnectionPoolState> owner,
                                             std::shared_ptr<detail::SqliteConnectionPoolEntry> entry) noexcept
    : owner_(std::move(owner)),
      entry_(std::move(entry)) {}

SqliteConnectionLease::SqliteConnectionLease(SqliteConnectionLease&& other) noexcept
    : owner_(std::move(other.owner_)),
      entry_(std::move(other.entry_)) {}

SqliteConnectionLease& SqliteConnectionLease::operator=(SqliteConnectionLease&& other) noexcept {
    if (this != &other) {
        Release();
        owner_ = std::move(other.owner_);
        entry_ = std::move(other.entry_);
    }
    return *this;
}

SqliteConnectionLease::~SqliteConnectionLease() {
    Release();
}

bool SqliteConnectionLease::valid() const noexcept {
    return owner_ && entry_ && entry_->leased;
}

SqliteConnectionLease::operator bool() const noexcept {
    return valid();
}

std::uint64_t SqliteConnectionLease::connection_id() const noexcept {
    return entry_ ? entry_->connection_id : 0;
}

SqliteConnectionKind SqliteConnectionLease::kind() const noexcept {
    return entry_ ? entry_->kind : SqliteConnectionKind::Read;
}

SqliteConnection& SqliteConnectionLease::connection() noexcept {
    return entry_->connection;
}

const SqliteConnection& SqliteConnectionLease::connection() const noexcept {
    return entry_->connection;
}

SqliteConnection* SqliteConnectionLease::operator->() noexcept {
    return &connection();
}

const SqliteConnection* SqliteConnectionLease::operator->() const noexcept {
    return &connection();
}

void SqliteConnectionLease::Release() {
    if (!owner_ || !entry_) {
        owner_.reset();
        entry_.reset();
        return;
    }

    auto entry = std::move(entry_);
    auto owner = std::move(owner_);
    owner->Release(std::move(entry));
}

SqliteConnectionPool::SqliteConnectionPool(SqliteConnectionPoolOptions options)
    : state_(std::make_shared<detail::SqliteConnectionPoolState>(std::move(options))) {
    state_->self = state_;
}

SqliteConnectionPool::~SqliteConnectionPool() {
    Close();
}

core::Status SqliteConnectionPool::Start() {
    return state_->Start();
}

void SqliteConnectionPool::Close() {
    state_->Close();
}

core::Result<SqliteConnectionLease> SqliteConnectionPool::AcquireRead() {
    return state_->Acquire(SqliteConnectionKind::Read);
}

core::Result<SqliteConnectionLease> SqliteConnectionPool::AcquireWrite() {
    return state_->Acquire(SqliteConnectionKind::Write);
}

core::Result<SqliteConnectionLease> SqliteConnectionPool::WaitAcquireRead() {
    return state_->WaitAcquire(SqliteConnectionKind::Read);
}

core::Result<SqliteConnectionLease> SqliteConnectionPool::WaitAcquireWrite() {
    return state_->WaitAcquire(SqliteConnectionKind::Write);
}

core::Result<SqliteConnectionLease> SqliteConnectionPool::WaitAcquireReadFor(std::chrono::milliseconds timeout) {
    return state_->WaitAcquireFor(SqliteConnectionKind::Read, timeout);
}

core::Result<SqliteConnectionLease> SqliteConnectionPool::WaitAcquireWriteFor(std::chrono::milliseconds timeout) {
    return state_->WaitAcquireFor(SqliteConnectionKind::Write, timeout);
}

SqliteConnectionPoolStats SqliteConnectionPool::Stats() const {
    return state_->Stats();
}

std::vector<SqliteConnectionSnapshot> SqliteConnectionPool::Snapshots() const {
    return state_->Snapshots();
}

} // namespace storage::sqlite
