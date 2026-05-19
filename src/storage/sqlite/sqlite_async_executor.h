#pragma once

#include "result.h"
#include "sqlite/sqlite_connection_pool.h"
#include "sqlite/sqlite_transaction.h"
#include "thread_pool.h"

#include <future>
#include <memory>
#include <exception>
#include <chrono>
#include <string>
#include <type_traits>
#include <utility>

namespace storage::sqlite {

struct SqliteAsyncExecutorOptions {
    SqliteConnectionPoolOptions pool;
    std::size_t worker_count = 2;
    std::size_t queue_capacity = 256;
    std::chrono::milliseconds acquire_timeout{5000};
    std::string thread_pool_name = "sqlite-async-executor";
};

class SqliteAsyncExecutor {
public:
    explicit SqliteAsyncExecutor(SqliteAsyncExecutorOptions options);
    ~SqliteAsyncExecutor();

    SqliteAsyncExecutor(const SqliteAsyncExecutor&) = delete;
    SqliteAsyncExecutor& operator=(const SqliteAsyncExecutor&) = delete;

    core::Status Start();
    void Shutdown(bool drain = true);

    SqliteConnectionPool& pool() noexcept;
    const SqliteConnectionPool& pool() const noexcept;

    template <typename Fn>
    auto SubmitRead(Fn&& fn) -> std::future<core::Result<std::invoke_result_t<Fn&, SqliteConnection&>>> {
        return SubmitWithLease(SqliteConnectionKind::Read, std::forward<Fn>(fn), "sqlite-read");
    }

    template <typename Fn>
    auto SubmitWrite(Fn&& fn) -> std::future<core::Result<std::invoke_result_t<Fn&, SqliteConnection&>>> {
        return SubmitWithLease(SqliteConnectionKind::Write, std::forward<Fn>(fn), "sqlite-write");
    }

    template <typename Fn>
    auto SubmitTransaction(Fn&& fn) -> std::future<core::Result<std::invoke_result_t<Fn&, SqliteConnection&, SqliteTransaction&>>> {
        using Value = std::invoke_result_t<Fn&, SqliteConnection&, SqliteTransaction&>;
        auto promise = std::make_shared<std::promise<core::Result<Value>>>();
        auto future = promise->get_future();

        auto status = thread_pool_.Submit(
            [this, promise, func = std::forward<Fn>(fn)]() mutable -> core::Status {
                auto lease_result = pool_.WaitAcquireWriteFor(acquire_timeout_);
                if (!lease_result.ok()) {
                    promise->set_value(lease_result.status());
                    return lease_result.status();
                }

                auto lease = std::move(lease_result).value();
                SqliteTransaction transaction(lease.connection());
                auto status = transaction.Begin();
                if (!status.ok()) {
                    promise->set_value(status);
                    return status;
                }

                try {
                    if constexpr (std::is_void_v<Value>) {
                        func(lease.connection(), transaction);
                        status = transaction.Commit();
                        promise->set_value(status);
                        return status;
                    } else {
                        Value value = func(lease.connection(), transaction);
                        status = transaction.Commit();
                        if (!status.ok()) {
                            promise->set_value(status);
                            return status;
                        }
                        promise->set_value(std::move(value));
                        return core::Status::Ok();
                    }
                } catch (const std::exception& e) {
                    (void)transaction.Rollback();
                    auto error = core::Status::Error(core::ErrorCode::InternalError, e.what());
                    promise->set_value(error);
                    return error;
                } catch (...) {
                    (void)transaction.Rollback();
                    auto error = core::Status::Error(core::ErrorCode::Unknown, "unknown SQLite transaction task error");
                    promise->set_value(error);
                    return error;
                }
            },
            {},
            "sqlite-transaction");

        if (!status.ok()) {
            promise->set_value(status);
        }
        return future;
    }

private:
    template <typename Fn>
    auto SubmitWithLease(SqliteConnectionKind kind, Fn&& fn, std::string task_name)
        -> std::future<core::Result<std::invoke_result_t<Fn&, SqliteConnection&>>> {
        using Value = std::invoke_result_t<Fn&, SqliteConnection&>;
        auto promise = std::make_shared<std::promise<core::Result<Value>>>();
        auto future = promise->get_future();

        auto status = thread_pool_.Submit(
            [this, kind, promise, func = std::forward<Fn>(fn)]() mutable -> core::Status {
                auto lease_result = kind == SqliteConnectionKind::Read
                    ? pool_.WaitAcquireReadFor(acquire_timeout_)
                    : pool_.WaitAcquireWriteFor(acquire_timeout_);
                if (!lease_result.ok()) {
                    promise->set_value(lease_result.status());
                    return lease_result.status();
                }

                try {
                    auto lease = std::move(lease_result).value();
                    if constexpr (std::is_void_v<Value>) {
                        func(lease.connection());
                        promise->set_value(core::Status::Ok());
                    } else {
                        promise->set_value(func(lease.connection()));
                    }
                } catch (const std::exception& e) {
                    auto error = core::Status::Error(core::ErrorCode::InternalError, e.what());
                    promise->set_value(error);
                    return error;
                } catch (...) {
                    auto error = core::Status::Error(core::ErrorCode::Unknown, "unknown SQLite async task error");
                    promise->set_value(error);
                    return error;
                }
                return core::Status::Ok();
            },
            {},
            std::move(task_name));

        if (!status.ok()) {
            promise->set_value(status);
        }
        return future;
    }

    std::chrono::milliseconds acquire_timeout_;
    SqliteConnectionPool pool_;
    core::ThreadPool thread_pool_;
};

} // namespace storage::sqlite
