#include "sqlite/sqlite_async_executor.h"

#include <utility>

namespace storage::sqlite {

SqliteAsyncExecutor::SqliteAsyncExecutor(SqliteAsyncExecutorOptions options)
    : acquire_timeout_(options.acquire_timeout),
      pool_(std::move(options.pool)),
      thread_pool_(core::ThreadPoolOptions{
          options.worker_count,
          options.queue_capacity,
          std::move(options.thread_pool_name)}) {}

SqliteAsyncExecutor::~SqliteAsyncExecutor() {
    Shutdown(true);
}

core::Status SqliteAsyncExecutor::Start() {
    auto status = pool_.Start();
    if (!status.ok()) {
        return status;
    }

    status = thread_pool_.Start();
    if (!status.ok()) {
        pool_.Close();
    }
    return status;
}

void SqliteAsyncExecutor::Shutdown(bool drain) {
    thread_pool_.Shutdown(drain);
    pool_.Close();
}

SqliteConnectionPool& SqliteAsyncExecutor::pool() noexcept {
    return pool_;
}

const SqliteConnectionPool& SqliteAsyncExecutor::pool() const noexcept {
    return pool_;
}

} // namespace storage::sqlite
