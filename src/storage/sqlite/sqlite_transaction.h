#pragma once

#include "result.h"

namespace storage::sqlite {

class SqliteConnection;

class SqliteTransaction {
public:
    explicit SqliteTransaction(SqliteConnection& connection);
    ~SqliteTransaction();

    SqliteTransaction(const SqliteTransaction&) = delete;
    SqliteTransaction& operator=(const SqliteTransaction&) = delete;
    SqliteTransaction(SqliteTransaction&& other) noexcept;
    SqliteTransaction& operator=(SqliteTransaction&& other) noexcept;

    core::Status Begin();
    core::Status Commit();
    core::Status Rollback();

    bool active() const noexcept;

private:
    SqliteConnection* connection_ = nullptr;
    bool active_ = false;
};

} // namespace storage::sqlite
