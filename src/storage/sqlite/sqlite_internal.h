#pragma once

#include "result.h"
#include "sqlite/sqlite_statement.h"

#include <string>

struct sqlite3;
struct sqlite3_stmt;

namespace storage::sqlite::internal {

core::Status MakeSqliteStatus(int sqlite_code, sqlite3* db, const std::string& operation);
core::Status MakeSqliteStatus(int sqlite_code, const std::string& operation);

SqliteStatement AdoptSqliteStatement(::sqlite3* db, ::sqlite3_stmt* statement);

} // namespace storage::sqlite::internal
