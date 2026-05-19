#include "sqlite/sqlite_internal.h"

#include <sqlite3.h>

#include <string>

namespace storage::sqlite::internal {

namespace {

core::ErrorCode MapSqliteCode(int sqlite_code) noexcept {
    switch (sqlite_code & 0xFF) {
    case SQLITE_OK:
    case SQLITE_ROW:
    case SQLITE_DONE:
        return core::ErrorCode::Ok;
    case SQLITE_NOMEM:
        return core::ErrorCode::OutOfMemory;
    case SQLITE_BUSY:
    case SQLITE_LOCKED:
        return core::ErrorCode::Unavailable;
    case SQLITE_NOTFOUND:
        return core::ErrorCode::NotFound;
    case SQLITE_FULL:
        return core::ErrorCode::ResourceExhausted;
    case SQLITE_CANTOPEN:
    case SQLITE_IOERR:
        return core::ErrorCode::Unavailable;
    case SQLITE_PERM:
    case SQLITE_READONLY:
    case SQLITE_AUTH:
        return core::ErrorCode::PermissionDenied;
    case SQLITE_CONSTRAINT:
        return core::ErrorCode::FailedPrecondition;
    case SQLITE_MISUSE:
        return core::ErrorCode::FailedPrecondition;
    case SQLITE_RANGE:
    case SQLITE_TOOBIG:
    case SQLITE_MISMATCH:
    case SQLITE_FORMAT:
        return core::ErrorCode::InvalidArgument;
    case SQLITE_ERROR:
    default:
        return core::ErrorCode::InternalError;
    }
}

std::string BuildMessage(int sqlite_code, const char* sqlite_message, const std::string& operation) {
    std::string message = operation;
    message += " failed";
    if (sqlite_message && sqlite_message[0] != '\0') {
        message += ": ";
        message += sqlite_message;
    }
    message += " (sqlite code ";
    message += std::to_string(sqlite_code);
    message += ")";
    return message;
}

} // namespace

core::Status MakeSqliteStatus(int sqlite_code, ::sqlite3* db, const std::string& operation) {
    const char* message = db ? sqlite3_errmsg(db) : sqlite3_errstr(sqlite_code);
    return core::Status::Error(MapSqliteCode(sqlite_code), BuildMessage(sqlite_code, message, operation));
}

core::Status MakeSqliteStatus(int sqlite_code, const std::string& operation) {
    return core::Status::Error(MapSqliteCode(sqlite_code), BuildMessage(sqlite_code, sqlite3_errstr(sqlite_code), operation));
}

} // namespace storage::sqlite::internal
