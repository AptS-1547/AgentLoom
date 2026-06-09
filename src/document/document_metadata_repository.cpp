#include "document_metadata_repository.h"

#include "sqlite/sqlite_statement.h"

#include <chrono>
#include <utility>

namespace agent::document {
namespace {

std::int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

core::Result<storage::sqlite::SqliteConnectionLease> AcquireWrite(storage::sqlite::SqliteConnectionPool& pool) {
    return pool.WaitAcquireWriteFor(std::chrono::milliseconds(5000));
}

core::Result<storage::sqlite::SqliteConnectionLease> AcquireRead(storage::sqlite::SqliteConnectionPool& pool) {
    return pool.WaitAcquireReadFor(std::chrono::milliseconds(5000));
}

DocumentMetadataRecord ReadRecord(const storage::sqlite::SqliteStatement& stmt) {
    DocumentMetadataRecord record;
    int i = 0;
    record.document_id = stmt.ColumnText(i++);
    record.content_hash = stmt.ColumnText(i++);
    record.owner_user_uuid = stmt.ColumnText(i++);
    record.session_id = stmt.ColumnText(i++);
    record.file_name = stmt.ColumnText(i++);
    record.file_type = stmt.ColumnText(i++);
    record.storage_path = stmt.ColumnText(i++);
    record.uploaded_at_ms = stmt.ColumnInt64(i++);
    record.last_analyzed_at_ms = stmt.ColumnInt64(i++);
    record.last_accessed_at_ms = stmt.ColumnInt64(i++);
    record.analysis_status = stmt.ColumnText(i++);
    record.analysis_trace_id = stmt.ColumnText(i++);
    record.size_bytes = stmt.ColumnInt64(i++);
    record.schema_version = stmt.ColumnText(i++);
    return record;
}

constexpr const char* kColumns =
    "document_id, content_hash, owner_user_uuid, session_id, file_name, file_type, "
    "storage_path, uploaded_at_ms, last_analyzed_at_ms, last_accessed_at_ms, "
    "analysis_status, analysis_trace_id, size_bytes, schema_version";

} // namespace

DocumentMetadataRepository::DocumentMetadataRepository(std::shared_ptr<storage::sqlite::SqliteConnectionPool> pool)
    : pool_(std::move(pool)) {}

core::Status DocumentMetadataRepository::EnsureSchema() {
    if (!pool_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "document metadata repository pool is not configured");
    }
    auto lease_result = AcquireWrite(*pool_);
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    return std::move(lease_result).value().connection().Execute(R"SQL(
CREATE TABLE IF NOT EXISTS document_files (
    document_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    owner_user_uuid TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    uploaded_at_ms INTEGER NOT NULL,
    last_analyzed_at_ms INTEGER NOT NULL DEFAULT 0,
    last_accessed_at_ms INTEGER NOT NULL DEFAULT 0,
    analysis_status TEXT NOT NULL,
    analysis_trace_id TEXT NOT NULL DEFAULT '',
    size_bytes INTEGER NOT NULL,
    schema_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_document_files_content_hash ON document_files(content_hash);
CREATE INDEX IF NOT EXISTS idx_document_files_owner_access ON document_files(owner_user_uuid, last_accessed_at_ms);
)SQL");
}

core::Status DocumentMetadataRepository::Upsert(DocumentMetadataRecord record) {
    if (!pool_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "document metadata repository pool is not configured");
    }
    if (record.document_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "document_id is required");
    }
    const auto now = NowMs();
    if (record.uploaded_at_ms == 0) {
        record.uploaded_at_ms = now;
    }
    if (record.last_accessed_at_ms == 0) {
        record.last_accessed_at_ms = record.uploaded_at_ms;
    }
    if (record.schema_version.empty()) {
        record.schema_version = "document_metadata.v1";
    }
    auto lease_result = AcquireWrite(*pool_);
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    auto lease = std::move(lease_result).value();
    auto stmt_result = lease.connection().Prepare(R"SQL(
INSERT INTO document_files (
    document_id, content_hash, owner_user_uuid, session_id, file_name, file_type,
    storage_path, uploaded_at_ms, last_analyzed_at_ms, last_accessed_at_ms,
    analysis_status, analysis_trace_id, size_bytes, schema_version
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
ON CONFLICT(document_id) DO UPDATE SET
    content_hash = excluded.content_hash,
    owner_user_uuid = excluded.owner_user_uuid,
    session_id = excluded.session_id,
    file_name = excluded.file_name,
    file_type = excluded.file_type,
    storage_path = excluded.storage_path,
    last_accessed_at_ms = excluded.last_accessed_at_ms,
    analysis_status = excluded.analysis_status,
    analysis_trace_id = excluded.analysis_trace_id,
    size_bytes = excluded.size_bytes,
    schema_version = excluded.schema_version
)SQL");
    if (!stmt_result.ok()) {
        return stmt_result.status();
    }
    auto stmt = std::move(stmt_result).value();
    int i = 1;
    if (auto s = stmt.BindText(i++, record.document_id); !s.ok()) return s;
    if (auto s = stmt.BindText(i++, record.content_hash); !s.ok()) return s;
    if (auto s = stmt.BindText(i++, record.owner_user_uuid); !s.ok()) return s;
    if (auto s = stmt.BindText(i++, record.session_id); !s.ok()) return s;
    if (auto s = stmt.BindText(i++, record.file_name); !s.ok()) return s;
    if (auto s = stmt.BindText(i++, record.file_type); !s.ok()) return s;
    if (auto s = stmt.BindText(i++, record.storage_path); !s.ok()) return s;
    if (auto s = stmt.BindInt64(i++, record.uploaded_at_ms); !s.ok()) return s;
    if (auto s = stmt.BindInt64(i++, record.last_analyzed_at_ms); !s.ok()) return s;
    if (auto s = stmt.BindInt64(i++, record.last_accessed_at_ms); !s.ok()) return s;
    if (auto s = stmt.BindText(i++, record.analysis_status); !s.ok()) return s;
    if (auto s = stmt.BindText(i++, record.analysis_trace_id); !s.ok()) return s;
    if (auto s = stmt.BindInt64(i++, record.size_bytes); !s.ok()) return s;
    if (auto s = stmt.BindText(i++, record.schema_version); !s.ok()) return s;
    auto step = stmt.Step();
    if (!step.ok()) {
        return step.status();
    }
    return core::Status::Ok();
}

core::Result<DocumentMetadataRecord> DocumentMetadataRepository::GetByDocumentId(const std::string& document_id) {
    if (document_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "document_id is required");
    }
    auto lease_result = AcquireRead(*pool_);
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    auto lease = std::move(lease_result).value();
    std::string sql = "SELECT ";
    sql += kColumns;
    sql += " FROM document_files WHERE document_id = ?1";
    auto stmt_result = lease.connection().Prepare(sql);
    if (!stmt_result.ok()) {
        return stmt_result.status();
    }
    auto stmt = std::move(stmt_result).value();
    if (auto s = stmt.BindText(1, document_id); !s.ok()) return s;
    auto step = stmt.Step();
    if (!step.ok()) {
        return step.status();
    }
    if (step.value() != storage::sqlite::SqliteStepResult::Row) {
        return core::Status::Error(core::ErrorCode::NotFound, "document metadata not found");
    }
    return ReadRecord(stmt);
}

core::Status DocumentMetadataRepository::MarkAnalyzing(const std::string& document_id, const std::string& trace_id) {
    auto lease_result = AcquireWrite(*pool_);
    if (!lease_result.ok()) return lease_result.status();
    auto lease = std::move(lease_result).value();
    auto stmt_result = lease.connection().Prepare(
        "UPDATE document_files SET analysis_status = 'analyzing', analysis_trace_id = ?1, "
        "last_accessed_at_ms = ?2 WHERE document_id = ?3");
    if (!stmt_result.ok()) return stmt_result.status();
    auto stmt = std::move(stmt_result).value();
    if (auto s = stmt.BindText(1, trace_id); !s.ok()) return s;
    if (auto s = stmt.BindInt64(2, NowMs()); !s.ok()) return s;
    if (auto s = stmt.BindText(3, document_id); !s.ok()) return s;
    auto step = stmt.Step();
    if (!step.ok()) return step.status();
    return core::Status::Ok();
}

core::Status DocumentMetadataRepository::MarkAnalyzed(const std::string& document_id,
                                                      const std::string& trace_id,
                                                      std::int64_t analyzed_at_ms) {
    auto lease_result = AcquireWrite(*pool_);
    if (!lease_result.ok()) return lease_result.status();
    auto lease = std::move(lease_result).value();
    auto stmt_result = lease.connection().Prepare(
        "UPDATE document_files SET analysis_status = 'completed', analysis_trace_id = ?1, "
        "last_analyzed_at_ms = ?2, last_accessed_at_ms = ?2 WHERE document_id = ?3");
    if (!stmt_result.ok()) return stmt_result.status();
    auto stmt = std::move(stmt_result).value();
    if (auto s = stmt.BindText(1, trace_id); !s.ok()) return s;
    if (auto s = stmt.BindInt64(2, analyzed_at_ms == 0 ? NowMs() : analyzed_at_ms); !s.ok()) return s;
    if (auto s = stmt.BindText(3, document_id); !s.ok()) return s;
    auto step = stmt.Step();
    if (!step.ok()) return step.status();
    return core::Status::Ok();
}

core::Status DocumentMetadataRepository::MarkFailed(const std::string& document_id, const std::string& trace_id) {
    auto lease_result = AcquireWrite(*pool_);
    if (!lease_result.ok()) return lease_result.status();
    auto lease = std::move(lease_result).value();
    auto stmt_result = lease.connection().Prepare(
        "UPDATE document_files SET analysis_status = ?1, analysis_trace_id = ?2, "
        "last_accessed_at_ms = ?3 WHERE document_id = ?4");
    if (!stmt_result.ok()) return stmt_result.status();
    auto stmt = std::move(stmt_result).value();
    if (auto s = stmt.BindText(1, "failed"); !s.ok()) return s;
    if (auto s = stmt.BindText(2, trace_id); !s.ok()) return s;
    if (auto s = stmt.BindInt64(3, NowMs()); !s.ok()) return s;
    if (auto s = stmt.BindText(4, document_id); !s.ok()) return s;
    auto step = stmt.Step();
    if (!step.ok()) return step.status();
    return core::Status::Ok();
}

core::Status DocumentMetadataRepository::TouchAccessed(const std::string& document_id, std::int64_t accessed_at_ms) {
    auto lease_result = AcquireWrite(*pool_);
    if (!lease_result.ok()) return lease_result.status();
    auto lease = std::move(lease_result).value();
    auto stmt_result = lease.connection().Prepare(
        "UPDATE document_files SET last_accessed_at_ms = ?1 WHERE document_id = ?2");
    if (!stmt_result.ok()) return stmt_result.status();
    auto stmt = std::move(stmt_result).value();
    if (auto s = stmt.BindInt64(1, accessed_at_ms == 0 ? NowMs() : accessed_at_ms); !s.ok()) return s;
    if (auto s = stmt.BindText(2, document_id); !s.ok()) return s;
    auto step = stmt.Step();
    if (!step.ok()) return step.status();
    return core::Status::Ok();
}

core::Status DocumentMetadataRepository::DeleteByDocumentId(const std::string& document_id) {
    if (document_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "document_id is required");
    }
    auto lease_result = AcquireWrite(*pool_);
    if (!lease_result.ok()) return lease_result.status();
    auto lease = std::move(lease_result).value();
    auto stmt_result = lease.connection().Prepare("DELETE FROM document_files WHERE document_id = ?1");
    if (!stmt_result.ok()) return stmt_result.status();
    auto stmt = std::move(stmt_result).value();
    if (auto s = stmt.BindText(1, document_id); !s.ok()) return s;
    auto step = stmt.Step();
    if (!step.ok()) return step.status();
    return core::Status::Ok();
}

} // namespace agent::document
