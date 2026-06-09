#pragma once

#include "result.h"
#include "sqlite/sqlite_connection_pool.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace agent::document {

struct DocumentMetadataRecord {
    std::string document_id;
    std::string content_hash;
    std::string owner_user_uuid;
    std::string session_id;
    std::string file_name;
    std::string file_type;
    std::string storage_path;
    std::int64_t uploaded_at_ms = 0;
    std::int64_t last_analyzed_at_ms = 0;
    std::int64_t last_accessed_at_ms = 0;
    std::string analysis_status = "uploaded";
    std::string analysis_trace_id;
    std::int64_t size_bytes = 0;
    std::string schema_version = "document_metadata.v1";
};

class DocumentMetadataRepository {
public:
    explicit DocumentMetadataRepository(std::shared_ptr<storage::sqlite::SqliteConnectionPool> pool);

    core::Status EnsureSchema();
    core::Status Upsert(DocumentMetadataRecord record);
    core::Result<DocumentMetadataRecord> GetByDocumentId(const std::string& document_id);
    core::Status MarkAnalyzing(const std::string& document_id, const std::string& trace_id);
    core::Status MarkAnalyzed(const std::string& document_id, const std::string& trace_id, std::int64_t analyzed_at_ms);
    core::Status MarkFailed(const std::string& document_id, const std::string& trace_id);
    core::Status TouchAccessed(const std::string& document_id, std::int64_t accessed_at_ms);
    core::Status DeleteByDocumentId(const std::string& document_id);

private:
    std::shared_ptr<storage::sqlite::SqliteConnectionPool> pool_;
};

} // namespace agent::document
