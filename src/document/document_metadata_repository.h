#pragma once

#include "result.h"
#include "sqlite/sqlite_migration.h"
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

class IDocumentMetadataRepository {
public:
    virtual ~IDocumentMetadataRepository() = default;

    virtual core::Status Upsert(DocumentMetadataRecord record) = 0;
    virtual core::Result<DocumentMetadataRecord> GetByDocumentId(
        const std::string& document_id) = 0;
    virtual core::Status MarkAnalyzing(
        const std::string& document_id,
        const std::string& trace_id) = 0;
    virtual core::Status MarkAnalyzed(
        const std::string& document_id,
        const std::string& trace_id,
        std::int64_t analyzed_at_ms) = 0;
    virtual core::Status MarkFailed(
        const std::string& document_id,
        const std::string& trace_id) = 0;
    virtual core::Status TouchAccessed(
        const std::string& document_id,
        std::int64_t accessed_at_ms) = 0;
    virtual core::Status DeleteByDocumentId(
        const std::string& document_id) = 0;
};

/// SQLite 兼容适配器。迁移 runner 接管 document_metadata namespace 前，保留 EnsureSchema 作为启动兼容入口。
class DocumentMetadataRepository final
    : public IDocumentMetadataRepository,
      public storage::sqlite::ISqliteMigrationSource {
public:
    explicit DocumentMetadataRepository(std::shared_ptr<storage::sqlite::SqliteConnectionPool> pool);

    core::Status EnsureSchema();
    core::Status Upsert(DocumentMetadataRecord record) override;
    core::Result<DocumentMetadataRecord> GetByDocumentId(const std::string& document_id) override;
    core::Status MarkAnalyzing(const std::string& document_id, const std::string& trace_id) override;
    core::Status MarkAnalyzed(const std::string& document_id, const std::string& trace_id, std::int64_t analyzed_at_ms) override;
    core::Status MarkFailed(const std::string& document_id, const std::string& trace_id) override;
    core::Status TouchAccessed(const std::string& document_id, std::int64_t accessed_at_ms) override;
    core::Status DeleteByDocumentId(const std::string& document_id) override;

    std::string_view MigrationNamespace() const noexcept override;
    std::span<const storage::sqlite::SqliteMigrationStep>
    MigrationSteps() const noexcept override;

private:
    std::shared_ptr<storage::sqlite::SqliteConnectionPool> pool_;
};

}
