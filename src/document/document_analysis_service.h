#pragma once

#include "document_types.h"
#include "document_file_store.h"
#include "logger_adapter.h"
#include "sqlite/sqlite_connection_pool.h"
#include "thread_pool.h"

#include <chrono>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace agent::document {

struct DocumentAnalyzeRequest {
    std::filesystem::path path;
    std::string file_name;
    std::string trace_id;
    std::string document_id;
    DocumentAnalysisOptions options;
    std::shared_ptr<llm::ILlmClient> llm_client;
    std::shared_ptr<IDocumentEmbeddingProvider> embedding_provider;
    std::shared_ptr<IDocumentLlmChunkCache> llm_chunk_cache;
    std::shared_ptr<semantic_cache::ISemanticCache> semantic_cache;
};

struct DocumentAnalyzeLatency {
    std::chrono::milliseconds compute_queue_wait{0};
    std::chrono::milliseconds compute_stage{0};
    std::chrono::milliseconds total{0};
};

struct DocumentAnalyzeResponse {
    std::string trace_id;
    std::string document_id;
    nlohmann::json result;
    DocumentAnalyzeLatency latency;
};

using DocumentAnalyzeCallback = std::function<void(core::Result<DocumentAnalyzeResponse>)>;

class DocumentAnalysisService {
public:
    DocumentAnalysisService(core::ThreadPool& compute_pool,
                            core::ThreadPool& io_pool,
                            core::LoggerAdapter logger = core::LoggerAdapter::ForModule("document"));
    ~DocumentAnalysisService();

    core::Status SubmitAnalyze(DocumentAnalyzeRequest request, DocumentAnalyzeCallback callback);

    core::Status SetRepository(std::shared_ptr<storage::sqlite::SqliteConnectionPool> repository_pool);
    core::Status SetFileStore(std::shared_ptr<DocumentFileStore> file_store);
    core::Result<DocumentMetadataRecord> ImportManagedFile(const std::filesystem::path& source_path,
                                                           std::string_view display_name = {},
                                                           std::string_view owner_user_uuid = {},
                                                           std::string_view session_id = {});
    core::Status TouchDocumentAccess(const std::string& document_id, std::int64_t accessed_at_ms = 0);
    core::Status RunRetentionCleanupOnceForTest(std::int64_t now_ms);
    core::Status RunRetentionCleanupOnce(std::int64_t now_ms);
    void SetRetentionCleanupOptions(std::chrono::hours retention, std::chrono::seconds cleanup_interval);

private:
    struct DocumentLruEntry {
        std::string document_id;
        std::int64_t last_accessed_at_ms = 0;
    };

    core::ThreadPool& compute_pool_;
    core::ThreadPool& io_pool_;
    std::shared_ptr<storage::sqlite::SqliteConnectionPool> repository_pool_;
    std::shared_ptr<DocumentMetadataRepository> metadata_repository_;
    std::shared_ptr<DocumentFileStore> file_store_;
    core::LoggerAdapter logger_;
    std::deque<DocumentLruEntry> document_lru_;
    std::mutex document_lru_mutex_;
    std::chrono::hours retention_{std::chrono::hours(24 * 7)};
    std::chrono::seconds cleanup_interval_{std::chrono::seconds(60)};
};

} // namespace agent::document
