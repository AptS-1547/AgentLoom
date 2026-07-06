#include "document_analysis_service.h"
#include "ooxml_extractor.h"
#include "sqlite/sqlite_statement.h"
#include "trace_context.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <initializer_list>
#include <sstream>
#include <string_view>
#include <vector>
#include <utility>

namespace agent::document {
namespace {

using Clock = std::chrono::steady_clock;

std::chrono::milliseconds Since(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start);
}

std::string ExtensionOf(const std::filesystem::path& path) {
    auto ext_u8 = path.extension().u8string();
    auto ext = std::string(ext_u8.begin(), ext_u8.end());
    if (!ext.empty() && ext.front() == '.') {
        ext.erase(ext.begin());
    }
    for (auto& ch : ext) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return ext;
}

std::string ExtensionOfName(std::string_view name) {
    const auto slash = name.find_last_of("/\\");
    const auto start = slash == std::string_view::npos ? 0 : slash + 1;
    const auto dot = name.find_last_of('.');
    if (dot == std::string_view::npos || dot < start) {
        return {};
    }
    auto ext = std::string(name.substr(dot + 1));
    for (auto& ch : ext) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return ext;
}

std::string PathToUtf8(const std::filesystem::path& path) {
    const auto utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

bool ContainsAny(std::string_view value, std::initializer_list<std::string_view> needles) {
    return std::any_of(needles.begin(), needles.end(), [&](std::string_view needle) {
        return value.find(needle) != std::string_view::npos;
    });
}

bool IsZipVerificationRejection(const core::Status& status) {
    switch (status.code()) {
    case core::ErrorCode::InvalidArgument:
    case core::ErrorCode::PermissionDenied:
    case core::ErrorCode::ResourceExhausted:
        break;
    default:
        return false;
    }
    return ContainsAny(status.message(), {
        "plain ZIP/OOXML package",
        "failed to create zip memory source",
        "failed to open zip archive",
        "zip archive",
        "zip entry",
        "unsafe zip entry path",
        "OOXML package",
        "DOCX package",
        "PPTX package",
        "self-extracting executable package",
        "office macro project",
        "office active content",
        "embedded executable package",
    });
}

std::string FileNameOrPathName(const std::filesystem::path& path, std::string_view file_name) {
    if (!file_name.empty()) {
        return std::string(file_name);
    }
    return PathToUtf8(path.filename());
}

nlohmann::json OptionalInt(std::optional<int> value) {
    return value ? nlohmann::json(*value) : nlohmann::json(nullptr);
}

nlohmann::json OptionalDouble(std::optional<double> value) {
    return value ? nlohmann::json(*value) : nlohmann::json(nullptr);
}

nlohmann::json BlockToJson(const DocumentBlock& block) {
    return {
        {"id", block.id},
        {"text", block.text},
        {"source", block.source},
        {"order", block.order},
        {"kind", block.kind},
        {"page", OptionalInt(block.page)},
        {"slide", OptionalInt(block.slide)},
        {"paragraphIndex", OptionalInt(block.paragraph_index)},
        {"headingLevel", OptionalInt(block.heading_level)},
        {"numbering", block.numbering},
        {"style", block.style},
        {"fontSize", OptionalDouble(block.font_size)},
        {"bold", block.bold},
        {"bulletLevel", OptionalInt(block.bullet_level)},
        {"confidence", block.confidence},
        {"metadata", block.metadata},
    };
}

nlohmann::json BlocksToJson(const std::vector<DocumentBlock>& blocks) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& block : blocks) {
        out.push_back(BlockToJson(block));
    }
    return out;
}

nlohmann::json ChunksToJson(const std::vector<ChunkTrunk>& chunks) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& chunk : chunks) {
        nlohmann::json slices = nlohmann::json::array();
        for (const auto& slice : chunk.slices) {
            slices.push_back({
                {"title", slice.title},
                {"summary", slice.summary},
                {"text", slice.text},
                {"kind", slice.kind},
                {"confidence", slice.confidence},
            });
        }
        out.push_back({
            {"chunkId", chunk.chunk_id},
            {"blockIds", chunk.block_ids},
            {"text", chunk.text},
            {"title", chunk.title},
            {"summary", chunk.summary},
            {"slices", slices},
            {"order", chunk.order},
            {"page", OptionalInt(chunk.page)},
            {"slide", OptionalInt(chunk.slide)},
            {"confidence", chunk.confidence},
            {"source", chunk.source},
            {"reused", chunk.reused},
            {"metadata", chunk.metadata},
        });
    }
    return out;
}

nlohmann::json RunNode(std::string name, nlohmann::json output, std::chrono::milliseconds latency) {
    return {
        {"nodeName", std::move(name)},
        {"output", output},
        {"costMilliSeconds", latency.count()},
    };
}

nlohmann::json ChunkMetricsToJson(const DocumentChunkBuildMetrics& metrics) {
    return {
        {"groupCount", metrics.group_count},
        {"embeddingRequestCount", metrics.embedding_request_count},
        {"embeddingMs", metrics.embedding_ms},
        {"llmCacheLookupCount", metrics.llm_cache_lookup_count},
        {"llmCacheHitCount", metrics.llm_cache_hit_count},
        {"llmCacheLookupMs", metrics.llm_cache_lookup_ms},
        {"llmCacheStoreCount", metrics.llm_cache_store_count},
        {"llmCacheStoreMs", metrics.llm_cache_store_ms},
        {"semanticCacheLookupCount", metrics.semantic_cache_lookup_count},
        {"semanticCacheHitCount", metrics.semantic_cache_hit_count},
        {"semanticCacheLookupMs", metrics.semantic_cache_lookup_ms},
        {"semanticCacheStoreCount", metrics.semantic_cache_store_count},
        {"semanticCacheStoreMs", metrics.semantic_cache_store_ms},
        {"llmDirectCount", metrics.llm_direct_count},
        {"llmDirectMs", metrics.llm_direct_ms},
        {"embeddingSampleMs", metrics.embedding_sample_ms},
        {"embeddingSampleBytes", metrics.embedding_sample_bytes},
    };
}

std::uint64_t EstimateTokenCount(const std::vector<DocumentBlock>& blocks) {
    std::uint64_t chars = 0;
    for (const auto& block : blocks) {
        chars += block.text.size();
    }
    return std::max<std::uint64_t>(1, chars / 2);
}

std::size_t CountChunksBySource(const std::vector<ChunkTrunk>& chunks, std::string_view source) {
    return static_cast<std::size_t>(std::count_if(
        chunks.begin(),
        chunks.end(),
        [source](const ChunkTrunk& chunk) {
            return chunk.source == source;
        }));
}

std::size_t CountReusedChunks(const std::vector<ChunkTrunk>& chunks) {
    return static_cast<std::size_t>(std::count_if(
        chunks.begin(),
        chunks.end(),
        [](const ChunkTrunk& chunk) {
            return chunk.reused;
        }));
}

core::Result<std::vector<DocumentBlock>> ExtractByType(const std::filesystem::path& path,
                                                       std::string_view file_type,
                                                       const DocumentAnalysisOptions& options) {
    OoxmlExtractor extractor(options.extract);
    if (file_type == "docx") {
        return extractor.ExtractDocxBlocks(path);
    }
    if (file_type == "pptx") {
        return extractor.ExtractPptxBlocks(path);
    }
    return core::Status::Error(core::ErrorCode::InvalidArgument, "unsupported document type: " + std::string(file_type));
}

core::Status EnsureRepositorySchema(storage::sqlite::SqliteConnection& connection) {
    return connection.Execute(R"SQL(
CREATE TABLE IF NOT EXISTS document_analysis_results (
    document_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
)SQL");
}

std::int64_t NowUnixMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string JsonStringField(const nlohmann::json& json, std::string_view key, std::string fallback = {}) {
    const auto it = json.find(std::string(key));
    if (it == json.end() || !it->is_string()) {
        return fallback;
    }
    return it->get<std::string>();
}

core::Status StoreAnalysisResult(storage::sqlite::SqliteConnectionPool& pool,
                                 const DocumentAnalyzeResponse& response) {
    auto lease_result = pool.AcquireWrite();
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    auto lease = std::move(lease_result).value();
    auto& connection = lease.connection();
    auto stmt_result = connection.Prepare(R"SQL(
INSERT INTO document_analysis_results (
    document_id,
    trace_id,
    file_name,
    file_type,
    schema_version,
    result_json,
    created_at_ms,
    updated_at_ms
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
ON CONFLICT(document_id) DO UPDATE SET
    trace_id = excluded.trace_id,
    file_name = excluded.file_name,
    file_type = excluded.file_type,
    schema_version = excluded.schema_version,
    result_json = excluded.result_json,
    updated_at_ms = excluded.updated_at_ms
)SQL");
    if (!stmt_result.ok()) {
        return stmt_result.status();
    }
    auto stmt = std::move(stmt_result).value();
    const auto now = NowUnixMs();
    const auto schema_version = JsonStringField(response.result, "schemaVersion", "document_analysis.v1");
    const auto file_name = JsonStringField(response.result, "fileName");
    const auto file_type = JsonStringField(response.result, "fileType");
    std::string result_json;
    try {
        result_json = response.result.dump(
            -1,
            ' ',
            false,
            nlohmann::json::error_handler_t::replace);
    } catch (const std::exception& ex) {
        return core::Status::Error(core::ErrorCode::InternalError, "serialize document analysis result failed: " + std::string(ex.what()));
    }
    if (auto status = stmt.BindText(1, response.document_id); !status.ok()) return status;
    if (auto status = stmt.BindText(2, response.trace_id); !status.ok()) return status;
    if (auto status = stmt.BindText(3, file_name); !status.ok()) return status;
    if (auto status = stmt.BindText(4, file_type); !status.ok()) return status;
    if (auto status = stmt.BindText(5, schema_version); !status.ok()) return status;
    if (auto status = stmt.BindText(6, result_json); !status.ok()) return status;
    if (auto status = stmt.BindInt64(7, now); !status.ok()) return status;
    if (auto status = stmt.BindInt64(8, now); !status.ok()) return status;

    auto step = stmt.Step();
    if (!step.ok()) {
        return step.status();
    }
    if (step.value() != storage::sqlite::SqliteStepResult::Done) {
        return core::Status::Error(core::ErrorCode::InternalError, "document result insert did not complete");
    }
    return core::Status::Ok();
}

} // namespace

core::Result<nlohmann::json> AnalyzeDocument(const std::filesystem::path& path,
                                             std::string_view file_name,
                                             const DocumentAnalysisOptions& options,
                                             std::shared_ptr<llm::ILlmClient> llm_client,
                                             std::shared_ptr<IDocumentEmbeddingProvider> embedding_provider,
                                             std::shared_ptr<IDocumentLlmChunkCache> llm_chunk_cache,
                                             std::shared_ptr<semantic_cache::ISemanticCache> semantic_cache) {
    const auto started = Clock::now();
    const auto resolved_file_name = FileNameOrPathName(path, file_name);
    auto file_type = ExtensionOfName(resolved_file_name);
    if (file_type.empty()) {
        file_type = ExtensionOf(path);
    }
    nlohmann::json run_nodes = nlohmann::json::array();
    nlohmann::json warnings = nlohmann::json::array();

    const auto extract_started = Clock::now();
    auto blocks = ExtractByType(path, file_type, options);
    if (!blocks.ok()) {
        return blocks.status();
    }
    if (blocks.value().empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "no analyzable document blocks were extracted");
    }
    run_nodes.push_back(RunNode(
        "参数提取2",
        {{"filetype", file_type}, {"blockCount", blocks.value().size()}},
        Since(extract_started)));

    const auto chunk_started = Clock::now();
    DocumentChunkBuildMetrics chunk_metrics;
    auto chunks = BuildLocalChunks(
        blocks.value(),
        options,
        std::move(llm_client),
        std::move(embedding_provider),
        std::move(llm_chunk_cache),
        std::move(semantic_cache),
        &chunk_metrics);
    const auto llm_count = CountChunksBySource(chunks, "llm");
    const auto cache_reuse_count = CountReusedChunks(chunks);
    run_nodes.push_back(RunNode(
        "SemanticChunkQueue",
        {
            {"chunkCount", chunks.size()},
            {"llmCount", llm_count},
            {"cacheReuseCount", cache_reuse_count},
            {"metrics", ChunkMetricsToJson(chunk_metrics)},
            {"status", llm_count > 0 ? "llm_fallback" : "local"},
        },
        Since(chunk_started)));

    const auto mindmap_started = Clock::now();
    auto mindmap = BuildMindmap(blocks.value(), resolved_file_name, chunks, embedding_provider);
    if (!mindmap.contains("children") || mindmap["children"].empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "rule matching did not produce any mindmap nodes");
    }
    run_nodes.push_back(RunNode(
        "条件判断1",
        {{"ConditionIndex", file_type == "jpg" || file_type == "jpeg" || file_type == "png" ? 1 : 0}},
        Since(mindmap_started)));

    const auto diagnosis_started = Clock::now();
    auto diagnosis = BuildDiagnosis(blocks.value(), mindmap, chunks);
    run_nodes.push_back(RunNode(
        "知识检索1",
        {{"KnowledgeList", nlohmann::json::array({"local-structure-analysis"})}},
        Since(diagnosis_started)));

    run_nodes.push_back(RunNode(
        "大模型2",
        {{"implemented", false}, {"status", "deterministic_pipeline"}, {"mindmap", mindmap}, {"diagnosis", diagnosis}},
        std::chrono::milliseconds(0)));

    return nlohmann::json{
        {"fileName", resolved_file_name},
        {"fileType", file_type},
        {"mindmap", mindmap},
        {"diagnosis", diagnosis},
        {"blocks", BlocksToJson(blocks.value())},
        {"chunks", ChunksToJson(chunks)},
        {"runNodes", run_nodes},
        {"elapsedMs", Since(started).count()},
        {"tokenCount", EstimateTokenCount(blocks.value())},
        {"warnings", warnings},
        {"schemaVersion", "document_analysis.v1"},
    };
}

DocumentAnalysisService::DocumentAnalysisService(core::ThreadPool& compute_pool,
                                                 core::ThreadPool& io_pool,
                                                 core::LoggerAdapter logger)
    : compute_pool_(compute_pool),
      io_pool_(io_pool),
      logger_(std::move(logger)) {}

DocumentAnalysisService::~DocumentAnalysisService() = default;

void DocumentAnalysisService::SetRetentionCleanupOptions(std::chrono::hours retention,
                                                         std::chrono::seconds cleanup_interval) {
    if (retention.count() > 0) {
        retention_ = retention;
    }
    if (cleanup_interval.count() > 0) {
        cleanup_interval_ = cleanup_interval;
    }
}

core::Status DocumentAnalysisService::TouchDocumentAccess(const std::string& document_id,
                                                          std::int64_t accessed_at_ms) {
    if (document_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "document_id is required");
    }
    const auto now = accessed_at_ms == 0 ? NowUnixMs() : accessed_at_ms;
    {
        std::lock_guard<std::mutex> lock(document_lru_mutex_);
        auto it = std::find_if(document_lru_.begin(), document_lru_.end(), [&](const DocumentLruEntry& entry) {
            return entry.document_id == document_id;
        });
        if (it != document_lru_.end()) {
            document_lru_.erase(it);
        }
        document_lru_.push_back(DocumentLruEntry{document_id, now});
    }
    if (metadata_repository_) {
        return metadata_repository_->TouchAccessed(document_id, now);
    }
    return core::Status::Ok();
}

core::Status DocumentAnalysisService::RemoveDocumentAccess(const std::string& document_id) {
    if (document_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "document_id is required");
    }
    std::lock_guard<std::mutex> lock(document_lru_mutex_);
    auto it = std::remove_if(document_lru_.begin(), document_lru_.end(), [&](const DocumentLruEntry& entry) {
        return entry.document_id == document_id;
    });
    document_lru_.erase(it, document_lru_.end());
    return core::Status::Ok();
}

core::Status DocumentAnalysisService::DeleteManagedDocument(const std::string& document_id,
                                                            const std::string& storage_path) {
    if (document_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "document_id is required");
    }
    if (auto status = RemoveDocumentAccess(document_id); !status.ok()) {
        return status;
    }
    if (file_store_ && !storage_path.empty()) {
        auto remove_status = file_store_->RemoveManagedFile(storage_path);
        if (!remove_status.ok() && remove_status.code() != core::ErrorCode::NotFound) {
            return remove_status;
        }
    }
    if (metadata_repository_) {
        return metadata_repository_->DeleteByDocumentId(document_id);
    }
    return core::Status::Ok();
}

core::Status DocumentAnalysisService::RunRetentionCleanupOnceForTest(std::int64_t now_ms) {
    return RunRetentionCleanupOnce(now_ms);
}

core::Status DocumentAnalysisService::RunRetentionCleanupOnce(std::int64_t now_ms) {
    if (!metadata_repository_ || !file_store_) {
        return core::Status::Ok();
    }
    const auto retention_ms = std::chrono::duration_cast<std::chrono::milliseconds>(retention_).count();
    std::vector<DocumentLruEntry> expired;
    {
        std::lock_guard<std::mutex> lock(document_lru_mutex_);
        std::sort(document_lru_.begin(), document_lru_.end(), [](const DocumentLruEntry& lhs, const DocumentLruEntry& rhs) {
            return lhs.last_accessed_at_ms < rhs.last_accessed_at_ms;
        });
        while (!document_lru_.empty()) {
            const auto age_ms = now_ms - document_lru_.front().last_accessed_at_ms;
            if (age_ms <= retention_ms) {
                break;
            }
            expired.push_back(std::move(document_lru_.front()));
            document_lru_.pop_front();
        }
    }
    for (const auto& entry : expired) {
        auto record = metadata_repository_->GetByDocumentId(entry.document_id);
        if (!record.ok()) {
            if (record.status().code() == core::ErrorCode::NotFound) {
                continue;
            }
            return record.status();
        }
        const auto latest_access = record.value().last_accessed_at_ms;
        if (now_ms - latest_access <= retention_ms) {
            if (auto touch_status = TouchDocumentAccess(entry.document_id, latest_access); !touch_status.ok()) {
                return touch_status;
            }
            continue;
        }
        auto remove_status = file_store_->RemoveManagedFile(record.value().storage_path);
        if (!remove_status.ok() && remove_status.code() != core::ErrorCode::NotFound) {
            return remove_status;
        }
        if (auto delete_status = metadata_repository_->DeleteByDocumentId(entry.document_id); !delete_status.ok()) {
            return delete_status;
        }
        logger_.info("[document_lru] evicted document_id={} age_ms={}",
                     entry.document_id,
                     now_ms - latest_access);
    }
    return core::Status::Ok();
}

core::Status DocumentAnalysisService::SetRepository(std::shared_ptr<storage::sqlite::SqliteConnectionPool> repository_pool) {
    if (!repository_pool) {
        repository_pool_.reset();
        metadata_repository_.reset();
        return core::Status::Ok();
    }
    {
        auto lease_result = repository_pool->AcquireWrite();
        if (!lease_result.ok()) {
            return lease_result.status();
        }
        auto lease = std::move(lease_result).value();
        if (auto status = EnsureRepositorySchema(lease.connection()); !status.ok()) {
            return status;
        }
    }
    auto metadata_repository = std::make_shared<DocumentMetadataRepository>(repository_pool);
    if (auto status = metadata_repository->EnsureSchema(); !status.ok()) {
        return status;
    }
    repository_pool_ = std::move(repository_pool);
    metadata_repository_ = std::move(metadata_repository);
    return core::Status::Ok();
}

core::Status DocumentAnalysisService::SetFileStore(std::shared_ptr<DocumentFileStore> file_store) {
    if (!file_store) {
        file_store_.reset();
        return core::Status::Ok();
    }
    if (auto status = file_store->EnsureRoot(); !status.ok()) {
        return status;
    }
    retention_ = file_store->retention();
    file_store_ = std::move(file_store);
    return core::Status::Ok();
}

core::Result<DocumentMetadataRecord> DocumentAnalysisService::ImportManagedFile(const std::filesystem::path& source_path,
                                                                                std::string_view display_name,
                                                                                std::string_view owner_user_uuid,
                                                                                std::string_view session_id) {
    if (!file_store_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "document file store is not configured");
    }
    if (!metadata_repository_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "document metadata repository is not configured");
    }
    auto record = file_store_->ImportLocalFile(source_path, display_name, owner_user_uuid, session_id);
    if (!record.ok()) {
        return record.status();
    }
    auto value = std::move(record).value();
    if (auto status = metadata_repository_->Upsert(value); !status.ok()) {
        return status;
    }
    if (auto status = TouchDocumentAccess(value.document_id, value.last_accessed_at_ms); !status.ok()) {
        return status;
    }
    return value;
}

core::Status DocumentAnalysisService::SubmitAnalyze(DocumentAnalyzeRequest request,
                                                    DocumentAnalyzeCallback callback) {
    if (!callback) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "document callback is required");
    }
    if (request.trace_id.empty()) {
        request.trace_id = core::GenerateTraceId();
    }
    if (request.document_id.empty()) {
        request.document_id = request.trace_id;
    }
    std::string managed_storage_path;
    if (request.path.empty()) {
        if (!metadata_repository_ || !file_store_) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "document path is required when document metadata store is not configured");
        }
        auto metadata = metadata_repository_->GetByDocumentId(request.document_id);
        if (!metadata.ok()) {
            return metadata.status();
        }
        if (!request.authenticated_user_uuid.empty() &&
            metadata.value().owner_user_uuid != request.authenticated_user_uuid) {
            return core::Status::Error(
                core::ErrorCode::PermissionDenied,
                "document does not belong to authenticated user");
        }
        managed_storage_path = metadata.value().storage_path;
        if (auto status = TouchDocumentAccess(request.document_id); !status.ok()) {
            logger_.warn("[trace={}] [document_analysis] touch access failed code={} reason={}",
                         request.trace_id,
                         static_cast<int>(status.code()),
                         status.message());
        }
        auto managed_path = file_store_->ResolveManagedPath(metadata.value().storage_path);
        if (!managed_path.ok()) {
            return managed_path.status();
        }
        request.path = std::move(managed_path).value();
        if (request.file_name.empty()) {
            request.file_name = metadata.value().file_name;
        }
        if (auto status = metadata_repository_->MarkAnalyzing(request.document_id, request.trace_id); !status.ok()) {
            logger_.warn("[trace={}] [document_analysis] mark analyzing failed code={} reason={}",
                         request.trace_id,
                         static_cast<int>(status.code()),
                         status.message());
        }
    }

    const auto submitted_at = Clock::now();
    auto status = compute_pool_.Submit(
        [this,
         request = std::move(request),
         managed_storage_path = std::move(managed_storage_path),
         callback = std::move(callback),
         submitted_at](core::ThreadPoolContext&) mutable -> core::Status {
            const auto compute_started_at = Clock::now();
            logger_.info("[trace={}] [document_analysis] analyze start path={}",
                         request.trace_id,
                         PathToUtf8(request.path));

            core::Result<nlohmann::json> analyzed(
                core::Status::Error(core::ErrorCode::Unknown, "document analysis did not run"));
            try {
                analyzed = AnalyzeDocument(
                    request.path,
                    request.file_name,
                    request.options,
                    request.llm_client,
                    request.embedding_provider,
                    request.llm_chunk_cache,
                    request.semantic_cache);
            } catch (const std::exception& e) {
                analyzed = core::Status::Error(
                    core::ErrorCode::InternalError,
                    std::string("document analysis threw exception: ") + e.what());
            } catch (...) {
                analyzed = core::Status::Error(
                    core::ErrorCode::InternalError,
                    "document analysis threw unknown exception");
            }
            if (!analyzed.ok()) {
                logger_.warn("[trace={}] [document_analysis] analyze failed code={} reason={}",
                             request.trace_id,
                             static_cast<int>(analyzed.status().code()),
                             analyzed.status().message());
                if (metadata_repository_) {
                    auto mark_status = metadata_repository_->MarkFailed(request.document_id, request.trace_id);
                    if (!mark_status.ok()) {
                        logger_.warn("[trace={}] [document_analysis] mark failed status failed code={} reason={}",
                                     request.trace_id,
                                     static_cast<int>(mark_status.code()),
                                     mark_status.message());
                    }
                }
                if (!managed_storage_path.empty() && IsZipVerificationRejection(analyzed.status())) {
                    auto delete_status = DeleteManagedDocument(request.document_id, managed_storage_path);
                    if (!delete_status.ok()) {
                        logger_.warn("[trace={}] [document_analysis] delete zip-rejected managed document failed document_id={} code={} reason={}",
                                     request.trace_id,
                                     request.document_id,
                                     static_cast<int>(delete_status.code()),
                                     delete_status.message());
                    } else {
                        logger_.warn("[trace={}] [document_analysis] deleted zip-rejected managed document document_id={}",
                                     request.trace_id,
                                     request.document_id);
                    }
                }
                callback(analyzed.status());
                return analyzed.status();
            }
            DocumentAnalyzeResponse response;
            response.trace_id = request.trace_id;
            response.document_id = request.document_id;
            response.result = std::move(analyzed).value();
            response.latency.compute_queue_wait = Since(submitted_at);
            response.latency.compute_stage = Since(compute_started_at);
            response.latency.total = Since(submitted_at);

            auto repository_pool = repository_pool_;
            auto metadata_repository = metadata_repository_;
            if (!repository_pool) {
                logger_.info("[trace={}] [document_analysis] analyze done latency_ms={}",
                             response.trace_id,
                             response.latency.total.count());
                callback(std::move(response));
                return core::Status::Ok();
            }

            auto io_status = io_pool_.Submit(
                [this,
                 repository_pool = std::move(repository_pool),
                 metadata_repository = std::move(metadata_repository),
                 response = std::move(response),
                 callback = std::move(callback),
                 submitted_at](
                    core::ThreadPoolContext&) mutable -> core::Status {
                    auto store_status = StoreAnalysisResult(*repository_pool, response);
                    response.latency.total = Since(submitted_at);
                    if (!store_status.ok()) {
                        logger_.warn("[trace={}] [document_analysis] store failed code={} reason={}",
                                     response.trace_id,
                                     static_cast<int>(store_status.code()),
                                     store_status.message());
                        callback(store_status);
                        return store_status;
                    }
                    if (metadata_repository) {
                        auto mark_status = metadata_repository->MarkAnalyzed(
                            response.document_id,
                            response.trace_id,
                            NowUnixMs());
                        if (!mark_status.ok()) {
                            logger_.warn("[trace={}] [document_analysis] mark analyzed failed code={} reason={}",
                                         response.trace_id,
                                         static_cast<int>(mark_status.code()),
                                         mark_status.message());
                        }
                    }
                    if (metadata_repository) {
                        auto touch_status = TouchDocumentAccess(response.document_id, NowUnixMs());
                        if (!touch_status.ok()) {
                            logger_.warn("[trace={}] [document_analysis] touch analyzed access failed code={} reason={}",
                                         response.trace_id,
                                         static_cast<int>(touch_status.code()),
                                         touch_status.message());
                        }
                    }
                    logger_.info("[trace={}] [document_analysis] analyze stored latency_ms={}",
                                 response.trace_id,
                                 response.latency.total.count());
                    callback(std::move(response));
                    return core::Status::Ok();
                },
                {},
                "document.store_result");
            if (!io_status.ok()) {
                callback(io_status);
            }
            return io_status;
        },
        {},
        "document.analyze");
    if (!status.ok()) {
        callback(status);
    }
    return status;
}

} // namespace agent::document
