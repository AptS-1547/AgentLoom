#include "persona_gateway_route_helpers.h"

namespace agent::service::gateway {
namespace {

using namespace route_detail;

DECLARE_AUTHENTICATED_HTTP_ROUTE(DocumentRegisterRoute, ::net::http::verb::post, "api", "document", "register") {
    if (!context.document_service) {
        const auto status = core::Status::Error(core::ErrorCode::FailedPrecondition, "document analysis service is not configured");
        SendJson(context.request, HttpStatusFor(status.code()), ErrorEnvelope(context.trace_id, status), context.trace_id);
        return;
    }
    if (!context.enable_path_register_test_endpoint) {
        const auto status = core::Status::Error(
            core::ErrorCode::PermissionDenied,
            "document path register test endpoint is disabled");
        SendJson(context.request, HttpStatusFor(status.code()), ErrorEnvelope(context.trace_id, status), context.trace_id);
        return;
    }
    const auto path_text = context.body.value("path", std::string{});
    if (path_text.empty()) {
        const auto status = core::Status::Error(core::ErrorCode::InvalidArgument, "document path is required");
        SendJson(context.request, HttpStatusFor(status.code()), ErrorEnvelope(context.trace_id, status), context.trace_id);
        return;
    }
    auto imported = context.document_service->ImportManagedFile(
        PathFromUtf8(path_text),
        context.body.value("fileName", context.body.value("file_name", std::string{})),
        context.identity.user_uuid,
        context.body.value("sessionId", context.body.value("session_id", std::string{})));
    if (!imported.ok()) {
        SendJson(context.request, HttpStatusFor(imported.status().code()), ErrorEnvelope(context.trace_id, imported.status()), context.trace_id);
        return;
    }
    SendJson(
        context.request,
        ::net::http::status::ok,
        DocumentMetadataEnvelope(context.trace_id, imported.value()),
        context.trace_id);
}

DECLARE_AUTHENTICATED_HTTP_ROUTE(DocumentAnalyzeRoute, ::net::http::verb::post, "api", "document", "analyze") {
    if (!context.document_service) {
        const auto status = core::Status::Error(core::ErrorCode::FailedPrecondition, "document analysis service is not configured");
        SendJson(context.request, HttpStatusFor(status.code()), ErrorEnvelope(context.trace_id, status), context.trace_id);
        return;
    }

    document::DocumentAnalyzeRequest req;
    req.trace_id = context.trace_id;
    const auto path_text = context.body.value("path", std::string{});
    const auto document_id = context.body.value("documentId", context.body.value("document_id", std::string{}));
    if (!path_text.empty() && !context.enable_path_analyze_test_endpoint) {
        const auto status = core::Status::Error(
            core::ErrorCode::PermissionDenied,
            "document path analyze test endpoint is disabled");
        SendJson(context.request, HttpStatusFor(status.code()), ErrorEnvelope(context.trace_id, status), context.trace_id);
        return;
    }
    req.path = PathFromUtf8(path_text);
    req.file_name = context.body.value("fileName", context.body.value("file_name", std::string{}));
    req.document_id = document_id;
    req.authenticated_user_uuid = context.identity.user_uuid;
    req.options.enable_embedding_clustering = context.body.value("enableEmbeddingClustering", true);
    req.options.chunk_similarity_threshold = context.body.value("chunkSimilarityThreshold", req.options.chunk_similarity_threshold);
    req.options.max_chunk_slices = context.body.value("maxChunkSlices", req.options.max_chunk_slices);
    req.options.enable_llm_chunk_fallback = context.body.value("enableLlmChunkFallback", false);
    req.options.chunk_llm_model = context.body.value("chunkLlmModel", std::string{});
    req.options.chunk_llm_max_tokens = context.body.value("chunkLlmMaxTokens", req.options.chunk_llm_max_tokens);
    if (req.options.enable_llm_chunk_fallback) {
        req.llm_client = context.llm_client;
    }
    req.embedding_provider = context.embedding_provider;
    req.llm_chunk_cache = context.llm_chunk_cache;
    req.semantic_cache = context.document_semantic_cache;

    auto trace_id = context.trace_id;
    auto request = context.request;
    auto status = context.document_service->SubmitAnalyze(
        std::move(req),
        [request = std::move(request), trace_id](core::Result<document::DocumentAnalyzeResponse> result) mutable {
            try {
                SendResult(request, std::move(result), trace_id, DocumentAnalyzeEnvelope);
            } catch (const std::exception& e) {
                const auto status = core::Status::Error(
                    core::ErrorCode::InternalError,
                    std::string("document analyze response serialization failed: ") + e.what());
                SendJson(request, HttpStatusFor(status.code()), ErrorEnvelope(trace_id, status), trace_id);
            } catch (...) {
                const auto status = core::Status::Error(
                    core::ErrorCode::InternalError,
                    "document analyze response serialization failed");
                SendJson(request, HttpStatusFor(status.code()), ErrorEnvelope(trace_id, status), trace_id);
            }
        });
    if (!status.ok()) {
        SendJson(context.request, HttpStatusFor(status.code()), ErrorEnvelope(trace_id, status), trace_id);
    }
}

DECLARE_AUTHENTICATED_WS_ROUTE(DocumentUploadStartWsRoute, "document.upload.start") {
    if (!context.document_service) {
        SendWsError(
            context.request,
            context.trace_id,
            core::Status::Error(core::ErrorCode::FailedPrecondition, "document analysis service is not configured"));
        return;
    }
    const auto payload = context.body.value("payload", Json::object());
    const auto file_name = payload.value("fileName", payload.value("file_name", std::string{}));
    if (file_name.empty()) {
        SendWsError(
            context.request,
            context.trace_id,
            core::Status::Error(core::ErrorCode::InvalidArgument, "fileName is required"));
        return;
    }
    const auto expected_size = payload.value("totalBytes", payload.value("total_bytes", std::uint64_t{0}));
    if (expected_size == 0) {
        SendWsError(
            context.request,
            context.trace_id,
            core::Status::Error(core::ErrorCode::InvalidArgument, "totalBytes is required"));
        return;
    }

    DocumentUploadSession session;
    session.upload_id = core::GenerateTraceId();
    session.file_name = file_name;
    session.owner_user_uuid = context.identity.user_uuid;
    session.session_id = payload.value("sessionId", payload.value("session_id", std::string{}));
    session.temp_path = UploadTempPath(session.upload_id);
    session.expected_size = expected_size;
    session.received_size = 0;
    session.connection_id = context.request->connection().connection_id;
    session.binary_mode = payload.value("mode", std::string{}) == "binary" ||
                          payload.value("encoding", std::string{}) == "binary";

    {
        std::lock_guard lock(context.upload_mutex);
        if (session.binary_mode) {
            const auto duplicate = std::find_if(
                context.document_uploads.begin(),
                context.document_uploads.end(),
                [connection_id = session.connection_id](const auto& entry) {
                    return entry.second.binary_mode && entry.second.connection_id == connection_id;
                });
            if (duplicate != context.document_uploads.end()) {
                SendWsError(
                    context.request,
                    context.trace_id,
                    core::Status::Error(core::ErrorCode::FailedPrecondition, "binary upload is already active on this websocket connection"));
                return;
            }
        }
        context.document_uploads.emplace(session.upload_id, session);
    }

    Json out{
        {"type", "document.upload.started"},
        {"payload", {
            {"ok", true},
            {"traceId", context.trace_id},
            {"uploadId", session.upload_id},
            {"mode", session.binary_mode ? "binary" : "base64"},
            {"receivedBytes", session.received_size},
            {"totalBytes", session.expected_size},
        }},
    };
    context.request->Send(TextFrame(context.request->memory_pool(), out.dump()));
}

DECLARE_AUTHENTICATED_WS_ROUTE(DocumentUploadChunkWsRoute, "document.upload.chunk") {
    const auto payload = context.body.value("payload", Json::object());
    const auto upload_id = payload.value("uploadId", payload.value("upload_id", std::string{}));
    const auto offset = payload.value("offset", std::uint64_t{0});
    const auto encoded = payload.value("data", std::string{});
    if (upload_id.empty() || encoded.empty()) {
        SendWsError(
            context.request,
            context.trace_id,
            core::Status::Error(core::ErrorCode::InvalidArgument, "uploadId and data are required"));
        return;
    }
    auto decoded = Base64Decode(encoded);
    if (!decoded.ok()) {
        SendWsError(context.request, context.trace_id, decoded.status());
        return;
    }

    DocumentUploadSession session;
    {
        std::lock_guard lock(context.upload_mutex);
        auto it = context.document_uploads.find(upload_id);
        if (it == context.document_uploads.end()) {
            SendWsError(
                context.request,
                context.trace_id,
                core::Status::Error(core::ErrorCode::NotFound, "upload session not found"));
            return;
        }
        if (offset != it->second.received_size) {
            SendWsError(
                context.request,
                context.trace_id,
                core::Status::Error(core::ErrorCode::InvalidArgument, "upload chunk offset does not match received size"));
            return;
        }
        if (it->second.received_size + decoded.value().size() > it->second.expected_size) {
            SendWsError(
                context.request,
                context.trace_id,
                core::Status::Error(core::ErrorCode::ResourceExhausted, "upload exceeds declared totalBytes"));
            return;
        }
        if (auto owner = EnsureDocumentUploadOwner(
                it->second,
                context.identity,
                context.request->connection().connection_id);
            !owner.ok()) {
            SendWsError(context.request, context.trace_id, owner);
            return;
        }
        session = it->second;
    }

    auto write_status = WriteUploadChunk(session.temp_path, decoded.value(), offset);
    if (!write_status.ok()) {
        SendWsError(context.request, context.trace_id, write_status);
        return;
    }

    std::uint64_t received = 0;
    {
        std::lock_guard lock(context.upload_mutex);
        auto it = context.document_uploads.find(upload_id);
        if (it == context.document_uploads.end()) {
            RemoveFileQuietly(session.temp_path);
            SendWsError(
                context.request,
                context.trace_id,
                core::Status::Error(core::ErrorCode::NotFound, "upload session not found"));
            return;
        }
        it->second.received_size += decoded.value().size();
        received = it->second.received_size;
    }

    Json out{
        {"type", "document.upload.chunk_ack"},
        {"payload", {
            {"ok", true},
            {"traceId", context.trace_id},
            {"uploadId", upload_id},
            {"receivedBytes", received},
        }},
    };
    context.request->Send(TextFrame(context.request->memory_pool(), out.dump()));
}

DECLARE_AUTHENTICATED_WS_ROUTE(DocumentUploadFinishWsRoute, "document.upload.finish") {
    if (!context.document_service) {
        SendWsError(
            context.request,
            context.trace_id,
            core::Status::Error(core::ErrorCode::FailedPrecondition, "document analysis service is not configured"));
        return;
    }
    const auto payload = context.body.value("payload", Json::object());
    const auto upload_id = payload.value("uploadId", payload.value("upload_id", std::string{}));
    DocumentUploadSession session;
    {
        std::lock_guard lock(context.upload_mutex);
        auto it = context.document_uploads.find(upload_id);
        if (it == context.document_uploads.end()) {
            SendWsError(
                context.request,
                context.trace_id,
                core::Status::Error(core::ErrorCode::NotFound, "upload session not found"));
            return;
        }
        session = it->second;
        if (session.received_size != session.expected_size) {
            SendWsError(
                context.request,
                context.trace_id,
                core::Status::Error(core::ErrorCode::FailedPrecondition, "upload is incomplete"));
            return;
        }
        if (auto owner = EnsureDocumentUploadOwner(
                session,
                context.identity,
                context.request->connection().connection_id);
            !owner.ok()) {
            SendWsError(context.request, context.trace_id, owner);
            return;
        }
        context.document_uploads.erase(it);
    }

    auto imported = context.document_service->ImportManagedFile(
        session.temp_path,
        session.file_name,
        session.owner_user_uuid,
        session.session_id);
    RemoveFileQuietly(session.temp_path);
    if (!imported.ok()) {
        SendWsError(context.request, context.trace_id, imported.status());
        return;
    }

    Json out{
        {"type", "document.upload.finished"},
        {"payload", DocumentMetadataEnvelope(context.trace_id, imported.value())},
    };
    context.request->Send(TextFrame(context.request->memory_pool(), out.dump()));
}

DECLARE_AUTHENTICATED_WS_ROUTE(DocumentUploadAbortWsRoute, "document.upload.abort") {
    const auto payload = context.body.value("payload", Json::object());
    const auto upload_id = payload.value("uploadId", payload.value("upload_id", std::string{}));
    std::filesystem::path temp_path;
    {
        std::lock_guard lock(context.upload_mutex);
        auto it = context.document_uploads.find(upload_id);
        if (it != context.document_uploads.end()) {
            if (auto owner = EnsureDocumentUploadOwner(
                    it->second,
                    context.identity,
                    context.request->connection().connection_id);
                !owner.ok()) {
                SendWsError(context.request, context.trace_id, owner);
                return;
            }
            temp_path = it->second.temp_path;
            context.document_uploads.erase(it);
        }
    }
    if (!temp_path.empty()) {
        RemoveFileQuietly(temp_path);
    }
    Json out{
        {"type", "document.upload.aborted"},
        {"payload", {
            {"ok", true},
            {"traceId", context.trace_id},
            {"uploadId", upload_id},
        }},
    };
    context.request->Send(TextFrame(context.request->memory_pool(), out.dump()));
}

} // namespace
} // namespace agent::service::gateway
