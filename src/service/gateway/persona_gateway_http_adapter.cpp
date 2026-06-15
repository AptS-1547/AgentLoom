#include "persona_gateway_http_adapter.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace agent::service::gateway {
namespace {

using Json = nlohmann::json;

std::string ErrorCodeName(core::ErrorCode code) {
    switch (code) {
    case core::ErrorCode::InvalidArgument: return "INVALID_ARGUMENT";
    case core::ErrorCode::NotFound: return "NOT_FOUND";
    case core::ErrorCode::AlreadyExists: return "ALREADY_EXISTS";
    case core::ErrorCode::PermissionDenied: return "PERMISSION_DENIED";
    case core::ErrorCode::FailedPrecondition: return "FAILED_PRECONDITION";
    case core::ErrorCode::ResourceExhausted: return "RESOURCE_EXHAUSTED";
    case core::ErrorCode::Unavailable: return "UNAVAILABLE";
    case core::ErrorCode::Timeout: return "TIMEOUT";
    case core::ErrorCode::InternalError: return "INTERNAL_ERROR";
    default: return "UNKNOWN";
    }
}

::net::http::status HttpStatusFor(core::ErrorCode code) {
    switch (code) {
    case core::ErrorCode::InvalidArgument: return ::net::http::status::bad_request;
    case core::ErrorCode::NotFound: return ::net::http::status::not_found;
    case core::ErrorCode::AlreadyExists: return ::net::http::status::conflict;
    case core::ErrorCode::PermissionDenied: return ::net::http::status::forbidden;
    case core::ErrorCode::FailedPrecondition: return ::net::http::status::conflict;
    case core::ErrorCode::ResourceExhausted: return ::net::http::status::too_many_requests;
    case core::ErrorCode::Unavailable: return ::net::http::status::service_unavailable;
    case core::ErrorCode::Timeout: return ::net::http::status::gateway_timeout;
    default: return ::net::http::status::internal_server_error;
    }
}

std::string HeaderValue(const ::net::BeastHttpRequest& req, ::net::http::field field) {
    auto it = req.find(field);
    if (it == req.end()) {
        return {};
    }
    return std::string(it->value());
}

std::string HeaderValue(const ::net::BeastHttpRequest& req, std::string_view field) {
    auto it = req.find(field);
    if (it == req.end()) {
        return {};
    }
    return std::string(it->value());
}

std::filesystem::path PathFromUtf8(std::string_view path) {
#ifdef _WIN32
    std::u8string utf8;
    utf8.reserve(path.size());
    for (const auto ch : path) {
        utf8.push_back(static_cast<char8_t>(ch));
    }
    return std::filesystem::path(std::move(utf8));
#else
    return std::filesystem::path(std::string(path));
#endif
}

std::string TraceFrom(const ::net::BeastHttpRequest& req, const Json* body = nullptr) {
    auto trace = HeaderValue(req, "X-Trace-Id");
    if (trace.empty()) {
        trace = HeaderValue(req, "X-Request-Id");
    }
    if (trace.empty() && body && body->is_object() && body->contains("traceId")) {
        trace = body->value("traceId", "");
    }
    if (trace.empty()) {
        trace = core::GenerateTraceId();
    }
    return trace;
}

Json ErrorEnvelope(std::string trace_id, const core::Status& status) {
    return Json{
        {"ok", false},
        {"traceId", trace_id},
        {"error", {
            {"code", ErrorCodeName(status.code())},
            {"message", status.message()},
        }},
    };
}

Json EmotionToJson(const persona::EmotionAnalysis& emotion) {
    return Json{
        {"primary", emotion.emotion.primary},
        {"intensity", emotion.emotion.intensity},
        {"behavior", emotion.behavior},
        {"tone", emotion.tone},
    };
}

std::string StatusName(persona::SessionStatus status) {
    switch (status) {
    case persona::SessionStatus::Creating: return "creating";
    case persona::SessionStatus::Active: return "active";
    case persona::SessionStatus::Disconnected: return "disconnected";
    case persona::SessionStatus::Closing: return "closing";
    case persona::SessionStatus::Closed: return "closed";
    }
    return "unknown";
}

Json MetricsToJson(const persona::SessionMetrics& metrics) {
    const auto avg = metrics.request_count == 0
        ? 0
        : metrics.total_latency.count() / static_cast<long long>(metrics.request_count);
    return Json{
        {"turnCount", metrics.turn_count},
        {"requestCount", metrics.request_count},
        {"failedRequestCount", metrics.failed_request_count},
        {"lastLatencyMs", metrics.last_latency.count()},
        {"totalLatencyMs", metrics.total_latency.count()},
        {"avgLatencyMs", avg},
    };
}

Json SessionToJson(const persona::SessionSnapshot& session) {
    return Json{
        {"sessionId", session.session_id},
        {"userUuid", session.user_uuid},
        {"personaId", session.persona_id},
        {"status", StatusName(session.status)},
        {"closeReason", session.close_reason},
        {"recentTurnCount", session.recent_turn_count},
        {"emotion", {
            {"valence", session.emotion_state.valence},
            {"arousal", session.emotion_state.arousal},
            {"primary", session.emotion_state.last_emotion},
            {"sustainedLabel", session.emotion_state.sustained_label},
        }},
        {"metrics", MetricsToJson(session.metrics)},
    };
}

Json SessionEnvelope(const SessionGatewayResponse& response) {
    return Json{
        {"ok", true},
        {"traceId", response.trace_id},
        {"sessionId", response.session_id},
        {"latencyMs", response.latency.count()},
        {"data", SessionToJson(response.session)},
    };
}

Json ChatEnvelope(const ChatGatewayResponse& response) {
    return Json{
        {"ok", true},
        {"traceId", response.trace_id},
        {"sessionId", response.session_id},
        {"latencyMs", response.latency.count()},
        {"data", {
            {"personaId", response.persona_id},
            {"turnIndex", response.turn_index},
            {"reply", {{"role", "assistant"}, {"content", response.content}}},
            {"userEmotion", EmotionToJson(response.user_emotion)},
            {"aiEmotion", EmotionToJson(response.ai_emotion)},
            {"memory", {{"l0Hit", response.l0_hit}, {"l3Hit", response.l3_hit}}},
            {"answerCache", {
                {"enabled", response.answer_cache.enabled},
                {"hit", response.answer_cache.hit},
                {"bypassed", response.answer_cache.bypassed},
                {"source", response.answer_cache.source},
                {"cacheKey", response.answer_cache.cache_key},
                {"similarityScore", response.answer_cache.similarity_score},
            }},
            {"pipelineLatency", {
                {"computeQueueWaitMs", response.pipeline_latency.compute_queue_wait.count()},
                {"computeStageMs", response.pipeline_latency.compute_stage.count()},
                {"ioQueueWaitMs", response.pipeline_latency.io_queue_wait.count()},
                {"ioStageMs", response.pipeline_latency.io_stage.count()},
                {"memoryContextMs", response.pipeline_latency.memory_context.count()},
                {"answerCacheMs", response.pipeline_latency.answer_cache.count()},
                {"promptBuildMs", response.pipeline_latency.prompt_build.count()},
                {"llmTotalMs", response.pipeline_latency.llm_total.count()},
                {"callbackToResponseMs", response.pipeline_latency.callback_to_response.count()},
                {"totalMs", response.pipeline_latency.total.count()},
            }},
        }},
    };
}

Json ClassroomEnvelope(const ClassroomGatewayResponse& response) {
    return Json{
        {"ok", true},
        {"traceId", response.trace_id},
        {"sessionId", response.session_id},
        {"latencyMs", response.latency.count()},
        {"data", {
            {"classroomId", response.classroom_id},
            {"speakerPersonaId", response.speaker_persona_id},
            {"content", response.content},
            {"shouldSpeak", response.should_speak},
            {"turnIndex", response.turn_index},
            {"userEmotion", EmotionToJson(response.user_emotion)},
            {"aiEmotion", EmotionToJson(response.ai_emotion)},
        }},
    };
}

Json ReportEnvelope(const TrainingReportGatewayResponse& response) {
    return Json{
        {"ok", true},
        {"traceId", response.trace_id},
        {"sessionId", response.session_id},
        {"latencyMs", response.latency.count()},
        {"data", {
            {"generatedAt", response.generated_at},
            {"totalTurns", response.total_turns},
            {"summary", response.summary},
            {"metrics", MetricsToJson(response.metrics)},
            {"schemaVersion", "training_report.v1"},
        }},
    };
}

Json ThreadPoolStatsToJson(const core::ThreadPoolStats& stats) {
    return Json{
        {"workerCount", stats.worker_count},
        {"queuedTasks", stats.queued_tasks},
        {"activeWorkers", stats.active_workers},
        {"submittedTasks", stats.submitted_tasks},
        {"completedTasks", stats.completed_tasks},
        {"failedTasks", stats.failed_tasks},
        {"rejectedTasks", stats.rejected_tasks},
    };
}

Json SystemStatsEnvelope(const SystemStatsGatewayResponse& response) {
    return Json{
        {"ok", true},
        {"traceId", response.trace_id},
        {"latencyMs", response.latency.count()},
        {"data", {
            {"sessionCount", response.session_count},
            {"pools", {
                {"compute", ThreadPoolStatsToJson(response.pools.compute)},
                {"io", ThreadPoolStatsToJson(response.pools.io)},
            }},
        }},
    };
}

Json DocumentAnalyzeEnvelope(const document::DocumentAnalyzeResponse& response) {
    return Json{
        {"ok", true},
        {"traceId", response.trace_id},
        {"documentId", response.document_id},
        {"latencyMs", response.latency.total.count()},
        {"data", response.result},
        {"pipelineLatency", {
            {"computeQueueWaitMs", response.latency.compute_queue_wait.count()},
            {"computeStageMs", response.latency.compute_stage.count()},
            {"totalMs", response.latency.total.count()},
        }},
    };
}

Json DocumentMetadataEnvelope(std::string trace_id, const document::DocumentMetadataRecord& record) {
    return Json{
        {"ok", true},
        {"traceId", std::move(trace_id)},
        {"documentId", record.document_id},
        {"data", {
            {"documentId", record.document_id},
            {"contentHash", record.content_hash},
            {"ownerUserUuid", record.owner_user_uuid},
            {"sessionId", record.session_id},
            {"fileName", record.file_name},
            {"fileType", record.file_type},
            {"uploadedAtMs", record.uploaded_at_ms},
            {"lastAnalyzedAtMs", record.last_analyzed_at_ms},
            {"lastAccessedAtMs", record.last_accessed_at_ms},
            {"analysisStatus", record.analysis_status},
            {"analysisTraceId", record.analysis_trace_id},
            {"sizeBytes", record.size_bytes},
            {"schemaVersion", record.schema_version},
        }},
    };
}

core::Result<Json> ParseJsonBody(const ::net::BeastHttpRequest& req) {
    if (req.body().empty()) {
        return Json::object();
    }
    try {
        return Json::parse(req.body());
    } catch (const Json::exception& e) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, e.what());
    }
}

std::vector<std::string> SplitPath(std::string_view target) {
    const auto q = target.find('?');
    if (q != std::string_view::npos) {
        target = target.substr(0, q);
    }
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start < target.size()) {
        while (start < target.size() && target[start] == '/') {
            ++start;
        }
        auto end = target.find('/', start);
        if (end == std::string_view::npos) {
            end = target.size();
        }
        if (end > start) {
            parts.emplace_back(target.substr(start, end - start));
        }
        start = end + 1;
    }
    return parts;
}

void SendJson(const std::shared_ptr<::net::IHttpRequest>& request,
              ::net::http::status status,
              const Json& body,
              std::string_view trace_id) {
    auto response = ::net::HttpResponse::Json(
        status,
        body.dump(-1, ' ', false, Json::error_handler_t::replace)).message;
    response.set("X-Trace-Id", trace_id);
    request->Respond(std::move(response));
}

void SendJsonWithHeaders(const std::shared_ptr<::net::IHttpRequest>& request,
                         ::net::http::status status,
                         const Json& body,
                         std::string_view trace_id,
                         const std::vector<std::pair<std::string, std::string>>& headers) {
    auto response = ::net::HttpResponse::Json(
        status,
        body.dump(-1, ' ', false, Json::error_handler_t::replace)).message;
    response.set("X-Trace-Id", trace_id);
    for (const auto& [key, value] : headers) {
        response.set(key, value);
    }
    request->Respond(std::move(response));
}

std::int64_t ToUnixSeconds(std::chrono::system_clock::time_point time) {
    return std::chrono::duration_cast<std::chrono::seconds>(time.time_since_epoch()).count();
}

template <typename T, typename Fn>
void SendResult(const std::shared_ptr<::net::IHttpRequest>& request,
                core::Result<T> result,
                std::string trace_id,
                Fn serializer) {
    if (!result.ok()) {
        SendJson(request, HttpStatusFor(result.status().code()), ErrorEnvelope(trace_id, result.status()), trace_id);
        return;
    }
    auto body = serializer(result.value());
    SendJson(request, ::net::http::status::ok, body, trace_id);
}

persona::PersonalityConfig PersonalityFromJson(const Json& body) {
    persona::PersonalityConfig personality;
    const auto persona_obj = body.value("personality", Json::object());
    personality.name = persona_obj.value("name", body.value("personaId", std::string{}));
    personality.description = persona_obj.value("description", std::string{});
    personality.traits = persona_obj.value("traits", std::vector<std::string>{});
    personality.openness = persona_obj.value("openness", personality.openness);
    personality.extraversion = persona_obj.value("extraversion", personality.extraversion);
    personality.humor_tendency = persona_obj.value("humorTendency", personality.humor_tendency);
    personality.empathy_level = persona_obj.value("empathyLevel", personality.empathy_level);
    personality.curiosity_level = persona_obj.value("curiosityLevel", personality.curiosity_level);
    personality.formality = persona_obj.value("formality", personality.formality);
    return personality;
}

std::optional<persona::EmotionPromptConfig> EmotionPromptConfigFromJson(const Json& body) {
    const auto it = body.find("emotionPrompts");
    if (it == body.end() || !it->is_object()) {
        return std::nullopt;
    }

    persona::EmotionPromptConfig config;
    config.emotion_map = it->value("emotionMap", std::map<std::string, std::string>{});
    if (config.emotion_map.empty()) {
        config.emotion_map = it->value("emotion_map", std::map<std::string, std::string>{});
    }
    config.emotion_reliability = it->value("emotionReliability", std::map<std::string, double>{});
    if (config.emotion_reliability.empty()) {
        config.emotion_reliability = it->value("emotion_reliability", std::map<std::string, double>{});
    }
    config.confidence_thresholds = it->value("confidenceThresholds", config.confidence_thresholds);
    if (!it->contains("confidenceThresholds")) {
        config.confidence_thresholds = it->value("confidence_thresholds", config.confidence_thresholds);
    }
    config.intensity_levels = it->value("intensityLevels", config.intensity_levels);
    if (!it->contains("intensityLevels")) {
        config.intensity_levels = it->value("intensity_levels", config.intensity_levels);
    }
    return config;
}

std::string MessagePayloadToString(const ::net::WebSocketMessage& message) {
    std::string out;
    for (const auto& fragment : message.fragments) {
        out.append(fragment.view());
    }
    return out;
}

int Base64Value(char ch) {
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
    if (ch >= '0' && ch <= '9') return ch - '0' + 52;
    if (ch == '+') return 62;
    if (ch == '/') return 63;
    return -1;
}

core::Result<std::string> Base64Decode(std::string_view input) {
    std::string out;
    out.reserve(input.size() * 3 / 4);
    int value = 0;
    int bits = -8;
    bool padding = false;
    for (char ch : input) {
        if (ch == '\r' || ch == '\n' || ch == ' ' || ch == '\t') {
            continue;
        }
        if (ch == '=') {
            padding = true;
            continue;
        }
        if (padding) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "invalid base64 padding");
        }
        const int decoded = Base64Value(ch);
        if (decoded < 0) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "invalid base64 character");
        }
        value = (value << 6) | decoded;
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<char>((value >> bits) & 0xff));
            bits -= 8;
        }
    }
    return out;
}

::net::WebSocketFrame TextFrame(core::RawMemoryPool& pool, std::string_view text) {
    ::net::WebSocketFrame frame;
    frame.kind = ::net::WebSocketMessageKind::Text;
    auto copied = ::net::SharedBuffer::Copy(pool, text);
    if (copied.ok()) {
        frame.payload = std::move(copied).value();
    }
    return frame;
}

void SendWsError(const std::shared_ptr<::net::IWebSocketStreamRequest>& request,
                 std::string_view trace_id,
                 const core::Status& status) {
    Json out{{"type", "error"}, {"payload", ErrorEnvelope(std::string(trace_id), status)}};
    request->Send(TextFrame(request->memory_pool(), out.dump()));
}

std::filesystem::path UploadTempPath(std::string_view upload_id) {
    std::string file_name = "agent_document_upload_";
    for (char ch : upload_id) {
        const auto safe = std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_';
        file_name.push_back(safe ? ch : '_');
    }
    file_name += ".tmp";
    return std::filesystem::temp_directory_path() / file_name;
}

core::Status WriteUploadChunk(const std::filesystem::path& path,
                              std::string_view data,
                              std::uint64_t offset) {
    std::fstream file;
    if (offset == 0) {
        file.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
    } else {
        file.open(path, std::ios::binary | std::ios::in | std::ios::out);
    }
    if (!file) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to open upload temp file");
    }
    file.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!file) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to seek upload temp file");
    }
    file.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!file) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to write upload chunk");
    }
    return core::Status::Ok();
}

void RemoveFileQuietly(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

core::Status AppendBinaryUploadFrame(
    std::unordered_map<std::string, PersonaGatewayHttpAdapter::DocumentUploadSession>& uploads,
    std::uint64_t connection_id,
    const ::net::WebSocketMessage& message,
    std::string* upload_id,
    std::uint64_t* received_size,
    std::uint64_t* expected_size) {
    auto it = std::find_if(
        uploads.begin(),
        uploads.end(),
        [connection_id](const auto& entry) {
            return entry.second.binary_mode && entry.second.connection_id == connection_id;
        });
    if (it == uploads.end()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "no active binary upload for websocket connection");
    }

    auto& session = it->second;
    if (message.fragments.empty()) {
        return core::Status::Ok();
    }
    std::uint64_t frame_bytes = 0;
    for (const auto& fragment : message.fragments) {
        frame_bytes += fragment.size();
    }
    if (session.received_size + frame_bytes > session.expected_size) {
        const auto temp_path = session.temp_path;
        uploads.erase(it);
        RemoveFileQuietly(temp_path);
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "binary upload exceeds declared totalBytes");
    }

    auto offset = session.received_size;
    for (const auto& fragment : message.fragments) {
        auto status = WriteUploadChunk(session.temp_path, fragment.view(), offset);
        if (!status.ok()) {
            return status;
        }
        offset += fragment.size();
    }
    session.received_size += frame_bytes;
    if (upload_id) {
        *upload_id = session.upload_id;
    }
    if (received_size) {
        *received_size = session.received_size;
    }
    if (expected_size) {
        *expected_size = session.expected_size;
    }
    return core::Status::Ok();
}

struct HttpRouteContext {
    PersonaGatewayService& service;
    std::shared_ptr<document::DocumentAnalysisService> document_service;
    std::shared_ptr<llm::ILlmClient> llm_client;
    std::shared_ptr<document::IDocumentEmbeddingProvider> embedding_provider;
    std::shared_ptr<document::IDocumentLlmChunkCache> llm_chunk_cache;
    std::shared_ptr<semantic_cache::ISemanticCache> document_semantic_cache;
    std::shared_ptr<IAuthRegistrationService> auth_registration;
    std::shared_ptr<::net::IHttpRequest> request;
    const ::net::BeastHttpRequest& message;
    const Json& body;
    const AuthIdentity& identity;
    std::string trace_id;
    std::vector<std::string> path_parts;
    std::unordered_map<std::string, std::string> path_params;
};

struct WsRouteContext {
    PersonaGatewayService& service;
    std::shared_ptr<document::DocumentAnalysisService> document_service;
    std::shared_ptr<llm::ILlmClient> llm_client;
    std::shared_ptr<document::IDocumentEmbeddingProvider> embedding_provider;
    std::shared_ptr<document::IDocumentLlmChunkCache> llm_chunk_cache;
    std::shared_ptr<semantic_cache::ISemanticCache> document_semantic_cache;
    std::shared_ptr<::net::IWebSocketStreamRequest> request;
    const Json& body;
    const AuthIdentity& identity;
    std::string trace_id;
    std::mutex& upload_mutex;
    std::unordered_map<std::string, PersonaGatewayHttpAdapter::DocumentUploadSession>& document_uploads;
};

class IHttpRoute {
public:
    virtual ~IHttpRoute() = default;
    virtual ::net::http::verb Method() const noexcept = 0;
    virtual std::vector<std::string_view> Pattern() const = 0;
    virtual bool RequiresAuth() const noexcept { return true; }
    virtual void Handle(HttpRouteContext& context) const = 0;

    bool Matches(::net::http::verb method,
                 const std::vector<std::string>& parts,
                 std::unordered_map<std::string, std::string>& params) const {
        if (method != Method()) {
            return false;
        }
        const auto pattern = Pattern();
        if (pattern.size() != parts.size()) {
            return false;
        }
        params.clear();
        for (std::size_t i = 0; i < pattern.size(); ++i) {
            const auto token = pattern[i];
            if (token.size() >= 2 && token.front() == '{' && token.back() == '}') {
                params.emplace(std::string(token.substr(1, token.size() - 2)), parts[i]);
                continue;
            }
            if (token != parts[i]) {
                return false;
            }
        }
        return true;
    }
};

using HttpRouteFactory = std::unique_ptr<IHttpRoute> (*)();

class IWsRoute {
public:
    virtual ~IWsRoute() = default;
    virtual std::string_view Type() const noexcept = 0;
    virtual bool RequiresAuth() const noexcept { return true; }
    virtual void Handle(WsRouteContext& context) const = 0;

    bool Matches(std::string_view type) const noexcept {
        return Type() == type;
    }
};

using WsRouteFactory = std::unique_ptr<IWsRoute> (*)();

class HttpRouteRegistry {
public:
    static HttpRouteRegistry& Instance() {
        static HttpRouteRegistry registry;
        return registry;
    }

    bool Register(std::string_view name, HttpRouteFactory factory) {
        auto duplicate = std::find_if(
            entries_.begin(),
            entries_.end(),
            [name](const Entry& entry) {
                return entry.name == name;
            });
        if (duplicate == entries_.end()) {
            entries_.push_back({name, factory});
        }
        return true;
    }

    std::vector<std::unique_ptr<IHttpRoute>> CreateRoutes() const {
        std::vector<std::unique_ptr<IHttpRoute>> routes;
        routes.reserve(entries_.size());
        for (const auto& entry : entries_) {
            routes.push_back(entry.factory());
        }
        return routes;
    }

private:
    struct Entry {
        std::string_view name;
        HttpRouteFactory factory = nullptr;
    };

    std::vector<Entry> entries_;
};

class WsRouteRegistry {
public:
    static WsRouteRegistry& Instance() {
        static WsRouteRegistry registry;
        return registry;
    }

    bool Register(std::string_view name, WsRouteFactory factory) {
        auto duplicate = std::find_if(
            entries_.begin(),
            entries_.end(),
            [name](const Entry& entry) {
                return entry.name == name;
            });
        if (duplicate == entries_.end()) {
            entries_.push_back({name, factory});
        }
        return true;
    }

    std::vector<std::unique_ptr<IWsRoute>> CreateRoutes() const {
        std::vector<std::unique_ptr<IWsRoute>> routes;
        routes.reserve(entries_.size());
        for (const auto& entry : entries_) {
            routes.push_back(entry.factory());
        }
        return routes;
    }

private:
    struct Entry {
        std::string_view name;
        WsRouteFactory factory = nullptr;
    };

    std::vector<Entry> entries_;
};

template <typename T>
class HttpRouteRegistrar {
public:
    HttpRouteRegistrar() {
        HttpRouteRegistry::Instance().Register(
            T::kRouteName,
            []() -> std::unique_ptr<IHttpRoute> {
                return std::make_unique<T>();
            });
    }
};

template <typename T>
class WsRouteRegistrar {
public:
    WsRouteRegistrar() {
        WsRouteRegistry::Instance().Register(
            T::kRouteName,
            []() -> std::unique_ptr<IWsRoute> {
                return std::make_unique<T>();
            });
    }
};

#define DECLARE_HTTP_ROUTE(ClassName, MethodValue, ...) \
class ClassName final : public IHttpRoute { \
public: \
    static constexpr std::string_view kRouteName = #ClassName; \
    ::net::http::verb Method() const noexcept override { return MethodValue; } \
    std::vector<std::string_view> Pattern() const override { return {__VA_ARGS__}; } \
    void Handle(HttpRouteContext& context) const override; \
}; \
static const HttpRouteRegistrar<ClassName> g_##ClassName##_registrar; \
void ClassName::Handle(HttpRouteContext& context) const

#define DECLARE_WS_ROUTE(ClassName, TypeValue) \
class ClassName final : public IWsRoute { \
public: \
    static constexpr std::string_view kRouteName = #ClassName; \
    std::string_view Type() const noexcept override { return TypeValue; } \
    void Handle(WsRouteContext& context) const override; \
}; \
static const WsRouteRegistrar<ClassName> g_##ClassName##_registrar; \
void ClassName::Handle(WsRouteContext& context) const

DECLARE_HTTP_ROUTE(AuthMeRoute, ::net::http::verb::get, "api", "auth", "me") {
    Json body{
        {"ok", true},
        {"traceId", context.trace_id},
        {"data", {
            {"authenticated", context.identity.authenticated},
            {"userUuid", context.identity.user_uuid},
            {"tenantId", context.identity.tenant_id},
            {"subject", context.identity.subject},
        }},
    };
    SendJson(context.request, ::net::http::status::ok, body, context.trace_id);
}

class AuthRegisterRoute final : public IHttpRoute {
public:
    static constexpr std::string_view kRouteName = "AuthRegisterRoute";
    ::net::http::verb Method() const noexcept override { return ::net::http::verb::post; }
    std::vector<std::string_view> Pattern() const override { return {"api", "auth", "register"}; }
    bool RequiresAuth() const noexcept override { return false; }
    void Handle(HttpRouteContext& context) const override {
        if (!context.auth_registration) {
            const auto status = core::Status::Error(core::ErrorCode::FailedPrecondition, "auth registration service is not configured");
            SendJson(context.request, HttpStatusFor(status.code()), ErrorEnvelope(context.trace_id, status), context.trace_id);
            return;
        }

        AuthRegistrationRequest req;
        req.user_uuid = context.body.value("userUuid", context.body.value("user_uuid", std::string{}));
        req.tenant_id = context.body.value("tenantId", context.body.value("tenant_id", std::string{"default"}));
        req.subject = context.body.value("subject", std::string{});
        const auto ttl_seconds = context.body.value("ttlSeconds", context.body.value("ttl_seconds", 0));
        if (ttl_seconds > 0) {
            req.ttl = std::chrono::seconds(ttl_seconds);
        }

        auto result = context.auth_registration->Register(req);
        if (!result.ok()) {
            SendJson(context.request, HttpStatusFor(result.status().code()), ErrorEnvelope(context.trace_id, result.status()), context.trace_id);
            return;
        }

        const auto& value = result.value();
        Json body{
            {"ok", true},
            {"traceId", context.trace_id},
            {"data", {
                {"authenticated", value.identity.authenticated},
                {"userUuid", value.identity.user_uuid},
                {"tenantId", value.identity.tenant_id},
                {"subject", value.identity.subject},
                {"tokenId", value.identity.token_id},
                {"issuedAt", ToUnixSeconds(value.issued_at)},
                {"expiresAt", ToUnixSeconds(value.identity.expires_at)},
                {"token", value.token},
            }},
        };
        SendJsonWithHeaders(
            context.request,
            ::net::http::status::ok,
            body,
            context.trace_id,
            {{"Set-Cookie", value.cookie_header}});
    }
};
static const HttpRouteRegistrar<AuthRegisterRoute> g_AuthRegisterRoute_registrar;

DECLARE_HTTP_ROUTE(CreateSessionRoute, ::net::http::verb::post, "api", "session", "create") {
    CreateSessionGatewayRequest req;
    req.trace_id = context.trace_id;
    req.session_id = context.body.value("sessionId", std::string{});
    req.user_uuid = context.identity.user_uuid;
    req.tenant_id = context.identity.tenant_id;
    req.classroom_id = context.body.value("classroomId", std::string{});
    req.persona_id = context.body.value("personaId", std::string{});
    req.context_ids = context.body.value("contextIds", std::vector<std::string>{});
    req.context_patterns = context.body.value("contextPatterns", std::vector<std::string>{});
    req.proactive_level = context.body.value("proactiveLevel", std::string{"off"});
    req.default_persona = context.body.value("defaultPersona", false);
    req.personality = PersonalityFromJson(context.body);
    req.emotion_prompt_config = EmotionPromptConfigFromJson(context.body);
    SendResult(context.request, context.service.CreateSession(std::move(req)), context.trace_id, SessionEnvelope);
}

DECLARE_HTTP_ROUTE(CloseSessionRoute, ::net::http::verb::post, "api", "session", "close") {
    CloseSessionGatewayRequest req;
    req.trace_id = context.trace_id;
    req.session_id = context.body.value("sessionId", std::string{});
    req.reason = context.body.value("reason", std::string{"client_close"});
    SendResult(context.request, context.service.CloseSession(std::move(req)), context.trace_id, SessionEnvelope);
}

DECLARE_HTTP_ROUTE(GetSessionRoute, ::net::http::verb::get, "api", "session", "{sessionId}") {
    SendResult(
        context.request,
        context.service.GetSession(context.path_params.at("sessionId"), context.trace_id),
        context.trace_id,
        SessionEnvelope);
}

DECLARE_HTTP_ROUTE(GetSessionEmotionRoute, ::net::http::verb::get, "api", "session", "{sessionId}", "emotion") {
    SendResult(
        context.request,
        context.service.GetSession(context.path_params.at("sessionId"), context.trace_id),
        context.trace_id,
        [](const SessionGatewayResponse& r) {
            return Json{
                {"ok", true},
                {"traceId", r.trace_id},
                {"sessionId", r.session_id},
                {"latencyMs", r.latency.count()},
                {"data", SessionToJson(r.session)["emotion"]},
            };
        });
}

DECLARE_HTTP_ROUTE(GetSessionMetricsRoute, ::net::http::verb::get, "api", "session", "{sessionId}", "metrics") {
    SendResult(
        context.request,
        context.service.GetSession(context.path_params.at("sessionId"), context.trace_id),
        context.trace_id,
        [](const SessionGatewayResponse& r) {
            return Json{
                {"ok", true},
                {"traceId", r.trace_id},
                {"sessionId", r.session_id},
                {"latencyMs", r.latency.count()},
                {"data", SessionToJson(r.session)["metrics"]},
            };
        });
}

DECLARE_HTTP_ROUTE(ChatMessageRoute, ::net::http::verb::post, "api", "chat", "message") {
    ChatGatewayRequest req;
    req.trace_id = context.trace_id;
    req.session_id = context.body.value("sessionId", std::string{});
    req.persona_id = context.body.value("personaId", std::string{});
    req.mode = context.body.value("mode", std::string{"chat"});
    req.message = context.body.value("message", std::string{});
    req.model = context.body.value("model", std::string{});
    req.stream = context.body.value("stream", false);
    auto trace_id = context.trace_id;
    auto request = context.request;
    auto status = context.service.SubmitChat(
        std::move(req),
        [request = std::move(request), trace_id](core::Result<ChatGatewayResponse> result) mutable {
            SendResult(request, std::move(result), trace_id, ChatEnvelope);
        });
    if (!status.ok()) {
        SendJson(context.request, HttpStatusFor(status.code()), ErrorEnvelope(trace_id, status), trace_id);
    }
}

DECLARE_HTTP_ROUTE(ClassroomMessageRoute, ::net::http::verb::post, "api", "classroom", "message") {
    ClassroomMessageGatewayRequest req;
    req.trace_id = context.trace_id;
    req.session_id = context.body.value("sessionId", std::string{});
    req.classroom_id = context.body.value("classroomId", std::string{});
    req.target_persona_id = context.body.value("targetPersonaId", context.body.value("personaId", std::string{}));
    req.context_id = context.body.value("contextId", std::string{});
    req.message = context.body.value("message", std::string{});
    req.broadcast = context.body.value("broadcast", false);
    req.model = context.body.value("model", std::string{});
    auto trace_id = context.trace_id;
    auto request = context.request;
    auto status = context.service.SubmitClassroomMessage(
        std::move(req),
        [request = std::move(request), trace_id](core::Result<ClassroomGatewayResponse> result) mutable {
            SendResult(request, std::move(result), trace_id, ClassroomEnvelope);
        });
    if (!status.ok()) {
        SendJson(context.request, HttpStatusFor(status.code()), ErrorEnvelope(trace_id, status), trace_id);
    }
}

DECLARE_HTTP_ROUTE(ClassroomProactiveRoute, ::net::http::verb::post, "api", "classroom", "proactive") {
    ClassroomProactiveGatewayRequest req;
    req.trace_id = context.trace_id;
    req.session_id = context.body.value("sessionId", std::string{});
    req.classroom_id = context.body.value("classroomId", std::string{});
    req.persona_id = context.body.value("personaId", std::string{});
    req.context_id = context.body.value("contextId", std::string{});
    req.model = context.body.value("model", std::string{});
    auto trace_id = context.trace_id;
    auto request = context.request;
    auto status = context.service.SubmitClassroomProactive(
        std::move(req),
        [request = std::move(request), trace_id](core::Result<ClassroomGatewayResponse> result) mutable {
            SendResult(request, std::move(result), trace_id, ClassroomEnvelope);
        });
    if (!status.ok()) {
        SendJson(context.request, HttpStatusFor(status.code()), ErrorEnvelope(trace_id, status), trace_id);
    }
}

DECLARE_HTTP_ROUTE(ClassroomPollRoute, ::net::http::verb::post, "api", "classroom", "poll") {
    ClassroomPollGatewayRequest req;
    req.trace_id = context.trace_id;
    req.classroom_id = context.body.value("classroomId", std::string{});
    req.persona_id = context.body.value("personaId", std::string{});
    req.context_id = context.body.value("contextId", std::string{});
    req.system_event = context.body.value("systemEvent", false);
    req.system_event_content = context.body.value("systemEventContent", std::string{});
    req.model = context.body.value("model", std::string{});
    auto trace_id = context.trace_id;
    auto request = context.request;
    auto status = context.service.SubmitClassroomPoll(
        std::move(req),
        [request = std::move(request), trace_id](core::Result<ClassroomGatewayResponse> result) mutable {
            SendResult(request, std::move(result), trace_id, ClassroomEnvelope);
        });
    if (!status.ok()) {
        SendJson(context.request, HttpStatusFor(status.code()), ErrorEnvelope(trace_id, status), trace_id);
    }
}

DECLARE_HTTP_ROUTE(TrainingReportRoute, ::net::http::verb::post, "api", "report", "training") {
    TrainingReportGatewayRequest req;
    req.trace_id = context.trace_id;
    req.session_id = context.body.value("sessionId", std::string{});
    req.include_raw_turns = context.body.value("includeRawTurns", true);
    SendResult(context.request, context.service.TrainingReport(std::move(req)), context.trace_id, ReportEnvelope);
}

DECLARE_HTTP_ROUTE(SystemStatsRoute, ::net::http::verb::get, "api", "system", "stats") {
    SendResult(context.request, context.service.SystemStats(context.trace_id), context.trace_id, SystemStatsEnvelope);
}

DECLARE_HTTP_ROUTE(DocumentRegisterRoute, ::net::http::verb::post, "api", "document", "register") {
    if (!context.document_service) {
        const auto status = core::Status::Error(core::ErrorCode::FailedPrecondition, "document analysis service is not configured");
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

DECLARE_HTTP_ROUTE(DocumentAnalyzeRoute, ::net::http::verb::post, "api", "document", "analyze") {
    if (!context.document_service) {
        const auto status = core::Status::Error(core::ErrorCode::FailedPrecondition, "document analysis service is not configured");
        SendJson(context.request, HttpStatusFor(status.code()), ErrorEnvelope(context.trace_id, status), context.trace_id);
        return;
    }

    document::DocumentAnalyzeRequest req;
    req.trace_id = context.trace_id;
    req.path = PathFromUtf8(context.body.value("path", std::string{}));
    req.file_name = context.body.value("fileName", context.body.value("file_name", std::string{}));
    req.document_id = context.body.value("documentId", context.body.value("document_id", std::string{}));
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

class HealthRoute final : public IHttpRoute {
public:
    static constexpr std::string_view kRouteName = "HealthRoute";
    ::net::http::verb Method() const noexcept override { return ::net::http::verb::get; }
    std::vector<std::string_view> Pattern() const override { return {"api", "health"}; }
    bool RequiresAuth() const noexcept override { return false; }

    void Handle(HttpRouteContext& context) const override {
        auto stats = context.service.SystemStats(context.trace_id);
        if (!stats.ok()) {
            SendJson(context.request, HttpStatusFor(stats.status().code()), ErrorEnvelope(context.trace_id, stats.status()), context.trace_id);
            return;
        }

        Json body{
            {"ok", true},
            {"traceId", context.trace_id},
            {"latencyMs", stats.value().latency.count()},
            {"data", {
                {"status", "ok"},
                {"sessionCount", stats.value().session_count},
                {"pools", {
                    {"compute", ThreadPoolStatsToJson(stats.value().pools.compute)},
                    {"io", ThreadPoolStatsToJson(stats.value().pools.io)},
                }},
            }},
        };
        SendJson(context.request, ::net::http::status::ok, body, context.trace_id);
    }
};
static const HttpRouteRegistrar<HealthRoute> g_HealthRoute_registrar;

DECLARE_WS_ROUTE(ChatMessageWsRoute, "chat.message") {
    const auto payload = context.body.value("payload", Json::object());
    ChatGatewayRequest req;
    req.trace_id = context.trace_id;
    req.session_id = payload.value("sessionId", std::string{});
    req.persona_id = payload.value("personaId", std::string{});
    req.mode = payload.value("mode", std::string{"ws_chat"});
    req.message = payload.value("message", std::string{});
    req.model = payload.value("model", std::string{});
    auto ws_request = context.request;
    auto status = context.service.SubmitChat(
        std::move(req),
        [request = std::move(ws_request), trace_id = context.trace_id](core::Result<ChatGatewayResponse> result) mutable {
            Json out = result.ok()
                ? Json{{"type", "chat.final"}, {"payload", ChatEnvelope(result.value())}}
                : Json{{"type", "error"}, {"payload", ErrorEnvelope(trace_id, result.status())}};
            request->Send(TextFrame(request->memory_pool(), out.dump()));
        });
    if (!status.ok()) {
        Json out{{"type", "error"}, {"payload", ErrorEnvelope(context.trace_id, status)}};
        context.request->Send(TextFrame(context.request->memory_pool(), out.dump()));
    }
}

DECLARE_WS_ROUTE(SessionCloseWsRoute, "session.close") {
    const auto payload = context.body.value("payload", Json::object());
    CloseSessionGatewayRequest req;
    req.trace_id = context.trace_id;
    req.session_id = payload.value("sessionId", std::string{});
    req.reason = payload.value("reason", std::string{"client_close"});
    auto result = context.service.CloseSession(std::move(req));
    Json out = result.ok()
        ? Json{{"type", "session.closed"}, {"payload", SessionEnvelope(result.value())}}
        : Json{{"type", "error"}, {"payload", ErrorEnvelope(context.trace_id, result.status())}};
    context.request->Send(TextFrame(context.request->memory_pool(), out.dump()));
}

DECLARE_WS_ROUTE(DocumentUploadStartWsRoute, "document.upload.start") {
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

    PersonaGatewayHttpAdapter::DocumentUploadSession session;
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

DECLARE_WS_ROUTE(DocumentUploadChunkWsRoute, "document.upload.chunk") {
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

    PersonaGatewayHttpAdapter::DocumentUploadSession session;
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

DECLARE_WS_ROUTE(DocumentUploadFinishWsRoute, "document.upload.finish") {
    if (!context.document_service) {
        SendWsError(
            context.request,
            context.trace_id,
            core::Status::Error(core::ErrorCode::FailedPrecondition, "document analysis service is not configured"));
        return;
    }
    const auto payload = context.body.value("payload", Json::object());
    const auto upload_id = payload.value("uploadId", payload.value("upload_id", std::string{}));
    PersonaGatewayHttpAdapter::DocumentUploadSession session;
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

DECLARE_WS_ROUTE(DocumentUploadAbortWsRoute, "document.upload.abort") {
    const auto payload = context.body.value("payload", Json::object());
    const auto upload_id = payload.value("uploadId", payload.value("upload_id", std::string{}));
    std::filesystem::path temp_path;
    {
        std::lock_guard lock(context.upload_mutex);
        auto it = context.document_uploads.find(upload_id);
        if (it != context.document_uploads.end()) {
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

PersonaGatewayHttpAdapter::PersonaGatewayHttpAdapter(PersonaGatewayService& service,
                                                     std::shared_ptr<IGatewayAuthenticator> authenticator,
                                                     std::shared_ptr<IAuthRegistrationService> auth_registration,
                                                     std::shared_ptr<document::DocumentAnalysisService> document_service,
                                                     std::shared_ptr<llm::ILlmClient> llm_client,
                                                     std::shared_ptr<document::IDocumentEmbeddingProvider> embedding_provider,
                                                     std::shared_ptr<document::IDocumentLlmChunkCache> llm_chunk_cache,
                                                     std::shared_ptr<semantic_cache::ISemanticCache> document_semantic_cache)
    : service_(service),
      authenticator_(std::move(authenticator)),
      auth_registration_(std::move(auth_registration)),
      document_service_(std::move(document_service)),
      llm_client_(std::move(llm_client)),
      embedding_provider_(std::move(embedding_provider)),
      llm_chunk_cache_(std::move(llm_chunk_cache)),
      document_semantic_cache_(std::move(document_semantic_cache)) {}

bool PersonaGatewayHttpAdapter::IsApiRequest(std::string_view target) noexcept {
    const auto q = target.find('?');
    if (q != std::string_view::npos) {
        target = target.substr(0, q);
    }
    return target == "/api" || target.starts_with("/api/");
}

void PersonaGatewayHttpAdapter::HandleHttp(std::shared_ptr<::net::IHttpRequest> request) {
    const auto& msg = request->message();
    if (!IsApiRequest(msg.target())) {
        const auto trace_id = TraceFrom(msg);
        const auto status = core::Status::Error(core::ErrorCode::NotFound, "route not found");
        SendJson(request, ::net::http::status::not_found, ErrorEnvelope(trace_id, status), trace_id);
        return;
    }

    const auto parts = SplitPath(msg.target());
    auto parsed = ParseJsonBody(msg);
    const Json* parsed_body = parsed.ok() ? &parsed.value() : nullptr;
    const auto trace_id = TraceFrom(msg, parsed_body);
    if (!parsed.ok()) {
        SendJson(request, ::net::http::status::bad_request, ErrorEnvelope(trace_id, parsed.status()), trace_id);
        return;
    }
    const auto& body = parsed.value();
    auto routes = HttpRouteRegistry::Instance().CreateRoutes();
    std::unordered_map<std::string, std::string> params;
    for (const auto& route : routes) {
        if (!route->Matches(msg.method(), parts, params)) {
            continue;
        }
        AuthIdentity identity;
        if (authenticator_ && route->RequiresAuth()) {
            auto auth = authenticator_->Authenticate(msg);
            if (!auth.ok()) {
                SendJson(request, HttpStatusFor(auth.status().code()), ErrorEnvelope(trace_id, auth.status()), trace_id);
                return;
            }
            identity = std::move(auth).value();
        }
        HttpRouteContext context{
            service_,
            document_service_,
            llm_client_,
            embedding_provider_,
            llm_chunk_cache_,
            document_semantic_cache_,
            auth_registration_,
            std::move(request),
            msg,
            body,
            identity,
            trace_id,
            parts,
            std::move(params),
        };
        route->Handle(context);
        return;
    }

    const auto status = core::Status::Error(core::ErrorCode::NotFound, "route not found");
    SendJson(request, ::net::http::status::not_found, ErrorEnvelope(trace_id, status), trace_id);
}

void PersonaGatewayHttpAdapter::HandleWebSocket(std::shared_ptr<::net::IWebSocketStreamRequest> request) {
    if (!request->message().ok()) {
        auto error = ErrorEnvelope(core::GenerateTraceId(), request->message().status);
        request->Send(TextFrame(request->memory_pool(), error.dump()));
        return;
    }

    if (request->message().kind == ::net::WebSocketMessageKind::Binary) {
        std::string upload_id;
        std::uint64_t received = 0;
        std::uint64_t total = 0;
        core::Status status;
        {
            std::lock_guard lock(document_upload_mutex_);
            status = AppendBinaryUploadFrame(
                document_uploads_,
                request->connection().connection_id,
                request->message(),
                &upload_id,
                &received,
                &total);
        }
        if (!status.ok()) {
            SendWsError(request, core::GenerateTraceId(), status);
            return;
        }
        Json out{
            {"type", "document.upload.chunk_ack"},
            {"payload", {
                {"ok", true},
                {"traceId", core::GenerateTraceId()},
                {"uploadId", upload_id},
                {"mode", "binary"},
                {"receivedBytes", received},
                {"totalBytes", total},
                {"finalFragment", request->message().final_fragment},
            }},
        };
        request->Send(TextFrame(request->memory_pool(), out.dump()));
        return;
    }

    const auto text = MessagePayloadToString(request->message());
    Json body;
    try {
        body = Json::parse(text);
    } catch (const Json::exception& e) {
        auto error = ErrorEnvelope(core::GenerateTraceId(), core::Status::Error(core::ErrorCode::InvalidArgument, e.what()));
        request->Send(TextFrame(request->memory_pool(), error.dump()));
        return;
    }

    const auto type = body.value("type", std::string{});
    const auto trace_id = body.value("traceId", core::GenerateTraceId());
    auto routes = WsRouteRegistry::Instance().CreateRoutes();
    for (const auto& route : routes) {
        if (!route->Matches(type)) {
            continue;
        }
        AuthIdentity identity;
        if (authenticator_ && route->RequiresAuth()) {
            auto auth = authenticator_->Authenticate(request->handshake_request());
            if (!auth.ok()) {
                Json out{{"type", "error"}, {"payload", ErrorEnvelope(trace_id, auth.status())}};
                request->Send(TextFrame(request->memory_pool(), out.dump()));
                return;
            }
            identity = std::move(auth).value();
        }
        WsRouteContext context{
            service_,
            document_service_,
            llm_client_,
            embedding_provider_,
            llm_chunk_cache_,
            document_semantic_cache_,
            std::move(request),
            body,
            identity,
            trace_id,
            document_upload_mutex_,
            document_uploads_,
        };
        route->Handle(context);
        return;
    }

    auto error = ErrorEnvelope(trace_id, core::Status::Error(core::ErrorCode::InvalidArgument, "unknown websocket message type"));
    request->Send(TextFrame(request->memory_pool(), Json{{"type", "error"}, {"payload", error}}.dump()));
}

} // namespace agent::service::gateway
