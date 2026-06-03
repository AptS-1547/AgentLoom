#include "persona_gateway_http_adapter.h"

#include <nlohmann/json.hpp>

#include <algorithm>
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
                {"memoryContextMs", response.pipeline_latency.memory_context.count()},
                {"answerCacheMs", response.pipeline_latency.answer_cache.count()},
                {"promptBuildMs", response.pipeline_latency.prompt_build.count()},
                {"llmTotalMs", response.pipeline_latency.llm_total.count()},
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
    auto response = ::net::HttpResponse::Json(status, body.dump()).message;
    response.set("X-Trace-Id", trace_id);
    request->Respond(std::move(response));
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

std::string MessagePayloadToString(const ::net::WebSocketMessage& message) {
    std::string out;
    for (const auto& fragment : message.fragments) {
        out.append(fragment.view());
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

struct HttpRouteContext {
    PersonaGatewayService& service;
    std::shared_ptr<::net::IHttpRequest> request;
    const ::net::BeastHttpRequest& message;
    const Json& body;
    std::string trace_id;
    std::vector<std::string> path_parts;
    std::unordered_map<std::string, std::string> path_params;
};

class IHttpRoute {
public:
    virtual ~IHttpRoute() = default;
    virtual ::net::http::verb Method() const noexcept = 0;
    virtual std::vector<std::string_view> Pattern() const = 0;
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

DECLARE_HTTP_ROUTE(CreateSessionRoute, ::net::http::verb::post, "api", "session", "create") {
    CreateSessionGatewayRequest req;
    req.trace_id = context.trace_id;
    req.session_id = context.body.value("sessionId", std::string{});
    req.user_uuid = context.body.value("userUuid", context.body.value("user_uuid", std::string{"local_user"}));
    req.tenant_id = context.body.value("tenantId", std::string{"default"});
    req.classroom_id = context.body.value("classroomId", std::string{});
    req.persona_id = context.body.value("personaId", std::string{});
    req.context_ids = context.body.value("contextIds", std::vector<std::string>{});
    req.context_patterns = context.body.value("contextPatterns", std::vector<std::string>{});
    req.proactive_level = context.body.value("proactiveLevel", std::string{"off"});
    req.default_persona = context.body.value("defaultPersona", false);
    req.personality = PersonalityFromJson(context.body);
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

} // namespace

PersonaGatewayHttpAdapter::PersonaGatewayHttpAdapter(PersonaGatewayService& service)
    : service_(service) {}

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
        HttpRouteContext context{
            service_,
            std::move(request),
            msg,
            body,
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
    if (type == "chat.message") {
        const auto payload = body.value("payload", Json::object());
        ChatGatewayRequest req;
        req.trace_id = trace_id;
        req.session_id = payload.value("sessionId", std::string{});
        req.persona_id = payload.value("personaId", std::string{});
        req.mode = payload.value("mode", std::string{"ws_chat"});
        req.message = payload.value("message", std::string{});
        req.model = payload.value("model", std::string{});
        auto ws_request = request;
        auto status = service_.SubmitChat(
            std::move(req),
            [request = std::move(ws_request), trace_id](core::Result<ChatGatewayResponse> result) mutable {
                Json out = result.ok()
                    ? Json{{"type", "chat.final"}, {"payload", ChatEnvelope(result.value())}}
                    : Json{{"type", "error"}, {"payload", ErrorEnvelope(trace_id, result.status())}};
                request->Send(TextFrame(request->memory_pool(), out.dump()));
            });
        if (!status.ok()) {
            Json out{{"type", "error"}, {"payload", ErrorEnvelope(trace_id, status)}};
            request->Send(TextFrame(request->memory_pool(), out.dump()));
        }
        return;
    }

    if (type == "session.close") {
        const auto payload = body.value("payload", Json::object());
        CloseSessionGatewayRequest req;
        req.trace_id = trace_id;
        req.session_id = payload.value("sessionId", std::string{});
        req.reason = payload.value("reason", std::string{"client_close"});
        auto result = service_.CloseSession(std::move(req));
        Json out = result.ok()
            ? Json{{"type", "session.closed"}, {"payload", SessionEnvelope(result.value())}}
            : Json{{"type", "error"}, {"payload", ErrorEnvelope(trace_id, result.status())}};
        request->Send(TextFrame(request->memory_pool(), out.dump()));
        return;
    }

    auto error = ErrorEnvelope(trace_id, core::Status::Error(core::ErrorCode::InvalidArgument, "unknown websocket message type"));
    request->Send(TextFrame(request->memory_pool(), Json{{"type", "error"}, {"payload", error}}.dump()));
}

} // namespace agent::service::gateway
