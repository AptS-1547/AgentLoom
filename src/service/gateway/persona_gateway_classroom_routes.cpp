#include "persona_gateway_route_helpers.h"

namespace agent::service::gateway {
namespace {

using namespace route_detail;

DECLARE_AUTHENTICATED_HTTP_ROUTE(ClassroomMessageRoute, ::net::http::verb::post, "api", "classroom", "message") {
    ClassroomMessageGatewayRequest req;
    req.trace_id = context.trace_id;
    req.session_id = context.body.value("sessionId", std::string{});
    req.authenticated_user_uuid = context.identity.user_uuid;
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

DECLARE_AUTHENTICATED_HTTP_ROUTE(ClassroomProactiveRoute, ::net::http::verb::post, "api", "classroom", "proactive") {
    ClassroomProactiveGatewayRequest req;
    req.trace_id = context.trace_id;
    req.session_id = context.body.value("sessionId", std::string{});
    req.authenticated_user_uuid = context.identity.user_uuid;
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

DECLARE_AUTHENTICATED_HTTP_ROUTE(ClassroomPollRoute, ::net::http::verb::post, "api", "classroom", "poll") {
    ClassroomPollGatewayRequest req;
    req.trace_id = context.trace_id;
    req.authenticated_user_uuid = context.identity.user_uuid;
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

} // namespace
} // namespace agent::service::gateway
