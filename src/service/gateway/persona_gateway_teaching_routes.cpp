#include "persona_gateway_route_helpers.h"

namespace agent::service::gateway {
namespace {

using namespace route_detail;

DECLARE_AUTHENTICATED_HTTP_ROUTE(TrainingReportRoute, ::net::http::verb::post, "api", "report", "training") {
    TrainingReportGatewayRequest req;
    req.trace_id = context.trace_id;
    req.session_id = context.body.value("sessionId", std::string{});
    req.authenticated_user_uuid = context.identity.user_uuid;
    req.include_raw_turns = context.body.value("includeRawTurns", true);
    SendResult(context.request, context.service.TrainingReport(std::move(req)), context.trace_id, ReportEnvelope);
}

} // namespace
} // namespace agent::service::gateway
