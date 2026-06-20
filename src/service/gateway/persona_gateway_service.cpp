#include "persona_gateway_service.h"

#include "trace_context.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <utility>

namespace agent::service::gateway {
namespace {

std::string NowIso8601Utc() {
    const auto now = std::chrono::system_clock::now();
    const auto secs = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &secs);
#else
    gmtime_r(&secs, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::chrono::milliseconds Since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
}

std::string DefaultPersonaName(std::string_view persona_id) {
    if (!persona_id.empty()) {
        return std::string(persona_id);
    }
    return "default_persona";
}

} // namespace

PersonaGatewayService::PersonaGatewayService(persona::SessionManager& sessions,
                                             persona::PersonaRuntime& runtime,
                                             IClassroomScheduler* classroom_scheduler,
                                             std::shared_ptr<evaluation::TeachingEvaluator> evaluator,
                                             std::shared_ptr<semantic_cache::RedisConnectionPool> l0_redis_pool,
                                             std::filesystem::path evaluation_config_path,
                                             core::LoggerAdapter logger)
    : sessions_(sessions),
      runtime_(runtime),
      classroom_scheduler_(classroom_scheduler),
      evaluator_(std::move(evaluator)),
      l0_redis_pool_(std::move(l0_redis_pool)),
      evaluation_config_path_(std::move(evaluation_config_path)),
      logger_(std::move(logger)) {}

core::Result<SessionGatewayResponse> PersonaGatewayService::CreateSession(CreateSessionGatewayRequest request) {
    const auto started = std::chrono::steady_clock::now();
    request.trace_id = EnsureTrace(std::move(request.trace_id));
    if (request.persona_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "persona_id is required");
    }
    if (request.personality.name.empty()) {
        request.personality.name = DefaultPersonaName(request.persona_id);
    }

    persona::CreateSessionRequest create;
    create.user_uuid = std::move(request.user_uuid);
    create.persona_id = std::move(request.persona_id);
    create.session_id = std::move(request.session_id);
    create.trace_id = request.trace_id;
    create.personality = std::move(request.personality);
    create.emotion_prompt_config = std::move(request.emotion_prompt_config);
    create.emotion_state_config = request.emotion_state_config;
    create.time_awareness = true;

    auto snapshot = sessions_.CreateSession(std::move(create));
    if (!snapshot.ok()) {
        return snapshot.status();
    }
    if (classroom_scheduler_ && !request.classroom_id.empty()) {
        auto registration = BuildClassroomRegistration(request, snapshot.value());
        auto registered_status = classroom_scheduler_->RegisterPersona(std::move(registration));
        if (!registered_status.ok()) {
            sessions_.CloseSession(snapshot.value().session_id, request.trace_id);
            return registered_status;
        }
    }

    SessionGatewayResponse response;
    response.trace_id = request.trace_id;
    response.session_id = snapshot.value().session_id;
    response.latency = Since(started);
    response.session = std::move(snapshot).value();
    return response;
}

core::Result<SessionGatewayResponse> PersonaGatewayService::GetSession(std::string_view session_id,
                                                                       std::string trace_id) {
    const auto started = std::chrono::steady_clock::now();
    trace_id = EnsureTrace(std::move(trace_id));
    auto snapshot = sessions_.GetSessionSnapshot(session_id);
    if (!snapshot.ok()) {
        return snapshot.status();
    }
    SessionGatewayResponse response;
    response.trace_id = trace_id;
    response.session_id = snapshot.value().session_id;
    response.latency = Since(started);
    response.session = std::move(snapshot).value();
    return response;
}

core::Result<SessionGatewayResponse> PersonaGatewayService::CloseSession(CloseSessionGatewayRequest request) {
    const auto started = std::chrono::steady_clock::now();
    request.trace_id = EnsureTrace(std::move(request.trace_id));
    auto before = sessions_.GetSessionSnapshot(request.session_id);
    if (!before.ok()) {
        return before.status();
    }
    auto close = sessions_.CloseSession(request.session_id, request.trace_id);
    if (!close.ok()) {
        return close;
    }
    auto snapshot = std::move(before).value();
    if (classroom_scheduler_) {
        auto unregister_status = classroom_scheduler_->UnregisterSession(request.session_id);
        if (!unregister_status.ok() && unregister_status.code() != core::ErrorCode::NotFound) {
            logger_.warn("[trace={}] [gateway] classroom unregister failed session={} error={}",
                         request.trace_id,
                         request.session_id,
                         unregister_status.message());
        }
    }
    snapshot.status = persona::SessionStatus::Closed;
    snapshot.close_reason = request.reason.empty() ? "client_close" : request.reason;

    SessionGatewayResponse response;
    response.trace_id = request.trace_id;
    response.session_id = snapshot.session_id;
    response.latency = Since(started);
    response.session = std::move(snapshot);
    return response;
}

core::Result<ChatGatewayResponse> PersonaGatewayService::Chat(ChatGatewayRequest request) {
    std::promise<core::Result<ChatGatewayResponse>> promise;
    auto future = promise.get_future();
    auto status = SubmitChat(
        std::move(request),
        [&promise](core::Result<ChatGatewayResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    if (!status.ok()) {
        return status;
    }
    if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        return core::Status::Error(core::ErrorCode::Timeout, "persona gateway callback timed out");
    }
    return future.get();
}

core::Result<ClassroomGatewayResponse> PersonaGatewayService::ClassroomMessage(
    ClassroomMessageGatewayRequest request) {
    std::promise<core::Result<ClassroomGatewayResponse>> promise;
    auto future = promise.get_future();
    auto status = SubmitClassroomMessage(
        std::move(request),
        [&promise](core::Result<ClassroomGatewayResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    if (!status.ok()) {
        return status;
    }
    if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        return core::Status::Error(core::ErrorCode::Timeout, "classroom message callback timed out");
    }
    return future.get();
}

core::Result<ClassroomGatewayResponse> PersonaGatewayService::ClassroomProactive(
    ClassroomProactiveGatewayRequest request) {
    std::promise<core::Result<ClassroomGatewayResponse>> promise;
    auto future = promise.get_future();
    auto status = SubmitClassroomProactive(
        std::move(request),
        [&promise](core::Result<ClassroomGatewayResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    if (!status.ok()) {
        return status;
    }
    if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        return core::Status::Error(core::ErrorCode::Timeout, "classroom proactive callback timed out");
    }
    return future.get();
}

core::Result<ClassroomGatewayResponse> PersonaGatewayService::ClassroomPoll(ClassroomPollGatewayRequest request) {
    std::promise<core::Result<ClassroomGatewayResponse>> promise;
    auto future = promise.get_future();
    auto status = SubmitClassroomPoll(
        std::move(request),
        [&promise](core::Result<ClassroomGatewayResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    if (!status.ok()) {
        return status;
    }
    if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        return core::Status::Error(core::ErrorCode::Timeout, "classroom poll callback timed out");
    }
    return future.get();
}

core::Result<TrainingReportGatewayResponse> PersonaGatewayService::TrainingReport(
    TrainingReportGatewayRequest request) {
    const auto started = std::chrono::steady_clock::now();
    request.trace_id = EnsureTrace(std::move(request.trace_id));
    auto snapshot = sessions_.GetSessionSnapshot(request.session_id);
    if (!snapshot.ok()) {
        return snapshot.status();
    }

    TrainingReportGatewayResponse response;
    response.trace_id = request.trace_id;
    response.session_id = snapshot.value().session_id;
    response.latency = Since(started);
    response.generated_at = NowIso8601Utc();
    response.total_turns = snapshot.value().metrics.turn_count;
    response.metrics = snapshot.value().metrics;
    response.summary = "Training report evaluation pipeline is pending; session metrics are available.";
    if (evaluator_ && l0_redis_pool_ && !evaluation_config_path_.empty()) {
        evaluation::TeachingEvaluationRequest eval_req;
        eval_req.user_uuid = snapshot.value().user_uuid;
        eval_req.session_id = snapshot.value().session_id;
        eval_req.trace_id = request.trace_id;
        eval_req.config_path = evaluation_config_path_;
        eval_req.redis_pool = l0_redis_pool_;
        auto evaluated = evaluator_->Evaluate(eval_req);
        if (evaluated.ok()) {
            response.evaluation = std::move(evaluated).value();
            response.summary = "Training report evaluation completed.";
        } else {
            response.evaluation = {
                {"error", evaluated.status().message()},
            };
            response.summary = "Training report evaluation unavailable; session metrics are available.";
        }
    }
    return response;
}

core::Result<SystemStatsGatewayResponse> PersonaGatewayService::SystemStats(std::string trace_id) {
    const auto started = std::chrono::steady_clock::now();
    SystemStatsGatewayResponse response;
    response.trace_id = EnsureTrace(std::move(trace_id));
    response.latency = Since(started);
    response.session_count = sessions_.SessionCount();
    response.pools = sessions_.PoolStats();
    return response;
}

core::Status PersonaGatewayService::SubmitChat(ChatGatewayRequest request, ChatCallback callback) {
    const auto started = std::chrono::steady_clock::now();
    if (!callback) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "chat callback is required");
    }
    request.trace_id = EnsureTrace(std::move(request.trace_id));
    if (request.session_id.empty() || request.message.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "session_id and message are required");
    }
    auto before = sessions_.GetSessionSnapshot(request.session_id);
    if (!before.ok()) {
        return before.status();
    }
    if (before.value().status != persona::SessionStatus::Active) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "session is not active");
    }
    if (request.persona_id.empty()) {
        request.persona_id = before.value().persona_id;
    }

    persona::ChatRequest chat;
    chat.session_id = request.session_id;
    chat.user_input = request.message;
    chat.trace_id = request.trace_id;
    chat.model = request.model;
    chat.context_id = request.mode;

    auto status = runtime_.SubmitChat(
        std::move(chat),
        [this, request = std::move(request), callback = std::move(callback), started](
            core::Result<persona::ChatResponse> result) mutable {
            if (!result.ok()) {
                sessions_.RecordRequestMetrics(request.session_id, Since(started), false, request.trace_id);
                callback(result.status());
                return;
            }
            callback(ToChatGatewayResponse(request, result.value(), started));
        });
    if (!status.ok()) {
        sessions_.RecordRequestMetrics(request.session_id, Since(started), false, request.trace_id);
    }
    return status;
}

core::Status PersonaGatewayService::SubmitClassroomMessage(ClassroomMessageGatewayRequest request,
                                                           ClassroomCallback callback) {
    if (!callback) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "classroom callback is required");
    }
    request.trace_id = EnsureTrace(std::move(request.trace_id));
    auto route = ResolveClassroomRoute(request);
    if (!route.ok()) {
        return route.status();
    }
    if (classroom_scheduler_) {
        auto user_status = classroom_scheduler_->OnUserMessage(route.value().classroom_id, route.value().persona_id);
        if (!user_status.ok()) {
            return user_status;
        }
    }
    ChatGatewayRequest chat;
    chat.trace_id = request.trace_id;
    chat.session_id = route.value().session_id;
    chat.persona_id = route.value().persona_id;
    chat.mode = request.broadcast ? "classroom_broadcast" : "classroom_message";
    chat.message = std::move(request.message);
    chat.model = std::move(request.model);
    const auto classroom_id = route.value().classroom_id;
    return SubmitChat(
        std::move(chat),
        [this, callback = std::move(callback), classroom_id](core::Result<ChatGatewayResponse> result) mutable {
            if (!result.ok()) {
                callback(result.status());
                return;
            }
            callback(ToClassroomGatewayResponse(result.value(), classroom_id));
        });
}

core::Status PersonaGatewayService::SubmitClassroomProactive(ClassroomProactiveGatewayRequest request,
                                                             ClassroomCallback callback) {
    if (!callback) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "classroom callback is required");
    }
    request.trace_id = EnsureTrace(std::move(request.trace_id));
    auto route = ResolveClassroomRoute(request);
    if (!route.ok()) {
        return route.status();
    }
    ChatGatewayRequest chat;
    chat.trace_id = request.trace_id;
    chat.session_id = route.value().session_id;
    chat.persona_id = route.value().persona_id;
    chat.mode = "classroom_proactive";
    chat.message = "[proactive] frontend requested proactive generation";
    chat.model = std::move(request.model);
    const auto classroom_id = route.value().classroom_id;
    return SubmitChat(
        std::move(chat),
        [this, callback = std::move(callback), classroom_id](core::Result<ChatGatewayResponse> result) mutable {
            if (!result.ok()) {
                callback(result.status());
                return;
            }
            if (classroom_scheduler_ && !result.value().content.empty()) {
                classroom_scheduler_->OnProactiveDelivered(classroom_id, result.value().persona_id);
            }
            callback(ToClassroomGatewayResponse(result.value(), classroom_id, !result.value().content.empty()));
        });
}

core::Status PersonaGatewayService::SubmitClassroomPoll(ClassroomPollGatewayRequest request,
                                                        ClassroomCallback callback) {
    if (!callback) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "classroom callback is required");
    }
    if (!classroom_scheduler_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "classroom scheduler is not configured");
    }
    request.trace_id = EnsureTrace(std::move(request.trace_id));
    ClassroomPollRequest poll;
    poll.trace_id = request.trace_id;
    poll.classroom_id = request.classroom_id;
    poll.persona_hint = request.persona_id;
    poll.context_id = request.context_id;
    poll.system_event = request.system_event;
    poll.system_event_content = request.system_event_content;
    auto decision = classroom_scheduler_->Poll(poll);
    if (!decision.ok()) {
        return decision.status();
    }
    if (!decision.value().should_speak) {
        ClassroomGatewayResponse response;
        response.trace_id = request.trace_id;
        response.classroom_id = request.classroom_id;
        response.session_id = decision.value().session_id;
        response.speaker_persona_id = decision.value().persona_id;
        response.should_speak = false;
        callback(std::move(response));
        return core::Status::Ok();
    }

    ChatGatewayRequest chat;
    chat.trace_id = request.trace_id;
    chat.session_id = decision.value().session_id;
    chat.persona_id = decision.value().persona_id;
    chat.mode = "classroom_proactive";
    chat.message = decision.value().trigger;
    chat.model = std::move(request.model);
    const auto classroom_id = decision.value().classroom_id;
    return SubmitChat(
        std::move(chat),
        [this, callback = std::move(callback), classroom_id](core::Result<ChatGatewayResponse> result) mutable {
            if (!result.ok()) {
                callback(result.status());
                return;
            }
            if (classroom_scheduler_ && !result.value().content.empty()) {
                classroom_scheduler_->OnProactiveDelivered(classroom_id, result.value().persona_id);
            }
            callback(ToClassroomGatewayResponse(result.value(), classroom_id, !result.value().content.empty()));
        });
}

core::Result<ChatGatewayResponse> PersonaGatewayService::ToChatGatewayResponse(
    const ChatGatewayRequest& request,
    const persona::ChatResponse& result,
    std::chrono::steady_clock::time_point started) {
    ChatGatewayResponse response;
    response.trace_id = request.trace_id;
    response.session_id = request.session_id;
    response.persona_id = request.persona_id;
    response.latency = Since(started);
    response.content = result.response;
    response.user_emotion = result.user_emotion;
    response.ai_emotion = result.ai_emotion;
    response.l0_hit = result.l0_hit;
    response.l3_hit = result.l3_hit;
    response.answer_cache = result.answer_cache;
    response.pipeline_latency = result.latency;
    response.turn_index = result.turn_index;
    return response;
}

ClassroomGatewayResponse PersonaGatewayService::ToClassroomGatewayResponse(const ChatGatewayResponse& result,
                                                                           std::string classroom_id,
                                                                           bool should_speak) const {
    ClassroomGatewayResponse response;
    response.trace_id = result.trace_id;
    response.classroom_id = std::move(classroom_id);
    response.session_id = result.session_id;
    response.latency = result.latency;
    response.speaker_persona_id = result.persona_id;
    response.content = result.content;
    response.should_speak = should_speak;
    response.user_emotion = result.user_emotion;
    response.ai_emotion = result.ai_emotion;
    response.turn_index = result.turn_index;
    return response;
}

core::Result<ClassroomRouteResult> PersonaGatewayService::ResolveClassroomRoute(
    const ClassroomMessageGatewayRequest& request) {
    if (!classroom_scheduler_) {
        if (request.session_id.empty()) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "session_id is required without classroom scheduler");
        }
        ClassroomRouteResult result;
        result.classroom_id = request.classroom_id;
        result.persona_id = request.target_persona_id;
        result.session_id = request.session_id;
        result.context_id = request.context_id;
        return result;
    }
    ClassroomRouteRequest route;
    route.classroom_id = request.classroom_id;
    route.context_id = request.context_id;
    route.persona_hint = request.target_persona_id;
    route.session_id = request.session_id;
    return classroom_scheduler_->Resolve(route);
}

core::Result<ClassroomRouteResult> PersonaGatewayService::ResolveClassroomRoute(
    const ClassroomProactiveGatewayRequest& request) {
    if (!classroom_scheduler_) {
        if (request.session_id.empty()) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "session_id is required without classroom scheduler");
        }
        ClassroomRouteResult result;
        result.classroom_id = request.classroom_id;
        result.persona_id = request.persona_id;
        result.session_id = request.session_id;
        result.context_id = request.context_id;
        return result;
    }
    ClassroomRouteRequest route;
    route.classroom_id = request.classroom_id;
    route.context_id = request.context_id;
    route.persona_hint = request.persona_id;
    route.session_id = request.session_id;
    return classroom_scheduler_->Resolve(route);
}

ClassroomPersonaRegistration PersonaGatewayService::BuildClassroomRegistration(
    const CreateSessionGatewayRequest& request,
    const persona::SessionSnapshot& snapshot) const {
    ClassroomPersonaRegistration registration;
    registration.classroom_id = request.classroom_id;
    registration.persona_id = snapshot.persona_id;
    registration.session_id = snapshot.session_id;
    registration.context_ids.insert(request.context_ids.begin(), request.context_ids.end());
    registration.context_patterns = request.context_patterns;
    registration.default_persona = request.default_persona;
    registration.agent.proactive_level = ProactiveLevelFromString(request.proactive_level);
    return registration;
}

core::Result<persona::ChatResponse> PersonaGatewayService::SubmitChatAndWait(persona::ChatRequest request) {
    std::promise<core::Result<persona::ChatResponse>> promise;
    auto future = promise.get_future();
    auto status = runtime_.SubmitChat(
        std::move(request),
        [&promise](core::Result<persona::ChatResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    if (!status.ok()) {
        return status;
    }
    if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        return core::Status::Error(core::ErrorCode::Timeout, "persona runtime callback timed out");
    }
    return future.get();
}

std::string PersonaGatewayService::EnsureTrace(std::string trace_id) const {
    if (!trace_id.empty()) {
        return trace_id;
    }
    if (auto current = core::CurrentTraceId(); current != "-") {
        return std::string(current);
    }
    return core::GenerateTraceId();
}

} // namespace agent::service::gateway
