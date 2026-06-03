#pragma once

#include "classroom_scheduler.h"
#include "gateway_models.h"
#include "logger_adapter.h"
#include "persona_runtime.h"
#include "session_manager.h"

#include <functional>
#include <future>
#include <memory>

namespace agent::service::gateway {

class PersonaGatewayService {
public:
    using ChatCallback = std::function<void(core::Result<ChatGatewayResponse>)>;
    using ClassroomCallback = std::function<void(core::Result<ClassroomGatewayResponse>)>;

    PersonaGatewayService(persona::SessionManager& sessions,
                          persona::PersonaRuntime& runtime,
                          IClassroomScheduler* classroom_scheduler = nullptr,
                          core::LoggerAdapter logger = core::LoggerAdapter::ForModule("service"));

    core::Result<SessionGatewayResponse> CreateSession(CreateSessionGatewayRequest request);
    core::Result<SessionGatewayResponse> GetSession(std::string_view session_id, std::string trace_id);
    core::Result<SessionGatewayResponse> CloseSession(CloseSessionGatewayRequest request);
    core::Result<ChatGatewayResponse> Chat(ChatGatewayRequest request);
    core::Result<ClassroomGatewayResponse> ClassroomMessage(ClassroomMessageGatewayRequest request);
    core::Result<ClassroomGatewayResponse> ClassroomProactive(ClassroomProactiveGatewayRequest request);
    core::Result<ClassroomGatewayResponse> ClassroomPoll(ClassroomPollGatewayRequest request);
    core::Result<TrainingReportGatewayResponse> TrainingReport(TrainingReportGatewayRequest request);

    core::Status SubmitChat(ChatGatewayRequest request, ChatCallback callback);
    core::Status SubmitClassroomMessage(ClassroomMessageGatewayRequest request, ClassroomCallback callback);
    core::Status SubmitClassroomProactive(ClassroomProactiveGatewayRequest request, ClassroomCallback callback);
    core::Status SubmitClassroomPoll(ClassroomPollGatewayRequest request, ClassroomCallback callback);

private:
    core::Result<ChatGatewayResponse> ToChatGatewayResponse(const ChatGatewayRequest& request,
                                                            const persona::ChatResponse& result,
                                                            std::chrono::steady_clock::time_point started);
    ClassroomGatewayResponse ToClassroomGatewayResponse(const ChatGatewayResponse& result,
                                                        std::string classroom_id,
                                                        bool should_speak = true) const;
    core::Result<ClassroomRouteResult> ResolveClassroomRoute(const ClassroomMessageGatewayRequest& request);
    core::Result<ClassroomRouteResult> ResolveClassroomRoute(const ClassroomProactiveGatewayRequest& request);
    ClassroomPersonaRegistration BuildClassroomRegistration(const CreateSessionGatewayRequest& request,
                                                            const persona::SessionSnapshot& snapshot) const;
    core::Result<persona::ChatResponse> SubmitChatAndWait(persona::ChatRequest request);
    std::string EnsureTrace(std::string trace_id) const;

    persona::SessionManager& sessions_;
    persona::PersonaRuntime& runtime_;
    IClassroomScheduler* classroom_scheduler_ = nullptr;
    core::LoggerAdapter logger_;
};

} // namespace agent::service::gateway
