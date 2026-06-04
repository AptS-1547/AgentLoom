#pragma once

#include "persona_gateway_http_adapter.h"
#include "persona_gateway_service.h"
#include "persona_runtime.h"
#include "http_server.h"
#include "static_file_handler.h"
#include "thread_pool.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace agent::service::gateway {

struct PersonaGatewayServerOptions {
    ::net::HttpServerOptions http;
    GatewayAuthOptions auth;
    core::ThreadPoolOptions compute_pool;
    core::ThreadPoolOptions io_pool;
    persona::SessionOptions session;
    persona::PersonaRuntimeOptions runtime;
    std::optional<::net::StaticFileOptions> static_files;
    std::string websocket_path = "/ws/session";
};

struct PersonaGatewayServerDependencies {
    std::shared_ptr<persona::IMemoryContextProvider> memory_provider;
    std::shared_ptr<persona::IEmotionAnalyzer> emotion_analyzer;
    std::shared_ptr<llm::ILlmClient> llm_client;
};

class PersonaGatewayServer final {
public:
    PersonaGatewayServer(PersonaGatewayServerOptions options,
                         PersonaGatewayServerDependencies dependencies,
                         core::LoggerAdapter logger = core::LoggerAdapter::ForModule("gateway"));
    ~PersonaGatewayServer();

    PersonaGatewayServer(const PersonaGatewayServer&) = delete;
    PersonaGatewayServer& operator=(const PersonaGatewayServer&) = delete;

    core::Status Start();
    void Stop();

    bool running() const noexcept;
    std::uint16_t port() const noexcept;

    persona::SessionManager& sessions() noexcept;
    PersonaGatewayService& service() noexcept;
    ::net::HttpServer& http_server() noexcept;

private:
    void HandleHttp(std::shared_ptr<::net::IHttpRequest> request);
    void HandleWebSocket(std::shared_ptr<::net::IWebSocketStreamRequest> request);
    core::Status ValidateDependencies() const;
    core::Status EnsureAuthSessionStore();

    PersonaGatewayServerOptions options_;
    PersonaGatewayServerDependencies dependencies_;
    core::LoggerAdapter logger_;
    core::ThreadPool compute_pool_;
    core::ThreadPool io_pool_;
    persona::SessionManager sessions_;
    persona::PersonaRuntime runtime_;
    ClassroomScheduler classroom_scheduler_;
    PersonaGatewayService service_;
    std::shared_ptr<IAuthSessionStore> auth_session_store_;
    std::shared_ptr<IGatewayAuthenticator> authenticator_;
    std::shared_ptr<IAuthRegistrationService> auth_registration_;
    PersonaGatewayHttpAdapter adapter_;
    ::net::HttpServer http_server_;
    std::shared_ptr<::net::StaticFileHandler> static_files_;
    bool started_ = false;
};

} // namespace agent::service::gateway
