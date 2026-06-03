#include "persona_gateway_server.h"

#include "http_types.h"

#include <utility>

namespace agent::service::gateway {

namespace {

core::ThreadPoolOptions WithDefaultPoolName(core::ThreadPoolOptions options, std::string name) {
    if (options.name.empty() || options.name == "core-thread-pool") {
        options.name = std::move(name);
    }
    return options;
}

::net::HttpServerOptions ResolveHttpOptions(const PersonaGatewayServerOptions& options) {
    auto http = options.http;
    if (options.static_files) {
        http.static_files = std::nullopt;
    }
    return http;
}

} // namespace

PersonaGatewayServer::PersonaGatewayServer(PersonaGatewayServerOptions options,
                                           PersonaGatewayServerDependencies dependencies,
                                           core::LoggerAdapter logger)
    : options_(std::move(options)),
      dependencies_(std::move(dependencies)),
      logger_(std::move(logger)),
      compute_pool_(WithDefaultPoolName(options_.compute_pool, "gateway-compute-pool")),
      io_pool_(WithDefaultPoolName(options_.io_pool, "gateway-io-pool")),
      sessions_(compute_pool_, io_pool_, options_.session, logger_),
      runtime_(sessions_,
               dependencies_.memory_provider,
               dependencies_.emotion_analyzer,
               dependencies_.llm_client,
               options_.runtime,
               nullptr,
               logger_),
      classroom_scheduler_({}, core::LoggerAdapter::ForModule("classroom")),
      service_(sessions_, runtime_, &classroom_scheduler_, logger_),
      adapter_(service_),
      http_server_(ResolveHttpOptions(options_)) {
    if (options_.static_files) {
        static_files_ = std::make_shared<::net::StaticFileHandler>(*options_.static_files);
    }

    http_server_.SetHttpRequestHandler([this](std::shared_ptr<::net::IHttpRequest> request) {
        HandleHttp(std::move(request));
    });
    http_server_.SetWebSocketStreamHandler(options_.websocket_path, [this](std::shared_ptr<::net::IWebSocketStreamRequest> request) {
        HandleWebSocket(std::move(request));
    });
}

PersonaGatewayServer::~PersonaGatewayServer() {
    Stop();
}

core::Status PersonaGatewayServer::Start() {
    if (started_) {
        return core::Status::Ok();
    }
    auto dependency_status = ValidateDependencies();
    if (!dependency_status.ok()) {
        return dependency_status;
    }

    auto compute_status = compute_pool_.Start();
    if (!compute_status.ok()) {
        return compute_status;
    }

    auto io_status = io_pool_.Start();
    if (!io_status.ok()) {
        compute_pool_.Shutdown(false);
        return io_status;
    }

    auto http_status = http_server_.Start();
    if (!http_status.ok()) {
        io_pool_.Shutdown(false);
        compute_pool_.Shutdown(false);
        return http_status;
    }

    started_ = true;
    logger_.info("[gateway] started http_port={} ws_path={}", http_server_.port(), options_.websocket_path);
    return core::Status::Ok();
}

void PersonaGatewayServer::Stop() {
    if (!started_ && !http_server_.running()) {
        return;
    }
    http_server_.Stop();
    sessions_.CleanupExpired();
    io_pool_.Shutdown(true);
    compute_pool_.Shutdown(true);
    started_ = false;
    logger_.info("[gateway] stopped");
}

bool PersonaGatewayServer::running() const noexcept {
    return http_server_.running();
}

std::uint16_t PersonaGatewayServer::port() const noexcept {
    return http_server_.port();
}

persona::SessionManager& PersonaGatewayServer::sessions() noexcept {
    return sessions_;
}

PersonaGatewayService& PersonaGatewayServer::service() noexcept {
    return service_;
}

::net::HttpServer& PersonaGatewayServer::http_server() noexcept {
    return http_server_;
}

void PersonaGatewayServer::HandleHttp(std::shared_ptr<::net::IHttpRequest> request) {
    if (!request) {
        return;
    }
    if (PersonaGatewayHttpAdapter::IsApiRequest(request->message().target())) {
        adapter_.HandleHttp(std::move(request));
        return;
    }

    if (static_files_) {
        request->Respond(static_files_->Handle(request->message()));
        return;
    }

    auto response = ::net::HttpResponse::Text(::net::http::status::not_found, "not found").message;
    response.version(request->message().version());
    response.keep_alive(request->message().keep_alive());
    response.prepare_payload();
    request->Respond(std::move(response));
}

void PersonaGatewayServer::HandleWebSocket(std::shared_ptr<::net::IWebSocketStreamRequest> request) {
    adapter_.HandleWebSocket(std::move(request));
}

core::Status PersonaGatewayServer::ValidateDependencies() const {
    if (!dependencies_.memory_provider) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "memory context provider is required");
    }
    if (!dependencies_.emotion_analyzer) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "emotion analyzer is required");
    }
    if (!dependencies_.llm_client) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "llm client is required");
    }
    return core::Status::Ok();
}

} // namespace agent::service::gateway
