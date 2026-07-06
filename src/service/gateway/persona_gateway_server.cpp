#include "persona_gateway_server.h"

#include "document_file_store.h"
#include "http_types.h"
#include "l0_memory_cache_adapter.h"
#include "redis_connection_pool.h"
#include "sqlite/sqlite_connection_pool.h"

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
    if (!http.request_filter.enabled) {
        http.request_filter.enabled = true;
        http.request_filter.reject_control_chars = true;
        http.request_filter.reject_suspicious_patterns = true;
    }
    if (options.static_files) {
        http.static_files = std::nullopt;
    }
    return http;
}

std::shared_ptr<IAuthSessionStore> MakeAuthSessionStore(
    const GatewayAuthOptions& options,
    std::shared_ptr<agent::semantic_cache::RedisConnectionPool>& auth_redis) {
    if (options.session_store_backend == "redis") {
        agent::semantic_cache::RedisPoolOptions redis_options;
        redis_options.host = options.redis_host;
        redis_options.port = options.redis_port;
        redis_options.password = options.redis_password;
        redis_options.pool_size = options.redis_pool_size;
        redis_options.command_timeout = options.redis_command_timeout;
        auth_redis = std::make_shared<agent::semantic_cache::RedisConnectionPool>(redis_options);
        return std::make_shared<RedisAuthSessionStore>(auth_redis, options.redis_key_prefix);
    }
    if (options.session_database_path.empty()) {
        return nullptr;
    }
    return std::make_shared<SqliteAuthSessionStore>(options.session_database_path);
}

std::shared_ptr<IPersonaMetadataStore> MakePersonaMetadataStore(
    const GatewayAuthOptions& options,
    const PersonaGatewayServerDependencies& dependencies,
    const std::shared_ptr<agent::semantic_cache::RedisConnectionPool>& auth_redis) {
    if (dependencies.persona_metadata_store) {
        return dependencies.persona_metadata_store;
    }
    if (options.session_database_path.empty()) {
        return std::make_shared<InMemoryPersonaMetadataStore>();
    }
    auto primary = std::make_shared<SqlitePersonaMetadataStore>(options.session_database_path);
    if (!auth_redis) {
        return primary;
    }
    auto cache = std::make_shared<RedisPersonaMetadataCache>(
        auth_redis,
        options.redis_key_prefix.empty() ? "agent:gateway:persona" : options.redis_key_prefix + ":persona");
    return std::make_shared<CachedPersonaMetadataStore>(std::move(primary), std::move(cache));
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
               dependencies_.tool_memory_provider,
               dependencies_.skill_session_manager,
               logger_),
      classroom_scheduler_({}, core::LoggerAdapter::ForModule("classroom")),
      auth_session_store_(MakeAuthSessionStore(options_.auth, auth_redis_)),
      persona_metadata_store_(MakePersonaMetadataStore(options_.auth, dependencies_, auth_redis_)),
      service_(sessions_,
               runtime_,
               &classroom_scheduler_,
               std::make_shared<evaluation::TeachingEvaluator>(),
               dependencies_.l0_redis_pool,
               persona_metadata_store_,
               dependencies_.evaluation_config_path,
               logger_),
      document_service_(std::make_shared<document::DocumentAnalysisService>(
          compute_pool_,
          io_pool_,
          core::LoggerAdapter::ForModule("document"))),
      authenticator_(std::make_shared<JwtCookieAuthenticator>(options_.auth, auth_session_store_)),
      auth_registration_(std::make_shared<JwtAuthRegistrationService>(options_.auth, auth_session_store_)),
      adapter_(service_,
               authenticator_,
               auth_registration_,
               document_service_,
               dependencies_.llm_client,
               dependencies_.document_embedding_provider,
               dependencies_.document_llm_chunk_cache,
               dependencies_.document_semantic_cache,
               dependencies_.skill_session_manager,
               PersonaGatewayHttpAdapterOptions{
                   .enable_dev_registration = options_.auth.enable_dev_registration,
                   .enable_path_register_test_endpoint =
                       options_.document_store.enable_path_register_test_endpoint,
                   .enable_path_analyze_test_endpoint =
                       options_.document_store.enable_path_analyze_test_endpoint}),
      http_server_(ResolveHttpOptions(options_)),
      maintenance_(core::LoggerAdapter::ForModule("gateway")) {
    if (dependencies_.l0_memory_adapter) {
        auto l0 = dependencies_.l0_memory_adapter;
        sessions_.SetSessionClosedCallback([l0 = std::move(l0)](const persona::SessionSnapshot& snapshot) {
            l0->ReleaseSession(snapshot.session_id);
        });
    }
    if (options_.static_files) {
        static_files_ = std::make_shared<::net::StaticFileHandler>(*options_.static_files);
    }

    static_cast<void>(maintenance_.RegisterTask(std::make_shared<SessionMaintenanceTask>(
        sessions_,
        std::chrono::seconds(30),
        core::LoggerAdapter::ForModule("gateway"))));
    for (const auto& task : dependencies_.maintenance_tasks) {
        static_cast<void>(maintenance_.RegisterTask(task));
    }

    http_server_.SetHttpRequestHandler([this](std::shared_ptr<::net::IHttpRequest> request) {
        HandleHttp(std::move(request));
    });
    http_server_.SetWebSocketStreamHandler(options_.websocket_path, [this](std::shared_ptr<::net::IWebSocketStreamRequest> request) {
        HandleWebSocket(std::move(request));
    });
    http_server_.SetWebSocketCloseHandler([this](const ::net::ConnectionCloseInfo& close_info) {
        HandleWebSocketClose(close_info);
    });
}

PersonaGatewayServer::~PersonaGatewayServer() {
    Stop();
}

core::Status PersonaGatewayServer::RegisterMaintenanceTask(std::shared_ptr<IRuntimeMaintenanceTask> task) {
    if (started_ || maintenance_.running()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                   "maintenance tasks must be registered before server start");
    }
    return maintenance_.RegisterTask(std::move(task));
}

core::Status PersonaGatewayServer::Start() {
    if (started_) {
        return core::Status::Ok();
    }
    auto dependency_status = ValidateDependencies();
    if (!dependency_status.ok()) {
        return dependency_status;
    }
    auto auth_store_status = EnsureAuthSessionStore();
    if (!auth_store_status.ok()) {
        return auth_store_status;
    }
    auto persona_metadata_status = EnsurePersonaMetadataStore();
    if (!persona_metadata_status.ok()) {
        return persona_metadata_status;
    }
    auto document_store_status = EnsureDocumentStore();
    if (!document_store_status.ok()) {
        return document_store_status;
    }

    auto compute_status = compute_pool_.Start();
    if (!compute_status.ok()) {
        ShutdownDocumentStore();
        return compute_status;
    }

    auto io_status = io_pool_.Start();
    if (!io_status.ok()) {
        ShutdownDocumentStore();
        compute_pool_.Shutdown(false);
        return io_status;
    }

    if (options_.document_store.enabled) {
        auto document_task_status = maintenance_.RegisterTask(std::make_shared<DocumentRetentionMaintenanceTask>(
            document_service_,
            std::chrono::seconds(options_.document_store.cleanup_interval_seconds)));
        if (!document_task_status.ok() && document_task_status.code() != core::ErrorCode::AlreadyExists) {
            ShutdownDocumentStore();
            io_pool_.Shutdown(false);
            compute_pool_.Shutdown(false);
            return document_task_status;
        }
    }

    auto maintenance_status = maintenance_.Start();
    if (!maintenance_status.ok()) {
        ShutdownDocumentStore();
        io_pool_.Shutdown(false);
        compute_pool_.Shutdown(false);
        return maintenance_status;
    }

    auto http_status = http_server_.Start();
    if (!http_status.ok()) {
        maintenance_.Stop();
        ShutdownDocumentStore();
        io_pool_.Shutdown(false);
        compute_pool_.Shutdown(false);
        return http_status;
    }

    started_ = true;
    logger_.info("[gateway] started http_port={} ws_path={}", http_server_.port(), options_.websocket_path);
    return core::Status::Ok();
}

core::Status PersonaGatewayServer::EnsureAuthSessionStore() {
    if (auth_redis_ && !auth_redis_->running()) {
        auto start = auth_redis_->Start();
        if (!start.ok()) {
            return start;
        }
    }
    if (!auth_session_store_) {
        return core::Status::Ok();
    }
    return auth_session_store_->EnsureSchema();
}

core::Status PersonaGatewayServer::EnsurePersonaMetadataStore() {
    if (!persona_metadata_store_) {
        return core::Status::Ok();
    }
    return persona_metadata_store_->EnsureSchema();
}

core::Status PersonaGatewayServer::EnsureDocumentStore() {
    if (!options_.document_store.enabled) {
        return core::Status::Ok();
    }
    if (options_.document_store.root.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "gateway document store root is required");
    }
    if (options_.document_store.database_path.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "gateway document store database_path is required");
    }

    storage::sqlite::SqliteConnectionPoolOptions pool_options;
    pool_options.path = options_.document_store.database_path.string();
    pool_options.read_connection_count = options_.document_store.read_connection_count;
    pool_options.write_connection_count = options_.document_store.write_connection_count;
    pool_options.busy_timeout_ms = options_.document_store.busy_timeout_ms;
    document_repository_pool_ = std::make_shared<storage::sqlite::SqliteConnectionPool>(pool_options);
    auto start = document_repository_pool_->Start();
    if (!start.ok()) {
        return start;
    }
    auto repository_status = document_service_->SetRepository(document_repository_pool_);
    if (!repository_status.ok()) {
        document_repository_pool_->Close();
        document_repository_pool_.reset();
        return repository_status;
    }

    auto file_store = std::make_shared<document::DocumentFileStore>(
        document::DocumentFileStoreOptions{
            options_.document_store.root,
            std::chrono::hours(options_.document_store.retention_hours)});
    auto file_store_status = document_service_->SetFileStore(std::move(file_store));
    if (!file_store_status.ok()) {
        ShutdownDocumentStore();
        return file_store_status;
    }
    document_service_->SetRetentionCleanupOptions(
        std::chrono::hours(options_.document_store.retention_hours),
        std::chrono::seconds(options_.document_store.cleanup_interval_seconds));
    logger_.info("[gateway] document store enabled root={} db={}",
                 options_.document_store.root.string(),
                 options_.document_store.database_path.string());
    return core::Status::Ok();
}

void PersonaGatewayServer::ShutdownDocumentStore() {
    if (document_service_) {
        static_cast<void>(document_service_->SetFileStore(nullptr));
        static_cast<void>(document_service_->SetRepository(nullptr));
    }
    if (document_repository_pool_) {
        document_repository_pool_->Close();
        document_repository_pool_.reset();
    }
}

void PersonaGatewayServer::Stop() {
    if (!started_ && !http_server_.running()) {
        return;
    }
    http_server_.Stop();
    maintenance_.Stop();
    io_pool_.Shutdown(true);
    compute_pool_.Shutdown(true);
    if (auth_redis_) {
        auth_redis_->Shutdown();
    }
    ShutdownDocumentStore();
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

void PersonaGatewayServer::HandleWebSocketClose(const ::net::ConnectionCloseInfo& close_info) {
    if (close_info.target != options_.websocket_path) {
        return;
    }
    adapter_.CleanupUnfinishedDocumentUploadsForConnection(close_info.connection_id);
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
