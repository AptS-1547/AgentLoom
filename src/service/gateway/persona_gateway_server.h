#pragma once

#include "persona_gateway_http_adapter.h"
#include "persona_gateway_service.h"
#include "persona_runtime.h"
#include "runtime_maintenance_service.h"
#include "document_analysis_service.h"
#include "teaching_evaluator.h"
#include "http_server.h"
#include "static_file_handler.h"
#include "thread_pool.h"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace agent::semantic_cache {
class L0MemoryCacheAdapter;
class RedisConnectionPool;
}

namespace storage::sqlite {
class SqliteConnectionPool;
}

namespace agent::service::gateway {

struct GatewayDocumentStoreOptions {
    bool enabled = false;
    std::filesystem::path root;
    std::filesystem::path database_path;
    std::size_t read_connection_count = 2;
    std::size_t write_connection_count = 1;
    int busy_timeout_ms = 5000;
    int retention_hours = 24 * 7;
    int cleanup_interval_seconds = 60;
    bool enable_path_register_test_endpoint = false;
    bool enable_path_analyze_test_endpoint = false;
};

struct PersonaGatewayServerOptions {
    ::net::HttpServerOptions http;
    GatewayAuthOptions auth;
    core::ThreadPoolOptions compute_pool;
    core::ThreadPoolOptions io_pool;
    persona::SessionOptions session;
    persona::PersonaRuntimeOptions runtime;
    std::optional<::net::StaticFileOptions> static_files;
    GatewayDocumentStoreOptions document_store;
    std::string websocket_path = "/ws/session";
};

struct PersonaGatewayServerDependencies {
    std::shared_ptr<persona::IMemoryContextProvider> memory_provider;
    std::shared_ptr<persona::IEmotionAnalyzer> emotion_analyzer;
    std::shared_ptr<persona::IToolMemoryProvider> tool_memory_provider;
    std::shared_ptr<persona::ISkillSessionManager> skill_session_manager;
    std::shared_ptr<llm::ILlmClient> llm_client;
    std::shared_ptr<document::IDocumentEmbeddingProvider> document_embedding_provider;
    std::shared_ptr<document::IDocumentLlmChunkCache> document_llm_chunk_cache;
    std::shared_ptr<semantic_cache::ISemanticCache> document_semantic_cache;
    std::shared_ptr<semantic_cache::RedisConnectionPool> l0_redis_pool;
    std::shared_ptr<semantic_cache::L0MemoryCacheAdapter> l0_memory_adapter;
    std::shared_ptr<IPersonaMetadataStore> persona_metadata_store;
    std::filesystem::path evaluation_config_path;
    std::vector<std::shared_ptr<IRuntimeMaintenanceTask>> maintenance_tasks;
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
    core::Status RegisterMaintenanceTask(std::shared_ptr<IRuntimeMaintenanceTask> task);

    bool running() const noexcept;
    std::uint16_t port() const noexcept;

    persona::SessionManager& sessions() noexcept;
    PersonaGatewayService& service() noexcept;
    ::net::HttpServer& http_server() noexcept;

private:
    void HandleHttp(std::shared_ptr<::net::IHttpRequest> request);
    void HandleWebSocket(std::shared_ptr<::net::IWebSocketStreamRequest> request);
    void HandleWebSocketClose(const ::net::ConnectionCloseInfo& close_info);
    core::Status ValidateDependencies() const;
    core::Status EnsureAuthSessionStore();
    core::Status EnsurePersonaMetadataStore();
    core::Status EnsureDocumentStore();
    void ShutdownDocumentStore();

    PersonaGatewayServerOptions options_;
    PersonaGatewayServerDependencies dependencies_;
    core::LoggerAdapter logger_;
    core::ThreadPool compute_pool_;
    core::ThreadPool io_pool_;
    persona::SessionManager sessions_;
    persona::PersonaRuntime runtime_;
    ClassroomScheduler classroom_scheduler_;
    std::shared_ptr<agent::semantic_cache::RedisConnectionPool> auth_redis_;
    std::shared_ptr<IAuthSessionStore> auth_session_store_;
    std::shared_ptr<IPersonaMetadataStore> persona_metadata_store_;
    PersonaGatewayService service_;
    std::shared_ptr<document::DocumentAnalysisService> document_service_;
    std::shared_ptr<storage::sqlite::SqliteConnectionPool> document_repository_pool_;
    std::shared_ptr<IGatewayAuthenticator> authenticator_;
    std::shared_ptr<IAuthRegistrationService> auth_registration_;
    PersonaGatewayHttpAdapter adapter_;
    ::net::HttpServer http_server_;
    RuntimeMaintenanceService maintenance_;
    std::shared_ptr<::net::StaticFileHandler> static_files_;
    bool started_ = false;
};

} // namespace agent::service::gateway
