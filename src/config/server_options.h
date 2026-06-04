#pragma once

#include <chrono>
#include "http_server.h"
#include "onnx_model.h"
#include "request_options.h"
#include "server_common.h"
#include "text_embedding_model.h"
#include "vector_cache.h"
#include "vlm_cache.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

struct VramGuardOptions {
    int monitor_interval_seconds = 10;
    size_t warning_free_bytes = 1024 * 1024 * 1024;
    size_t unload_free_bytes = 512 * 1024 * 1024;
    size_t min_free_before_load_bytes = 0;
    bool reload_after_unload = false;
    bool unload_on_oom_error = true;
};

struct EmbeddingModelOptions {
    std::string tokenizer_path;
    std::string onnx_model_path;
    std::string execution_provider = "auto";
    bool allow_cpu_fallback = true;
    int cuda_device_id = 0;
    int intra_op_num_threads = 0;
    int inter_op_num_threads = 0;
    std::string pooling_strategy = "mean";
    bool normalize = true;
    int expected_dimension = 0;
    bool require_token_type_ids = false;
};

struct LlmOptions {
    std::string base_url;
    std::string api_key_env = "AGENT_LLM_API_KEY";
    std::string api_key_file;
    /// Resolved at Validate time — do not set in config file.
    std::string api_key;
    std::string model = "deepseek-chat";
    int timeout_ms = 30000;
    int max_retries = 2;
    /// Optional CA bundle (PEM) for HTTPS verification.  Relative to config
    /// file directory.  Leave empty on Linux to use the system trust store.
    std::string ca_bundle_path;
    std::unordered_map<std::string, std::filesystem::path> prompts;
};

struct GatewayAuthConfigOptions {
    bool enabled = false;
    bool allow_dev_identity = true;
    bool require_auth_for_api = false;
    std::string cookie_name = "agent_auth";
    std::string public_key_pem;
    std::string public_key_file;
    std::string private_key_pem;
    std::string private_key_file;
    std::string issuer;
    std::string audience;
    int clock_skew_seconds = 60;
    int token_ttl_seconds = 28800;
    bool cookie_http_only = true;
    bool cookie_secure = false;
    std::string cookie_same_site = "Lax";
    bool require_session_record = false;
    bool auto_provision_session = true;
    std::string session_database_path;
};

struct GatewayStaticFilesConfigOptions {
    bool enabled = false;
    std::filesystem::path root;
    std::string index_file = "index.html";
    bool spa_fallback = true;
};

struct GatewayThreadPoolConfigOptions {
    std::size_t worker_count = 0;
    std::size_t queue_capacity = 0;
};

struct PersonaGatewayConfigOptions {
    std::string websocket_path = "/ws/session";
    GatewayStaticFilesConfigOptions static_files;
    GatewayThreadPoolConfigOptions compute_pool;
    GatewayThreadPoolConfigOptions io_pool;
    int session_idle_timeout_minutes = 15;
    std::size_t session_max_recent_turns = 20;
    std::size_t runtime_recent_raw_turns = 10;
    std::string runtime_default_model;
    bool request_filter_enabled = true;
    bool reject_control_chars = true;
    bool reject_suspicious_patterns = true;
};

template <typename GatewayAuthOptionsT>
GatewayAuthOptionsT ToGatewayAuthOptions(const GatewayAuthConfigOptions& config) {
    GatewayAuthOptionsT options;
    options.enabled = config.enabled;
    options.allow_dev_identity = config.allow_dev_identity;
    options.require_auth_for_api = config.require_auth_for_api;
    options.cookie_name = config.cookie_name;
    options.public_key_pem = config.public_key_pem;
    options.private_key_pem = config.private_key_pem;
    options.issuer = config.issuer;
    options.audience = config.audience;
    options.clock_skew = std::chrono::seconds(config.clock_skew_seconds);
    options.token_ttl = std::chrono::seconds(config.token_ttl_seconds);
    options.cookie_http_only = config.cookie_http_only;
    options.cookie_secure = config.cookie_secure;
    options.cookie_same_site = config.cookie_same_site;
    options.require_session_record = config.require_session_record;
    options.auto_provision_session = config.auto_provision_session;
    options.session_database_path = config.session_database_path;
    return options;
}

struct MultimodalServerOptions {
    server_common::GrpcServerOptions grpc;
    net::HttpServerOptions http;
    bert::ModelRuntimeOptions bert_runtime;
    EmbeddingModelOptions embedding;
    LlmOptions llm;
    request_validation::AuthOptions auth;
    GatewayAuthConfigOptions gateway_auth;
    PersonaGatewayConfigOptions persona_gateway;
    request_validation::RequestLimits limits;
    VramGuardOptions vram;
    vlm_cache::Options vlm_cache;
    vlm_cache::VectorOptions vlm_cache_vector;
    std::string auth_token_file;
    std::string auth_token_env = "AGENT_BACKEND_AUTH_TOKEN";
    std::string auth_source;
    std::string bert_model;
    std::string vit_model;
    std::string llm_model;
    std::string mmproj;
    int n_gpu_layers = -1;
    std::filesystem::path config_file_path;
};

template <typename PersonaGatewayServerOptionsT, typename StaticFileOptionsT>
PersonaGatewayServerOptionsT ToPersonaGatewayServerOptions(const MultimodalServerOptions& config) {
    PersonaGatewayServerOptionsT options;
    options.http = config.http;
    options.http.request_filter.enabled = config.persona_gateway.request_filter_enabled;
    options.http.request_filter.reject_control_chars = config.persona_gateway.reject_control_chars;
    options.http.request_filter.reject_suspicious_patterns = config.persona_gateway.reject_suspicious_patterns;
    options.auth = ToGatewayAuthOptions<decltype(options.auth)>(config.gateway_auth);
    options.websocket_path = config.persona_gateway.websocket_path;
    options.compute_pool.worker_count = config.persona_gateway.compute_pool.worker_count;
    options.compute_pool.queue_capacity = config.persona_gateway.compute_pool.queue_capacity;
    options.io_pool.worker_count = config.persona_gateway.io_pool.worker_count;
    options.io_pool.queue_capacity = config.persona_gateway.io_pool.queue_capacity;
    options.session.idle_timeout = std::chrono::minutes(config.persona_gateway.session_idle_timeout_minutes);
    options.session.max_recent_turns = config.persona_gateway.session_max_recent_turns;
    options.runtime.recent_raw_turns = config.persona_gateway.runtime_recent_raw_turns;
    options.runtime.default_model = config.persona_gateway.runtime_default_model.empty()
        ? config.llm.model
        : config.persona_gateway.runtime_default_model;
    if (config.persona_gateway.static_files.enabled) {
        StaticFileOptionsT static_files;
        static_files.root = config.persona_gateway.static_files.root;
        static_files.index_file = config.persona_gateway.static_files.index_file;
        static_files.spa_fallback = config.persona_gateway.static_files.spa_fallback;
        options.static_files = std::move(static_files);
    }
    return options;
}
