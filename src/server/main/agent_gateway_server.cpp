// Production-style Persona Gateway server entry.
//
// Starts one C++ HTTP server that hosts static frontend files, Gateway API
// routes, WebSocket sessions, document analysis, cache layers, and runtime
// maintenance tasks.

#include "persona_gateway_server.h"
#include "grpc_emotion_analyzer.h"
#include "emotion_fusion_analyzer.h"
#include "semantic_cache_types.h"
#include "document_llm_chunk_cache.h"
#include "isemantic_cache.h"
#include "openai_llm_client.h"
#include "local_llm_client.h"
#include "beast_http_client.h"
#include "tls_context.h"
#include "logger.h"
#include "redis_connection_pool.h"
#include "semantic_cache_pipeline.h"
#include "l0_memory_cache_adapter.h"
#include "sqlite/sqlite_connection.h"
#include "sqlite/sqlite_connection_pool.h"
#include "sqlite_vector_repository.h"
#include "vector_partition_registry.h"
#include "vector_index_manager.h"
#include "long_term_memory_compressor.h"
#include "embedding_pipeline.h"
#include "hf_tokenizer.h"
#include "onnx_text_embedding_model.h"

#ifdef _WIN32
#include "crash_dump.h"
#endif

#include <nlohmann/json.hpp>

#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_set>
#include <string>
#include <thread>

using Json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

std::atomic_bool g_stop_requested{false};

int Fail(const std::string& msg) {
    std::cerr << "[agent-gateway] FAIL: " << msg << std::endl;
    return 1;
}

void OnSignal(int) {
    g_stop_requested.store(true);
}

std::string ReadTextFile(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open file: " + path.string());
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::string ReadFirstLine(const fs::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open file: " + path.string());
    }
    std::string line;
    std::getline(file, line);
    return line;
}

fs::path ResolvePath(const fs::path& config_path, const fs::path& path) {
    if (path.empty() || path.is_absolute()) {
        return path;
    }
    return config_path.parent_path() / path;
}

fs::path FindRepoRoot(const fs::path& start) {
    auto current = fs::absolute(start);
    if (fs::is_regular_file(current)) {
        current = current.parent_path();
    }
    for (int depth = 0; depth < 8 && !current.empty(); ++depth) {
        if (fs::exists(current / "CMakeLists.txt") && fs::exists(current / "src") && fs::exists(current / "tools")) {
            return current;
        }
        current = current.parent_path();
    }
    return fs::current_path();
}

std::optional<std::string> ReadEnv(const std::string& name) {
    if (name.empty()) {
        return std::nullopt;
    }
#ifdef _WIN32
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, name.c_str()) != 0 || value == nullptr) {
        return std::nullopt;
    }
    std::string out(value);
    free(value);
#else
    const char* value = std::getenv(name.c_str());
    if (!value) {
        return std::nullopt;
    }
    std::string out(value);
#endif
    return out.empty() ? std::nullopt : std::optional<std::string>(std::move(out));
}

std::string GetString(const Json& object, std::string_view name, std::string fallback = {}) {
    auto it = object.find(std::string(name));
    return it != object.end() && it->is_string() ? it->get<std::string>() : fallback;
}

int GetInt(const Json& object, std::string_view name, int fallback) {
    auto it = object.find(std::string(name));
    return it != object.end() && it->is_number_integer() ? it->get<int>() : fallback;
}

bool GetBool(const Json& object, std::string_view name, bool fallback) {
    auto it = object.find(std::string(name));
    return it != object.end() && it->is_boolean() ? it->get<bool>() : fallback;
}

std::size_t GetSize(const Json& object, std::string_view name, std::size_t fallback) {
    auto it = object.find(std::string(name));
    return it != object.end() && it->is_number_integer() ? it->get<std::size_t>() : fallback;
}

float GetFloat(const Json& object, std::string_view name, float fallback) {
    auto it = object.find(std::string(name));
    return it != object.end() && it->is_number() ? it->get<float>() : fallback;
}

struct BioDeleter {
    void operator()(BIO* bio) const noexcept { BIO_free(bio); }
};

struct PKeyDeleter {
    void operator()(EVP_PKEY* key) const noexcept { EVP_PKEY_free(key); }
};

struct GeneratedKeyPair {
    std::string private_key_pem;
    std::string public_key_pem;
};

GeneratedKeyPair GenerateRsaKeyPair() {
    std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)> ctx(
        EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr),
        EVP_PKEY_CTX_free);
    if (!ctx ||
        EVP_PKEY_keygen_init(ctx.get()) != 1 ||
        EVP_PKEY_CTX_set_rsa_keygen_bits(ctx.get(), 2048) != 1) {
        throw std::runtime_error("failed to initialize RSA key generator");
    }

    EVP_PKEY* raw_key = nullptr;
    if (EVP_PKEY_keygen(ctx.get(), &raw_key) != 1 || raw_key == nullptr) {
        throw std::runtime_error("failed to generate RSA key pair");
    }
    std::unique_ptr<EVP_PKEY, PKeyDeleter> key(raw_key);

    std::unique_ptr<BIO, BioDeleter> private_bio(BIO_new(BIO_s_mem()));
    std::unique_ptr<BIO, BioDeleter> public_bio(BIO_new(BIO_s_mem()));
    if (!private_bio || !public_bio) {
        throw std::runtime_error("failed to allocate OpenSSL BIO");
    }
    if (PEM_write_bio_PrivateKey(private_bio.get(), key.get(), nullptr, nullptr, 0, nullptr, nullptr) != 1) {
        throw std::runtime_error("failed to write private key PEM");
    }
    if (PEM_write_bio_PUBKEY(public_bio.get(), key.get()) != 1) {
        throw std::runtime_error("failed to write public key PEM");
    }

    BUF_MEM* private_mem = nullptr;
    BUF_MEM* public_mem = nullptr;
    BIO_get_mem_ptr(private_bio.get(), &private_mem);
    BIO_get_mem_ptr(public_bio.get(), &public_mem);
    if (!private_mem || !public_mem) {
        throw std::runtime_error("failed to read generated key PEM");
    }
    return {
        std::string(private_mem->data, private_mem->length),
        std::string(public_mem->data, public_mem->length),
    };
}

class NoopSemanticCache final : public agent::semantic_cache::ISemanticCache {
public:
    core::Result<agent::semantic_cache::CacheLookupResult> Lookup(
        const agent::semantic_cache::CacheLookupRequest&) override {
        agent::semantic_cache::CacheLookupResult result;
        result.hit = false;
        return result;
    }

    core::Status Store(const agent::semantic_cache::CacheStoreRequest&) override {
        return core::Status::Ok();
    }
};

class PlaceholderLlmClient final : public agent::llm::ILlmClient {
public:
    core::Result<agent::llm::ChatCompletionResponse> Complete(
        const agent::llm::ChatCompletionRequest& req) override {
        agent::llm::ChatCompletionResponse response;
        response.id = "placeholder-e2e";
        response.model = req.model.empty() ? "placeholder-e2e" : req.model;
        response.content =
            "This is a placeholder E2E response from agent_gateway_server. "
            "Configure llm.base_url and an API key to measure the real maximum-latency chain.";
        response.completion_tokens = 32;
        response.total_tokens = 32;
        return response;
    }
};

class DocumentEmbeddingProvider final : public agent::document::IDocumentEmbeddingProvider {
public:
    explicit DocumentEmbeddingProvider(std::shared_ptr<vector::EmbeddingPipeline> pipeline)
        : pipeline_(std::move(pipeline)) {}

    core::Result<std::vector<float>> EmbedText(std::string_view text) override {
        return pipeline_->Encode(text);
    }

private:
    std::shared_ptr<vector::EmbeddingPipeline> pipeline_;
};

struct L0MemoryCacheBundle {
    std::shared_ptr<agent::semantic_cache::ISemanticCache> cache;
    std::shared_ptr<agent::semantic_cache::RedisConnectionPool> redis_pool;
};

struct ToolConfig {
    fs::path config_path;
    fs::path repo_root;
    fs::path dump_dir;
    Json root;
    agent::service::gateway::PersonaGatewayServerOptions gateway;
    logging::LoggerOptions logging;
    agent::llm::OpenAiLlmClientOptions cloud_llm;
    bool cloud_llm_enabled = false;
    bool local_llm_enabled = false;
    bool allow_placeholder_llm = false;
    bool l0_enabled = false;
    fs::path tokenizer_path;
    fs::path embedding_model_path;
    fs::path emotion_tokenizer_path;
    std::string embedding_provider = "auto";
    int embedding_dimension = 384;
    std::string l0_redis_host = "127.0.0.1";
    int l0_redis_port = 5000;
    fs::path l0_sqlite_path;
    std::size_t l0_max_cached_records = 1000;
    std::size_t l0_top_k = 5;
    std::size_t l0_neighbors_per_hit = 1;
    float l0_similarity_floor = 0.78f;
    std::string l0_user_uuid = "e2e-l0";
    agent::llm::GrpcLocalLlmClientOptions local_llm;
    bool grpc_emotion_enabled = false;
    agent::service::persona::GrpcEmotionAnalyzerOptions grpc_emotion;
    agent::service::persona::EmotionFusionAnalyzerOptions emotion_fusion;
    bool emotion_fusion_enabled = true;
    std::vector<agent::service::persona::EmotionKeywordRule> emotion_keyword_rules;
    bool disable_tls_verify_on_windows = true;
    bool document_llm_chunk_cache_enabled = false;
    std::string document_llm_chunk_cache_redis_host = "127.0.0.1";
    int document_llm_chunk_cache_redis_port = 5000;
    std::string document_llm_chunk_cache_key_prefix = "agent:gateway:document:llm_chunk";
    int document_llm_chunk_cache_ttl_seconds = 7 * 24 * 60 * 60;
    std::size_t document_llm_chunk_cache_redis_pool_size = 4;
    bool document_semantic_cache_enabled = false;
    std::string document_semantic_cache_redis_host = "127.0.0.1";
    int document_semantic_cache_redis_port = 5000;
    fs::path document_semantic_cache_sqlite_path;
    std::string document_semantic_cache_user_uuid = "document-semantic-cache";
    std::size_t document_semantic_cache_max_cached_records = 10000;
    std::size_t document_semantic_cache_top_k = 6;
    float document_semantic_cache_similarity_floor = 0.94f;
    bool l3_enabled = false;
    fs::path l3_sqlite_path;
    std::string l3_collection_name = "l3_memory";
    std::string l3_embedding_fingerprint = "minilm-l6-v2";
    std::string l3_tokenizer_fingerprint = "minilm-l6-v2";
    std::string l3_corpus_version = "v1";
    std::string l3_policy_version = "v1";
    std::string l3_index_backend = "exact";
    std::size_t l3_max_resident_partitions = 16;
    int l3_max_records_per_batch = 1000;
    int l3_compression_max_tokens = 800;
    float l3_compression_temperature = 0.1f;
    fs::path l3_registry_sqlite_path;
    bool l3_flush_enabled = false;
    int l3_flush_interval_seconds = 300;
    int l3_flush_hour = 3;
    int l3_flush_minute = 0;
    int l3_flush_date_offset_days = 0;
    bool l3_flush_defer_when_sessions_active = true;
    std::vector<std::string> l3_user_uuids;
};

std::string ResolveApiKey(const fs::path& config_path, const Json& llm) {
    const auto env_name = GetString(llm, "api_key_env", "AGENT_LLM_API_KEY");
    if (auto key = ReadEnv(env_name)) {
        return *key;
    }

    const auto api_key_file = GetString(llm, "api_key_file");
    if (!api_key_file.empty()) {
        auto key_path = ResolvePath(config_path, api_key_file);
        auto key = ReadFirstLine(key_path);
        if (key.empty()) {
            throw std::runtime_error("API key file is empty: " + key_path.string());
        }
        return key;
    }
    return {};
}

std::vector<agent::service::persona::EmotionKeywordRule> ParseEmotionKeywordRules(const Json& emotion_fusion) {
    auto rules = agent::service::persona::DefaultEmotionKeywordRules();
    auto it = emotion_fusion.find("keyword_rules");
    if (it == emotion_fusion.end() || !it->is_object()) {
        return rules;
    }
    for (auto label = it->begin(); label != it->end(); ++label) {
        if (!label.value().is_array()) {
            continue;
        }
        for (const auto& item : label.value()) {
            if (item.is_string()) {
                rules.push_back(agent::service::persona::EmotionKeywordRule{
                    label.key(),
                    item.get<std::string>(),
                    1.0,
                });
                continue;
            }
            if (item.is_object()) {
                const auto pattern = GetString(item, "pattern");
                if (!pattern.empty()) {
                    rules.push_back(agent::service::persona::EmotionKeywordRule{
                        label.key(),
                        pattern,
                        GetFloat(item, "score", 1.0f),
                    });
                }
            }
        }
    }
    return rules;
}

ToolConfig LoadConfig(const fs::path& config_path) {
    std::ifstream file(config_path);
    if (!file) {
        throw std::runtime_error("cannot open config file: " + config_path.string());
    }

    ToolConfig config;
    config.config_path = config_path;
    config.repo_root = FindRepoRoot(config_path);
    config.dump_dir = config.repo_root / "dumps";
    config.root = Json::parse(file);

    const auto logging = config.root.value("logging", Json::object());
    config.logging.log_dir = ResolvePath(config_path, GetString(logging, "log_dir", "../logs/persona_gateway_e2e"));
    config.logging.logger_name = GetString(logging, "logger_name", "agent_gateway_server");
    config.logging.file_name = GetString(logging, "file_name", "agent_gateway_server.log");
    config.logging.enable_console = GetBool(logging, "enable_console", true);
    config.logging.use_daily_rotation = GetBool(logging, "use_daily_rotation", false);
    config.logging.max_file_size_bytes = static_cast<std::size_t>(GetInt(logging, "max_file_size_bytes", 10 * 1024 * 1024));
    config.logging.max_files = static_cast<std::size_t>(GetInt(logging, "max_files", 5));
    config.logging.module_names = {"gateway", "service", "classroom", "gateway-auth"};

    const auto gateway = config.root.value("persona_gateway", Json::object());
    config.gateway.http.address = GetString(gateway, "address", "127.0.0.1");
    config.gateway.http.port = static_cast<std::uint16_t>(GetInt(gateway, "port", 18080));
    config.gateway.http.io_threads = GetInt(gateway, "http_io_threads", 1);
    config.gateway.websocket_path = GetString(gateway, "websocket_path", "/ws/session");

    const auto compute_pool = gateway.value("compute_pool", Json::object());
    config.gateway.compute_pool.worker_count = GetSize(compute_pool, "worker_count", 2);
    config.gateway.compute_pool.queue_capacity = GetSize(compute_pool, "queue_capacity", 256);

    const auto io_pool = gateway.value("io_pool", Json::object());
    config.gateway.io_pool.worker_count = GetSize(io_pool, "worker_count", 2);
    config.gateway.io_pool.queue_capacity = GetSize(io_pool, "queue_capacity", 256);

    config.gateway.session.idle_timeout = std::chrono::minutes(GetInt(gateway, "session_idle_timeout_minutes", 15));
    config.gateway.session.max_recent_turns = GetSize(gateway, "session_max_recent_turns", 20);
    config.gateway.runtime.recent_raw_turns = GetSize(gateway, "runtime_recent_raw_turns", 10);
    config.gateway.runtime.default_model = GetString(gateway, "runtime_default_model");

    const auto request_filter = gateway.value("request_filter", Json::object());
    config.gateway.http.request_filter.enabled = GetBool(request_filter, "enabled", true);
    config.gateway.http.request_filter.reject_control_chars = GetBool(request_filter, "reject_control_chars", true);
    config.gateway.http.request_filter.reject_suspicious_patterns = GetBool(request_filter, "reject_suspicious_patterns", true);

    const auto static_files = gateway.value("static_files", Json::object());
    if (GetBool(static_files, "enabled", true)) {
        net::StaticFileOptions files;
        files.root = ResolvePath(config_path, GetString(static_files, "root", "../dist"));
        files.index_file = GetString(static_files, "index_file", "index.html");
        files.spa_fallback = GetBool(static_files, "spa_fallback", true);
        config.gateway.static_files = files;
    }

    const auto auth = config.root.value("gateway_auth", Json::object());
    config.gateway.auth.enabled = GetBool(auth, "enabled", true);
    config.gateway.auth.allow_dev_identity = GetBool(auth, "allow_dev_identity", false);
    config.gateway.auth.require_auth_for_api = GetBool(auth, "require_auth_for_api", true);
    config.gateway.auth.cookie_name = GetString(auth, "cookie_name", "agent_auth");
    config.gateway.auth.issuer = GetString(auth, "issuer", "agent-e2e");
    config.gateway.auth.audience = GetString(auth, "audience", "agent-gateway");
    config.gateway.auth.clock_skew = std::chrono::seconds(GetInt(auth, "clock_skew_seconds", 60));
    config.gateway.auth.token_ttl = std::chrono::seconds(GetInt(auth, "token_ttl_seconds", 28800));
    config.gateway.auth.cookie_http_only = GetBool(auth, "cookie_http_only", true);
    config.gateway.auth.cookie_secure = GetBool(auth, "cookie_secure", false);
    config.gateway.auth.cookie_same_site = GetString(auth, "cookie_same_site", "Lax");
    config.gateway.auth.require_session_record = GetBool(auth, "require_session_record", false);
    config.gateway.auth.auto_provision_session = GetBool(auth, "auto_provision_session", true);
    config.gateway.auth.session_store_backend = GetString(auth, "session_store_backend", "sqlite");

    const auto session_db = GetString(auth, "session_database_path", "../tmp/persona_gateway_e2e_auth.db");
    config.gateway.auth.session_database_path = ResolvePath(config_path, session_db).string();
    config.gateway.auth.redis_host = GetString(auth, "redis_host", "127.0.0.1");
    config.gateway.auth.redis_port = GetString(auth, "redis_port", "5000");
    config.gateway.auth.redis_password = GetString(auth, "redis_password");
    config.gateway.auth.redis_pool_size = GetSize(auth, "redis_pool_size", 16);
    config.gateway.auth.redis_command_timeout = std::chrono::milliseconds(GetInt(auth, "redis_command_timeout_ms", 5000));
    config.gateway.auth.redis_key_prefix = GetString(auth, "redis_key_prefix", "agent:e2e:gateway:auth");

    config.gateway.auth.public_key_pem = GetString(auth, "public_key_pem");
    config.gateway.auth.private_key_pem = GetString(auth, "private_key_pem");
    const auto public_key_file = GetString(auth, "public_key_file");
    const auto private_key_file = GetString(auth, "private_key_file");
    if (!public_key_file.empty()) {
        config.gateway.auth.public_key_pem = ReadTextFile(ResolvePath(config_path, public_key_file));
    }
    if (!private_key_file.empty()) {
        config.gateway.auth.private_key_pem = ReadTextFile(ResolvePath(config_path, private_key_file));
    }
    if (config.gateway.auth.enabled &&
        (config.gateway.auth.public_key_pem.empty() || config.gateway.auth.private_key_pem.empty()) &&
        GetBool(auth, "generate_dev_keys", true)) {
        auto keys = GenerateRsaKeyPair();
        config.gateway.auth.private_key_pem = std::move(keys.private_key_pem);
        config.gateway.auth.public_key_pem = std::move(keys.public_key_pem);
    }

    const auto llm = config.root.value("llm", Json::object());
    config.cloud_llm_enabled = GetBool(llm, "enabled", true);
    config.allow_placeholder_llm = GetBool(llm, "allow_placeholder", false);
    config.disable_tls_verify_on_windows = GetBool(llm, "disable_tls_verify_on_windows", true);
    config.cloud_llm.base_url = GetString(llm, "base_url");
    config.cloud_llm.default_model = GetString(llm, "model", "deepseek-chat");
    config.cloud_llm.timeout_ms = GetInt(llm, "timeout_ms", 30000);
    config.cloud_llm.retry_policy.max_retries = GetInt(llm, "max_retries", 2);
    config.cloud_llm.api_key = ResolveApiKey(config_path, llm);
    config.cloud_llm.require_api_key = GetBool(llm, "require_api_key", true);
    if (config.gateway.runtime.default_model.empty()) {
        config.gateway.runtime.default_model = config.cloud_llm.default_model;
    }

    const auto local = config.root.value("local_llm", Json::object());
    config.local_llm_enabled = GetBool(local, "enabled", false);
    config.local_llm.target = GetString(local, "target", "127.0.0.1:50051");
    config.local_llm.deadline = std::chrono::milliseconds(GetInt(local, "deadline_ms", 30000));
    config.local_llm.auth_token = GetString(local, "auth_token");
    config.local_llm.auth_metadata_key = GetString(local, "auth_metadata_key", "authorization");

    const auto emotion = config.root.value("emotion_analyzer", Json::object());
    const auto emotion_backend = GetString(emotion, "backend", "neutral");
    config.grpc_emotion_enabled = GetBool(emotion, "enabled", emotion_backend == "grpc" || emotion_backend == "grpc_multimodal");
    config.grpc_emotion.target = GetString(emotion, "target", "127.0.0.1:50051");
    config.grpc_emotion.deadline = std::chrono::milliseconds(GetInt(emotion, "deadline_ms", 3000));
    config.grpc_emotion.auth_token = GetString(emotion, "auth_token");
    config.grpc_emotion.auth_metadata_key = GetString(emotion, "auth_metadata_key", "authorization");
    config.grpc_emotion.tokenizer_options.max_length = GetSize(emotion, "max_length", 128);
    config.grpc_emotion.tokenizer_options.truncation = GetBool(emotion, "truncation", true);
    config.grpc_emotion.tokenizer_options.padding = GetBool(emotion, "padding", true);
    config.grpc_emotion.tokenizer_options.pad_to_longest_in_batch = false;
    config.grpc_emotion.tokenizer_options.add_special_tokens = GetBool(emotion, "add_special_tokens", true);
    config.emotion_tokenizer_path = ResolvePath(
        config_path,
        GetString(emotion, "tokenizer_path", config.tokenizer_path.string()));

    const auto emotion_fusion = config.root.value("emotion_fusion", Json::object());
    config.emotion_fusion_enabled = GetBool(emotion_fusion, "enabled", true);
    config.emotion_fusion.enabled = config.emotion_fusion_enabled;
    config.emotion_fusion.bert_weight = GetFloat(emotion_fusion, "bert_weight", static_cast<float>(config.emotion_fusion.bert_weight));
    config.emotion_fusion.default_reliability =
        GetFloat(emotion_fusion, "default_reliability", static_cast<float>(config.emotion_fusion.default_reliability));
    config.emotion_fusion.accept_confidence =
        GetFloat(emotion_fusion, "accept_confidence", static_cast<float>(config.emotion_fusion.accept_confidence));
    config.emotion_fusion.ambiguity_margin =
        GetFloat(emotion_fusion, "ambiguity_margin", static_cast<float>(config.emotion_fusion.ambiguity_margin));
    config.emotion_fusion.head_bias =
        GetFloat(emotion_fusion, "head_bias", static_cast<float>(config.emotion_fusion.head_bias));
    config.emotion_fusion.bert_signal_weight =
        GetFloat(emotion_fusion, "bert_signal_weight", static_cast<float>(config.emotion_fusion.bert_signal_weight));
    const auto evidence_weight =
        GetFloat(emotion_fusion, "evidence_signal_weight", static_cast<float>(config.emotion_fusion.keyword_signal_weight));
    config.emotion_fusion.keyword_signal_weight =
        GetFloat(emotion_fusion, "keyword_signal_weight", evidence_weight);
    config.emotion_fusion.vector_signal_weight =
        GetFloat(emotion_fusion, "vector_signal_weight", evidence_weight);
    config.emotion_fusion.llm_signal_weight =
        GetFloat(emotion_fusion, "llm_signal_weight", static_cast<float>(config.emotion_fusion.llm_signal_weight));
    config.emotion_fusion.margin_signal_weight =
        GetFloat(emotion_fusion, "margin_signal_weight", static_cast<float>(config.emotion_fusion.margin_signal_weight));
    config.emotion_fusion.llm_gate_confidence =
        GetFloat(emotion_fusion, "llm_gate_confidence", static_cast<float>(config.emotion_fusion.llm_gate_confidence));
    config.emotion_fusion.llm_gate_min_delta =
        GetFloat(emotion_fusion, "llm_gate_min_delta", static_cast<float>(config.emotion_fusion.llm_gate_min_delta));
    if (auto it = emotion_fusion.find("label_reliability"); it != emotion_fusion.end() && it->is_object()) {
        for (auto label = it->begin(); label != it->end(); ++label) {
            if (label.value().is_number()) {
                config.emotion_fusion.label_reliability[label.key()] = label.value().get<double>();
            }
        }
    }
    if (auto it = emotion_fusion.find("source_weights"); it != emotion_fusion.end() && it->is_object()) {
        for (auto source = it->begin(); source != it->end(); ++source) {
            if (source.value().is_number()) {
                config.emotion_fusion.source_weights[source.key()] = source.value().get<double>();
            }
        }
    }
    config.emotion_keyword_rules = ParseEmotionKeywordRules(emotion_fusion);

    const auto embedding = config.root.value("embedding", Json::object());
    config.tokenizer_path = ResolvePath(config_path, GetString(embedding, "tokenizer_path", "../onnx_models/minilm/tokenizer.json"));
    config.embedding_model_path = ResolvePath(config_path, GetString(embedding, "model_path", "../onnx_models/minilm/model.onnx"));
    config.embedding_provider = GetString(embedding, "execution_provider", "auto");
    config.embedding_dimension = GetInt(embedding, "dimension", 384);

    const auto l0 = config.root.value("l0_memory", Json::object());
    config.l0_enabled = GetBool(l0, "enabled", true);
    config.l0_redis_host = GetString(l0, "redis_host", "127.0.0.1");
    config.l0_redis_port = GetInt(l0, "redis_port", 5000);
    config.l0_sqlite_path = ResolvePath(config_path, GetString(l0, "sqlite_path", "../data/persona_gateway_e2e/l0_memory.db"));
    config.l0_max_cached_records = GetSize(l0, "max_cached_records", 1000);
    config.l0_top_k = GetSize(l0, "top_k", 5);
    config.l0_neighbors_per_hit = GetSize(l0, "neighbors_per_hit", 1);
    auto similarity = l0.find("similarity_floor");
    if (similarity != l0.end() && similarity->is_number()) {
        config.l0_similarity_floor = similarity->get<float>();
    }
    config.l0_user_uuid = GetString(l0, "user_uuid", "e2e-l0");
    if (config.l0_user_uuid.empty()) {
        config.l0_user_uuid = "gateway-l0";
    }

    const auto document_store = config.root.value("document_store", Json::object());
    config.gateway.document_store.enabled = GetBool(document_store, "enabled", false);
    config.gateway.document_store.root = ResolvePath(config_path, GetString(document_store, "root", "../data/agent_gateway/documents"));
    config.gateway.document_store.database_path = ResolvePath(config_path, GetString(document_store, "database_path", "../data/agent_gateway/document_store.db"));
    config.gateway.document_store.read_connection_count = GetSize(document_store, "read_connection_count", 2);
    config.gateway.document_store.write_connection_count = GetSize(document_store, "write_connection_count", 1);
    config.gateway.document_store.busy_timeout_ms = GetInt(document_store, "busy_timeout_ms", 5000);
    config.gateway.document_store.retention_hours = GetInt(document_store, "retention_hours", 24 * 7);
    config.gateway.document_store.cleanup_interval_seconds = GetInt(document_store, "cleanup_interval_seconds", 60);

    const auto document_llm_cache = config.root.value("document_llm_chunk_cache", Json::object());
    config.document_llm_chunk_cache_enabled = GetBool(document_llm_cache, "enabled", false);
    config.document_llm_chunk_cache_redis_host =
        GetString(document_llm_cache, "redis_host", config.l0_redis_host);
    config.document_llm_chunk_cache_redis_port =
        GetInt(document_llm_cache, "redis_port", config.l0_redis_port);
    config.document_llm_chunk_cache_key_prefix =
        GetString(document_llm_cache, "key_prefix", "agent:gateway:document:llm_chunk");
    config.document_llm_chunk_cache_ttl_seconds =
        GetInt(document_llm_cache, "ttl_seconds", 7 * 24 * 60 * 60);
    config.document_llm_chunk_cache_redis_pool_size =
        GetSize(document_llm_cache, "redis_pool_size", 4);

    const auto document_semantic = config.root.value("document_semantic_cache", Json::object());
    config.document_semantic_cache_enabled = GetBool(document_semantic, "enabled", false);
    config.document_semantic_cache_redis_host =
        GetString(document_semantic, "redis_host", config.l0_redis_host);
    config.document_semantic_cache_redis_port =
        GetInt(document_semantic, "redis_port", config.l0_redis_port);
    config.document_semantic_cache_sqlite_path =
        ResolvePath(config_path, GetString(document_semantic, "sqlite_path", "../data/agent_gateway/document_semantic_cache.db"));
    config.document_semantic_cache_user_uuid =
        GetString(document_semantic, "user_uuid", "document-semantic-cache");
    config.document_semantic_cache_max_cached_records =
        GetSize(document_semantic, "max_cached_records", 10000);
    config.document_semantic_cache_top_k = GetSize(document_semantic, "top_k", 6);
    config.document_semantic_cache_similarity_floor =
        GetFloat(document_semantic, "similarity_floor", 0.94f);

    const auto l3 = config.root.value("l3_memory", Json::object());
    config.l3_enabled = GetBool(l3, "enabled", false);
    config.l3_sqlite_path = ResolvePath(config_path, GetString(l3, "sqlite_path", "../data/agent_gateway/l3_memory.db"));
    config.l3_collection_name = GetString(l3, "collection_name", "l3_memory");
    config.l3_embedding_fingerprint = GetString(l3, "embedding_fingerprint", "minilm-l6-v2");
    config.l3_tokenizer_fingerprint = GetString(l3, "tokenizer_fingerprint", "minilm-l6-v2");
    config.l3_corpus_version = GetString(l3, "corpus_version", "v1");
    config.l3_policy_version = GetString(l3, "policy_version", "v1");
    config.l3_index_backend = GetString(l3, "index_backend", "exact");
    config.l3_max_resident_partitions = GetSize(l3, "max_resident_partitions", 16);
    config.l3_max_records_per_batch = GetInt(l3, "max_records_per_batch", 1000);
    config.l3_compression_max_tokens = GetInt(l3, "compression_max_tokens", 800);
    config.l3_compression_temperature = GetFloat(l3, "compression_temperature", 0.1f);
    config.l3_registry_sqlite_path =
        ResolvePath(config_path, GetString(l3, "registry_sqlite_path", "../data/agent_gateway/l3_registry.db"));
    if (auto it = l3.find("user_uuids"); it != l3.end() && it->is_array()) {
        for (const auto& item : *it) {
            if (item.is_string() && !item.get<std::string>().empty()) {
                config.l3_user_uuids.push_back(item.get<std::string>());
            }
        }
    }
    if (!config.l0_user_uuid.empty()) {
        config.l3_user_uuids.push_back(config.l0_user_uuid);
    }
    std::unordered_set<std::string> seen_users;
    std::vector<std::string> unique_users;
    for (auto& user : config.l3_user_uuids) {
        if (seen_users.insert(user).second) {
            unique_users.push_back(std::move(user));
        }
    }
    config.l3_user_uuids = std::move(unique_users);

    const auto l3_flush = config.root.value("l3_flush_scheduler", Json::object());
    config.l3_flush_enabled = GetBool(l3_flush, "enabled", config.l3_enabled);
    config.l3_flush_interval_seconds = GetInt(l3_flush, "check_interval_seconds", 300);
    config.l3_flush_hour = GetInt(l3_flush, "flush_hour", 3);
    config.l3_flush_minute = GetInt(l3_flush, "flush_minute", 0);
    config.l3_flush_date_offset_days = GetInt(l3_flush, "flush_date_offset_days", 0);
    config.l3_flush_defer_when_sessions_active =
        GetBool(l3_flush, "defer_when_sessions_active", true);

    return config;
}

core::Result<std::shared_ptr<agent::llm::ILlmClient>> CreateLlmClient(const ToolConfig& config) {
    std::shared_ptr<agent::llm::ILlmClient> primary;
    std::shared_ptr<agent::llm::ILlmClient> fallback;

    if (config.cloud_llm_enabled && !config.cloud_llm.base_url.empty()) {
        agent::net::TlsClientOptions tls_opts;
#ifdef _WIN32
        if (config.disable_tls_verify_on_windows) {
            tls_opts.verify_mode = agent::net::TlsVerifyMode::None;
        }
#endif
        auto tls = agent::net::TlsContext::CreateClient(tls_opts);
        if (!tls.ok()) {
            return tls.status();
        }
        agent::net::BeastHttpClientOptions http_opts;
        http_opts.tls_context = std::move(tls).value();
        auto http = agent::net::BeastHttpClient::Create(std::move(http_opts));
        if (!http.ok()) {
            return http.status();
        }
        auto http_client = std::shared_ptr<agent::net::IHttpClient>(std::move(http).value());

        auto cloud = agent::llm::OpenAiLlmClient::Create(config.cloud_llm, *http_client);
        if (!cloud.ok()) {
            return cloud.status();
        }
        struct ClientWithTransport final : public agent::llm::ILlmClient {
            std::shared_ptr<agent::net::IHttpClient> transport;
            std::unique_ptr<agent::llm::OpenAiLlmClient> client;
            core::Result<agent::llm::ChatCompletionResponse> Complete(
                const agent::llm::ChatCompletionRequest& req) override {
                return client->Complete(req);
            }
        };
        auto holder = std::make_shared<ClientWithTransport>();
        holder->transport = std::move(http_client);
        holder->client = std::move(cloud).value();
        primary = holder;
    }

    if (config.local_llm_enabled) {
        auto local_llm = std::make_shared<agent::llm::GrpcLocalLlmClient>(config.local_llm);
        agent::llm::LocalLlmChatClientOptions local_options;
        local_options.default_model = config.gateway.runtime.default_model.empty()
            ? "local-llm"
            : config.gateway.runtime.default_model;
        fallback = std::make_shared<agent::llm::LocalLlmChatClient>(std::move(local_llm), local_options);
    }

    if (primary && fallback) {
        return std::shared_ptr<agent::llm::ILlmClient>(
            std::make_shared<agent::llm::FallbackLlmClient>(primary, fallback));
    }
    if (primary) {
        return primary;
    }
    if (fallback) {
        return fallback;
    }
    if (config.allow_placeholder_llm) {
        return std::shared_ptr<agent::llm::ILlmClient>(std::make_shared<PlaceholderLlmClient>());
    }
    return core::Status::Error(core::ErrorCode::FailedPrecondition, "no LLM client configured");
}

core::Result<std::shared_ptr<agent::service::persona::IEmotionAnalyzer>> CreateEmotionAnalyzer(const ToolConfig& config) {
    if (!config.grpc_emotion_enabled) {
        return std::shared_ptr<agent::service::persona::IEmotionAnalyzer>(
            std::make_shared<agent::service::persona::NeutralEmotionAnalyzer>());
    }
    if (!fs::exists(config.emotion_tokenizer_path)) {
        return core::Status::Error(core::ErrorCode::NotFound, "emotion tokenizer not found: " + config.emotion_tokenizer_path.string());
    }

    auto tokenizer = vector::HfTokenizer::LoadFromFile(config.emotion_tokenizer_path);
    if (!tokenizer.ok()) {
        return tokenizer.status();
    }
    auto tokenizer_ptr = std::make_shared<vector::HfTokenizer>(std::move(tokenizer).value());
    std::shared_ptr<agent::service::persona::IEmotionAnalyzer> grpc =
        std::make_shared<agent::service::persona::GrpcEmotionAnalyzer>(
            config.grpc_emotion,
            std::move(tokenizer_ptr));
    if (!config.emotion_fusion_enabled) {
        return grpc;
    }
    std::vector<std::shared_ptr<agent::service::persona::IEmotionEvidenceProvider>> providers;
    providers.push_back(std::make_shared<agent::service::persona::KeywordEmotionEvidenceProvider>(
        config.emotion_keyword_rules));
    return std::shared_ptr<agent::service::persona::IEmotionAnalyzer>(
        std::make_shared<agent::service::persona::FusedEmotionAnalyzer>(
            std::move(grpc),
            config.emotion_fusion,
            std::move(providers)));
}

core::Result<L0MemoryCacheBundle> CreateL0MemoryCache(const ToolConfig& config) {
    if (!config.l0_enabled) {
        L0MemoryCacheBundle bundle;
        bundle.cache = std::make_shared<NoopSemanticCache>();
        return bundle;
    }
    if (!fs::exists(config.tokenizer_path)) {
        return core::Status::Error(core::ErrorCode::NotFound, "tokenizer not found: " + config.tokenizer_path.string());
    }
    if (!fs::exists(config.embedding_model_path)) {
        return core::Status::Error(core::ErrorCode::NotFound, "embedding model not found: " + config.embedding_model_path.string());
    }

    auto tokenizer = vector::HfTokenizer::LoadFromFile(config.tokenizer_path);
    if (!tokenizer.ok()) {
        return tokenizer.status();
    }
    auto tokenizer_ptr = std::make_shared<vector::HfTokenizer>(std::move(tokenizer).value());

    vector::EmbeddingModelOptions model_options;
    model_options.model_path = config.embedding_model_path;
    model_options.execution_provider = config.embedding_provider;
    model_options.allow_cpu_fallback = true;
    model_options.expected_dimension = static_cast<std::size_t>(config.embedding_dimension);
    model_options.pooling = vector::PoolingStrategy::Mean;
    model_options.normalize = true;
    auto model = vector::OnnxTextEmbeddingModel::Load(model_options);
    if (!model.ok()) {
        return model.status();
    }
    std::shared_ptr<vector::IEmbeddingModel> model_ptr(std::move(model).value());
    auto embedding = std::make_shared<vector::EmbeddingPipeline>(std::move(tokenizer_ptr), std::move(model_ptr));

    agent::semantic_cache::RedisPoolOptions redis_options;
    redis_options.host = config.l0_redis_host;
    redis_options.port = std::to_string(config.l0_redis_port);
    redis_options.pool_size = 4;
    auto redis = std::make_shared<agent::semantic_cache::RedisConnectionPool>(redis_options);
    auto redis_start = redis->Start();
    if (!redis_start.ok()) {
        return redis_start;
    }

    fs::create_directories(config.l0_sqlite_path.parent_path());
    auto sqlite = storage::sqlite::SqliteConnection::Open(config.l0_sqlite_path.string());
    if (!sqlite.ok()) {
        redis->Shutdown();
        return sqlite.status();
    }

    auto index = std::make_shared<agent::semantic_cache::cache_vector::VectorIndexManager>(
        config.l0_user_uuid,
        redis,
        std::move(sqlite).value(),
        config.l0_max_cached_records);

    agent::semantic_cache::L0MemoryCacheAdapterOptions options;
    options.top_k = config.l0_top_k;
    options.neighbors_per_hit = config.l0_neighbors_per_hit;
    options.similarity_floor = config.l0_similarity_floor;
    L0MemoryCacheBundle bundle;
    bundle.redis_pool = redis;
    bundle.cache = std::make_shared<agent::semantic_cache::L0MemoryCacheAdapter>(
            std::move(embedding),
            std::move(index),
            options);
    return bundle;
}

core::Result<std::shared_ptr<vector::EmbeddingPipeline>> CreateEmbeddingPipeline(const ToolConfig& config) {
    if (!fs::exists(config.tokenizer_path)) {
        return core::Status::Error(core::ErrorCode::NotFound, "tokenizer not found: " + config.tokenizer_path.string());
    }
    if (!fs::exists(config.embedding_model_path)) {
        return core::Status::Error(core::ErrorCode::NotFound, "embedding model not found: " + config.embedding_model_path.string());
    }

    auto tokenizer = vector::HfTokenizer::LoadFromFile(config.tokenizer_path);
    if (!tokenizer.ok()) {
        return tokenizer.status();
    }
    auto tokenizer_ptr = std::make_shared<vector::HfTokenizer>(std::move(tokenizer).value());

    vector::EmbeddingModelOptions model_options;
    model_options.model_path = config.embedding_model_path;
    model_options.execution_provider = config.embedding_provider;
    model_options.allow_cpu_fallback = true;
    model_options.expected_dimension = static_cast<std::size_t>(config.embedding_dimension);
    model_options.pooling = vector::PoolingStrategy::Mean;
    model_options.normalize = true;
    auto model = vector::OnnxTextEmbeddingModel::Load(model_options);
    if (!model.ok()) {
        return model.status();
    }
    std::shared_ptr<vector::IEmbeddingModel> model_ptr(std::move(model).value());
    return std::make_shared<vector::EmbeddingPipeline>(std::move(tokenizer_ptr), std::move(model_ptr));
}

core::Result<std::shared_ptr<agent::document::IDocumentEmbeddingProvider>> CreateDocumentEmbeddingProvider(
    const ToolConfig& config) {
    auto pipeline = CreateEmbeddingPipeline(config);
    if (!pipeline.ok()) {
        return pipeline.status();
    }
    return std::shared_ptr<agent::document::IDocumentEmbeddingProvider>(
        std::make_shared<DocumentEmbeddingProvider>(std::move(pipeline).value()));
}

core::Result<std::shared_ptr<agent::document::IDocumentLlmChunkCache>> CreateDocumentLlmChunkCache(
    const ToolConfig& config) {
    if (!config.document_llm_chunk_cache_enabled) {
        return std::shared_ptr<agent::document::IDocumentLlmChunkCache>{};
    }

    agent::semantic_cache::RedisPoolOptions redis_options;
    redis_options.host = config.document_llm_chunk_cache_redis_host;
    redis_options.port = std::to_string(config.document_llm_chunk_cache_redis_port);
    redis_options.pool_size = config.document_llm_chunk_cache_redis_pool_size;
    auto redis = std::make_shared<agent::semantic_cache::RedisConnectionPool>(redis_options);
    auto status = redis->Start();
    if (!status.ok()) {
        return status;
    }

    agent::document::RedisDocumentLlmChunkCacheOptions options;
    options.key_prefix = config.document_llm_chunk_cache_key_prefix;
    options.ttl = std::chrono::seconds(config.document_llm_chunk_cache_ttl_seconds);
    return std::shared_ptr<agent::document::IDocumentLlmChunkCache>(
        std::make_shared<agent::document::RedisDocumentLlmChunkCache>(std::move(redis), std::move(options)));
}

core::Result<std::shared_ptr<agent::semantic_cache::ISemanticCache>> CreateDocumentSemanticCache(
    const ToolConfig& config) {
    if (!config.document_semantic_cache_enabled) {
        return std::shared_ptr<agent::semantic_cache::ISemanticCache>{};
    }
    if (!fs::exists(config.tokenizer_path)) {
        return core::Status::Error(core::ErrorCode::NotFound, "tokenizer not found: " + config.tokenizer_path.string());
    }
    if (!fs::exists(config.embedding_model_path)) {
        return core::Status::Error(core::ErrorCode::NotFound, "embedding model not found: " + config.embedding_model_path.string());
    }

    auto tokenizer = vector::HfTokenizer::LoadFromFile(config.tokenizer_path);
    if (!tokenizer.ok()) {
        return tokenizer.status();
    }
    auto tokenizer_ptr = std::make_shared<vector::HfTokenizer>(std::move(tokenizer).value());

    vector::EmbeddingModelOptions model_options;
    model_options.model_path = config.embedding_model_path;
    model_options.execution_provider = config.embedding_provider;
    model_options.allow_cpu_fallback = true;
    model_options.expected_dimension = static_cast<std::size_t>(config.embedding_dimension);
    model_options.pooling = vector::PoolingStrategy::Mean;
    model_options.normalize = true;
    auto model = vector::OnnxTextEmbeddingModel::Load(model_options);
    if (!model.ok()) {
        return model.status();
    }
    std::shared_ptr<vector::OnnxTextEmbeddingModel> model_ptr(std::move(model).value());

    agent::semantic_cache::RedisPoolOptions redis_options;
    redis_options.host = config.document_semantic_cache_redis_host;
    redis_options.port = std::to_string(config.document_semantic_cache_redis_port);
    redis_options.pool_size = 4;
    auto redis = std::make_shared<agent::semantic_cache::RedisConnectionPool>(redis_options);
    auto redis_start = redis->Start();
    if (!redis_start.ok()) {
        return redis_start;
    }

    fs::create_directories(config.document_semantic_cache_sqlite_path.parent_path());
    auto sqlite = storage::sqlite::SqliteConnection::Open(config.document_semantic_cache_sqlite_path.string());
    if (!sqlite.ok()) {
        redis->Shutdown();
        return sqlite.status();
    }

    auto index = std::make_shared<agent::semantic_cache::cache_vector::VectorIndexManager>(
        config.document_semantic_cache_user_uuid,
        redis,
        std::move(sqlite).value(),
        config.document_semantic_cache_max_cached_records);

    agent::semantic_cache::SemanticCachePipelineDeps deps;
    deps.tokenizer = std::move(tokenizer_ptr);
    deps.embedding_model = std::move(model_ptr);
    deps.index_manager = std::move(index);

    agent::semantic_cache::SemanticCachePipelineOptions options;
    options.top_k = config.document_semantic_cache_top_k;
    options.similarity_floor = config.document_semantic_cache_similarity_floor;
    options.enable_global_scope = true;

    auto pipeline = agent::semantic_cache::SemanticCachePipeline::Create(std::move(options), std::move(deps));
    if (!pipeline.ok()) {
        redis->Shutdown();
        return pipeline.status();
    }
    return std::shared_ptr<agent::semantic_cache::ISemanticCache>(std::move(pipeline).value());
}

core::Result<std::shared_ptr<agent::memory::LongTermMemoryCompressor>> CreateL3MemoryCompressor(
    const ToolConfig& config,
    std::shared_ptr<agent::llm::ILlmClient> llm_client) {
    if (!config.l3_enabled) {
        return std::shared_ptr<agent::memory::LongTermMemoryCompressor>{};
    }

    agent::semantic_cache::RedisPoolOptions redis_options;
    redis_options.host = config.l0_redis_host;
    redis_options.port = std::to_string(config.l0_redis_port);
    redis_options.pool_size = 4;
    auto redis = std::make_shared<agent::semantic_cache::RedisConnectionPool>(redis_options);
    auto redis_start = redis->Start();
    if (!redis_start.ok()) {
        return redis_start;
    }

    fs::create_directories(config.l3_sqlite_path.parent_path());
    storage::sqlite::SqliteConnectionPoolOptions pool_options;
    pool_options.path = config.l3_sqlite_path.string();
    pool_options.read_connection_count = 4;
    pool_options.write_connection_count = 1;
    pool_options.enable_wal = true;
    auto sqlite_pool = std::make_shared<storage::sqlite::SqliteConnectionPool>(pool_options);
    auto sqlite_start = sqlite_pool->Start();
    if (!sqlite_start.ok()) {
        redis->Shutdown();
        return sqlite_start;
    }

    auto vector_repo = std::make_shared<agent::vector_storage::SqliteVectorRepository>(sqlite_pool);
    auto schema_status = vector_repo->EnsureSchema();
    if (!schema_status.ok()) {
        sqlite_pool->Close();
        redis->Shutdown();
        return schema_status;
    }

    agent::vector_storage::CollectionDescriptor collection;
    collection.name = config.l3_collection_name;
    collection.embedding_model_fingerprint = config.l3_embedding_fingerprint;
    collection.tokenizer_fingerprint = config.l3_tokenizer_fingerprint;
    collection.pooling_strategy = "mean";
    collection.normalization = "l2";
    collection.dimension = static_cast<std::size_t>(config.embedding_dimension);
    collection.corpus_version = config.l3_corpus_version;
    collection.policy_version = config.l3_policy_version;
    auto collection_id = vector_repo->EnsureCollection(collection);
    if (!collection_id.ok()) {
        sqlite_pool->Close();
        redis->Shutdown();
        return collection_id.status();
    }

    auto partition_registry = std::make_shared<agent::vector_storage::PartitionRegistry>(vector_repo);

    fs::create_directories(config.l3_registry_sqlite_path.parent_path());
    storage::sqlite::SqliteConnectionPoolOptions registry_pool_options;
    registry_pool_options.path = config.l3_registry_sqlite_path.string();
    registry_pool_options.read_connection_count = 2;
    registry_pool_options.write_connection_count = 1;
    registry_pool_options.enable_wal = true;
    auto registry_pool = std::make_shared<storage::sqlite::SqliteConnectionPool>(registry_pool_options);
    auto registry_start = registry_pool->Start();
    if (!registry_start.ok()) {
        sqlite_pool->Close();
        redis->Shutdown();
        return registry_start;
    }

    agent::vector::IndexManagerOptions index_options;
    index_options.max_resident_partitions = config.l3_max_resident_partitions;
    index_options.backend = config.l3_index_backend;
    index_options.log_hydration = false;
    auto index_manager = std::make_shared<agent::vector::VectorIndexManager>(
        vector_repo,
        partition_registry,
        static_cast<std::size_t>(config.embedding_dimension),
        index_options);

    auto embedding_pipeline = CreateEmbeddingPipeline(config);
    if (!embedding_pipeline.ok()) {
        sqlite_pool->Close();
        redis->Shutdown();
        return embedding_pipeline.status();
    }

    agent::memory::LongTermMemoryCompressorOptions compressor_options;
    compressor_options.redis_pool = std::move(redis);
    compressor_options.llm_client = std::move(llm_client);
    compressor_options.vector_repo = std::move(vector_repo);
    compressor_options.partition_registry = std::move(partition_registry);
    compressor_options.index_manager = std::move(index_manager);
    compressor_options.embedding_pipeline = std::move(embedding_pipeline).value();
    compressor_options.collection_id = collection_id.value();
    compressor_options.mode = agent::memory::CompressorMode::Production;
    compressor_options.max_records_per_batch = config.l3_max_records_per_batch;
    compressor_options.compression_temperature = config.l3_compression_temperature;
    compressor_options.compression_max_tokens = config.l3_compression_max_tokens;
    compressor_options.registry_pool = std::move(registry_pool);

    auto compressor = agent::memory::LongTermMemoryCompressor::Create(std::move(compressor_options));
    if (!compressor.ok()) {
        sqlite_pool->Close();
        return compressor.status();
    }
    return std::shared_ptr<agent::memory::LongTermMemoryCompressor>(std::move(compressor).value());
}

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, OnSignal);
    std::signal(SIGTERM, OnSignal);
    bool logging_initialized = false;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json> [--no-stdin-stop]\n";
        return 2;
    }
    bool stop_on_stdin = true;
    for (int i = 2; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--no-stdin-stop") {
            stop_on_stdin = false;
        }
    }

    try {
        const fs::path config_path = argv[1];
        const auto repo_root = FindRepoRoot(config_path);
        const auto dump_dir = repo_root / "dumps";
        fs::create_directories(dump_dir);
#ifdef _WIN32
        crash_dump::Install(dump_dir.string());
#endif

        auto config = LoadConfig(argv[1]);
        fs::create_directories(config.logging.log_dir);
        if (!logging::Initialize(config.logging)) {
            return Fail("logging initialization failed");
        }
        logging_initialized = true;
        LOG_INFO("[agent-gateway] log file: {}", logging::GetLogFilePath(config.logging).string());
        LOG_INFO("[agent-gateway] dump dir: {}", config.dump_dir.string());
        LOG_INFO("[agent-gateway] config: {}", fs::absolute(config.config_path).string());

        auto llm = CreateLlmClient(config);
        if (!llm.ok()) {
            logging::Shutdown();
            return Fail("LLM client: " + llm.status().message());
        }
        auto llm_client = std::move(llm).value();

        auto l0 = CreateL0MemoryCache(config);
        if (!l0.ok()) {
            logging::Shutdown();
            return Fail("L0 memory: " + l0.status().message());
        }
        auto l0_bundle = std::move(l0).value();
        LOG_INFO("[agent-gateway] L0 memory: {} redis={}:{} tokenizer={} model={}",
                 config.l0_enabled ? "enabled" : "disabled",
                 config.l0_redis_host,
                 config.l0_redis_port,
                 config.tokenizer_path.string(),
                 config.embedding_model_path.string());

        auto l3 = CreateL3MemoryCompressor(config, llm_client);
        if (!l3.ok()) {
            logging::Shutdown();
            return Fail("L3 memory: " + l3.status().message());
        }
        auto l3_memory = std::move(l3).value();
        if (l3_memory) {
            LOG_INFO("[agent-gateway] L3 memory enabled db={} users={}",
                     config.l3_sqlite_path.string(),
                     config.l3_user_uuids.size());
        }

        auto document_embedding = CreateDocumentEmbeddingProvider(config);
        if (!document_embedding.ok()) {
            logging::Shutdown();
            return Fail("document embedding: " + document_embedding.status().message());
        }
        auto document_llm_chunk_cache = CreateDocumentLlmChunkCache(config);
        if (!document_llm_chunk_cache.ok()) {
            logging::Shutdown();
            return Fail("document LLM chunk cache: " + document_llm_chunk_cache.status().message());
        }
        auto document_semantic_cache = CreateDocumentSemanticCache(config);
        if (!document_semantic_cache.ok()) {
            logging::Shutdown();
            return Fail("document semantic cache: " + document_semantic_cache.status().message());
        }
        if (document_semantic_cache.value()) {
            LOG_INFO("[agent-gateway] document semantic cache enabled redis={}:{} db={}",
                     config.document_semantic_cache_redis_host,
                     config.document_semantic_cache_redis_port,
                     config.document_semantic_cache_sqlite_path.string());
        }

        agent::service::persona::SemanticMemoryContextProviderOptions memory_options;
        auto memory = std::make_shared<agent::service::persona::SemanticMemoryContextProvider>(
            l0_bundle.cache,
            l3_memory,
            memory_options);
        auto emotion = CreateEmotionAnalyzer(config);
        if (!emotion.ok()) {
            logging::Shutdown();
            return Fail("emotion analyzer: " + emotion.status().message());
        }
        LOG_INFO("[agent-gateway] emotion analyzer backend={}",
                 config.grpc_emotion_enabled ? "grpc" : "neutral");

        agent::service::gateway::PersonaGatewayServerDependencies dependencies;
        dependencies.memory_provider = std::move(memory);
        dependencies.emotion_analyzer = std::move(emotion).value();
        dependencies.llm_client = llm_client;
        dependencies.document_embedding_provider = std::move(document_embedding).value();
        dependencies.document_llm_chunk_cache = std::move(document_llm_chunk_cache).value();
        dependencies.document_semantic_cache = std::move(document_semantic_cache).value();
        dependencies.l0_redis_pool = l0_bundle.redis_pool;
        dependencies.evaluation_config_path = config.repo_root / "config" / "evaluation_indicators.json";

        if (config.gateway.auth.session_store_backend != "redis") {
            fs::create_directories(fs::path(config.gateway.auth.session_database_path).parent_path());
        }
        if (config.gateway.static_files) {
            if (!fs::exists(config.gateway.static_files->root)) {
                logging::Shutdown();
                return Fail("static dist root does not exist: " + config.gateway.static_files->root.string());
            }
        }

        agent::service::gateway::PersonaGatewayServer server(
            std::move(config.gateway),
            std::move(dependencies));
        if (l3_memory && config.l3_flush_enabled) {
            agent::service::gateway::L3MemoryFlushMaintenanceOptions flush_options;
            flush_options.interval = std::chrono::seconds(config.l3_flush_interval_seconds);
            flush_options.flush_hour = config.l3_flush_hour;
            flush_options.flush_minute = config.l3_flush_minute;
            flush_options.flush_date_offset_days = config.l3_flush_date_offset_days;
            flush_options.defer_when_sessions_active = config.l3_flush_defer_when_sessions_active;
            flush_options.user_uuids = config.l3_user_uuids;
            auto task = std::make_shared<agent::service::gateway::L3MemoryFlushMaintenanceTask>(
                l3_memory,
                server.sessions(),
                std::move(flush_options));
            auto register_task = server.RegisterMaintenanceTask(std::move(task));
            if (!register_task.ok()) {
                logging::Shutdown();
                return Fail("L3 flush maintenance: " + register_task.message());
            }
        }
        auto start = server.Start();
        if (!start.ok()) {
            logging::Shutdown();
            return Fail("server start: " + start.message());
        }

        std::cout << "[agent-gateway] started\n";
        std::cout << "[agent-gateway] frontend: http://127.0.0.1:" << server.port() << "/\n";
        std::cout << "[agent-gateway] auth:     POST http://127.0.0.1:" << server.port() << "/api/auth/register\n";
        std::cout << "[agent-gateway] ws:       ws://127.0.0.1:" << server.port() << "/ws/session\n";
        std::cout << "[agent-gateway] press Enter or Ctrl+C to stop\n";

        std::thread input_thread;
        if (stop_on_stdin) {
            input_thread = std::thread([] {
                std::string line;
                std::getline(std::cin, line);
                g_stop_requested.store(true);
            });
        }

        while (!g_stop_requested.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        server.Stop();
        logging::Shutdown();
        logging_initialized = false;
        if (input_thread.joinable()) {
            input_thread.detach();
        }
        std::cout << "[agent-gateway] stopped\n";
        return 0;
    } catch (const std::exception& e) {
        if (logging_initialized) {
            LOG_ERROR("[agent-gateway] fatal: {}", e.what());
            logging::Shutdown();
        }
        return Fail(e.what());
    }
}
