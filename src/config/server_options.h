#pragma once

#include "http_server.h"
#include "onnx_model.h"
#include "request_options.h"
#include "server_common.h"
#include "text_embedding_model.h"
#include "vector_cache.h"
#include "vlm_cache.h"

#include <cstddef>
#include <filesystem>
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

struct MultimodalServerOptions {
    server_common::GrpcServerOptions grpc;
    net::HttpServerOptions http;
    bert::ModelRuntimeOptions bert_runtime;
    EmbeddingModelOptions embedding;
    LlmOptions llm;
    request_validation::AuthOptions auth;
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
