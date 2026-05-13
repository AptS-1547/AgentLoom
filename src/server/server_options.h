#pragma once

#include "onnx_model.h"
#include "request_validation.h"
#include "server_common.h"
#include "vector_cache.h"
#include "vlm_cache.h"

#include <cstddef>
#include <string>

struct VramGuardOptions {
    int monitor_interval_seconds = 10;
    size_t warning_free_bytes = 1024 * 1024 * 1024;
    size_t unload_free_bytes = 512 * 1024 * 1024;
    size_t min_free_before_load_bytes = 0;
    bool reload_after_unload = false;
    bool unload_on_oom_error = true;
};

struct MultimodalServerOptions {
    server_common::GrpcServerOptions grpc;
    bert::ModelRuntimeOptions bert_runtime;
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
};
