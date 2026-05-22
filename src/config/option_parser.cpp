#include "option_parser.h"

#include "config_section.h"
#include "server_config.h"

#include <iostream>
#include <string_view>

MultimodalServerOptions ParseMultimodalOptions(int argc, char** argv) {
    MultimodalServerOptions options;
    options.grpc.max_receive_message_mb = 100;
    options.grpc.max_send_message_mb = 10;

    if (auto config_path = server_config::FindConfigPath(argc, argv)) {
        server_config::LoadConfigFile(*config_path, options);
    }

    server_config::ApplyCliFallbackOptions(argc, argv, options);
    server_config::ValidateOptions(options);
    return options;
}

bool IsHelpRequested(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            return true;
        }
    }
    return false;
}

void PrintUsage(const char* program) {
    std::cerr << "Usage: " << program << " --llm <model.gguf> [options]\n"
              << "\nModel options:\n"
              << "  --bert <model.onnx>   BERT model path\n"
              << "  --vit <model.onnx>    ViT model path\n"
              << "  --llm <model.gguf>    LLM model path (required)\n"
              << "  --mmproj <proj.gguf>  Vision projector path\n"
              << "  --ngl <n>             GPU layers, -1 for all (default: -1)\n"
              << "  --provider <auto|cpu|cuda>  ONNX execution provider (default: auto)\n"
              << "  --cuda-device <id>    CUDA device id (default: 0)\n"
              << "  --config <file>       Load JSON config file before CLI overrides\n"
              << "\nServer options:\n"
              << "  --host <host>         Listen address (default: 127.0.0.1)\n"
              << "  --port <port>         Listen port (default: 50051)\n"
              << "  --log-dir <dir>       Log directory (default: logs)\n"
              << "  --grpc-num-cqs <n>    gRPC completion queues (default: auto)\n"
              << "  --grpc-min-pollers <n>  Min poller threads (default: 1)\n"
              << "  --grpc-max-pollers <n>  Max poller threads (default: auto)\n"
              << "  --max-recv-mb <n>     Max receive message size MB (default: 100)\n"
              << "  --max-send-mb <n>     Max send message size MB (default: 10)\n"
              << "  --auth-token <token>  Require metadata auth token (default: disabled)\n"
              << "  --auth-token-file <path> Read metadata auth token from file\n"
              << "  --auth-token-env <name> Read metadata auth token from env (default: AGENT_BACKEND_AUTH_TOKEN)\n"
              << "  --auth-header <name>  Auth metadata key (default: x-agent-auth)\n"
              << "  --max-image-mb <n>    Max image payload MB (default: 20)\n"
              << "  --max-image-pixels <n>  Max decoded image pixels (default: 16777216)\n"
              << "  --max-image-width <n> Max image width (default: 8192)\n"
              << "  --max-image-height <n> Max image height (default: 8192)\n"
              << "  --max-prompt-bytes <n> Max prompt bytes (default: 8192)\n"
              << "  --max-seq-len <n>     Max BERT sequence length (default: 512)\n"
              << "  --max-batch-size <n>  Max BERT batch size (default: 64)\n"
              << "  --max-vlm-tokens <n>  Max VLM output tokens (default: 2048)\n"
              << "  --max-context-size <n> Max VLM context size (default: 8192)\n"
              << "  --max-token-id <n>    Max accepted token id (default: 10000000)\n"
              << "  --vlm-cache-enabled   Enable exact VLM result cache\n"
              << "  --vlm-cache-persist   Persist exact VLM cache under --vlm-cache-dir\n"
              << "  --vlm-cache-dir <dir> VLM cache directory (default: cache/vlm)\n"
              << "  --vlm-cache-max-entries <n> Max cache entries, 0 disables entry limit (default: 512)\n"
              << "  --vlm-cache-max-mb <n> Max cache bytes MB, 0 disables byte limit (default: 1024)\n"
              << "  --vlm-cache-ttl-seconds <n> Cache TTL, 0 disables TTL (default: 3600)\n"
              << "  --no-vlm-cache-store-images Do not retain image payloads in VLM cache records\n"
              << "  --no-vlm-cache-store-prompts Do not retain prompts in VLM cache records\n"
              << "  --no-vlm-cache-stale-on-failure Disable automatic stale cache fallback\n"
              << "  --no-vlm-cache-default Require per-request allow_cache=true\n"
              << "  --vram-monitor-interval-seconds <n> VRAM watchdog interval, 0 disables (default: 10)\n"
              << "  --vram-warning-free-mb <n> Warn when GPU free VRAM is below MB, 0 disables (default: 1024)\n"
              << "  --vram-unload-free-mb <n> Unload LLM when GPU free VRAM is below MB, 0 disables (default: 512)\n"
              << "  --vram-min-free-before-load-mb <n> Reject LLM load when GPU free VRAM is below MB, 0 disables\n"
              << "  --vram-reload-after-unload Reload LLM immediately after watchdog unload\n"
              << "  --no-vram-unload-on-oom-error Disable automatic unload after OOM-like VLM errors\n"
              << "  --stats-log-interval-seconds <n>  Stats log interval (default: 30)\n"
              << "  --slow-request-ms <n> Slow request threshold (default: 250)\n"
              << "\nLLM client options:\n"
              << "  --llm-base-url <url>      OpenAI-compatible API base URL\n"
              << "  --llm-api-key-env <name>  Env var holding the API key (default: AGENT_LLM_API_KEY)\n"
              << "  --llm-api-key-file <path> File containing the API key\n"
              << "  --llm-model <name>        Model name (default: deepseek-chat)\n"
              << "  --llm-timeout <ms>        Request timeout ms (default: 30000)\n"
              << "  --llm-max-retries <n>     Max retries on 5xx (default: 2)\n";
}
