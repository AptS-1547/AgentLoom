#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#include "logger.h"
#include "multimodal_grpc_service.h"
#include "multimodal_service.h"
#include "option_parser.h"
#include "server_common.h"

#include <grpc/grpc.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    if (IsHelpRequested(argc, argv)) {
        PrintUsage(argv[0]);
        return 0;
    }

    MultimodalServerOptions options;
    try {
        options = ParseMultimodalOptions(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[Server] " << e.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

    logging::LoggerOptions log_options;
    log_options.log_dir = options.grpc.log_dir;
    log_options.logger_name = "multimodal_inference_server";
    log_options.file_name = "multimodal_inference_server.log";
    if (!logging::Initialize(log_options)) {
        return 1;
    }

    const std::string server_address = options.grpc.host + ":" + options.grpc.port;
    const auto log_file = logging::GetLogFilePath(log_options);

    LOG_INFO("[Server] Log file: {}", log_file.string());
    LOG_INFO("[Server] Bind address: {}", server_address);
    LOG_INFO("[Server] LLM: {}", options.llm_model);
    if (!options.mmproj.empty()) LOG_INFO("[Server] MMProj: {}", options.mmproj);
    if (!options.bert_model.empty()) LOG_INFO("[Server] BERT: {}", options.bert_model);
    LOG_INFO("[Server] Auth: {} source={}",
             options.auth.token.empty() ? "disabled" : "enabled",
             options.auth_source);
    LOG_INFO("[Server] Limits: image_mb={} image_pixels={} prompt_bytes={} seq_len={} batch={} vlm_tokens={} context={}",
             options.limits.max_image_bytes / (1024 * 1024),
             options.limits.max_image_pixels,
             options.limits.max_prompt_bytes,
             options.limits.max_sequence_length,
             options.limits.max_batch_size,
             options.limits.max_vlm_tokens,
             options.limits.max_context_size);
    LOG_INFO("[Server] VRAM guard: interval={}s warn_free_mb={} unload_free_mb={} min_free_before_load_mb={} reload_after_unload={} unload_on_oom_error={}",
             options.vram.monitor_interval_seconds,
             options.vram.warning_free_bytes / (1024 * 1024),
             options.vram.unload_free_bytes / (1024 * 1024),
             options.vram.min_free_before_load_bytes / (1024 * 1024),
             options.vram.reload_after_unload,
             options.vram.unload_on_oom_error);
    LOG_INFO("[Server] VLM cache: enabled={} persist={} dir={} max_entries={} max_mb={} ttl_seconds={} stale_on_failure={} default_allow_cache={}",
             options.vlm_cache.enabled,
             options.vlm_cache.persist,
             options.vlm_cache.cache_dir.string(),
             options.vlm_cache.max_entries,
             options.vlm_cache.max_bytes / (1024 * 1024),
             options.vlm_cache.ttl_seconds,
             options.vlm_cache.allow_stale_on_failure,
             options.vlm_cache.default_allow_cache);
    LOG_INFO("[Server] GPU layers: {}, gRPC CQs: {}, pollers: {}-{}, stats_interval: {}s, slow_request: {}ms",
             options.n_gpu_layers,
             options.grpc.grpc_num_cqs,
             options.grpc.grpc_min_pollers,
             options.grpc.grpc_max_pollers,
             options.grpc.stats_log_interval_seconds,
             options.grpc.slow_request_ms);

    server_common::RuntimeStats stats;
    service::MultimodalService multimodal_service(options);
    server::grpc_service::MultimodalGrpcService grpc_service(options, stats, multimodal_service);

    grpc::EnableDefaultHealthCheckService(true);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&grpc_service);
    builder.SetMaxReceiveMessageSize(options.grpc.max_receive_message_mb * 1024 * 1024);
    builder.SetMaxSendMessageSize(options.grpc.max_send_message_mb * 1024 * 1024);
    builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::NUM_CQS, options.grpc.grpc_num_cqs);
    builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, options.grpc.grpc_min_pollers);
    builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, options.grpc.grpc_max_pollers);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) {
        LOG_ERROR("[Server] Failed to start gRPC server");
        logging::Shutdown();
        return 1;
    }

    if (auto* health = server->GetHealthCheckService(); health != nullptr) {
        health->SetServingStatus("multimodal_inference.MultimodalInference", true);
        health->SetServingStatus(true);
        LOG_INFO("[Server] Health check service enabled");
    }

    LOG_INFO("[Server] Ready");

    std::jthread stats_thread(
        [&stats, server_address, interval = options.grpc.stats_log_interval_seconds]
        (std::stop_token stop_token) {
            server_common::LogStatsPeriodically(stats, server_address, stop_token, interval);
        });

    server->Wait();
    LOG_INFO("[Server] Shutdown complete");
    logging::Shutdown();

    return 0;
}
