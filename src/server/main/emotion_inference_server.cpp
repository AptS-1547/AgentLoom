#include "async_emotion_grpc_service.h"
#include "async_emotion_inference_handler.h"
#include "emotion_inference_service.h"
#include "logger.h"
#include "option_parser.h"
#include "server_common.h"

#include <grpc/grpc.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include <exception>
#include <algorithm>
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
        std::cerr << "[EmotionServer] " << e.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

    logging::LoggerOptions log_options;
    log_options.log_dir = options.grpc.log_dir;
    log_options.logger_name = "emotion_inference_server";
    log_options.file_name = "emotion_inference_server.log";
    if (!logging::Initialize(log_options)) {
        return 1;
    }

    const std::string server_address = options.grpc.host + ":" + options.grpc.port;
    const auto log_file = logging::GetLogFilePath(log_options);

    LOG_INFO("[EmotionServer] Log file: {}", log_file.string());
    LOG_INFO("[EmotionServer] Bind address: {}", server_address);
    LOG_INFO("[EmotionServer] BERT: {}", options.bert_model.empty() ? "<empty>" : options.bert_model);
    LOG_INFO("[EmotionServer] BERT provider: {} intra_threads={} inter_threads={}",
             options.bert_runtime.execution_provider,
             options.bert_runtime.intra_op_num_threads,
             options.bert_runtime.inter_op_num_threads);
    LOG_INFO("[EmotionServer] Auth: {} source={}",
             options.auth.token.empty() ? "disabled" : "enabled",
             options.auth_source);
    LOG_INFO("[EmotionServer] Limits: seq_len={} batch={}",
             options.limits.max_sequence_length,
             options.limits.max_batch_size);

    server_common::RuntimeStats stats;
    auto emotion_service = std::make_shared<service::EmotionInferenceService>(options);

    core::ThreadPoolOptions worker_options;
    worker_options.worker_count = static_cast<std::size_t>(
        std::max(1, options.grpc.grpc_max_pollers));
    worker_options.queue_capacity = worker_options.worker_count * 32;
    worker_options.name = "emotion-grpc-worker";
    auto worker_pool = std::make_shared<core::ThreadPool>(std::move(worker_options));
    if (const auto status = worker_pool->Start(); !status.ok()) {
        LOG_ERROR("[EmotionServer] Failed to start async worker pool: {}", status.message());
        logging::Shutdown();
        return 1;
    }

    grpc_runtime::AsyncGrpcRuntimeOptions runtime_options;
    runtime_options.max_inflight_calls = std::max<std::size_t>(
        256, worker_pool->Stats().worker_count * 32);
    runtime_options.task_name = "emotion-grpc";
    auto runtime_result = grpc_runtime::AsyncGrpcRuntime::Create(
        worker_pool, runtime_options, core::LoggerAdapter::ForModule("emotion-grpc"));
    if (!runtime_result.ok()) {
        LOG_ERROR("[EmotionServer] Failed to create async gRPC runtime: {}",
                  runtime_result.status().message());
        worker_pool->Shutdown(false);
        logging::Shutdown();
        return 1;
    }
    auto runtime = std::move(runtime_result).value();
    auto handler = std::make_shared<server::grpc_service::AsyncEmotionInferenceHandler>(
        options, stats, emotion_service);
    std::shared_ptr<grpc_runtime::IAsyncUnaryRpcHandler<
        multimodal_inference::EmotionRequest,
        multimodal_inference::EmotionResponse>> emotion_handler = handler;
    std::shared_ptr<grpc_runtime::IAsyncUnaryRpcHandler<
        multimodal_inference::EmotionBatchRequest,
        multimodal_inference::EmotionBatchResponse>> batch_handler = handler;
    server::grpc_service::AsyncEmotionGrpcService grpc_service(
        runtime, std::move(emotion_handler), std::move(batch_handler));

    grpc::EnableDefaultHealthCheckService(true);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&grpc_service);
    builder.SetMaxReceiveMessageSize(options.grpc.max_receive_message_mb * 1024 * 1024);
    builder.SetMaxSendMessageSize(options.grpc.max_send_message_mb * 1024 * 1024);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) {
        LOG_ERROR("[EmotionServer] Failed to start gRPC server");
        worker_pool->Shutdown(false);
        logging::Shutdown();
        return 1;
    }

    if (auto* health = server->GetHealthCheckService(); health != nullptr) {
        health->SetServingStatus("multimodal_inference.MultimodalInference", true);
        health->SetServingStatus(true);
        LOG_INFO("[EmotionServer] Health check service enabled");
    }

    LOG_INFO("[EmotionServer] Ready");

    std::jthread stats_thread(
        [&stats, server_address, interval = options.grpc.stats_log_interval_seconds]
        (std::stop_token stop_token) {
            server_common::LogStatsPeriodically(stats, server_address, stop_token, interval);
        });

    server->Wait();
    // gRPC 已停止接收并等待 reactor 结束，随后排空业务任务，保证模型最后析构。
    worker_pool->Shutdown(true);
    LOG_INFO("[EmotionServer] Shutdown complete");
    logging::Shutdown();

    return 0;
}
