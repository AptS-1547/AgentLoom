#include "emotion_grpc_service.h"
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
    service::EmotionInferenceService emotion_service(options);
    server::grpc_service::EmotionGrpcService grpc_service(options, stats, emotion_service);

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
        LOG_ERROR("[EmotionServer] Failed to start gRPC server");
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
    LOG_INFO("[EmotionServer] Shutdown complete");
    logging::Shutdown();

    return 0;
}
