/**
 * @file multimodal_inference_server.cpp
 * @brief 统一多模态推理 gRPC 服务端
 *
 * 整合 BERT 情绪分类（ONNX Runtime）、ViT 显著度检测（ONNX Runtime）
 * 和 VLM 推理（llama.cpp）到单一 gRPC 服务端。
 */

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#include "server_common.h"
#include "llama_runner.h"
#include "onnx_model.h"
#include "logger.h"

#include <grpc/grpc.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include "multimodal_inference.grpc.pb.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;
using grpc::StatusCode;

using multimodal_inference::MultimodalInference;
using multimodal_inference::EmotionRequest;
using multimodal_inference::EmotionResponse;
using multimodal_inference::EmotionBatchRequest;
using multimodal_inference::EmotionBatchResponse;
using multimodal_inference::SaliencyRequest;
using multimodal_inference::SaliencyResponse;
using multimodal_inference::VLMRequest;
using multimodal_inference::VLMResponse;
using multimodal_inference::VLMToken;

struct MultimodalServerOptions {
    server_common::GrpcServerOptions grpc;
    bert::ModelRuntimeOptions bert_runtime;
    std::string bert_model;
    std::string vit_model;
    std::string llm_model;
    std::string mmproj;
    int n_gpu_layers = -1;
};

class MultimodalInferenceServiceImpl final : public MultimodalInference::Service {
public:
    MultimodalInferenceServiceImpl(
        const MultimodalServerOptions& options,
        server_common::RuntimeStats& stats)
        : stats_(stats),
          slow_request_ms_(options.grpc.slow_request_ms),
          llm_model_path_(options.llm_model),
          mmproj_path_(options.mmproj),
          n_gpu_layers_(options.n_gpu_layers) {

        // 加载 BERT 模型（立即加载，模型小）
        if (!options.bert_model.empty()) {
            LOG_INFO("Loading BERT model: {}", options.bert_model);
            if (!bert_model_.LoadModel(options.bert_model, options.bert_runtime)) {
                LOG_ERROR("Failed to load BERT model");
            }
        }

        // LLM 采用 lazy load（首次请求时加载，模型大）
        idle_thread_ = std::jthread([this](std::stop_token stop_token) {
            IdleUnloadLoop(stop_token);
        });
    }

    const server_common::RuntimeStats& GetStats() const { return stats_; }

    // ========== BERT 情绪分类（单条） ==========

    Status PredictEmotion(ServerContext* context,
                         const EmotionRequest* request,
                         EmotionResponse* response) override {
        (void)context;
        server_common::ScopedRequestStats request_stats(
            stats_, "PredictEmotion", 1, false, slow_request_ms_);

        if (!bert_model_.IsLoaded()) {
            response->set_error("BERT model not loaded");
            request_stats.MarkFailure("BERT model not loaded");
            return Status(StatusCode::FAILED_PRECONDITION, "BERT model not loaded");
        }

        // 输入验证
        if (request->input_ids_size() == 0 || request->attention_mask_size() == 0) {
            response->set_error("Empty input");
            request_stats.MarkFailure("Empty input");
            return Status(StatusCode::INVALID_ARGUMENT, "Empty input");
        }

        if (request->input_ids_size() != request->attention_mask_size()) {
            response->set_error("input_ids and attention_mask size mismatch");
            request_stats.MarkFailure("input_ids and attention_mask size mismatch");
            return Status(StatusCode::INVALID_ARGUMENT, "Size mismatch");
        }

        if (request->personality_size() != 11) {
            response->set_error("personality must have 11 elements");
            request_stats.MarkFailure("personality must have 11 elements");
            return Status(StatusCode::INVALID_ARGUMENT, "Invalid personality size");
        }

        std::vector<int64_t> input_ids(request->input_ids().begin(), request->input_ids().end());
        std::vector<int64_t> attention_mask(request->attention_mask().begin(), request->attention_mask().end());
        std::vector<float> personality(request->personality().begin(), request->personality().end());

        auto result = bert_model_.Predict(input_ids, attention_mask, personality);

        if (!result.success) {
            response->set_error(result.error_message);
            request_stats.MarkFailure(result.error_message);
            return Status(StatusCode::INTERNAL, result.error_message);
        }

        response->mutable_emotion_logits()->Reserve(static_cast<int>(result.emotion_logits.size()));
        for (float v : result.emotion_logits) response->add_emotion_logits(v);

        response->mutable_behavior_logits()->Reserve(static_cast<int>(result.behavior_logits.size()));
        for (float v : result.behavior_logits) response->add_behavior_logits(v);

        response->mutable_tone_logits()->Reserve(static_cast<int>(result.tone_logits.size()));
        for (float v : result.tone_logits) response->add_tone_logits(v);

        response->set_intensity(result.intensity);

        response->mutable_response_length_logits()->Reserve(static_cast<int>(result.response_length_logits.size()));
        for (float v : result.response_length_logits) response->add_response_length_logits(v);

        request_stats.MarkSuccess();
        return Status::OK;
    }

    // ========== BERT 情绪分类（批量） ==========

    Status PredictEmotionBatch(ServerContext* context,
                              const EmotionBatchRequest* request,
                              EmotionBatchResponse* response) override {
        (void)context;
        size_t batch_size = request->batch_size();
        size_t seq_len = request->seq_length();
        server_common::ScopedRequestStats request_stats(
            stats_, "PredictEmotionBatch", batch_size, true, slow_request_ms_);

        if (!bert_model_.IsLoaded()) {
            response->set_error("BERT model not loaded");
            request_stats.MarkFailure("BERT model not loaded");
            return Status(StatusCode::FAILED_PRECONDITION, "BERT model not loaded");
        }

        if (batch_size == 0) {
            response->set_error("Empty batch");
            request_stats.MarkFailure("Empty batch");
            return Status(StatusCode::INVALID_ARGUMENT, "Empty batch");
        }

        size_t expected_input_size = batch_size * seq_len;
        size_t expected_personality_size = batch_size * 11;

        if (static_cast<size_t>(request->input_ids_size()) != expected_input_size ||
            static_cast<size_t>(request->attention_mask_size()) != expected_input_size ||
            static_cast<size_t>(request->personality_size()) != expected_personality_size) {
            response->set_error("Input size mismatch");
            request_stats.MarkFailure("Input size mismatch");
            return Status(StatusCode::INVALID_ARGUMENT, "Input size mismatch");
        }

        std::vector<int64_t> input_ids(request->input_ids().begin(), request->input_ids().end());
        std::vector<int64_t> attention_mask(request->attention_mask().begin(), request->attention_mask().end());
        std::vector<float> personality(request->personality().begin(), request->personality().end());

        auto results = bert_model_.PredictBatch(input_ids, attention_mask, personality, batch_size, seq_len);

        if (results.empty()) {
            response->set_error("Batch inference failed");
            request_stats.MarkFailure("Batch inference failed");
            return Status(StatusCode::INTERNAL, "Batch inference failed");
        }

        response->mutable_emotion_logits()->Reserve(static_cast<int>(batch_size * results.front().emotion_logits.size()));
        response->mutable_behavior_logits()->Reserve(static_cast<int>(batch_size * results.front().behavior_logits.size()));
        response->mutable_tone_logits()->Reserve(static_cast<int>(batch_size * results.front().tone_logits.size()));
        response->mutable_intensity()->Reserve(static_cast<int>(batch_size));
        response->mutable_response_length_logits()->Reserve(static_cast<int>(batch_size * results.front().response_length_logits.size()));

        for (const auto& result : results) {
            for (float v : result.emotion_logits) response->add_emotion_logits(v);
            for (float v : result.behavior_logits) response->add_behavior_logits(v);
            for (float v : result.tone_logits) response->add_tone_logits(v);
            response->add_intensity(result.intensity);
            for (float v : result.response_length_logits) response->add_response_length_logits(v);
        }

        request_stats.MarkSuccess();
        return Status::OK;
    }

    // ========== ViT 显著度检测 ==========

    Status DetectSaliency(ServerContext* context,
                         const SaliencyRequest* request,
                         SaliencyResponse* response) override {
        (void)context;
        server_common::ScopedRequestStats request_stats(
            stats_, "DetectSaliency", 1, false, slow_request_ms_);

        response->set_saliency_score(0.0f);
        response->set_trigger_vlm(false);
        response->set_inference_ms(0.0f);

        request_stats.MarkFailure("ViT not implemented yet");
        return Status(StatusCode::UNIMPLEMENTED, "ViT not implemented yet");
    }

    // ========== VLM 流式推理 ==========

    Status GenerateVLM(ServerContext* context,
                      const VLMRequest* request,
                      ServerWriter<VLMToken>* writer) override {
        (void)context;
        server_common::ScopedRequestStats request_stats(
            stats_, "GenerateVLM", 1, false, slow_request_ms_);

        if (!EnsureLLMLoaded()) {
            request_stats.MarkFailure("Failed to load LLM");
            return Status(StatusCode::INTERNAL, "Failed to load LLM");
        }

        last_llm_request_time_ = std::chrono::steady_clock::now();

        auto params = BuildGenerateParams(*request);
        auto image_data = ExtractImageData(*request);

        auto result = llm_runner_.Generate(
            image_data,
            request->prompt(),
            params,
            [writer](const std::string& token) {
                VLMToken t;
                t.set_token(token);
                t.set_is_final(false);
                writer->Write(t);
            }
        );

        VLMToken final_token;
        final_token.set_is_final(true);
        writer->Write(final_token);

        if (result.success) {
            request_stats.MarkSuccess();
            return Status::OK;
        } else {
            request_stats.MarkFailure(result.error_message);
            return Status(StatusCode::INTERNAL, result.error_message);
        }
    }

    // ========== VLM 同步推理 ==========

    Status GenerateVLMSync(ServerContext* context,
                          const VLMRequest* request,
                          VLMResponse* response) override {
        (void)context;
        server_common::ScopedRequestStats request_stats(
            stats_, "GenerateVLMSync", 1, false, slow_request_ms_);

        if (!EnsureLLMLoaded()) {
            response->set_error("Failed to load LLM");
            request_stats.MarkFailure("Failed to load LLM");
            return Status(StatusCode::INTERNAL, "Failed to load LLM");
        }

        last_llm_request_time_ = std::chrono::steady_clock::now();

        auto params = BuildGenerateParams(*request);
        auto image_data = ExtractImageData(*request);

        auto result = llm_runner_.Generate(image_data, request->prompt(), params);

        response->set_text(result.text);
        response->set_image_encode_ms(result.image_encode_ms);
        response->set_prompt_eval_ms(result.prompt_eval_ms);
        response->set_eval_ms(result.eval_ms);
        response->set_prompt_tokens(result.prompt_tokens);
        response->set_generated_tokens(result.generated_tokens);

        if (result.success) {
            request_stats.MarkSuccess();
            return Status::OK;
        } else {
            response->set_error(result.error_message);
            request_stats.MarkFailure(result.error_message);
            return Status(StatusCode::INTERNAL, result.error_message);
        }
    }

private:
    bool EnsureLLMLoaded() {
        if (llm_runner_.IsLoaded()) {
            return true;
        }

        std::lock_guard lock(llm_load_mutex_);
        if (llm_runner_.IsLoaded()) {
            return true;
        }

        LOG_INFO("Lazy loading LLM model");
        return llm_runner_.LoadModel(llm_model_path_, mmproj_path_, n_gpu_layers_);
    }

    static llm::GenerateParams BuildGenerateParams(const VLMRequest& request) {
        llm::GenerateParams params;
        if (request.max_tokens() > 0) params.max_tokens = request.max_tokens();
        if (request.temperature() > 0) params.temperature = request.temperature();
        if (request.context_size() > 0) params.context_size = request.context_size();
        if (request.top_p() > 0) params.top_p = request.top_p();
        if (request.top_k() > 0) params.top_k = request.top_k();
        return params;
    }

    static std::vector<uint8_t> ExtractImageData(const VLMRequest& request) {
        if (request.image_source_case() == VLMRequest::kImageData) {
            return {request.image_data().begin(), request.image_data().end()};
        }
        return {};
    }

    void IdleUnloadLoop(std::stop_token stop_token) {
        const auto idle_timeout = std::chrono::minutes(5);

        while (!stop_token.stop_requested()) {
            for (int second = 0; second < 30; ++second) {
                if (stop_token.stop_requested()) return;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            if (llm_runner_.IsLoaded()) {
                auto now = std::chrono::steady_clock::now();
                auto idle_time = now - last_llm_request_time_;

                if (idle_time > idle_timeout) {
                    LOG_INFO("LLM idle for {} minutes, unloading to free VRAM",
                            std::chrono::duration_cast<std::chrono::minutes>(idle_time).count());
                    llm_runner_.Unload();
                }
            }
        }
    }

    // 模型实例
    bert::OnnxBERTModel bert_model_;
    llm::LlamaRunner llm_runner_;

    // 统计
    server_common::RuntimeStats& stats_;
    int slow_request_ms_ = 250;

    // LLM lazy load 配置
    std::string llm_model_path_;
    std::string mmproj_path_;
    int n_gpu_layers_;
    std::mutex llm_load_mutex_;
    std::chrono::steady_clock::time_point last_llm_request_time_;
    std::jthread idle_thread_;
};

// ========== 参数解析 ==========

MultimodalServerOptions ParseMultimodalOptions(int argc, char** argv) {
    MultimodalServerOptions options;
    options.grpc.max_receive_message_mb = 100;
    options.grpc.max_send_message_mb = 10;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--bert" && i + 1 < argc) {
            options.bert_model = argv[++i];
        } else if (arg == "--vit" && i + 1 < argc) {
            options.vit_model = argv[++i];
        } else if (arg == "--llm" && i + 1 < argc) {
            options.llm_model = argv[++i];
        } else if (arg == "--mmproj" && i + 1 < argc) {
            options.mmproj = argv[++i];
        } else if (arg == "--ngl" && i + 1 < argc) {
            options.n_gpu_layers = server_common::ParseIntValue(arg, argv[++i]);
        } else if (arg == "--provider" && i + 1 < argc) {
            options.bert_runtime.execution_provider = argv[++i];
        } else if (arg == "--cuda-device" && i + 1 < argc) {
            options.bert_runtime.cuda_device_id = server_common::ParseIntValue(arg, argv[++i]);
        }
    }

    server_common::ParseGrpcServerOptions(argc, argv, 1, options.grpc);
    return options;
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
              << "\nServer options:\n"
              << "  --host <host>         Listen address (default: 127.0.0.1)\n"
              << "  --port <port>         Listen port (default: 50051)\n"
              << "  --log-dir <dir>       Log directory (default: logs)\n"
              << "  --grpc-num-cqs <n>    gRPC completion queues (default: auto)\n"
              << "  --grpc-min-pollers <n>  Min poller threads (default: 1)\n"
              << "  --grpc-max-pollers <n>  Max poller threads (default: auto)\n"
              << "  --max-recv-mb <n>     Max receive message size MB (default: 100)\n"
              << "  --max-send-mb <n>     Max send message size MB (default: 10)\n"
              << "  --stats-log-interval-seconds <n>  Stats log interval (default: 30)\n"
              << "  --slow-request-ms <n> Slow request threshold (default: 250)\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        PrintUsage(argv[0]);
        return 1;
    }

    MultimodalServerOptions options;
    try {
        options = ParseMultimodalOptions(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[Server] " << e.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

    if (options.llm_model.empty()) {
        std::cerr << "Error: --llm is required\n";
        PrintUsage(argv[0]);
        return 1;
    }

    // 初始化日志
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
    LOG_INFO("[Server] GPU layers: {}, gRPC CQs: {}, pollers: {}-{}, "
             "stats_interval: {}s, slow_request: {}ms",
             options.n_gpu_layers,
             options.grpc.grpc_num_cqs,
             options.grpc.grpc_min_pollers,
             options.grpc.grpc_max_pollers,
             options.grpc.stats_log_interval_seconds,
             options.grpc.slow_request_ms);

    // 创建服务
    server_common::RuntimeStats stats;
    MultimodalInferenceServiceImpl service(options, stats);

    // 配置 gRPC 服务端
    grpc::EnableDefaultHealthCheckService(true);
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(options.grpc.max_receive_message_mb * 1024 * 1024);
    builder.SetMaxSendMessageSize(options.grpc.max_send_message_mb * 1024 * 1024);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::NUM_CQS, options.grpc.grpc_num_cqs);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::MIN_POLLERS, options.grpc.grpc_min_pollers);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::MAX_POLLERS, options.grpc.grpc_max_pollers);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    if (!server) {
        LOG_ERROR("[Server] Failed to start gRPC server");
        logging::Shutdown();
        return 1;
    }

    // 健康检查
    if (auto* health = server->GetHealthCheckService(); health != nullptr) {
        health->SetServingStatus("multimodal_inference.MultimodalInference", true);
        health->SetServingStatus(true);
        LOG_INFO("[Server] Health check service enabled");
    }

    LOG_INFO("[Server] Ready");

    // 周期性统计日志
    std::jthread stats_thread(
        [&stats, server_address, interval = options.grpc.stats_log_interval_seconds]
        (std::stop_token stop_token) {
            server_common::LogStatsPeriodically(stats, server_address, stop_token, interval);
        }
    );

    server->Wait();
    LOG_INFO("[Server] Shutdown complete");
    logging::Shutdown();

    return 0;
}
