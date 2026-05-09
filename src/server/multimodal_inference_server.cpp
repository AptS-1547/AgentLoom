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
#include "request_validation.h"
#include "server_config.h"
#include "server_options.h"
#include "vlm_cache.h"

#include <grpc/grpc.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include "multimodal_inference.grpc.pb.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <string>
#include <thread>
#include <utility>
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

class MultimodalInferenceServiceImpl final : public MultimodalInference::Service {
public:
    MultimodalInferenceServiceImpl(
        const MultimodalServerOptions& options,
        server_common::RuntimeStats& stats)
        : stats_(stats),
          slow_request_ms_(options.grpc.slow_request_ms),
          auth_options_(options.auth),
          request_limits_(options.limits),
          vram_options_(options.vram),
          vlm_cache_(options.vlm_cache),
          llm_model_path_(options.llm_model),
          mmproj_path_(options.mmproj),
          model_fingerprint_(BuildModelFingerprint(options.llm_model, options.mmproj, options.n_gpu_layers)),
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

        if (vram_options_.monitor_interval_seconds > 0 &&
            (vram_options_.warning_free_bytes > 0 || vram_options_.unload_free_bytes > 0)) {
            vram_thread_ = std::jthread([this](std::stop_token stop_token) {
                VramMonitorLoop(stop_token);
            });
        }
    }

    const server_common::RuntimeStats& GetStats() const { return stats_; }

    // ========== BERT 情绪分类（单条） ==========

    Status PredictEmotion(ServerContext* context,
                         const EmotionRequest* request,
                         EmotionResponse* response) override {
        server_common::ScopedRequestStats request_stats(
            stats_, "PredictEmotion", 1, false, slow_request_ms_);

        if (auto auth_status = CheckAuth(*context, request_stats); !auth_status.ok()) {
            response->set_error(auth_status.error_message());
            return auth_status;
        }

        if (auto validation = request_validation::ValidateEmotionRequest(*request, request_limits_);
            !validation.ok) {
            response->set_error(validation.error);
            request_stats.MarkFailure(validation.error);
            return Status(StatusCode::INVALID_ARGUMENT, validation.error);
        }

        if (!bert_model_.IsLoaded()) {
            response->set_error("BERT model not loaded");
            request_stats.MarkFailure("BERT model not loaded");
            return Status(StatusCode::FAILED_PRECONDITION, "BERT model not loaded");
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
        const size_t reported_batch_size =
            request->batch_size() > 0 ? static_cast<size_t>(request->batch_size()) : 0;
        server_common::ScopedRequestStats request_stats(
            stats_, "PredictEmotionBatch", reported_batch_size, true, slow_request_ms_);

        if (auto auth_status = CheckAuth(*context, request_stats); !auth_status.ok()) {
            response->set_error(auth_status.error_message());
            return auth_status;
        }

        if (auto validation = request_validation::ValidateEmotionBatchRequest(*request, request_limits_);
            !validation.ok) {
            response->set_error(validation.error);
            request_stats.MarkFailure(validation.error);
            return Status(StatusCode::INVALID_ARGUMENT, validation.error);
        }

        const size_t batch_size = static_cast<size_t>(request->batch_size());
        const size_t seq_len = static_cast<size_t>(request->seq_length());

        if (!bert_model_.IsLoaded()) {
            response->set_error("BERT model not loaded");
            request_stats.MarkFailure("BERT model not loaded");
            return Status(StatusCode::FAILED_PRECONDITION, "BERT model not loaded");
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
        server_common::ScopedRequestStats request_stats(
            stats_, "DetectSaliency", 1, false, slow_request_ms_);

        if (auto auth_status = CheckAuth(*context, request_stats); !auth_status.ok()) {
            return auth_status;
        }

        if (auto validation = request_validation::ValidateSaliencyRequest(*request, request_limits_);
            !validation.ok) {
            request_stats.MarkFailure(validation.error);
            return Status(StatusCode::INVALID_ARGUMENT, validation.error);
        }

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
        server_common::ScopedRequestStats request_stats(
            stats_, "GenerateVLM", 1, false, slow_request_ms_);

        if (auto auth_status = CheckAuth(*context, request_stats); !auth_status.ok()) {
            return auth_status;
        }

        if (auto validation = request_validation::ValidateVLMRequest(*request, request_limits_);
            !validation.ok) {
            request_stats.MarkFailure(validation.error);
            return Status(StatusCode::INVALID_ARGUMENT, validation.error);
        }

        auto params = BuildGenerateParams(*request);
        auto image_data = ExtractImageData(*request);
        auto key = BuildVLMCacheKey(image_data, request->prompt(), params);
        const bool use_cache = ShouldUseVLMCache(*request);

        if (use_cache && !request->force_refresh()) {
            if (auto cached = vlm_cache_.Get(key.cache_key)) {
                WriteCachedStream(writer, *cached, false, "exact_cache");
                request_stats.MarkSuccess();
                return Status::OK;
            }
        }

        std::string load_error;
        if (!EnsureLLMLoaded(load_error)) {
            if (load_error.empty()) {
                load_error = "Failed to load LLM";
            }
            if (auto fallback = GetStaleVLMFallback(*request, key.cache_key)) {
                WriteCachedStream(writer, *fallback, true, "stale_cache");
                request_stats.MarkSuccess();
                return Status::OK;
            }
            request_stats.MarkFailure(load_error);
            const auto status_code = load_error.find("GPU memory low") != std::string::npos
                ? StatusCode::RESOURCE_EXHAUSTED
                : StatusCode::INTERNAL;
            return Status(status_code, load_error);
        }

        last_llm_request_time_ = std::chrono::steady_clock::now();

        bool stream_wrote_token = false;
        auto result = llm_runner_.Generate(
            image_data,
            request->prompt(),
            params,
            [writer, &stream_wrote_token](const std::string& token) {
                stream_wrote_token = true;
                VLMToken t;
                t.set_token(token);
                t.set_is_final(false);
                writer->Write(t);
            }
        );

        if (result.success) {
            StoreVLMCacheResult(*request, image_data, params, key, result);
            VLMToken final_token;
            final_token.set_is_final(true);
            FillVLMTokenCacheMetadata(&final_token, key, false, false, "fresh");
            writer->Write(final_token);
            request_stats.MarkSuccess();
            return Status::OK;
        } else {
            HandleVLMFailure(result.error_message);
            if (!stream_wrote_token) {
                if (auto fallback = GetStaleVLMFallback(*request, key.cache_key)) {
                    WriteCachedStream(writer, *fallback, true, "stale_cache");
                    request_stats.MarkSuccess();
                    return Status::OK;
                }
            }
            VLMToken final_token;
            final_token.set_is_final(true);
            final_token.set_error(result.error_message);
            FillVLMTokenCacheMetadata(&final_token, key, false, false, "error");
            writer->Write(final_token);
            request_stats.MarkFailure(result.error_message);
            return Status(StatusCode::INTERNAL, result.error_message);
        }
    }

    // ========== VLM 同步推理 ==========

    Status GenerateVLMSync(ServerContext* context,
                          const VLMRequest* request,
                          VLMResponse* response) override {
        server_common::ScopedRequestStats request_stats(
            stats_, "GenerateVLMSync", 1, false, slow_request_ms_);

        if (auto auth_status = CheckAuth(*context, request_stats); !auth_status.ok()) {
            response->set_error(auth_status.error_message());
            return auth_status;
        }

        if (auto validation = request_validation::ValidateVLMRequest(*request, request_limits_);
            !validation.ok) {
            response->set_error(validation.error);
            request_stats.MarkFailure(validation.error);
            return Status(StatusCode::INVALID_ARGUMENT, validation.error);
        }

        auto params = BuildGenerateParams(*request);
        auto image_data = ExtractImageData(*request);
        auto key = BuildVLMCacheKey(image_data, request->prompt(), params);
        const bool use_cache = ShouldUseVLMCache(*request);

        if (use_cache && !request->force_refresh()) {
            if (auto cached = vlm_cache_.Get(key.cache_key)) {
                FillVLMResponseFromCache(response, *cached, false, "exact_cache");
                request_stats.MarkSuccess();
                return Status::OK;
            }
        }

        std::string load_error;
        if (!EnsureLLMLoaded(load_error)) {
            if (load_error.empty()) {
                load_error = "Failed to load LLM";
            }
            if (auto fallback = GetStaleVLMFallback(*request, key.cache_key)) {
                FillVLMResponseFromCache(response, *fallback, true, "stale_cache");
                request_stats.MarkSuccess();
                return Status::OK;
            }
            response->set_error(load_error);
            request_stats.MarkFailure(load_error);
            const auto status_code = load_error.find("GPU memory low") != std::string::npos
                ? StatusCode::RESOURCE_EXHAUSTED
                : StatusCode::INTERNAL;
            return Status(status_code, load_error);
        }

        last_llm_request_time_ = std::chrono::steady_clock::now();

        auto result = llm_runner_.Generate(image_data, request->prompt(), params);

        FillVLMResponseFromResult(response, result);

        if (result.success) {
            StoreVLMCacheResult(*request, image_data, params, key, result);
            FillVLMResponseCacheMetadata(response, key, false, false, "fresh");
            request_stats.MarkSuccess();
            return Status::OK;
        } else {
            HandleVLMFailure(result.error_message);
            if (auto fallback = GetStaleVLMFallback(*request, key.cache_key)) {
                FillVLMResponseFromCache(response, *fallback, true, "stale_cache");
                request_stats.MarkSuccess();
                return Status::OK;
            }
            response->set_error(result.error_message);
            FillVLMResponseCacheMetadata(response, key, false, false, "error");
            request_stats.MarkFailure(result.error_message);
            return Status(StatusCode::INTERNAL, result.error_message);
        }
    }

private:
    static void AppendFileIdentity(std::ostringstream& out, const std::filesystem::path& path) {
        if (path.empty()) {
            out << "empty\n";
            return;
        }

        std::error_code ec;
        const auto canonical = std::filesystem::weakly_canonical(path, ec);
        out << (ec ? path.string() : canonical.string()) << '\n';

        ec.clear();
        const auto size = std::filesystem::file_size(path, ec);
        out << (ec ? 0 : size) << '\n';

        ec.clear();
        const auto write_time = std::filesystem::last_write_time(path, ec);
        out << (ec ? 0 : write_time.time_since_epoch().count()) << '\n';
    }

    static std::string BuildModelFingerprint(
        const std::filesystem::path& llm_model,
        const std::filesystem::path& mmproj,
        int n_gpu_layers) {
        std::ostringstream out;
        out << "model-fingerprint-v1\n";
        AppendFileIdentity(out, llm_model);
        AppendFileIdentity(out, mmproj);
        out << n_gpu_layers << '\n';
        return vlm_cache::Sha256Hex(out.str());
    }

    bool ShouldUseVLMCache(const VLMRequest& request) const {
        return vlm_cache_.options().enabled &&
               (request.allow_cache() || vlm_cache_.options().default_allow_cache);
    }

    bool ShouldAllowStaleVLMCache(const VLMRequest& request) const {
        return vlm_cache_.options().enabled &&
               (request.allow_stale_cache() || vlm_cache_.options().allow_stale_on_failure);
    }

    vlm_cache::KeyInfo BuildVLMCacheKey(
        const std::vector<uint8_t>& image_data,
        const std::string& prompt,
        const llm::GenerateParams& params) const {
        if (!vlm_cache_.options().enabled) {
            return {};
        }
        return vlm_cache_.BuildKey(image_data, prompt, params, model_fingerprint_);
    }

    std::optional<vlm_cache::Result> GetStaleVLMFallback(
        const VLMRequest& request,
        const std::string& exact_cache_key) {
        if (!ShouldAllowStaleVLMCache(request)) {
            return std::nullopt;
        }
        if (!exact_cache_key.empty()) {
            if (auto exact = vlm_cache_.Get(exact_cache_key)) {
                return exact;
            }
        }
        if (request.session_id().empty()) {
            return std::nullopt;
        }
        return vlm_cache_.GetLatestFallback(
            request.session_id(),
            request.task_type(),
            model_fingerprint_);
    }

    static void FillVLMResponseFromResult(VLMResponse* response, const llm::VLMResult& result) {
        response->set_text(result.text);
        response->set_image_encode_ms(result.image_encode_ms);
        response->set_prompt_eval_ms(result.prompt_eval_ms);
        response->set_eval_ms(result.eval_ms);
        response->set_prompt_tokens(result.prompt_tokens);
        response->set_generated_tokens(result.generated_tokens);
    }

    static void FillVLMResponseCacheMetadata(
        VLMResponse* response,
        const vlm_cache::KeyInfo& key,
        bool cache_hit,
        bool cache_stale,
        const std::string& result_source) {
        response->set_cache_hit(cache_hit);
        response->set_cache_stale(cache_stale);
        response->set_result_source(result_source);
        if (!key.cache_key.empty()) {
            response->set_cache_key(key.cache_key);
            response->set_image_sha256(key.image_sha256);
            response->set_prompt_sha256(key.prompt_sha256);
        }
    }

    static void FillVLMResponseFromCache(
        VLMResponse* response,
        const vlm_cache::Result& cached,
        bool cache_stale,
        const std::string& result_source) {
        response->Clear();
        response->set_text(cached.text);
        response->set_image_encode_ms(cached.image_encode_ms);
        response->set_prompt_eval_ms(cached.prompt_eval_ms);
        response->set_eval_ms(cached.eval_ms);
        response->set_prompt_tokens(cached.prompt_tokens);
        response->set_generated_tokens(cached.generated_tokens);
        response->set_cache_hit(true);
        response->set_cache_stale(cache_stale);
        response->set_cache_key(cached.cache_key);
        response->set_image_sha256(cached.image_sha256);
        response->set_prompt_sha256(cached.prompt_sha256);
        response->set_result_source(result_source);
    }

    static void FillVLMTokenCacheMetadata(
        VLMToken* token,
        const vlm_cache::KeyInfo& key,
        bool cache_hit,
        bool cache_stale,
        const std::string& result_source) {
        token->set_cache_hit(cache_hit);
        token->set_cache_stale(cache_stale);
        token->set_result_source(result_source);
        if (!key.cache_key.empty()) {
            token->set_cache_key(key.cache_key);
        }
    }

    static void FillVLMTokenFromCache(
        VLMToken* token,
        const vlm_cache::Result& cached,
        bool cache_stale,
        const std::string& result_source) {
        token->set_cache_hit(true);
        token->set_cache_stale(cache_stale);
        token->set_cache_key(cached.cache_key);
        token->set_result_source(result_source);
    }

    static void WriteCachedStream(
        ServerWriter<VLMToken>* writer,
        const vlm_cache::Result& cached,
        bool cache_stale,
        const std::string& result_source) {
        if (!cached.text.empty()) {
            VLMToken token;
            token.set_token(cached.text);
            token.set_is_final(false);
            FillVLMTokenFromCache(&token, cached, cache_stale, result_source);
            writer->Write(token);
        }

        VLMToken final_token;
        final_token.set_is_final(true);
        FillVLMTokenFromCache(&final_token, cached, cache_stale, result_source);
        writer->Write(final_token);
    }

    void StoreVLMCacheResult(
        const VLMRequest& request,
        const std::vector<uint8_t>& image_data,
        const llm::GenerateParams& params,
        const vlm_cache::KeyInfo& key,
        const llm::VLMResult& result) {
        if (!result.success || !ShouldUseVLMCache(request) || key.cache_key.empty()) {
            return;
        }

        vlm_cache::StoreRecord record;
        record.key = key;
        record.model_fingerprint = model_fingerprint_;
        record.session_id = request.session_id();
        record.request_id = request.request_id();
        record.task_type = request.task_type();
        if (vlm_cache_.options().store_prompts) {
            record.prompt = request.prompt();
        }
        if (vlm_cache_.options().store_images) {
            record.image_data = image_data;
        }
        record.params = params;
        record.result.text = result.text;
        record.result.image_encode_ms = result.image_encode_ms;
        record.result.prompt_eval_ms = result.prompt_eval_ms;
        record.result.eval_ms = result.eval_ms;
        record.result.prompt_tokens = result.prompt_tokens;
        record.result.generated_tokens = result.generated_tokens;
        vlm_cache_.Put(std::move(record));
    }

    static double BytesToMiB(size_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    }

    std::optional<llm::DeviceMemoryInfo> LowestFreeGpuDevice() const {
        auto snapshot = llm_runner_.GetMemorySnapshot();
        std::optional<llm::DeviceMemoryInfo> lowest;
        for (const auto& device : snapshot.devices) {
            if (!device.is_gpu || device.total_bytes == 0) {
                continue;
            }
            if (!lowest || device.free_bytes < lowest->free_bytes) {
                lowest = device;
            }
        }
        return lowest;
    }

    bool IsBelowFreeWatermark(size_t watermark_bytes, std::string& message) const {
        if (watermark_bytes == 0) {
            return false;
        }
        auto device = LowestFreeGpuDevice();
        if (!device) {
            return false;
        }
        if (device->free_bytes >= watermark_bytes) {
            return false;
        }

        message = "GPU memory low on " + device->name +
            ": free=" + std::to_string(static_cast<int64_t>(BytesToMiB(device->free_bytes))) +
            "MiB threshold=" + std::to_string(static_cast<int64_t>(BytesToMiB(watermark_bytes))) +
            "MiB total=" + std::to_string(static_cast<int64_t>(BytesToMiB(device->total_bytes))) + "MiB";
        return true;
    }

    void UnloadLLMForVramPressure(const std::string& reason) {
        std::lock_guard lock(llm_load_mutex_);
        if (!llm_runner_.IsLoaded()) {
            return;
        }

        auto snapshot = llm_runner_.GetMemorySnapshot();
        LOG_WARN("[VramGuard] unloading LLM: reason={} model_size_mb={:.2f}",
                 reason,
                 BytesToMiB(static_cast<size_t>(snapshot.model_size_bytes)));
        llm_runner_.Unload();

        if (vram_options_.reload_after_unload) {
            LOG_INFO("[VramGuard] reloading LLM after unload");
            if (!llm_runner_.LoadModel(llm_model_path_, mmproj_path_, n_gpu_layers_)) {
                LOG_ERROR("[VramGuard] LLM reload failed after unload");
            } else {
                last_llm_request_time_ = std::chrono::steady_clock::now();
            }
        }
    }

    void HandleVramWatermark(const char* source) {
        std::string unload_message;
        if (IsBelowFreeWatermark(vram_options_.unload_free_bytes, unload_message)) {
            UnloadLLMForVramPressure(std::string(source) + ": " + unload_message);
            return;
        }

        std::string warning_message;
        if (IsBelowFreeWatermark(vram_options_.warning_free_bytes, warning_message)) {
            LOG_WARN("[VramGuard] {}", warning_message);
        }
    }

    void VramMonitorLoop(std::stop_token stop_token) {
        while (!stop_token.stop_requested()) {
            for (int second = 0; second < vram_options_.monitor_interval_seconds; ++second) {
                if (stop_token.stop_requested()) return;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            HandleVramWatermark("watchdog");
        }
    }

    static bool IsLikelyVramFailure(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value.find("out of memory") != std::string::npos ||
               value.find("oom") != std::string::npos ||
               value.find("cuda error") != std::string::npos ||
               value.find("cuda out") != std::string::npos ||
               value.find("failed to allocate") != std::string::npos;
    }

    void HandleVLMFailure(const std::string& error_message) {
        if (vram_options_.unload_on_oom_error && IsLikelyVramFailure(error_message)) {
            UnloadLLMForVramPressure("VLM failure: " + error_message);
        }
    }

    Status CheckAuth(
        const ServerContext& context,
        server_common::ScopedRequestStats& request_stats) const {
        auto validation = request_validation::ValidateAuth(context, auth_options_);
        if (!validation.ok) {
            request_stats.MarkFailure(validation.error);
            return Status(StatusCode::UNAUTHENTICATED, validation.error);
        }
        return Status::OK;
    }

    bool EnsureLLMLoaded(std::string& error_message) {
        if (llm_runner_.IsLoaded()) {
            return true;
        }

        std::lock_guard lock(llm_load_mutex_);
        if (llm_runner_.IsLoaded()) {
            return true;
        }

        if (std::string pressure_message; IsBelowFreeWatermark(vram_options_.min_free_before_load_bytes, pressure_message)) {
            error_message = pressure_message;
            return false;
        }

        LOG_INFO("Lazy loading LLM model");
        if (!llm_runner_.LoadModel(llm_model_path_, mmproj_path_, n_gpu_layers_)) {
            error_message = "Failed to load LLM";
            return false;
        }
        return true;
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
    request_validation::AuthOptions auth_options_;
    request_validation::RequestLimits request_limits_;
    VramGuardOptions vram_options_;
    vlm_cache::VLMCache vlm_cache_;

    // LLM lazy load 配置
    std::string llm_model_path_;
    std::string mmproj_path_;
    std::string model_fingerprint_;
    int n_gpu_layers_;
    std::mutex llm_load_mutex_;
    std::chrono::steady_clock::time_point last_llm_request_time_;
    std::jthread idle_thread_;
    std::jthread vram_thread_;
};

// ========== 参数解析 ==========

int ParsePositiveOption(const std::string& flag, const char* value) {
    int parsed = server_common::ParseIntValue(flag, value);
    if (parsed <= 0) {
        throw std::runtime_error(flag + " must be positive");
    }
    return parsed;
}

int ParseNonNegativeOption(const std::string& flag, const char* value) {
    int parsed = server_common::ParseIntValue(flag, value);
    if (parsed < 0) {
        throw std::runtime_error(flag + " must be non-negative");
    }
    return parsed;
}

size_t ParseMegabytesOption(const std::string& flag, const char* value) {
    return static_cast<size_t>(ParsePositiveOption(flag, value)) * 1024 * 1024;
}

size_t ParseOptionalMegabytesOption(const std::string& flag, const char* value) {
    return static_cast<size_t>(ParseNonNegativeOption(flag, value)) * 1024 * 1024;
}

std::string TrimToken(std::string value) {
    if (value.size() >= 3 &&
        static_cast<unsigned char>(value[0]) == 0xEF &&
        static_cast<unsigned char>(value[1]) == 0xBB &&
        static_cast<unsigned char>(value[2]) == 0xBF) {
        value.erase(0, 3);
    }

    const auto not_space = [](unsigned char c) {
        return !std::isspace(c);
    };

    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

void ValidateResolvedToken(const std::string& token, const std::string& source) {
    if (token.empty()) {
        throw std::runtime_error("Auth token from " + source + " is empty");
    }
    for (unsigned char c : token) {
        if (c == 0 || c < 0x21 || c == 0x7F) {
            throw std::runtime_error("Auth token from " + source + " contains invalid characters");
        }
    }
}

std::string ReadAuthTokenFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open auth token file: " + path.string());
    }
    std::string value(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    return TrimToken(std::move(value));
}

std::optional<std::string> ReadAuthTokenEnv(const std::string& name) {
    if (name.empty()) {
        return std::nullopt;
    }
    const char* value = std::getenv(name.c_str());
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return TrimToken(value);
}

void ResolveAuthToken(MultimodalServerOptions& options) {
    if (!options.auth.token.empty()) {
        options.auth.token = TrimToken(options.auth.token);
        ValidateResolvedToken(options.auth.token, "--auth-token");
        options.auth_source = "command-line";
        return;
    }

    if (!options.auth_token_file.empty()) {
        options.auth.token = ReadAuthTokenFile(options.auth_token_file);
        ValidateResolvedToken(options.auth.token, "--auth-token-file");
        options.auth_source = "file";
        return;
    }

    if (auto env_token = ReadAuthTokenEnv(options.auth_token_env)) {
        options.auth.token = *env_token;
        ValidateResolvedToken(options.auth.token, options.auth_token_env);
        options.auth_source = "environment";
        return;
    }

    options.auth_source = "disabled";
}

MultimodalServerOptions ParseMultimodalOptions(int argc, char** argv) {
    MultimodalServerOptions options;
    options.grpc.max_receive_message_mb = 100;
    options.grpc.max_send_message_mb = 10;

    if (auto config_path = server_config::FindConfigPath(argc, argv)) {
        server_config::LoadConfigFile(*config_path, options);
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--config" && i + 1 < argc) {
            ++i;
        } else if (arg.rfind("--config=", 0) == 0) {
        } else if (arg == "--bert" && i + 1 < argc) {
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
        } else if (arg == "--auth-token" && i + 1 < argc) {
            options.auth.token = argv[++i];
        } else if (arg == "--auth-token-file" && i + 1 < argc) {
            options.auth_token_file = argv[++i];
        } else if (arg == "--auth-token-env" && i + 1 < argc) {
            options.auth_token_env = argv[++i];
        } else if (arg == "--auth-header" && i + 1 < argc) {
            options.auth.metadata_key = argv[++i];
        } else if (arg == "--max-image-mb" && i + 1 < argc) {
            options.limits.max_image_bytes = ParseMegabytesOption(arg, argv[++i]);
        } else if (arg == "--max-image-pixels" && i + 1 < argc) {
            options.limits.max_image_pixels = static_cast<size_t>(ParsePositiveOption(arg, argv[++i]));
        } else if (arg == "--max-image-width" && i + 1 < argc) {
            options.limits.max_image_width = static_cast<uint32_t>(ParsePositiveOption(arg, argv[++i]));
        } else if (arg == "--max-image-height" && i + 1 < argc) {
            options.limits.max_image_height = static_cast<uint32_t>(ParsePositiveOption(arg, argv[++i]));
        } else if (arg == "--max-prompt-bytes" && i + 1 < argc) {
            options.limits.max_prompt_bytes = static_cast<size_t>(ParsePositiveOption(arg, argv[++i]));
        } else if (arg == "--max-seq-len" && i + 1 < argc) {
            options.limits.max_sequence_length = ParsePositiveOption(arg, argv[++i]);
        } else if (arg == "--max-batch-size" && i + 1 < argc) {
            options.limits.max_batch_size = ParsePositiveOption(arg, argv[++i]);
        } else if (arg == "--max-vlm-tokens" && i + 1 < argc) {
            options.limits.max_vlm_tokens = ParsePositiveOption(arg, argv[++i]);
        } else if (arg == "--max-context-size" && i + 1 < argc) {
            options.limits.max_context_size = ParsePositiveOption(arg, argv[++i]);
        } else if (arg == "--max-token-id" && i + 1 < argc) {
            options.limits.max_token_id = ParsePositiveOption(arg, argv[++i]);
        } else if (arg == "--vlm-cache-enabled") {
            options.vlm_cache.enabled = true;
        } else if (arg == "--vlm-cache-persist") {
            options.vlm_cache.persist = true;
        } else if (arg == "--vlm-cache-dir" && i + 1 < argc) {
            options.vlm_cache.cache_dir = argv[++i];
        } else if (arg == "--vlm-cache-max-entries" && i + 1 < argc) {
            options.vlm_cache.max_entries = static_cast<size_t>(ParseNonNegativeOption(arg, argv[++i]));
        } else if (arg == "--vlm-cache-max-mb" && i + 1 < argc) {
            options.vlm_cache.max_bytes = ParseOptionalMegabytesOption(arg, argv[++i]);
        } else if (arg == "--vlm-cache-ttl-seconds" && i + 1 < argc) {
            options.vlm_cache.ttl_seconds = ParseNonNegativeOption(arg, argv[++i]);
        } else if (arg == "--no-vlm-cache-store-images") {
            options.vlm_cache.store_images = false;
        } else if (arg == "--no-vlm-cache-store-prompts") {
            options.vlm_cache.store_prompts = false;
        } else if (arg == "--no-vlm-cache-stale-on-failure") {
            options.vlm_cache.allow_stale_on_failure = false;
        } else if (arg == "--no-vlm-cache-default") {
            options.vlm_cache.default_allow_cache = false;
        } else if (arg == "--vram-monitor-interval-seconds" && i + 1 < argc) {
            options.vram.monitor_interval_seconds = ParseNonNegativeOption(arg, argv[++i]);
        } else if (arg == "--vram-warning-free-mb" && i + 1 < argc) {
            options.vram.warning_free_bytes = ParseOptionalMegabytesOption(arg, argv[++i]);
        } else if (arg == "--vram-unload-free-mb" && i + 1 < argc) {
            options.vram.unload_free_bytes = ParseOptionalMegabytesOption(arg, argv[++i]);
        } else if (arg == "--vram-min-free-before-load-mb" && i + 1 < argc) {
            options.vram.min_free_before_load_bytes = ParseOptionalMegabytesOption(arg, argv[++i]);
        } else if (arg == "--vram-reload-after-unload") {
            options.vram.reload_after_unload = true;
        } else if (arg == "--no-vram-unload-on-oom-error") {
            options.vram.unload_on_oom_error = false;
        }
    }

    server_common::ParseGrpcServerOptions(argc, argv, 1, options.grpc);
    if (options.auth.metadata_key.empty()) {
        throw std::runtime_error("--auth-header must not be empty");
    }
    if (options.limits.max_context_size < options.limits.min_context_size) {
        throw std::runtime_error("--max-context-size is below the minimum context size");
    }
    if (options.vram.warning_free_bytes > 0 &&
        options.vram.unload_free_bytes > 0 &&
        options.vram.unload_free_bytes > options.vram.warning_free_bytes) {
        throw std::runtime_error("--vram-unload-free-mb must be less than or equal to --vram-warning-free-mb");
    }
    if (options.vlm_cache.persist && options.vlm_cache.cache_dir.empty()) {
        throw std::runtime_error("--vlm-cache-dir must not be empty when persistence is enabled");
    }
    ResolveAuthToken(options);
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
