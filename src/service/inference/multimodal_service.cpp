#include "multimodal_service.h"

#include "llama_runner.h"
#include "onnx_model.h"
#include "onnx_session_utils.h"
#include "redis_connection_pool.h"
#include "vector_cache.h"
#include "vlm_cache.h"
#include "../../core/logger_adapter.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace service {

namespace {

static core::LoggerAdapter logger = core::LoggerAdapter::ForModule("service");

using multimodal_inference::EmotionBatchRequest;
using multimodal_inference::EmotionBatchResponse;
using multimodal_inference::EmotionRequest;
using multimodal_inference::EmotionResponse;
using multimodal_inference::SaliencyRequest;
using multimodal_inference::SaliencyResponse;
using multimodal_inference::VLMRequest;
using multimodal_inference::VLMResponse;
using multimodal_inference::VLMToken;

core::Status Error(core::ErrorCode code, std::string message) {
    return core::Status::Error(code, std::move(message));
}

bert::ModelRuntimeOptions ToModelRuntimeOptions(const BertRuntimeConfigOptions& config) {
    bert::ModelRuntimeOptions options;
    options.execution_provider = config.execution_provider;
    options.allow_cpu_fallback = config.allow_cpu_fallback;
    options.cuda_device_id = config.cuda_device_id;
    options.intra_op_num_threads = config.intra_op_num_threads;
    options.inter_op_num_threads = config.inter_op_num_threads;
    options.enable_cpu_mem_arena = config.enable_cpu_mem_arena;
    options.enable_mem_pattern = config.enable_mem_pattern;
    return options;
}

vlm_cache::Options ToVlmCacheOptions(const VlmCacheConfigOptions& config) {
    vlm_cache::Options options;
    options.enabled = config.enabled;
    options.persist = config.persist;
    options.cache_dir = config.cache_dir;
    options.max_entries = config.max_entries;
    options.max_bytes = config.max_bytes;
    options.ttl_seconds = config.ttl_seconds;
    options.store_images = config.store_images;
    options.store_prompts = config.store_prompts;
    options.allow_stale_on_failure = config.allow_stale_on_failure;
    options.default_allow_cache = config.default_allow_cache;
    return options;
}

vlm_cache::VectorOptions ToVlmCacheVectorOptions(const VlmCacheVectorOptions& config) {
    vlm_cache::VectorOptions options;
    options.enabled = config.enabled;
    options.sim_threshold_high = config.sim_threshold_high;
    options.sim_threshold_mid = config.sim_threshold_mid;
    options.max_saliency_for_mid = config.max_saliency_for_mid;
    options.max_entries_per_bucket = config.max_entries_per_bucket;
    options.ttl_seconds = config.ttl_seconds;
    options.persist = config.persist;
    options.vector_dir = config.vector_dir;
    return options;
}

double BytesToMiB(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

class MemoryPromptKvCache final : public llm::IPromptKvCache {
public:
    explicit MemoryPromptKvCache(std::size_t max_bytes)
        : max_bytes_(max_bytes) {}

    std::optional<llm::PromptKvCacheEntry> Load(const std::string& key) override {
        std::lock_guard lock(mutex_);
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    void Store(const std::string& key, const llm::PromptKvCacheEntry& entry) override {
        if (key.empty() || entry.state.empty() || entry.prefix_tokens <= 0) {
            return;
        }
        const auto entry_bytes = entry.state.size();
        if (max_bytes_ > 0 && entry_bytes > max_bytes_) {
            spdlog::warn("[PromptKvCache] skip oversized memory state: key={} bytes={} max_bytes={}",
                         key, entry_bytes, max_bytes_);
            return;
        }

        std::lock_guard lock(mutex_);
        if (auto existing = entries_.find(key); existing != entries_.end()) {
            current_bytes_ -= existing->second.state.size();
            existing->second = entry;
            current_bytes_ += entry_bytes;
        } else {
            entries_.emplace(key, entry);
            order_.push_back(key);
            current_bytes_ += entry_bytes;
        }
        EvictIfNeeded();
    }

private:
    void EvictIfNeeded() {
        if (max_bytes_ == 0) {
            return;
        }
        while (current_bytes_ > max_bytes_ && !order_.empty()) {
            auto key = std::move(order_.front());
            order_.pop_front();
            auto it = entries_.find(key);
            if (it == entries_.end()) {
                continue;
            }
            current_bytes_ -= it->second.state.size();
            entries_.erase(it);
        }
    }

    std::mutex mutex_;
    std::unordered_map<std::string, llm::PromptKvCacheEntry> entries_;
    std::deque<std::string> order_;
    std::size_t current_bytes_ = 0;
    std::size_t max_bytes_ = 0;
};

class RedisPromptKvCache final : public llm::IPromptKvCache {
public:
    RedisPromptKvCache(std::shared_ptr<agent::semantic_cache::RedisConnectionPool> redis,
                       std::string key_prefix,
                       std::chrono::seconds ttl,
                       std::size_t max_bytes)
        : redis_(std::move(redis)),
          key_prefix_(std::move(key_prefix)),
          ttl_(ttl),
          max_bytes_(max_bytes) {}

    std::optional<llm::PromptKvCacheEntry> Load(const std::string& key) override {
        if (!redis_ || !redis_->running()) {
            return std::nullopt;
        }
        auto value = redis_->Get(Key(key));
        if (!value.ok()) {
            return std::nullopt;
        }
        return Decode(value.value());
    }

    void Store(const std::string& key, const llm::PromptKvCacheEntry& entry) override {
        if (!redis_ || !redis_->running() || entry.state.empty() || entry.prefix_tokens <= 0) {
            return;
        }
        auto payload = Encode(entry);
        if (payload.size() > max_bytes_) {
            spdlog::warn("[PromptKvCache] skip oversized state: key={} bytes={} max_bytes={}",
                         key, payload.size(), max_bytes_);
            return;
        }
        auto status = redis_->Set(Key(key), payload, ttl_);
        if (!status.ok()) {
            spdlog::warn("[PromptKvCache] redis store failed: key={} error={}", key, status.message());
        }
    }

private:
    static constexpr std::string_view kMagic = "VLMKV1";

    std::string Key(const std::string& key) const {
        return key_prefix_ + ":" + key;
    }

    static void AppendU32(std::string& out, std::uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
        }
    }

    static std::uint32_t ReadU32(const std::string& value, std::size_t offset) {
        std::uint32_t result = 0;
        for (int i = 0; i < 4; ++i) {
            result |= static_cast<std::uint32_t>(
                static_cast<unsigned char>(value[offset + i])) << (i * 8);
        }
        return result;
    }

    static std::string Encode(const llm::PromptKvCacheEntry& entry) {
        std::string out;
        out.reserve(kMagic.size() + 8 + entry.state.size());
        out.append(kMagic);
        AppendU32(out, static_cast<std::uint32_t>(entry.prefix_tokens));
        AppendU32(out, static_cast<std::uint32_t>(entry.state.size()));
        out.append(reinterpret_cast<const char*>(entry.state.data()), entry.state.size());
        return out;
    }

    static std::optional<llm::PromptKvCacheEntry> Decode(const std::string& value) {
        if (value.size() < kMagic.size() + 8 ||
            std::string_view(value.data(), kMagic.size()) != kMagic) {
            return std::nullopt;
        }
        const auto prefix_tokens = ReadU32(value, kMagic.size());
        const auto state_size = ReadU32(value, kMagic.size() + 4);
        if (prefix_tokens == 0 || value.size() != kMagic.size() + 8 + state_size) {
            return std::nullopt;
        }
        llm::PromptKvCacheEntry entry;
        entry.prefix_tokens = static_cast<int32_t>(prefix_tokens);
        entry.state.resize(state_size);
        std::memcpy(entry.state.data(), value.data() + kMagic.size() + 8, state_size);
        return entry;
    }

    std::shared_ptr<agent::semantic_cache::RedisConnectionPool> redis_;
    std::string key_prefix_;
    std::chrono::seconds ttl_;
    std::size_t max_bytes_ = 0;
};

} // namespace

class MultimodalService::Impl {
public:
    static constexpr std::int32_t kMinTokensForVectorCache = 8;

    explicit Impl(const MultimodalServerOptions& options)
        : vram_options_(options.vram),
          vlm_cache_(ToVlmCacheOptions(options.vlm_cache)),
          vector_index_(ToVlmCacheVectorOptions(options.vlm_cache_vector)),
          prompt_kv_options_(options.vlm_prompt_kv_cache),
          llm_model_path_(options.llm_model),
          mmproj_path_(options.mmproj),
          model_fingerprint_(BuildModelFingerprint(options.llm_model, options.mmproj, options.n_gpu_layers)),
          n_gpu_layers_(options.n_gpu_layers),
          runner_pool_size_(options.runner_pool_size),
          llama_threads_(options.llama_threads),
          mmproj_threads_(options.mmproj_threads) {
        InitializePromptKvCache();

        if (!options.bert_model.empty()) {
            logger.info("Loading BERT model: {}", options.bert_model);
            if (!bert_model_.LoadModel(options.bert_model, ToModelRuntimeOptions(options.bert_runtime))) {
                logger.error("Failed to load BERT model");
            }
        }

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

    ~Impl() {
        if (prompt_kv_redis_) {
            prompt_kv_redis_->Shutdown();
        }
    }

    core::Status PredictEmotion(const EmotionRequest& request, EmotionResponse& response) {
        if (!bert_model_.IsLoaded()) {
            response.set_error("BERT model not loaded");
            return Error(core::ErrorCode::FailedPrecondition, "BERT model not loaded");
        }

        std::vector<std::int64_t> input_ids(request.input_ids().begin(), request.input_ids().end());
        std::vector<std::int64_t> attention_mask(request.attention_mask().begin(), request.attention_mask().end());
        std::vector<float> personality(request.personality().begin(), request.personality().end());

        auto result = bert_model_.Predict(input_ids, attention_mask, personality);
        if (!result.success) {
            response.set_error(result.error_message);
            return Error(core::ErrorCode::InternalError, result.error_message);
        }

        response.mutable_emotion_logits()->Reserve(static_cast<int>(result.emotion_logits.size()));
        for (float v : result.emotion_logits) response.add_emotion_logits(v);

        response.mutable_behavior_logits()->Reserve(static_cast<int>(result.behavior_logits.size()));
        for (float v : result.behavior_logits) response.add_behavior_logits(v);

        response.mutable_tone_logits()->Reserve(static_cast<int>(result.tone_logits.size()));
        for (float v : result.tone_logits) response.add_tone_logits(v);

        response.set_intensity(result.intensity);

        response.mutable_response_length_logits()->Reserve(static_cast<int>(result.response_length_logits.size()));
        for (float v : result.response_length_logits) response.add_response_length_logits(v);

        return core::Status::Ok();
    }

    core::Status PredictEmotionBatch(const EmotionBatchRequest& request, EmotionBatchResponse& response) {
        const auto batch_size = static_cast<std::size_t>(request.batch_size());
        const auto seq_len = static_cast<std::size_t>(request.seq_length());

        if (!bert_model_.IsLoaded()) {
            response.set_error("BERT model not loaded");
            return Error(core::ErrorCode::FailedPrecondition, "BERT model not loaded");
        }

        std::vector<std::int64_t> input_ids(request.input_ids().begin(), request.input_ids().end());
        std::vector<std::int64_t> attention_mask(request.attention_mask().begin(), request.attention_mask().end());
        std::vector<float> personality(request.personality().begin(), request.personality().end());

        auto results = bert_model_.PredictBatch(input_ids, attention_mask, personality, batch_size, seq_len);
        if (results.empty()) {
            response.set_error("Batch inference failed");
            return Error(core::ErrorCode::InternalError, "Batch inference failed");
        }

        response.mutable_emotion_logits()->Reserve(static_cast<int>(batch_size * results.front().emotion_logits.size()));
        response.mutable_behavior_logits()->Reserve(static_cast<int>(batch_size * results.front().behavior_logits.size()));
        response.mutable_tone_logits()->Reserve(static_cast<int>(batch_size * results.front().tone_logits.size()));
        response.mutable_intensity()->Reserve(static_cast<int>(batch_size));
        response.mutable_response_length_logits()->Reserve(static_cast<int>(batch_size * results.front().response_length_logits.size()));

        for (const auto& result : results) {
            for (float v : result.emotion_logits) response.add_emotion_logits(v);
            for (float v : result.behavior_logits) response.add_behavior_logits(v);
            for (float v : result.tone_logits) response.add_tone_logits(v);
            response.add_intensity(result.intensity);
            for (float v : result.response_length_logits) response.add_response_length_logits(v);
        }

        return core::Status::Ok();
    }

    core::Status DetectSaliency(const SaliencyRequest&, SaliencyResponse& response) {
        response.set_saliency_score(0.0f);
        response.set_trigger_vlm(false);
        response.set_inference_ms(0.0f);
        return Error(core::ErrorCode::Unimplemented, "ViT not implemented yet");
    }

    core::Status GenerateVLM(const VLMRequest& request, const VlmTokenEmitter& emit) {
        if (!emit) {
            return Error(core::ErrorCode::InvalidArgument, "VLM token emitter is empty");
        }

        auto params = BuildGenerateParamsWithEmbedding(request);
        auto image_data = ExtractImageData(request);
        ConfigurePromptKvCache(request, image_data, &params);
        auto key = BuildVLMCacheKey(image_data, request.prompt(), params);
        const bool use_cache = ShouldUseVLMCache(request);

        if (use_cache && !request.force_refresh()) {
            if (auto cached = vlm_cache_.Get(key.cache_key)) {
                EmitCachedStream(emit, *cached, false, "exact_cache");
                return core::Status::Ok();
            }

            vlm_cache::Result vec_result;
            std::string vec_source;
            if (TryVectorCache(request, image_data, key, &vec_result, &vec_source)) {
                EmitCachedStream(emit, vec_result, false, vec_source);
                return core::Status::Ok();
            }
        }

        std::string load_error;
        if (!EnsureLLMLoaded(load_error)) {
            if (load_error.empty()) {
                load_error = "Failed to load LLM";
            }
            if (auto fallback = GetStaleVLMFallback(request, key.cache_key)) {
                EmitCachedStream(emit, *fallback, true, "stale_cache");
                return core::Status::Ok();
            }
            const auto code = load_error.find("GPU memory low") != std::string::npos
                ? core::ErrorCode::ResourceExhausted
                : core::ErrorCode::InternalError;
            return Error(code, load_error);
        }

        last_llm_request_time_ = std::chrono::steady_clock::now();

        bool stream_wrote_token = false;
        llm::VLMResult result;
        {
            auto runner = llm_runner_pool_.Acquire();
            if (!runner) {
                return Error(core::ErrorCode::FailedPrecondition, "LLM runner pool not loaded");
            }

            result = runner->Generate(
                image_data,
                request.prompt(),
                params,
                [&emit, &stream_wrote_token](const std::string& token) {
                    stream_wrote_token = true;
                    VLMToken event;
                    event.set_token(token);
                    event.set_is_final(false);
                    emit(std::move(event));
                });
        }

        if (result.success) {
            StoreVLMCacheResult(request, image_data, params, key, result);
            StoreVectorCacheEntry(key, result);
            VLMToken final_token;
            final_token.set_is_final(true);
            FillVLMTokenCacheMetadata(&final_token, key, false, false, "fresh");
            emit(std::move(final_token));
            return core::Status::Ok();
        }

        HandleVLMFailure(result.error_message);
        if (!stream_wrote_token) {
            if (auto fallback = GetStaleVLMFallback(request, key.cache_key)) {
                EmitCachedStream(emit, *fallback, true, "stale_cache");
                return core::Status::Ok();
            }
        }

        VLMToken final_token;
        final_token.set_is_final(true);
        final_token.set_error(result.error_message);
        FillVLMTokenCacheMetadata(&final_token, key, false, false, "error");
        emit(std::move(final_token));
        return Error(core::ErrorCode::InternalError, result.error_message);
    }

    core::Status GenerateVLMSync(const VLMRequest& request, VLMResponse& response) {
        auto params = BuildGenerateParamsWithEmbedding(request);
        auto image_data = ExtractImageData(request);
        ConfigurePromptKvCache(request, image_data, &params);
        auto key = BuildVLMCacheKey(image_data, request.prompt(), params);
        const bool use_cache = ShouldUseVLMCache(request);

        if (use_cache && !request.force_refresh()) {
            if (auto cached = vlm_cache_.Get(key.cache_key)) {
                FillVLMResponseFromCache(&response, *cached, false, "exact_cache");
                return core::Status::Ok();
            }

            vlm_cache::Result vec_result;
            std::string vec_source;
            if (TryVectorCache(request, image_data, key, &vec_result, &vec_source)) {
                FillVLMResponseFromCache(&response, vec_result, false, vec_source);
                return core::Status::Ok();
            }
        }

        std::string load_error;
        if (!EnsureLLMLoaded(load_error)) {
            if (load_error.empty()) {
                load_error = "Failed to load LLM";
            }
            if (auto fallback = GetStaleVLMFallback(request, key.cache_key)) {
                FillVLMResponseFromCache(&response, *fallback, true, "stale_cache");
                return core::Status::Ok();
            }
            response.set_error(load_error);
            const auto code = load_error.find("GPU memory low") != std::string::npos
                ? core::ErrorCode::ResourceExhausted
                : core::ErrorCode::InternalError;
            return Error(code, load_error);
        }

        last_llm_request_time_ = std::chrono::steady_clock::now();

        llm::VLMResult result;
        {
            auto runner = llm_runner_pool_.Acquire();
            if (!runner) {
                response.set_error("LLM runner pool not loaded");
                return Error(core::ErrorCode::FailedPrecondition, "LLM runner pool not loaded");
            }

            result = runner->Generate(image_data, request.prompt(), params);
        }
        FillVLMResponseFromResult(&response, result);

        if (result.success) {
            StoreVLMCacheResult(request, image_data, params, key, result);
            StoreVectorCacheEntry(key, result);
            FillVLMResponseCacheMetadata(&response, key, false, false, "fresh");
            return core::Status::Ok();
        }

        HandleVLMFailure(result.error_message);
        if (auto fallback = GetStaleVLMFallback(request, key.cache_key)) {
            FillVLMResponseFromCache(&response, *fallback, true, "stale_cache");
            return core::Status::Ok();
        }
        response.set_error(result.error_message);
        FillVLMResponseCacheMetadata(&response, key, false, false, "error");
        return Error(core::ErrorCode::InternalError, result.error_message);
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
        const std::vector<std::uint8_t>& image_data,
        const std::string& prompt,
        const llm::GenerateParams& params) const {
        if (!vlm_cache_.options().enabled) {
            return {};
        }
        return vlm_cache_.BuildKey(image_data, prompt, params, model_fingerprint_);
    }

    void InitializePromptKvCache() {
        if (!prompt_kv_options_.enabled) {
            return;
        }

        if (prompt_kv_options_.backend == "memory") {
            prompt_kv_cache_ = std::make_shared<MemoryPromptKvCache>(prompt_kv_options_.max_bytes);
            logger.info("[PromptKvCache] enabled: backend=memory max_mb={}",
                        prompt_kv_options_.max_bytes / (1024 * 1024));
            return;
        }

        agent::semantic_cache::RedisPoolOptions redis_options;
        redis_options.host = prompt_kv_options_.redis_host;
        redis_options.port = std::to_string(prompt_kv_options_.redis_port);
        redis_options.password = prompt_kv_options_.redis_password;
        redis_options.pool_size = prompt_kv_options_.redis_pool_size;
        redis_options.command_timeout = std::chrono::milliseconds(prompt_kv_options_.redis_command_timeout_ms);

        prompt_kv_redis_ = std::make_shared<agent::semantic_cache::RedisConnectionPool>(redis_options);
        auto status = prompt_kv_redis_->Start();
        if (!status.ok()) {
            logger.warn("[PromptKvCache] disabled: {}", status.message());
            prompt_kv_redis_.reset();
            return;
        }

        prompt_kv_cache_ = std::make_shared<RedisPromptKvCache>(
            prompt_kv_redis_,
            prompt_kv_options_.key_prefix,
            std::chrono::seconds(prompt_kv_options_.ttl_seconds),
            prompt_kv_options_.max_bytes);
        logger.info("[PromptKvCache] enabled: backend=redis redis={}:{} prefix={} ttl_seconds={} max_mb={}",
                    prompt_kv_options_.redis_host,
                    prompt_kv_options_.redis_port,
                    prompt_kv_options_.key_prefix,
                    prompt_kv_options_.ttl_seconds,
                    prompt_kv_options_.max_bytes / (1024 * 1024));
    }

    std::string BuildPromptKvCacheKey(
        const VLMRequest& request,
        const std::vector<std::uint8_t>& image_data) const {
        if (!prompt_kv_cache_ || image_data.empty() || request.session_id().empty()) {
            return {};
        }
        std::ostringstream out;
        out << "prompt-kv-v1\n"
            << model_fingerprint_ << '\n'
            << request.session_id() << '\n'
            << vlm_cache::Sha256Hex(image_data.data(), image_data.size()) << '\n';
        return vlm_cache::Sha256Hex(out.str());
    }

    void ConfigurePromptKvCache(
        const VLMRequest& request,
        const std::vector<std::uint8_t>& image_data,
        llm::GenerateParams* params) const {
        if (!params) {
            return;
        }
        params->prompt_kv_cache_key = BuildPromptKvCacheKey(request, image_data);
        params->enable_prompt_kv_cache = !params->prompt_kv_cache_key.empty();
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

    static void EmitCachedStream(
        const VlmTokenEmitter& emit,
        const vlm_cache::Result& cached,
        bool cache_stale,
        const std::string& result_source) {
        if (!cached.text.empty()) {
            VLMToken token;
            token.set_token(cached.text);
            token.set_is_final(false);
            FillVLMTokenFromCache(&token, cached, cache_stale, result_source);
            emit(std::move(token));
        }

        VLMToken final_token;
        final_token.set_is_final(true);
        FillVLMTokenFromCache(&final_token, cached, cache_stale, result_source);
        emit(std::move(final_token));
    }

    void StoreVLMCacheResult(
        const VLMRequest& request,
        const std::vector<std::uint8_t>& image_data,
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

    bool TryVectorCache(
        const VLMRequest& request,
        const std::vector<std::uint8_t>& image_data,
        const vlm_cache::KeyInfo& key,
        vlm_cache::Result* out_result,
        std::string* out_source) {
        if (!vector_index_.options().enabled || image_data.empty()) {
            return false;
        }
        if (!llm_runner_pool_.IsLoaded()) {
            return false;
        }

        auto runner = llm_runner_pool_.Acquire();
        if (!runner) {
            return false;
        }
        auto encode_result = runner->EncodeImageOnly(image_data);
        if (!encode_result.success || encode_result.image_embedding.empty()) {
            logger.debug("[VectorCache] EncodeImageOnly failed: {}", encode_result.error_message);
            return false;
        }

        const float saliency = request.saliency_hint();
        auto hit = vector_index_.Query(
            key.prompt_sha256, model_fingerprint_,
            encode_result.image_embedding, saliency);

        if (!hit) {
            return false;
        }

        auto cached = vlm_cache_.Get(hit->cache_key);
        if (!cached) {
            vector_index_.Evict(key.prompt_sha256, hit->cache_key);
            logger.debug("[VectorCache] hit key {} not in VLMCache, evicted from vector index", hit->cache_key);
            return false;
        }

        *out_result = std::move(*cached);
        *out_source = hit->tentative ? "vector_cache_tentative" : "vector_cache";
        spdlog::info("[VectorCache] {} hit: sim={:.4f} key={}",
                     hit->tentative ? "tentative" : "confident",
                     hit->similarity, hit->cache_key);
        return true;
    }

    void StoreVectorCacheEntry(
        const vlm_cache::KeyInfo& key,
        const llm::VLMResult& result) {
        if (!vector_index_.options().enabled || result.image_embedding.empty()) {
            return;
        }
        if (result.generated_tokens < kMinTokensForVectorCache) {
            spdlog::debug("[VectorCache] skip store: generated_tokens={} < min={}",
                          result.generated_tokens, kMinTokensForVectorCache);
            return;
        }
        vlm_cache::VectorEntry entry;
        entry.cache_key = key.cache_key;
        entry.model_fingerprint = model_fingerprint_;
        entry.embedding = result.image_embedding;
        vector_index_.Put(key.prompt_sha256, std::move(entry));
    }

    std::optional<llm::DeviceMemoryInfo> LowestFreeGpuDevice() const {
        auto snapshot = llm_runner_pool_.GetMemorySnapshot();
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

    bool IsBelowFreeWatermark(std::size_t watermark_bytes, std::string& message) const {
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
            ": free=" + std::to_string(static_cast<std::int64_t>(BytesToMiB(device->free_bytes))) +
            "MiB threshold=" + std::to_string(static_cast<std::int64_t>(BytesToMiB(watermark_bytes))) +
            "MiB total=" + std::to_string(static_cast<std::int64_t>(BytesToMiB(device->total_bytes))) + "MiB";
        return true;
    }

    void UnloadLLMForVramPressure(const std::string& reason) {
        std::lock_guard lock(llm_load_mutex_);
        if (!llm_runner_pool_.IsLoaded()) {
            return;
        }

        auto snapshot = llm_runner_pool_.GetMemorySnapshot();
        spdlog::warn("[VramGuard] unloading LLM: reason={} model_size_mb={:.2f}",
                     reason,
                     BytesToMiB(static_cast<std::size_t>(snapshot.model_size_bytes)));
        llm_runner_pool_.Unload();

        if (vram_options_.reload_after_unload) {
            spdlog::info("[VramGuard] reloading LLM after unload");
            if (!llm_runner_pool_.Load(llm_model_path_, mmproj_path_, n_gpu_layers_,
                                       runner_pool_size_, llama_threads_, mmproj_threads_,
                                       prompt_kv_cache_)) {
                spdlog::error("[VramGuard] LLM reload failed after unload");
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
            spdlog::warn("[VramGuard] {}", warning_message);
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

    bool EnsureLLMLoaded(std::string& error_message) {
        if (llm_runner_pool_.IsLoaded()) {
            return true;
        }

        std::lock_guard lock(llm_load_mutex_);
        if (llm_runner_pool_.IsLoaded()) {
            return true;
        }

        if (std::string pressure_message; IsBelowFreeWatermark(vram_options_.min_free_before_load_bytes, pressure_message)) {
            error_message = pressure_message;
            return false;
        }

        spdlog::info("Lazy loading LLM model runner pool: slots={} llama_threads={} mmproj_threads={}",
                     runner_pool_size_,
                     llama_threads_,
                     mmproj_threads_);
        if (!llm_runner_pool_.Load(llm_model_path_, mmproj_path_, n_gpu_layers_,
                                   runner_pool_size_, llama_threads_, mmproj_threads_,
                                   prompt_kv_cache_)) {
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

    llm::GenerateParams BuildGenerateParamsWithEmbedding(const VLMRequest& request) const {
        auto params = BuildGenerateParams(request);
        if (vector_index_.options().enabled) {
            params.capture_image_embedding = true;
        }
        return params;
    }

    static std::vector<std::uint8_t> ExtractImageData(const VLMRequest& request) {
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

            if (llm_runner_pool_.IsLoaded()) {
                auto now = std::chrono::steady_clock::now();
                auto idle_time = now - last_llm_request_time_;

                if (idle_time > idle_timeout) {
                    spdlog::info("LLM idle for {} minutes, unloading to free VRAM",
                                 std::chrono::duration_cast<std::chrono::minutes>(idle_time).count());
                    llm_runner_pool_.Unload();
                }
            }
        }
    }

    bert::OnnxBERTModel bert_model_;
    llm::LlamaRunnerPool llm_runner_pool_;
    VramGuardOptions vram_options_;
    vlm_cache::VLMCache vlm_cache_;
    vlm_cache::VectorIndex vector_index_;
    VlmPromptKvCacheOptions prompt_kv_options_;
    std::shared_ptr<agent::semantic_cache::RedisConnectionPool> prompt_kv_redis_;
    std::shared_ptr<llm::IPromptKvCache> prompt_kv_cache_;
    std::string llm_model_path_;
    std::string mmproj_path_;
    std::string model_fingerprint_;
    int n_gpu_layers_ = -1;
    std::size_t runner_pool_size_ = 1;
    int llama_threads_ = 8;
    int mmproj_threads_ = 8;
    std::mutex llm_load_mutex_;
    std::chrono::steady_clock::time_point last_llm_request_time_;
    std::jthread idle_thread_;
    std::jthread vram_thread_;
};

MultimodalService::MultimodalService(const MultimodalServerOptions& options)
    : impl_(std::make_unique<Impl>(options)) {}

MultimodalService::~MultimodalService() = default;

core::Status MultimodalService::PredictEmotion(const EmotionRequest& request, EmotionResponse& response) {
    return impl_->PredictEmotion(request, response);
}

core::Status MultimodalService::PredictEmotionBatch(const EmotionBatchRequest& request, EmotionBatchResponse& response) {
    return impl_->PredictEmotionBatch(request, response);
}

core::Status MultimodalService::DetectSaliency(const SaliencyRequest& request, SaliencyResponse& response) {
    return impl_->DetectSaliency(request, response);
}

core::Status MultimodalService::GenerateVLM(const VLMRequest& request, VlmTokenEmitter emit) {
    return impl_->GenerateVLM(request, emit);
}

core::Status MultimodalService::GenerateVLMSync(const VLMRequest& request, VLMResponse& response) {
    return impl_->GenerateVLMSync(request, response);
}

} // namespace service
