/**
 * @file llama_runner.cpp
 * @brief llama.cpp VLM 推理实现
 */

#include "llama_runner.h"
#include "llama_handles.h"
#include "../core/logger_adapter.h"

#include <llama.h>
#include <ggml-backend.h>
#include <mtmd.h>
#include <mtmd-helper.h>
#include <common.h>
#include <log.h>

#include <chrono>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <utility>

namespace llm {

static core::LoggerAdapter logger = core::LoggerAdapter::ForModule("models");

std::string ExtractVlmImagePromptPrefix(std::string_view prompt) {
    const std::string_view marker = mtmd_default_marker();
    const auto marker_pos = prompt.find(marker);
    if (marker_pos == std::string_view::npos) {
        return {};
    }
    return std::string(prompt.substr(0, marker_pos));
}

void EnsureLlamaRuntimeInitialized() {
    static std::once_flag init_flag;
    std::call_once(init_flag, [] {
        ggml_time_init();
        common_init();
        mtmd_helper_log_set(common_log_default_callback, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
    });
}

MtmdBitmapHandle MakeBitmapFromBuffer(mtmd_context* ctx, const std::vector<uint8_t>& image_data) {
    if (!ctx || image_data.empty()) {
        return {};
    }

    spdlog::info("mtmd bitmap init from buffer: bytes={}", image_data.size());
    auto wrapper = mtmd_helper_bitmap_init_from_buf(ctx, image_data.data(), image_data.size(), false);
    if (!wrapper.bitmap) {
        spdlog::warn("mtmd bitmap init from buffer returned null");
        return {};
    }

    spdlog::info("mtmd bitmap init from buffer done");
    return core::make_unique_handle<mtmd_bitmap>(wrapper.bitmap);
}

// PIMPL 实现
struct LlamaSharedRuntime::Impl {
    LlamaModelHandle model;
    MtmdContextHandle mtmd_ctx;
    std::filesystem::path model_path;
    std::filesystem::path mmproj_path;
    int n_gpu_layers = -1;
    mutable std::mutex mutex;
    mutable std::mutex mtmd_mutex;

    ~Impl() {
        Cleanup();
    }

    void Cleanup() {
        mtmd_ctx.reset();
        model.reset();
    }
};

struct LlamaRunner::Impl {
    std::shared_ptr<LlamaSharedRuntime> runtime;
    LlamaModelHandle owned_model;
    MtmdContextHandle owned_mtmd_ctx;
    LlamaContextHandle ctx;
    LlamaSamplerHandle sampler;

    std::filesystem::path model_path;
    std::filesystem::path mmproj_path;
    int n_gpu_layers = -1;
    std::string kv_prefix_key;
    llama_pos kv_prefix_tokens = 0;
    bool kv_prefix_valid = false;
    std::shared_ptr<IPromptKvCache> prompt_kv_cache;

    ~Impl() {
        Cleanup();
    }

    void Cleanup() {
        sampler.reset();
        ctx.reset();
        runtime.reset();
        owned_mtmd_ctx.reset();
        owned_model.reset();
        kv_prefix_key.clear();
        kv_prefix_tokens = 0;
        kv_prefix_valid = false;
        prompt_kv_cache.reset();
    }

    llama_model* Model() const {
        return runtime ? runtime->Model() : owned_model.get();
    }

    mtmd_context* MtmdContext() const {
        return runtime ? runtime->MtmdContext() : owned_mtmd_ctx.get();
    }

    std::mutex& MtmdMutex() const {
        if (runtime) {
            return runtime->MtmdMutex();
        }
        static std::mutex fallback_mutex;
        return fallback_mutex;
    }
};

struct LlamaRunnerPool::Impl {
    std::shared_ptr<LlamaSharedRuntime> runtime;
    std::vector<std::unique_ptr<LlamaRunner>> runners;
    std::vector<std::size_t> available;
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::size_t active = 0;
    bool loaded = false;
    bool unloading = false;
};

LlamaRunner::LlamaRunner() : impl_(std::make_unique<Impl>()) {
    EnsureLlamaRuntimeInitialized();
}

LlamaRunner::~LlamaRunner() {
    Unload();
}

LlamaRunner::LlamaRunner(LlamaRunner&& other) noexcept
    : impl_(std::move(other.impl_)), loaded_(other.loaded_) {
    other.loaded_ = false;
}

LlamaRunner& LlamaRunner::operator=(LlamaRunner&& other) noexcept {
    if (this != &other) {
        Unload();
        impl_ = std::move(other.impl_);
        loaded_ = other.loaded_;
        other.loaded_ = false;
    }
    return *this;
}

LlamaSharedRuntime::LlamaSharedRuntime() : impl_(std::make_unique<Impl>()) {
    EnsureLlamaRuntimeInitialized();
}

LlamaSharedRuntime::~LlamaSharedRuntime() {
    Unload();
}

bool LlamaSharedRuntime::Load(const std::filesystem::path& model_path,
                              const std::filesystem::path& mmproj_path,
                              int n_gpu_layers,
                              int mmproj_threads) {
    std::lock_guard lock(impl_->mutex);

    if (impl_->model) {
        logger.warn("Shared Llama runtime already loaded, unloading first");
        impl_->Cleanup();
    }

    logger.info("Loading shared Llama model: {}", model_path.string());

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    impl_->model.reset(llama_load_model_from_file(model_path.string().c_str(), model_params));
    if (!impl_->model) {
        logger.error("Failed to load shared Llama model: {}", model_path.string());
        return false;
    }

    if (!mmproj_path.empty()) {
        spdlog::info("Loading shared mmproj: {}", mmproj_path.string());

        mtmd_context_params mmproj_params = mtmd_context_params_default();
        mmproj_params.use_gpu = (n_gpu_layers != 0);
        mmproj_params.n_threads = mmproj_threads;

        impl_->mtmd_ctx.reset(mtmd_init_from_file(mmproj_path.string().c_str(), impl_->model.get(), mmproj_params));
        if (!impl_->mtmd_ctx) {
            spdlog::error("Failed to load shared mmproj: {}", mmproj_path.string());
            impl_->Cleanup();
            return false;
        }
    }

    impl_->model_path = model_path;
    impl_->mmproj_path = mmproj_path;
    impl_->n_gpu_layers = n_gpu_layers;
    spdlog::info("Shared Llama runtime loaded successfully");
    return true;
}

void LlamaSharedRuntime::Unload() {
    std::lock_guard lock(impl_->mutex);
    impl_->Cleanup();
}

bool LlamaSharedRuntime::IsLoaded() const {
    std::lock_guard lock(impl_->mutex);
    return static_cast<bool>(impl_->model);
}

llama_model* LlamaSharedRuntime::Model() const {
    return impl_->model.get();
}

mtmd_context* LlamaSharedRuntime::MtmdContext() const {
    return impl_->mtmd_ctx.get();
}

std::mutex& LlamaSharedRuntime::MtmdMutex() const {
    return impl_->mtmd_mutex;
}

const std::filesystem::path& LlamaSharedRuntime::ModelPath() const {
    return impl_->model_path;
}

const std::filesystem::path& LlamaSharedRuntime::MmprojPath() const {
    return impl_->mmproj_path;
}

int LlamaSharedRuntime::GpuLayers() const {
    return impl_->n_gpu_layers;
}

bool LlamaRunner::LoadModel(const std::filesystem::path& model_path,
                            const std::filesystem::path& mmproj_path,
                            int n_gpu_layers,
                            int llama_threads,
                            int mmproj_threads) {
    std::lock_guard lock(mutex_);

    if (loaded_) {
        logger.warn("Model already loaded, unloading first");
        impl_->Cleanup();
        loaded_ = false;
    }

    logger.info("Loading model: {}", model_path.string());

    // 加载文本模型
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    impl_->owned_model.reset(llama_load_model_from_file(model_path.string().c_str(), model_params));
    if (!impl_->owned_model) {
        logger.error("Failed to load model: {}", model_path.string());
        return false;
    }

    // 创建推理 context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = llama_threads;

    impl_->ctx.reset(llama_new_context_with_model(impl_->owned_model.get(), ctx_params));
    if (!impl_->ctx) {
        logger.error("Failed to create context");
        impl_->Cleanup();
        return false;
    }

    // 创建采样器链
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    impl_->sampler.reset(llama_sampler_chain_init(sampler_params));

    llama_sampler_chain_add(impl_->sampler.get(), llama_sampler_init_top_k(40));
    llama_sampler_chain_add(impl_->sampler.get(), llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(impl_->sampler.get(), llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(impl_->sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // 加载 mmproj 视觉投影层（可选）
    if (!mmproj_path.empty()) {
        spdlog::info("Loading mmproj: {}", mmproj_path.string());

        mtmd_context_params mmproj_params = mtmd_context_params_default();
        mmproj_params.use_gpu = (n_gpu_layers != 0);
        mmproj_params.n_threads = mmproj_threads;

        impl_->owned_mtmd_ctx.reset(mtmd_init_from_file(mmproj_path.string().c_str(), impl_->owned_model.get(), mmproj_params));

        if (!impl_->owned_mtmd_ctx) {
            spdlog::error("Failed to load mmproj: {}", mmproj_path.string());
            impl_->Cleanup();
            return false;
        }
    }

    impl_->model_path = model_path;
    impl_->mmproj_path = mmproj_path;
    impl_->n_gpu_layers = n_gpu_layers;
    loaded_ = true;

    spdlog::info("Model loaded successfully");
    return true;
}

bool LlamaRunner::LoadFromRuntime(std::shared_ptr<LlamaSharedRuntime> runtime,
                                  int llama_threads) {
    std::lock_guard lock(mutex_);

    if (!runtime || !runtime->IsLoaded() || !runtime->Model()) {
        logger.error("Cannot create Llama runner slot from unloaded runtime");
        return false;
    }

    if (loaded_) {
        impl_->Cleanup();
        loaded_ = false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = llama_threads;

    impl_->ctx.reset(llama_new_context_with_model(runtime->Model(), ctx_params));
    if (!impl_->ctx) {
        logger.error("Failed to create runner slot context");
        return false;
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    impl_->sampler.reset(llama_sampler_chain_init(sampler_params));
    llama_sampler_chain_add(impl_->sampler.get(), llama_sampler_init_top_k(40));
    llama_sampler_chain_add(impl_->sampler.get(), llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(impl_->sampler.get(), llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(impl_->sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    impl_->runtime = std::move(runtime);
    impl_->model_path = impl_->runtime->ModelPath();
    impl_->mmproj_path = impl_->runtime->MmprojPath();
    impl_->n_gpu_layers = impl_->runtime->GpuLayers();
    loaded_ = true;
    return true;
}

void LlamaRunner::SetPromptKvCache(std::shared_ptr<IPromptKvCache> cache) {
    std::lock_guard lock(mutex_);
    impl_->prompt_kv_cache = std::move(cache);
}

VLMResult LlamaRunner::Generate(const std::vector<uint8_t>& image_data,
                                const std::string& prompt,
                                const GenerateParams& params,
                                std::function<void(const std::string&)> token_callback,
                                std::shared_ptr<PreparedImageEmbedding> prepared_image) {
    std::lock_guard lock(mutex_);

    VLMResult result;

    if (!loaded_) {
        result.error_message = "Model not loaded";
        return result;
    }

    llama_model* model = impl_->Model();
    mtmd_context* mtmd_ctx = impl_->MtmdContext();
    if (!model) {
        result.error_message = "Model handle not loaded";
        return result;
    }
    if (!mtmd_ctx) {
        result.error_message = "No mmproj loaded";
        return result;
    }

    const bool prompt_kv_enabled =
        params.enable_prompt_kv_cache &&
        !params.prompt_kv_cache_key.empty() &&
        !image_data.empty();

    auto clear_prompt_kv_state = [&] {
        if (impl_->ctx) {
            llama_memory_clear(llama_get_memory(impl_->ctx.get()), true);
        }
        impl_->kv_prefix_valid = false;
        impl_->kv_prefix_key.clear();
        impl_->kv_prefix_tokens = 0;
    };
    auto fail = [&](std::string message) {
        result.error_message = std::move(message);
        if (prompt_kv_enabled) {
            clear_prompt_kv_state();
        }
        return result;
    };

    bool prompt_kv_cache_hit = false;
    llama_pos cached_prefix_tokens = 0;
    const std::string prompt_kv_key = params.prompt_kv_cache_key;

    if (prompt_kv_enabled &&
        impl_->kv_prefix_valid &&
        impl_->kv_prefix_key == prompt_kv_key &&
        impl_->kv_prefix_tokens > 0) {
        prompt_kv_cache_hit = true;
        cached_prefix_tokens = impl_->kv_prefix_tokens;
    } else {
        llama_memory_clear(llama_get_memory(impl_->ctx.get()), true);
        impl_->kv_prefix_valid = false;
        impl_->kv_prefix_key.clear();
        impl_->kv_prefix_tokens = 0;

        if (prompt_kv_enabled && impl_->prompt_kv_cache) {
            if (auto cached = impl_->prompt_kv_cache->Load(prompt_kv_key);
                cached && !cached->state.empty() && cached->prefix_tokens > 0) {
                const size_t restored = llama_state_seq_set_data_ext(
                    impl_->ctx.get(),
                    cached->state.data(),
                    cached->state.size(),
                    0,
                    LLAMA_STATE_SEQ_FLAGS_NONE);
                if (restored == cached->state.size()) {
                    prompt_kv_cache_hit = true;
                    cached_prefix_tokens = cached->prefix_tokens;
                    impl_->kv_prefix_key = prompt_kv_key;
                    impl_->kv_prefix_tokens = cached_prefix_tokens;
                    impl_->kv_prefix_valid = true;
                    spdlog::info("VLM prompt KV cache restored: key={} bytes={} prefix_tokens={}",
                                 prompt_kv_key,
                                 cached->state.size(),
                                 cached->prefix_tokens);
                } else {
                    spdlog::warn("VLM prompt KV cache restore failed: key={} bytes={} restored={}",
                                 prompt_kv_key,
                                 cached->state.size(),
                                 restored);
                    llama_memory_clear(llama_get_memory(impl_->ctx.get()), true);
                }
            }
        }
    }
    llama_sampler_reset(impl_->sampler.get());
    result.prompt_kv_cache_hit = prompt_kv_cache_hit;

    // 1. 从内存解码图片（如果有）
    MtmdBitmapHandle bitmap;
    std::unique_lock<std::mutex> mtmd_lock(impl_->MtmdMutex(), std::defer_lock);
    if (mtmd_ctx) {
        mtmd_lock.lock();
    }

    if (!image_data.empty()) {
        auto t_img_start = std::chrono::high_resolution_clock::now();

        bitmap = MakeBitmapFromBuffer(mtmd_ctx, image_data);

        if (!bitmap) {
            return fail("Failed to decode image from memory");
        }

        auto t_img_end = std::chrono::high_resolution_clock::now();
        result.image_encode_ms =
            std::chrono::duration<float, std::milli>(t_img_end - t_img_start).count() +
            (prepared_image ? prepared_image->image_encode_ms : 0.0f);
    }

    // 2. 分词（文本 + 图片 marker）
    auto t_prompt_start = std::chrono::high_resolution_clock::now();

    std::string user_content = prompt;
    if (bitmap && user_content.find(mtmd_default_marker()) == std::string::npos) {
        user_content = std::string(mtmd_default_marker()) + user_content;
    }

    spdlog::info("VLM prompt formatting started: prompt_bytes={} has_bitmap={}", prompt.size(), bitmap.get() != nullptr);
    std::string full_prompt = user_content;
    const char* tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl) {
        llama_chat_message msg{"user", user_content.c_str()};
        std::vector<char> tmpl_buf(user_content.size() * 2 + 512);
        int32_t tmpl_len = llama_chat_apply_template(
            tmpl, &msg, 1, /*add_ass=*/true, tmpl_buf.data(), static_cast<int32_t>(tmpl_buf.size()));
        if (tmpl_len < 0) {
            return fail("llama_chat_apply_template failed");
        }
        if (tmpl_len > static_cast<int32_t>(tmpl_buf.size())) {
            tmpl_buf.resize(tmpl_len);
            tmpl_len = llama_chat_apply_template(
                tmpl, &msg, 1, true, tmpl_buf.data(), static_cast<int32_t>(tmpl_buf.size()));
        }
        full_prompt.assign(tmpl_buf.data(), static_cast<size_t>(tmpl_len));
    }
    spdlog::info("VLM prompt formatting done: full_prompt_bytes={}", full_prompt.size());

    mtmd_input_text text;
    text.text = full_prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    auto chunks = MakeMtmdInputChunks();
    const mtmd_bitmap* bitmaps_arr[] = {bitmap.get()};
    spdlog::info("mtmd_tokenize started");
    int32_t tokenize_res = mtmd_tokenize(
        mtmd_ctx,
        chunks.get(),
        &text,
        bitmap ? bitmaps_arr : nullptr,
        bitmap ? 1 : 0
    );

    if (tokenize_res != 0) {
        return fail("Failed to tokenize prompt (code=" + std::to_string(tokenize_res) + ")");
    }
    spdlog::info("mtmd_tokenize done");

    // 3. 逐 chunk 处理，支持 image embedding 捕获
    llama_pos n_past = prompt_kv_cache_hit ? cached_prefix_tokens : 0;
    const size_t n_chunks = mtmd_input_chunks_size(chunks.get());
    spdlog::info("VLM chunk evaluation started: chunks={}", n_chunks);

    bool skipped_cached_prefix = !prompt_kv_cache_hit;
    bool stored_prompt_prefix = prompt_kv_cache_hit;
    bool prepared_image_consumed = false;
    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.get(), i);
        const auto chunk_type = mtmd_input_chunk_get_type(chunk);
        spdlog::info("VLM chunk {} started: type={} tokens={} pos={}",
                     i,
                     static_cast<int>(chunk_type),
                     mtmd_input_chunk_get_n_tokens(chunk),
                     mtmd_input_chunk_get_n_pos(chunk));

        if (prompt_kv_cache_hit && !skipped_cached_prefix) {
            if (chunk_type != MTMD_INPUT_CHUNK_TYPE_TEXT) {
                skipped_cached_prefix = true;
            }
            spdlog::info("VLM prompt KV cache skipping prefix chunk: chunk={} type={}",
                         i,
                         static_cast<int>(chunk_type));
            continue;
        }

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            spdlog::info("mtmd_helper_eval_chunk_single started: chunk={}", i);
            if (mtmd_helper_eval_chunk_single(mtmd_ctx, impl_->ctx.get(), chunk,
                                              n_past, 0, 512, true, &n_past) != 0) {
                return fail("Failed to evaluate chunk");
            }
            spdlog::info("mtmd_helper_eval_chunk_single done: chunk={} n_past={}", i, n_past);
        } else {
            const int32_t n_embd = llama_model_n_embd_inp(model);
            const int32_t n_tokens = static_cast<int32_t>(mtmd_input_chunk_get_n_tokens(chunk));
            MtmdBatchHandle media_batch;
            float* embd = nullptr;

            if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE &&
                prepared_image && !prepared_image_consumed) {
                const auto expected_values =
                    static_cast<std::size_t>(n_tokens) * static_cast<std::size_t>(n_embd);
                if (prepared_image->image_embedding_dim != n_embd ||
                    prepared_image->image_embedding_tokens != n_tokens ||
                    prepared_image->image_token_embeddings.size() != expected_values) {
                    return fail("Prepared image embedding shape does not match tokenized image chunk");
                }
                embd = prepared_image->image_token_embeddings.data();
                prepared_image_consumed = true;
                spdlog::info(
                    "VLM reused prepared image embedding: chunk={} tokens={} dim={}",
                    i,
                    n_tokens,
                    n_embd);
            } else {
                media_batch = MakeMtmdBatch(mtmd_ctx);
                if (!media_batch) {
                    return fail("Failed to create mtmd media batch");
                }

                spdlog::info("mtmd_batch_add_chunk started: chunk={}", i);
                if (mtmd_batch_add_chunk(media_batch.get(), chunk) != 0) {
                    return fail("Failed to add media chunk to mtmd batch");
                }
                spdlog::info("mtmd_batch_add_chunk done: chunk={}", i);

                spdlog::info("mtmd_batch_encode started: chunk={}", i);
                if (mtmd_batch_encode(media_batch.get()) != 0) {
                    return fail("Failed to encode media batch");
                }
                spdlog::info("mtmd_batch_encode done: chunk={}", i);
                embd = mtmd_batch_get_output_embd(media_batch.get(), chunk);
                if (!embd) {
                    return fail("Failed to get media embedding from mtmd batch");
                }
            }

            if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE
                && (params.capture_image_embedding || prompt_kv_enabled)
                && result.image_embedding.empty()) {
                result.image_embedding.assign(n_embd, 0.0f);
                result.image_token_embeddings.assign(
                    embd,
                    embd + static_cast<std::size_t>(n_tokens) * n_embd);
                for (int32_t t = 0; t < n_tokens; ++t) {
                    for (int32_t d = 0; d < n_embd; ++d) {
                        result.image_embedding[d] += embd[t * n_embd + d];
                    }
                }
                const float inv = 1.0f / std::max(1, n_tokens);
                for (auto& v : result.image_embedding) v *= inv;
                result.image_embedding_dim = n_embd;
                result.image_embedding_tokens = n_tokens;
            }

            spdlog::info("mtmd_helper_decode_image_chunk started: chunk={}", i);
            if (mtmd_helper_decode_image_chunk(mtmd_ctx, impl_->ctx.get(), chunk,
                                               embd, n_past, 0, 512, &n_past, nullptr, nullptr) != 0) {
                return fail("Failed to decode image chunk");
            }
            spdlog::info("mtmd_helper_decode_image_chunk done: chunk={} n_past={}", i, n_past);

            if (prompt_kv_enabled && !stored_prompt_prefix) {
                const size_t state_size = llama_state_seq_get_size_ext(
                    impl_->ctx.get(),
                    0,
                    LLAMA_STATE_SEQ_FLAGS_NONE);
                if (state_size > 0) {
                    PromptKvCacheEntry entry;
                    entry.state.resize(state_size);
                    const size_t written = llama_state_seq_get_data_ext(
                        impl_->ctx.get(),
                        entry.state.data(),
                        entry.state.size(),
                        0,
                        LLAMA_STATE_SEQ_FLAGS_NONE);
                    if (written == entry.state.size()) {
                        entry.prefix_tokens = n_past;
                        entry.session_id = params.prompt_kv_session_id;
                        entry.prefix_fingerprint = params.prompt_kv_prefix_fingerprint;
                        entry.image_embedding = result.image_embedding;
                        entry.image_token_embeddings = result.image_token_embeddings;
                        entry.image_embedding_dim = result.image_embedding_dim;
                        entry.image_embedding_tokens = result.image_embedding_tokens;
                        impl_->kv_prefix_key = prompt_kv_key;
                        impl_->kv_prefix_tokens = n_past;
                        impl_->kv_prefix_valid = true;
                        if (impl_->prompt_kv_cache) {
                            impl_->prompt_kv_cache->Store(prompt_kv_key, entry);
                            for (const auto& alias : params.prompt_kv_cache_alias_keys) {
                                impl_->prompt_kv_cache->StoreAlias(alias, prompt_kv_key);
                            }
                        }
                        stored_prompt_prefix = true;
                        spdlog::info("VLM prompt KV cache stored: key={} bytes={} prefix_tokens={}",
                                     prompt_kv_key,
                                     entry.state.size(),
                                     entry.prefix_tokens);
                    } else {
                        spdlog::warn("VLM prompt KV cache state export failed: key={} bytes={} written={}",
                                     prompt_kv_key,
                                     entry.state.size(),
                                     written);
                    }
                }
            }
        }
    }
    spdlog::info("VLM chunk evaluation done: n_past={}", n_past);
    if (mtmd_lock.owns_lock()) {
        mtmd_lock.unlock();
    }
    result.prompt_tokens = n_past;

    auto t_prompt_end = std::chrono::high_resolution_clock::now();
    result.prompt_eval_ms = std::chrono::duration<float, std::milli>(t_prompt_end - t_prompt_start).count();

    // 4. 生成 token 循环
    auto t_gen_start = std::chrono::high_resolution_clock::now();

    std::string generated_text;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    auto gen_batch = MakeLlamaBatch(1, 0, 1);

    spdlog::info("VLM token generation started: max_tokens={}", params.max_tokens);
    for (int i = 0; i < params.max_tokens; ++i) {
        llama_token token_id = llama_sampler_sample(impl_->sampler.get(), impl_->ctx.get(), -1);

        if (llama_vocab_is_eog(vocab, token_id)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, token_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string token_str(buf, n);
            generated_text += token_str;

            if (token_callback) {
                token_callback(token_str);
            }
        }

        llama_sampler_accept(impl_->sampler.get(), token_id);

        common_batch_clear(gen_batch.get());
        common_batch_add(gen_batch.get(), token_id, n_past++, {0}, true);

        if (llama_decode(impl_->ctx.get(), gen_batch.get()) != 0) {
            return fail("Failed to decode generated token");
        }

        result.generated_tokens++;
    }
    spdlog::info("VLM token generation done: generated_tokens={}", result.generated_tokens);

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    result.eval_ms = std::chrono::duration<float, std::milli>(t_gen_end - t_gen_start).count();

    result.text = generated_text;
    result.success = true;

    if (impl_->kv_prefix_valid && impl_->kv_prefix_tokens > 0) {
        llama_memory_seq_rm(llama_get_memory(impl_->ctx.get()), 0, impl_->kv_prefix_tokens, -1);
    }

    if (result.generated_tokens > 0) {
        spdlog::info("Generated {} tokens in {:.2f} ms ({:.2f} tok/s)",
                 result.generated_tokens,
                 result.eval_ms,
                 result.generated_tokens * 1000.0f / result.eval_ms);
    }

    return result;
}

VLMResult LlamaRunner::EncodeImageOnly(const std::vector<uint8_t>& image_data) {
    std::lock_guard lock(mutex_);

    VLMResult result;

    if (!loaded_) {
        result.error_message = "Model not loaded";
        return result;
    }
    llama_model* model = impl_->Model();
    mtmd_context* mtmd_ctx = impl_->MtmdContext();
    if (!model) {
        result.error_message = "Model handle not loaded";
        return result;
    }
    if (!mtmd_ctx) {
        result.error_message = "No mmproj loaded";
        return result;
    }
    if (image_data.empty()) {
        result.error_message = "Empty image data";
        return result;
    }

    llama_memory_clear(llama_get_memory(impl_->ctx.get()), true);

    auto t_start = std::chrono::high_resolution_clock::now();

    std::lock_guard mtmd_lock(impl_->MtmdMutex());

    auto bitmap = MakeBitmapFromBuffer(mtmd_ctx, image_data);
    if (!bitmap) {
        result.error_message = "Failed to decode image";
        return result;
    }

    std::string marker_prompt(mtmd_default_marker());
    mtmd_input_text text;
    text.text = marker_prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    auto chunks = MakeMtmdInputChunks();
    const mtmd_bitmap* bitmaps_arr[] = {bitmap.get()};
    if (mtmd_tokenize(mtmd_ctx, chunks.get(), &text, bitmaps_arr, 1) != 0) {
        result.error_message = "Failed to tokenize for embedding";
        return result;
    }

    const size_t n_chunks = mtmd_input_chunks_size(chunks.get());
    const int32_t n_embd = llama_model_n_embd_inp(model);

    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.get(), i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            continue;
        }

        auto media_batch = MakeMtmdBatch(mtmd_ctx);
        if (!media_batch) {
            result.error_message = "Failed to create mtmd media batch";
            return result;
        }
        if (mtmd_batch_add_chunk(media_batch.get(), chunk) != 0) {
            result.error_message = "Failed to add image chunk to mtmd batch";
            return result;
        }
        if (mtmd_batch_encode(media_batch.get()) != 0) {
            result.error_message = "Failed to encode image chunk";
            return result;
        }

        const float* embd = mtmd_batch_get_output_embd(media_batch.get(), chunk);
        if (!embd) {
            result.error_message = "Failed to get image embedding from mtmd batch";
            return result;
        }
        const int32_t n_tokens = static_cast<int32_t>(mtmd_input_chunk_get_n_tokens(chunk));

        result.image_embedding.assign(n_embd, 0.0f);
        result.image_token_embeddings.assign(
            embd,
            embd + static_cast<std::size_t>(n_tokens) * n_embd);
        for (int32_t t = 0; t < n_tokens; ++t) {
            for (int32_t d = 0; d < n_embd; ++d) {
                result.image_embedding[d] += embd[t * n_embd + d];
            }
        }
        const float inv = 1.0f / std::max(1, n_tokens);
        for (auto& v : result.image_embedding) v *= inv;
        result.image_embedding_dim = n_embd;
        result.image_embedding_tokens = n_tokens;
        break;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.image_encode_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    result.success = !result.image_embedding.empty();
    if (!result.success) {
        result.error_message = "No image chunk found";
    }

    llama_memory_clear(llama_get_memory(impl_->ctx.get()), true);
    llama_sampler_reset(impl_->sampler.get());

    return result;
}

void LlamaRunner::Unload() {
    std::lock_guard lock(mutex_);

    if (!loaded_) {
        return;
    }

    spdlog::info("Unloading model");
    impl_->Cleanup();
    loaded_ = false;
}

bool LlamaRunner::IsLoaded() const {
    std::lock_guard lock(mutex_);
    return loaded_;
}

std::string LlamaRunner::GetInfo() const {
    std::lock_guard lock(mutex_);

    if (!loaded_) {
        return "Model not loaded";
    }

    return "Llama model loaded: " + impl_->model_path.string();
}

MemorySnapshot LlamaRunner::GetMemorySnapshot() const {
    std::lock_guard lock(mutex_);

    MemorySnapshot snapshot;
    snapshot.loaded = loaded_;
    if (loaded_ && impl_->Model()) {
        snapshot.model_size_bytes = llama_model_size(impl_->Model());
    }

    const size_t device_count = ggml_backend_dev_count();
    snapshot.devices.reserve(device_count);
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (!device) {
            continue;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(device, &free_bytes, &total_bytes);

        const auto type = ggml_backend_dev_type(device);
        DeviceMemoryInfo info;
        if (const char* name = ggml_backend_dev_name(device)) {
            info.name = name;
        }
        if (const char* description = ggml_backend_dev_description(device)) {
            info.description = description;
        }
        info.free_bytes = free_bytes;
        info.total_bytes = total_bytes;
        info.is_gpu =
            type == GGML_BACKEND_DEVICE_TYPE_GPU ||
            type == GGML_BACKEND_DEVICE_TYPE_IGPU ||
            type == GGML_BACKEND_DEVICE_TYPE_META;
        snapshot.devices.push_back(std::move(info));
    }

    return snapshot;
}

LlamaRunnerPool::Lease::Lease(LlamaRunnerPool* pool, std::size_t index, LlamaRunner* runner)
    : pool_(pool), index_(index), runner_(runner) {}

LlamaRunnerPool::Lease::~Lease() {
    if (pool_ && runner_) {
        pool_->Release(index_);
    }
}

LlamaRunnerPool::Lease::Lease(Lease&& other) noexcept
    : pool_(other.pool_), index_(other.index_), runner_(other.runner_) {
    other.pool_ = nullptr;
    other.runner_ = nullptr;
}

LlamaRunnerPool::Lease& LlamaRunnerPool::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        if (pool_ && runner_) {
            pool_->Release(index_);
        }
        pool_ = other.pool_;
        index_ = other.index_;
        runner_ = other.runner_;
        other.pool_ = nullptr;
        other.runner_ = nullptr;
    }
    return *this;
}

LlamaRunner* LlamaRunnerPool::Lease::operator->() const {
    return runner_;
}

LlamaRunner& LlamaRunnerPool::Lease::operator*() const {
    return *runner_;
}

LlamaRunnerPool::Lease::operator bool() const noexcept {
    return runner_ != nullptr;
}

LlamaRunnerPool::LlamaRunnerPool() : impl_(std::make_unique<Impl>()) {}

LlamaRunnerPool::~LlamaRunnerPool() {
    Unload();
}

bool LlamaRunnerPool::Load(const std::filesystem::path& model_path,
                           const std::filesystem::path& mmproj_path,
                           int n_gpu_layers,
                           std::size_t pool_size,
                           int llama_threads,
                           int mmproj_threads,
                           std::shared_ptr<IPromptKvCache> prompt_kv_cache) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->loaded) {
        logger.warn("Llama runner pool already loaded, unloading first");
        impl_->available.clear();
        impl_->runners.clear();
        impl_->runtime.reset();
        impl_->loaded = false;
        impl_->active = 0;
        impl_->unloading = false;
    }

    pool_size = std::max<std::size_t>(1, pool_size);
    auto runtime = std::make_shared<LlamaSharedRuntime>();
    if (!runtime->Load(model_path, mmproj_path, n_gpu_layers, mmproj_threads)) {
        return false;
    }

    std::vector<std::unique_ptr<LlamaRunner>> runners;
    runners.reserve(pool_size);
    for (std::size_t i = 0; i < pool_size; ++i) {
        auto runner = std::make_unique<LlamaRunner>();
        if (!runner->LoadFromRuntime(runtime, llama_threads)) {
            logger.error("Failed to initialize Llama runner slot {}", i);
            return false;
        }
        runner->SetPromptKvCache(prompt_kv_cache);
        runners.push_back(std::move(runner));
    }

    impl_->runtime = std::move(runtime);
    impl_->runners = std::move(runners);
    impl_->available.clear();
    impl_->available.reserve(pool_size);
    for (std::size_t i = 0; i < pool_size; ++i) {
        impl_->available.push_back(pool_size - 1 - i);
    }
    impl_->active = 0;
    impl_->unloading = false;
    impl_->loaded = true;
    logger.info("Llama runner pool loaded: slots={}", pool_size);
    return true;
}

void LlamaRunnerPool::Unload() {
    std::unique_lock lock(impl_->mutex);
    impl_->unloading = true;
    impl_->loaded = false;
    impl_->cv.wait(lock, [this] {
        return impl_->active == 0;
    });
    impl_->available.clear();
    impl_->runners.clear();
    impl_->runtime.reset();
    impl_->unloading = false;
    impl_->cv.notify_all();
}

bool LlamaRunnerPool::IsLoaded() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->loaded;
}

LlamaRunnerPool::Lease LlamaRunnerPool::Acquire() {
    std::unique_lock lock(impl_->mutex);
    impl_->cv.wait(lock, [this] {
        return (!impl_->loaded && !impl_->unloading) ||
               (impl_->loaded && !impl_->unloading && !impl_->available.empty());
    });
    if (!impl_->loaded || impl_->unloading || impl_->available.empty()) {
        return {};
    }
    const std::size_t index = impl_->available.back();
    impl_->available.pop_back();
    ++impl_->active;
    return Lease(this, index, impl_->runners[index].get());
}

MemorySnapshot LlamaRunnerPool::GetMemorySnapshot() const {
    std::lock_guard lock(impl_->mutex);
    if (!impl_->loaded || impl_->runners.empty()) {
        MemorySnapshot snapshot;
        snapshot.loaded = false;
        return snapshot;
    }
    return impl_->runners.front()->GetMemorySnapshot();
}

void LlamaRunnerPool::Release(std::size_t index) {
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->active > 0) {
            --impl_->active;
        }
        if (!impl_->unloading && impl_->loaded && index < impl_->runners.size()) {
            impl_->available.push_back(index);
        }
    }
    impl_->cv.notify_all();
}

} // namespace llm
