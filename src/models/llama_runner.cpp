/**
 * @file llama_runner.cpp
 * @brief llama.cpp VLM 推理实现
 */

#include "llama_runner.h"
#include "server_common.h"

#include <llama.h>
#include <ggml-backend.h>
#include <mtmd.h>
#include <mtmd-helper.h>
#include <common.h>

#include <chrono>
#include <utility>

namespace llm {

// RAII 包装器：自动释放 llama_batch
struct BatchGuard {
    llama_batch batch;

    explicit BatchGuard(int32_t n_tokens, int32_t embd = 0, int32_t n_seq_max = 1)
        : batch(llama_batch_init(n_tokens, embd, n_seq_max)) {}

    ~BatchGuard() {
        llama_batch_free(batch);
    }

    BatchGuard(const BatchGuard&) = delete;
    BatchGuard& operator=(const BatchGuard&) = delete;

    BatchGuard(BatchGuard&& other) noexcept : batch(other.batch) {
        other.batch = {};
    }

    BatchGuard& operator=(BatchGuard&& other) noexcept {
        if (this != &other) {
            llama_batch_free(batch);
            batch = other.batch;
            other.batch = {};
        }
        return *this;
    }
};

// RAII 包装器：自动释放 mtmd_bitmap
struct BitmapGuard {
    mtmd_bitmap* ptr = nullptr;

    BitmapGuard() = default;
    explicit BitmapGuard(mtmd_bitmap* bmp) : ptr(bmp) {}

    ~BitmapGuard() {
        if (ptr) {
            mtmd_bitmap_free(ptr);
        }
    }

    BitmapGuard(const BitmapGuard&) = delete;
    BitmapGuard& operator=(const BitmapGuard&) = delete;

    BitmapGuard(BitmapGuard&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    BitmapGuard& operator=(BitmapGuard&& other) noexcept {
        if (this != &other) {
            if (ptr) mtmd_bitmap_free(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    mtmd_bitmap* get() const { return ptr; }
    explicit operator bool() const { return ptr != nullptr; }
};

// RAII 包装器：自动释放 mtmd_input_chunks
struct ChunksGuard {
    mtmd_input_chunks* ptr = nullptr;

    ChunksGuard() : ptr(mtmd_input_chunks_init()) {}

    ~ChunksGuard() {
        if (ptr) {
            mtmd_input_chunks_free(ptr);
        }
    }

    ChunksGuard(const ChunksGuard&) = delete;
    ChunksGuard& operator=(const ChunksGuard&) = delete;

    mtmd_input_chunks* get() const { return ptr; }
};

// PIMPL 实现
struct LlamaRunner::Impl {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    mtmd_context* mtmd_ctx = nullptr;
    llama_sampler* sampler = nullptr;

    std::filesystem::path model_path;
    std::filesystem::path mmproj_path;
    int n_gpu_layers = -1;

    ~Impl() {
        Cleanup();
    }

    void Cleanup() {
        if (sampler) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (mtmd_ctx) {
            mtmd_free(mtmd_ctx);
            mtmd_ctx = nullptr;
        }
        if (model) {
            llama_free_model(model);
            model = nullptr;
        }
    }
};

LlamaRunner::LlamaRunner() : impl_(std::make_unique<Impl>()) {
    llama_backend_init();
}

LlamaRunner::~LlamaRunner() {
    Unload();
    llama_backend_free();
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

bool LlamaRunner::LoadModel(const std::filesystem::path& model_path,
                            const std::filesystem::path& mmproj_path,
                            int n_gpu_layers) {
    std::lock_guard lock(mutex_);

    if (loaded_) {
        LOG_WARN("Model already loaded, unloading first");
        impl_->Cleanup();
        loaded_ = false;
    }

    LOG_INFO("Loading model: {}", model_path.string());

    // 加载文本模型
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    impl_->model = llama_load_model_from_file(model_path.string().c_str(), model_params);
    if (!impl_->model) {
        LOG_ERROR("Failed to load model: {}", model_path.string());
        return false;
    }

    // 创建推理 context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = 8;

    impl_->ctx = llama_new_context_with_model(impl_->model, ctx_params);
    if (!impl_->ctx) {
        LOG_ERROR("Failed to create context");
        impl_->Cleanup();
        return false;
    }

    // 创建采样器链
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    impl_->sampler = llama_sampler_chain_init(sampler_params);

    llama_sampler_chain_add(impl_->sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(impl_->sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(impl_->sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(impl_->sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // 加载 mmproj 视觉投影层（可选）
    if (!mmproj_path.empty()) {
        LOG_INFO("Loading mmproj: {}", mmproj_path.string());

        mtmd_context_params mmproj_params = mtmd_context_params_default();
        mmproj_params.use_gpu = (n_gpu_layers > 0);
        mmproj_params.n_threads = 8;

        impl_->mtmd_ctx = mtmd_init_from_file(mmproj_path.string().c_str(), impl_->model, mmproj_params);

        if (!impl_->mtmd_ctx) {
            LOG_ERROR("Failed to load mmproj: {}", mmproj_path.string());
            impl_->Cleanup();
            return false;
        }
    }

    impl_->model_path = model_path;
    impl_->mmproj_path = mmproj_path;
    impl_->n_gpu_layers = n_gpu_layers;
    loaded_ = true;

    LOG_INFO("Model loaded successfully");
    return true;
}

VLMResult LlamaRunner::Generate(const std::vector<uint8_t>& image_data,
                                const std::string& prompt,
                                const GenerateParams& params,
                                std::function<void(const std::string&)> token_callback) {
    std::lock_guard lock(mutex_);

    VLMResult result;

    if (!loaded_) {
        result.error_message = "Model not loaded";
        return result;
    }

    llama_memory_clear(llama_get_memory(impl_->ctx), true);
    llama_sampler_reset(impl_->sampler);

    // 1. 从内存解码图片（如果有）
    BitmapGuard bitmap;

    if (!image_data.empty() && impl_->mtmd_ctx) {
        auto t_img_start = std::chrono::high_resolution_clock::now();

        bitmap = BitmapGuard(mtmd_helper_bitmap_init_from_buf(
            impl_->mtmd_ctx,
            image_data.data(),
            image_data.size()
        ));

        if (!bitmap) {
            result.error_message = "Failed to decode image from memory";
            return result;
        }

        auto t_img_end = std::chrono::high_resolution_clock::now();
        result.image_encode_ms = std::chrono::duration<float, std::milli>(t_img_end - t_img_start).count();
    }

    // 2. 分词（文本 + 图片 marker）
    auto t_prompt_start = std::chrono::high_resolution_clock::now();

    std::string user_content = prompt;
    if (bitmap && user_content.find(mtmd_default_marker()) == std::string::npos) {
        user_content = std::string(mtmd_default_marker()) + user_content;
    }

    std::string full_prompt = user_content;
    const char* tmpl = llama_model_chat_template(impl_->model, nullptr);
    if (tmpl) {
        llama_chat_message msg{"user", user_content.c_str()};
        std::vector<char> tmpl_buf(user_content.size() * 2 + 512);
        int32_t tmpl_len = llama_chat_apply_template(
            tmpl, &msg, 1, /*add_ass=*/true, tmpl_buf.data(), static_cast<int32_t>(tmpl_buf.size()));
        if (tmpl_len < 0) {
            result.error_message = "llama_chat_apply_template failed";
            return result;
        }
        if (tmpl_len > static_cast<int32_t>(tmpl_buf.size())) {
            tmpl_buf.resize(tmpl_len);
            tmpl_len = llama_chat_apply_template(
                tmpl, &msg, 1, true, tmpl_buf.data(), static_cast<int32_t>(tmpl_buf.size()));
        }
        full_prompt.assign(tmpl_buf.data(), static_cast<size_t>(tmpl_len));
    }

    mtmd_input_text text;
    text.text = full_prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    ChunksGuard chunks;
    const mtmd_bitmap* bitmaps_arr[] = {bitmap.get()};
    int32_t tokenize_res = mtmd_tokenize(
        impl_->mtmd_ctx,
        chunks.get(),
        &text,
        bitmap ? bitmaps_arr : nullptr,
        bitmap ? 1 : 0
    );

    if (tokenize_res != 0) {
        result.error_message = "Failed to tokenize prompt (code=" + std::to_string(tokenize_res) + ")";
        return result;
    }

    // 3. 逐 chunk 处理，支持 image embedding 捕获
    llama_pos n_past = 0;
    const size_t n_chunks = mtmd_input_chunks_size(chunks.get());

    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.get(), i);
        const auto chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE
            && params.capture_image_embedding
            && result.image_embedding.empty()) {
            const int32_t n_embd = llama_model_n_embd_inp(impl_->model);
            if (mtmd_encode_chunk(impl_->mtmd_ctx, chunk) != 0) {
                result.error_message = "Failed to encode image chunk for embedding";
                return result;
            }
            float* embd = mtmd_get_output_embd(impl_->mtmd_ctx);
            const int32_t n_tokens = static_cast<int32_t>(mtmd_input_chunk_get_n_tokens(chunk));

            result.image_embedding.assign(n_embd, 0.0f);
            for (int32_t t = 0; t < n_tokens; ++t) {
                for (int32_t d = 0; d < n_embd; ++d) {
                    result.image_embedding[d] += embd[t * n_embd + d];
                }
            }
            const float inv = 1.0f / std::max(1, n_tokens);
            for (auto& v : result.image_embedding) v *= inv;
            result.image_embedding_dim = n_embd;
            result.image_embedding_tokens = n_tokens;

            if (mtmd_helper_decode_image_chunk(impl_->mtmd_ctx, impl_->ctx, chunk,
                                               embd, n_past, 0, 512, &n_past) != 0) {
                result.error_message = "Failed to decode image chunk";
                return result;
            }
        } else {
            if (mtmd_helper_eval_chunk_single(impl_->mtmd_ctx, impl_->ctx, chunk,
                                              n_past, 0, 512, true, &n_past) != 0) {
                result.error_message = "Failed to evaluate chunk";
                return result;
            }
        }
    }

    result.prompt_tokens = n_past;

    auto t_prompt_end = std::chrono::high_resolution_clock::now();
    result.prompt_eval_ms = std::chrono::duration<float, std::milli>(t_prompt_end - t_prompt_start).count();

    // 4. 生成 token 循环
    auto t_gen_start = std::chrono::high_resolution_clock::now();

    std::string generated_text;
    const llama_vocab* vocab = llama_model_get_vocab(impl_->model);
    BatchGuard gen_batch(1, 0, 1);

    for (int i = 0; i < params.max_tokens; ++i) {
        llama_token token_id = llama_sampler_sample(impl_->sampler, impl_->ctx, -1);

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

        llama_sampler_accept(impl_->sampler, token_id);

        common_batch_clear(gen_batch.batch);
        common_batch_add(gen_batch.batch, token_id, n_past++, {0}, true);

        if (llama_decode(impl_->ctx, gen_batch.batch) != 0) {
            result.error_message = "Failed to decode generated token";
            break;
        }

        result.generated_tokens++;
    }

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    result.eval_ms = std::chrono::duration<float, std::milli>(t_gen_end - t_gen_start).count();

    result.text = generated_text;
    result.success = true;

    if (result.generated_tokens > 0) {
        LOG_INFO("Generated {} tokens in {:.2f} ms ({:.2f} tok/s)",
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
    if (!impl_->mtmd_ctx) {
        result.error_message = "No mmproj loaded";
        return result;
    }
    if (image_data.empty()) {
        result.error_message = "Empty image data";
        return result;
    }

    llama_memory_clear(llama_get_memory(impl_->ctx), true);

    auto t_start = std::chrono::high_resolution_clock::now();

    BitmapGuard bitmap(mtmd_helper_bitmap_init_from_buf(
        impl_->mtmd_ctx, image_data.data(), image_data.size()));
    if (!bitmap) {
        result.error_message = "Failed to decode image";
        return result;
    }

    std::string marker_prompt(mtmd_default_marker());
    mtmd_input_text text;
    text.text = marker_prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    ChunksGuard chunks;
    const mtmd_bitmap* bitmaps_arr[] = {bitmap.get()};
    if (mtmd_tokenize(impl_->mtmd_ctx, chunks.get(), &text, bitmaps_arr, 1) != 0) {
        result.error_message = "Failed to tokenize for embedding";
        return result;
    }

    const size_t n_chunks = mtmd_input_chunks_size(chunks.get());
    const int32_t n_embd = llama_model_n_embd_inp(impl_->model);

    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.get(), i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            continue;
        }

        if (mtmd_encode_chunk(impl_->mtmd_ctx, chunk) != 0) {
            result.error_message = "Failed to encode image chunk";
            return result;
        }

        const float* embd = mtmd_get_output_embd(impl_->mtmd_ctx);
        const int32_t n_tokens = static_cast<int32_t>(mtmd_input_chunk_get_n_tokens(chunk));

        result.image_embedding.assign(n_embd, 0.0f);
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

    llama_memory_clear(llama_get_memory(impl_->ctx), true);
    llama_sampler_reset(impl_->sampler);

    return result;
}

void LlamaRunner::Unload() {
    std::lock_guard lock(mutex_);

    if (!loaded_) {
        return;
    }

    LOG_INFO("Unloading model");
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
    if (loaded_ && impl_->model) {
        snapshot.model_size_bytes = llama_model_size(impl_->model);
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

} // namespace llm
