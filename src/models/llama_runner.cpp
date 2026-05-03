/**
 * @file llama_runner.cpp
 * @brief llama.cpp VLM 推理实现
 */

#include "llama_runner.h"
#include "server_common.h"

#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>
#include <common.h>

#include <chrono>

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
        if (model) {
            llama_free_model(model);
            model = nullptr;
        }
        if (mtmd_ctx) {
            mtmd_free(mtmd_ctx);
            mtmd_ctx = nullptr;
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

    std::string full_prompt = prompt;
    if (bitmap && full_prompt.find(mtmd_default_marker()) == std::string::npos) {
        full_prompt = std::string(mtmd_default_marker()) + full_prompt;
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

    // 3. 使用 helper 函数处理所有 chunks（文本 + 图片 embedding）
    llama_pos n_past = 0;
    int32_t eval_res = mtmd_helper_eval_chunks(
        impl_->mtmd_ctx,
        impl_->ctx,
        chunks.get(),
        n_past,
        0,     // seq_id
        512,   // n_batch
        true,  // logits_last
        &n_past
    );

    if (eval_res != 0) {
        result.error_message = "Failed to evaluate chunks";
        return result;
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

void LlamaRunner::Unload() {
    std::lock_guard lock(mutex_);

    if (!loaded_) {
        return;
    }

    LOG_INFO("Unloading model");
    impl_->Cleanup();
    loaded_ = false;
}

std::string LlamaRunner::GetInfo() const {
    std::lock_guard lock(mutex_);

    if (!loaded_) {
        return "Model not loaded";
    }

    return "Llama model loaded: " + impl_->model_path.string();
}

} // namespace llm
