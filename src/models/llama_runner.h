/**
 * @file llama_runner.h
 * @brief llama.cpp VLM 推理封装
 */

#pragma once

#include "result.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

struct llama_model;
struct mtmd_context;

namespace llm {

class LlamaSharedRuntime;

std::string ExtractVlmImagePromptPrefix(std::string_view prompt);

/**
 * VLM 推理结果
 */
struct VLMResult {
    std::string text;

    float image_encode_ms = 0.0f;
    float prompt_eval_ms = 0.0f;
    float eval_ms = 0.0f;
    int32_t prompt_tokens = 0;
    int32_t generated_tokens = 0;
    bool prompt_kv_cache_hit = false;

    std::vector<float> image_embedding;
    std::vector<float> image_token_embeddings;
    int32_t image_embedding_dim = 0;
    int32_t image_embedding_tokens = 0;

    bool success = false;
    std::string error_message;
};

struct PreparedImageEmbedding {
    float image_encode_ms = 0.0f;
    std::vector<float> image_embedding;
    std::vector<float> image_token_embeddings;
    int32_t image_embedding_dim = 0;
    int32_t image_embedding_tokens = 0;
};

/**
 * 生成参数
 */
struct GenerateParams {
    int32_t max_tokens = 256;
    float temperature = 0.7f;
    int32_t context_size = 2048;
    float top_p = 0.9f;
    int32_t top_k = 40;
    bool capture_image_embedding = false;
    bool enable_prompt_kv_cache = false;
    std::string prompt_kv_cache_key;
    std::vector<std::string> prompt_kv_cache_alias_keys;
    std::string prompt_kv_session_id;
    std::string prompt_kv_prefix_fingerprint;
};

struct DeviceMemoryInfo {
    std::string name;
    std::string description;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    bool is_gpu = false;
};

struct MemorySnapshot {
    bool loaded = false;
    uint64_t model_size_bytes = 0;
    std::vector<DeviceMemoryInfo> devices;
};

struct PromptKvCacheEntry {
    std::vector<uint8_t> state;
    int32_t prefix_tokens = 0;
    std::string session_id;
    std::string prefix_fingerprint;
    std::vector<float> image_embedding;
    std::vector<float> image_token_embeddings;
    int32_t image_embedding_dim = 0;
    int32_t image_embedding_tokens = 0;
};

struct PromptKvNearQuery {
    std::string session_id;
    std::string prefix_fingerprint;
    std::span<const float> image_embedding;
    std::span<const float> image_token_embeddings;
    int32_t image_embedding_dim = 0;
    int32_t image_embedding_tokens = 0;
    float same_session_min_cosine = 0.99f;
    float cross_session_min_cosine = 0.995f;
    float same_session_min_mean_token_cosine = 0.99f;
    float cross_session_min_mean_token_cosine = 0.995f;
    float same_session_min_p05_token_cosine = 0.95f;
    float cross_session_min_p05_token_cosine = 0.98f;
    float same_session_max_relative_l2 = 0.15f;
    float cross_session_max_relative_l2 = 0.10f;
    bool allow_same_session = true;
    bool allow_cross_session = true;
};

struct PromptKvNearMatch {
    std::string key;
    std::string source_session_id;
    float cosine_similarity = 0.0f;
    float mean_token_cosine = 0.0f;
    float p05_token_cosine = 0.0f;
    float min_token_cosine = 0.0f;
    float relative_l2 = 0.0f;
    float max_abs_error = 0.0f;
    bool same_session = false;
    bool accepted = false;
};

class IPromptKvCache {
public:
    virtual ~IPromptKvCache() = default;
    virtual core::Result<bool> Probe(const std::string& key) = 0;
    virtual std::optional<PromptKvCacheEntry> Load(const std::string& key) = 0;
    virtual void Store(const std::string& key, const PromptKvCacheEntry& entry) = 0;
    virtual void StoreAlias(const std::string& alias, const std::string& key) = 0;
    virtual core::Result<std::optional<PromptKvNearMatch>> FindNear(
        const PromptKvNearQuery& query) = 0;
};

/**
 * llama.cpp VLM 推理封装
 *
 * 线程安全，支持 lazy load。
 */
class LlamaRunner {
public:
    LlamaRunner();
    ~LlamaRunner();

    LlamaRunner(const LlamaRunner&) = delete;
    LlamaRunner& operator=(const LlamaRunner&) = delete;
    LlamaRunner(LlamaRunner&&) noexcept;
    LlamaRunner& operator=(LlamaRunner&&) noexcept;

    /**
     * 加载模型
     * @param model_path 文本模型 GGUF 路径
     * @param mmproj_path 视觉投影层 GGUF 路径（可选）
     * @param n_gpu_layers GPU offload 层数，-1 表示全部
     */
    bool LoadModel(const std::filesystem::path& model_path,
                   const std::filesystem::path& mmproj_path = {},
                   int n_gpu_layers = -1,
                   int llama_threads = 8,
                   int mmproj_threads = 8);

    bool LoadFromRuntime(std::shared_ptr<LlamaSharedRuntime> runtime,
                         int llama_threads = 8);
    void SetPromptKvCache(std::shared_ptr<IPromptKvCache> cache);

    /**
     * 推理生成
     * @param image_data JPEG/PNG 二进制数据（可选，为空时纯文本推理）
     * @param prompt 文本 prompt
     * @param params 生成参数
     * @param token_callback 流式回调（可选，为 nullptr 时等价于同步生成）
     */
    VLMResult Generate(const std::vector<uint8_t>& image_data,
                       const std::string& prompt,
                       const GenerateParams& params = {},
                       std::function<void(const std::string&)> token_callback = nullptr,
                       std::shared_ptr<PreparedImageEmbedding> prepared_image = nullptr);

    VLMResult EncodeImageOnly(const std::vector<uint8_t>& image_data);

    void Unload();
    bool IsLoaded() const;
    std::string GetInfo() const;
    MemorySnapshot GetMemorySnapshot() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool loaded_ = false;
    mutable std::mutex mutex_;
};

class LlamaSharedRuntime {
public:
    LlamaSharedRuntime();
    ~LlamaSharedRuntime();

    LlamaSharedRuntime(const LlamaSharedRuntime&) = delete;
    LlamaSharedRuntime& operator=(const LlamaSharedRuntime&) = delete;

    bool Load(const std::filesystem::path& model_path,
              const std::filesystem::path& mmproj_path = {},
              int n_gpu_layers = -1,
              int mmproj_threads = 8);
    void Unload();
    bool IsLoaded() const;

    llama_model* Model() const;
    mtmd_context* MtmdContext() const;
    std::mutex& MtmdMutex() const;
    const std::filesystem::path& ModelPath() const;
    const std::filesystem::path& MmprojPath() const;
    int GpuLayers() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class LlamaRunnerPool {
public:
    class Lease {
    public:
        Lease() = default;
        ~Lease();

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;

        LlamaRunner* operator->() const;
        LlamaRunner& operator*() const;
        explicit operator bool() const noexcept;

    private:
        friend class LlamaRunnerPool;
        Lease(LlamaRunnerPool* pool, std::size_t index, LlamaRunner* runner);

        LlamaRunnerPool* pool_ = nullptr;
        std::size_t index_ = 0;
        LlamaRunner* runner_ = nullptr;
    };

    LlamaRunnerPool();
    ~LlamaRunnerPool();

    LlamaRunnerPool(const LlamaRunnerPool&) = delete;
    LlamaRunnerPool& operator=(const LlamaRunnerPool&) = delete;

    bool Load(const std::filesystem::path& model_path,
              const std::filesystem::path& mmproj_path = {},
              int n_gpu_layers = -1,
              std::size_t pool_size = 1,
              int llama_threads = 8,
              int mmproj_threads = 8,
              std::shared_ptr<IPromptKvCache> prompt_kv_cache = nullptr);
    void Unload();
    bool IsLoaded() const;
    Lease Acquire();
    MemorySnapshot GetMemorySnapshot() const;

private:
    void Release(std::size_t index);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llm
