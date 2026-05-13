/**
 * @file llama_runner.h
 * @brief llama.cpp VLM 推理封装
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace llm {

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

    std::vector<float> image_embedding;
    int32_t image_embedding_dim = 0;
    int32_t image_embedding_tokens = 0;

    bool success = false;
    std::string error_message;
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

/**
 * llama.cpp VLM 推理封装
 *
 * 线程安全，支持 lazy load。
 * 同一时刻只能有一个推理任务（单飞行锁）。
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
                   int n_gpu_layers = -1);

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
                       std::function<void(const std::string&)> token_callback = nullptr);

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

} // namespace llm
