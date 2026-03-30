/**
 * @file onnx_model.h
 * @brief ONNX Runtime BERT 模型封装
 */

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <optional>

// ONNX Runtime
#include <onnxruntime_cxx_api.h>

namespace bert {

/**
 * BERT 推理结果
 */
struct InferenceResult {
    // emotion_logits [10]
    std::vector<float> emotion_logits;
    // behavior_logits [12]
    std::vector<float> behavior_logits;
    // tone_logits [8]
    std::vector<float> tone_logits;
    // intensity [0-1]
    float intensity;
    // response_length_logits [3]
    std::vector<float> response_length_logits;

    bool success = false;
    std::string error_message;
};

/**
 * ONNX Runtime 会话配置
 */
struct ModelRuntimeOptions {
    // auto / cpu / cuda
    std::string execution_provider = "auto";
    // 当请求 CUDA 但运行环境不满足时，是否自动回退到 CPU
    bool allow_cpu_fallback = true;
    // CUDA 设备 ID
    int cuda_device_id = 0;
    // <= 0 表示使用服务端的默认调优值
    int intra_op_num_threads = 0;
    // <= 0 表示使用服务端的默认调优值
    int inter_op_num_threads = 0;
    // CPU allocator arena
    bool enable_cpu_mem_arena = true;
    // 内存 pattern cache
    bool enable_mem_pattern = true;
};

/**
 * ONNX BERT 模型封装
 *
 * 封装了 ONNX Runtime 的 Session，提供线程安全的推理接口。
 * 注意：Ort::Session 本身是线程安全的，可以并发调用 Run()。
 */
class OnnxBERTModel {
public:
    OnnxBERTModel();
    ~OnnxBERTModel();

    // 禁止拷贝，允许移动
    OnnxBERTModel(const OnnxBERTModel&) = delete;
    OnnxBERTModel& operator=(const OnnxBERTModel&) = delete;
    OnnxBERTModel(OnnxBERTModel&&) noexcept;
    OnnxBERTModel& operator=(OnnxBERTModel&&) noexcept;

    /**
     * 加载 ONNX 模型
     * @param model_path ONNX 模型文件路径
     * @return 是否成功
     */
    bool LoadModel(const std::filesystem::path& model_path,
                   const ModelRuntimeOptions& options = {});

    /**
     * 单条推理
     * @param input_ids token IDs [seq_len]
     * @param attention_mask attention mask [seq_len]
     * @param personality personality vector [11]
     * @return 推理结果
     */
    InferenceResult Predict(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>& attention_mask,
        const std::vector<float>& personality);

    /**
     * 批量推理
     * @param input_ids 展平的 token IDs [batch * seq_len]
     * @param attention_mask 展平的 attention mask [batch * seq_len]
     * @param personality 展平的 personality [batch * 11]
     * @param batch_size batch 大小
     * @param seq_len 序列长度
     * @return 推理结果列表
     */
    std::vector<InferenceResult> PredictBatch(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>& attention_mask,
        const std::vector<float>& personality,
        size_t batch_size,
        size_t seq_len);

    /**
     * 检查模型是否已加载
     */
    bool IsLoaded() const { return loaded_; }

    /**
     * 获取模型信息
     */
    std::string GetInfo() const;

    /**
     * 获取当前实际使用的执行提供器
     */
    std::string GetActiveExecutionProvider() const;

private:
    // PIMPL 实现
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool loaded_ = false;
};

} // namespace bert
