#pragma once

#include "hf_tokenizer.h"
#include "result.h"

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace vector {

enum class PoolingStrategy {
    Cls,
    Mean,
};

struct EmbeddingModelOptions {
    std::filesystem::path model_path;
    std::string execution_provider = "auto";
    bool allow_cpu_fallback = true;
    int cuda_device_id = 0;
    int intra_op_num_threads = 0;
    int inter_op_num_threads = 0;
    PoolingStrategy pooling = PoolingStrategy::Mean;
    bool normalize = true;
    std::size_t expected_dimension = 0;
    bool require_token_type_ids = false;
};

struct EmbeddingBatch {
    std::size_t batch_size = 0;
    std::size_t dimension = 0;
    std::vector<float> embeddings;

    std::span<const float> row(std::size_t i) const noexcept {
        if (i >= batch_size || dimension == 0) {
            return {};
        }
        return std::span<const float>(embeddings.data() + i * dimension, dimension);
    }
};

class IEmbeddingModel {
public:
    virtual ~IEmbeddingModel() = default;

    virtual core::Result<EmbeddingBatch> Embed(const TokenizedBatch& batch) const = 0;

    virtual std::size_t Dimension() const noexcept = 0;
    virtual PoolingStrategy Pooling() const noexcept = 0;
    virtual bool Normalized() const noexcept = 0;
};

namespace pooling {

core::Status MeanPool(
    const float* last_hidden_state,
    const std::int64_t* attention_mask,
    std::size_t batch_size,
    std::size_t sequence_length,
    std::size_t hidden_size,
    float* out_embeddings);

core::Status ClsPool(
    const float* last_hidden_state,
    std::size_t batch_size,
    std::size_t sequence_length,
    std::size_t hidden_size,
    float* out_embeddings);

void L2NormalizeRows(float* data, std::size_t batch_size, std::size_t dimension) noexcept;

} // namespace pooling

} // namespace vector
