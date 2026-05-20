#pragma once

#include "hf_tokenizer.h"
#include "onnx_text_embedding_model.h"
#include "result.h"

#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace vector {

struct EmbeddingPipelineOptions {
    EncodeOptions tokenizer_options;
};

class EmbeddingPipeline {
public:
    EmbeddingPipeline(
        std::shared_ptr<HfTokenizer> tokenizer,
        std::shared_ptr<IEmbeddingModel> model,
        EmbeddingPipelineOptions options = {});

    core::Result<std::vector<float>> Encode(std::string_view text) const;
    core::Result<EmbeddingBatch> EncodeBatch(std::span<const std::string_view> texts) const;

    std::size_t Dimension() const noexcept;
    PoolingStrategy Pooling() const noexcept;
    bool Normalized() const noexcept;

private:
    std::shared_ptr<HfTokenizer> tokenizer_;
    std::shared_ptr<IEmbeddingModel> model_;
    EmbeddingPipelineOptions options_;
};

} // namespace vector
