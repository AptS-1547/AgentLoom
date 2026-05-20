#include "embedding_pipeline.h"

#include <utility>

namespace vector {

EmbeddingPipeline::EmbeddingPipeline(
    std::shared_ptr<HfTokenizer> tokenizer,
    std::shared_ptr<IEmbeddingModel> model,
    EmbeddingPipelineOptions options)
    : tokenizer_(std::move(tokenizer)),
      model_(std::move(model)),
      options_(std::move(options)) {}

core::Result<std::vector<float>> EmbeddingPipeline::Encode(std::string_view text) const {
    if (!tokenizer_ || !model_) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "EmbeddingPipeline::Encode: tokenizer or model is null");
    }

    auto tokenized = tokenizer_->Encode(text, options_.tokenizer_options);
    if (!tokenized.ok()) {
        return tokenized.status();
    }

    auto embedded = model_->Embed(tokenized.value());
    if (!embedded.ok()) {
        return embedded.status();
    }

    const auto& batch = embedded.value();
    if (batch.batch_size != 1) {
        return core::Status(core::ErrorCode::InternalError,
                            "EmbeddingPipeline::Encode: batch size != 1");
    }

    std::vector<float> result(batch.dimension);
    std::copy_n(batch.embeddings.data(), batch.dimension, result.begin());
    return result;
}

core::Result<EmbeddingBatch> EmbeddingPipeline::EncodeBatch(
    std::span<const std::string_view> texts) const {
    if (!tokenizer_ || !model_) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "EmbeddingPipeline::EncodeBatch: tokenizer or model is null");
    }

    auto tokenized = tokenizer_->EncodeBatch(texts, options_.tokenizer_options);
    if (!tokenized.ok()) {
        return tokenized.status();
    }

    return model_->Embed(tokenized.value());
}

std::size_t EmbeddingPipeline::Dimension() const noexcept {
    return model_ ? model_->Dimension() : 0;
}

PoolingStrategy EmbeddingPipeline::Pooling() const noexcept {
    return model_ ? model_->Pooling() : PoolingStrategy::Mean;
}

bool EmbeddingPipeline::Normalized() const noexcept {
    return model_ ? model_->Normalized() : false;
}

} // namespace vector
