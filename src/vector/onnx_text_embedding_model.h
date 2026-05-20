#pragma once

#include "text_embedding_model.h"
#include "result.h"

#include <memory>

namespace vector {

class OnnxTextEmbeddingModel : public IEmbeddingModel {
public:
    OnnxTextEmbeddingModel();
    ~OnnxTextEmbeddingModel() override;

    OnnxTextEmbeddingModel(const OnnxTextEmbeddingModel&) = delete;
    OnnxTextEmbeddingModel& operator=(const OnnxTextEmbeddingModel&) = delete;
    OnnxTextEmbeddingModel(OnnxTextEmbeddingModel&&) noexcept;
    OnnxTextEmbeddingModel& operator=(OnnxTextEmbeddingModel&&) noexcept;

    static core::Result<std::unique_ptr<OnnxTextEmbeddingModel>> Load(
        EmbeddingModelOptions options);

    core::Result<EmbeddingBatch> Embed(const TokenizedBatch& batch) const override;

    std::size_t Dimension() const noexcept override;
    PoolingStrategy Pooling() const noexcept override;
    bool Normalized() const noexcept override;

    std::string GetInfo() const;
    std::string GetActiveExecutionProvider() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vector
