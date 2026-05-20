#pragma once

#include "result.h"
#include "vector_index.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vector {

class ExactVectorIndex : public IVectorIndex {
public:
    static core::Result<std::unique_ptr<ExactVectorIndex>> Create(
        VectorIndexOptions options);

    static core::Result<std::unique_ptr<ExactVectorIndex>> LoadFromFile(
        const std::filesystem::path& path);

    ExactVectorIndex(const ExactVectorIndex&) = delete;
    ExactVectorIndex& operator=(const ExactVectorIndex&) = delete;
    ExactVectorIndex(ExactVectorIndex&&) noexcept;
    ExactVectorIndex& operator=(ExactVectorIndex&&) noexcept;
    ~ExactVectorIndex() override;

    std::size_t Dimension() const noexcept override;
    std::size_t Size() const noexcept override;

    core::Status Add(
        std::span<const float> vectors,
        std::span<const std::int64_t> ids) override;

    core::Result<std::vector<VectorSearchResult>> Search(
        std::span<const float> query,
        std::size_t top_k) const override;

    core::Status Remove(std::span<const std::int64_t> ids) override;

    core::Status Save(const std::filesystem::path& path) const override;
    core::Status Reset() override;

    std::string BackendName() const override;

private:
    explicit ExactVectorIndex(std::size_t dimension) noexcept;

    std::size_t dimension_ = 0;
    std::vector<std::int64_t> ids_;
    std::vector<float> data_;
};

} // namespace vector
