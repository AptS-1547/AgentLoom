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

class FaissVectorIndex : public IVectorIndex {
public:
    static core::Result<std::unique_ptr<FaissVectorIndex>> Create(
        VectorIndexOptions options);

    static core::Result<std::unique_ptr<FaissVectorIndex>> LoadFromFile(
        const std::filesystem::path& path,
        std::size_t expected_dimension = 0);

    FaissVectorIndex(const FaissVectorIndex&) = delete;
    FaissVectorIndex& operator=(const FaissVectorIndex&) = delete;
    FaissVectorIndex(FaissVectorIndex&&) noexcept;
    FaissVectorIndex& operator=(FaissVectorIndex&&) noexcept;
    ~FaissVectorIndex() override;

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
    FaissVectorIndex() noexcept;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vector
