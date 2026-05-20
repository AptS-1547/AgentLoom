#pragma once

#include "result.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace vector {

struct VectorSearchResult {
    std::int64_t id = 0;
    float score = 0.0f;
};

struct VectorIndexOptions {
    std::size_t dimension = 0;
};

class IVectorIndex {
public:
    virtual ~IVectorIndex() = default;

    virtual std::size_t Dimension() const noexcept = 0;
    virtual std::size_t Size() const noexcept = 0;

    virtual core::Status Add(
        std::span<const float> vectors,
        std::span<const std::int64_t> ids) = 0;

    virtual core::Result<std::vector<VectorSearchResult>> Search(
        std::span<const float> query,
        std::size_t top_k) const = 0;

    virtual core::Status Remove(std::span<const std::int64_t> ids) = 0;

    virtual core::Status Save(const std::filesystem::path& path) const = 0;
    virtual core::Status Reset() = 0;

    virtual std::string BackendName() const = 0;
};

} // namespace vector
