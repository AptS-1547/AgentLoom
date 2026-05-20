#include "exact_vector_index.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <unordered_set>

namespace vector {

namespace {

constexpr const char kExactIndexMagic[8] = {'E', 'X', 'I', 'D', 'X', '0', '0', '1'};
constexpr std::uint32_t kExactIndexVersion = 1;

template <typename T>
bool ReadPod(std::istream& is, T& value) {
    return static_cast<bool>(is.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

template <typename T>
bool WritePod(std::ostream& os, const T& value) {
    return static_cast<bool>(os.write(reinterpret_cast<const char*>(&value), sizeof(T)));
}

} // namespace

ExactVectorIndex::ExactVectorIndex(std::size_t dimension) noexcept
    : dimension_(dimension) {}

ExactVectorIndex::~ExactVectorIndex() = default;
ExactVectorIndex::ExactVectorIndex(ExactVectorIndex&&) noexcept = default;
ExactVectorIndex& ExactVectorIndex::operator=(ExactVectorIndex&&) noexcept = default;

core::Result<std::unique_ptr<ExactVectorIndex>> ExactVectorIndex::Create(
    VectorIndexOptions options) {
    if (options.dimension == 0) {
        return core::Status(core::ErrorCode::InvalidArgument,
                            "ExactVectorIndex::Create: dimension must be > 0");
    }
    return std::unique_ptr<ExactVectorIndex>(new ExactVectorIndex(options.dimension));
}

std::size_t ExactVectorIndex::Dimension() const noexcept {
    return dimension_;
}

std::size_t ExactVectorIndex::Size() const noexcept {
    return ids_.size();
}

std::string ExactVectorIndex::BackendName() const {
    return "exact_ip";
}

core::Status ExactVectorIndex::Add(
    std::span<const float> vectors,
    std::span<const std::int64_t> ids) {
    if (ids.empty()) {
        return core::Status::Ok();
    }
    if (vectors.size() != ids.size() * dimension_) {
        return core::Status(core::ErrorCode::InvalidArgument,
                            "ExactVectorIndex::Add: vectors.size() must equal ids.size() * dimension");
    }

    const std::size_t old_count = ids_.size();
    ids_.insert(ids_.end(), ids.begin(), ids.end());
    data_.insert(data_.end(), vectors.begin(), vectors.end());

    (void)old_count;
    return core::Status::Ok();
}

core::Result<std::vector<VectorSearchResult>> ExactVectorIndex::Search(
    std::span<const float> query,
    std::size_t top_k) const {
    if (query.size() != dimension_) {
        return core::Status(core::ErrorCode::InvalidArgument,
                            "ExactVectorIndex::Search: query dimension mismatch");
    }
    if (top_k == 0) {
        return std::vector<VectorSearchResult>{};
    }

    const std::size_t n = ids_.size();
    if (n == 0) {
        return std::vector<VectorSearchResult>{};
    }

    std::vector<VectorSearchResult> all;
    all.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float* row = data_.data() + i * dimension_;
        float ip = 0.0f;
        for (std::size_t d = 0; d < dimension_; ++d) {
            ip += row[d] * query[d];
        }
        all.push_back({ids_[i], ip});
    }

    const std::size_t k = std::min(top_k, n);
    std::vector<VectorSearchResult> out(k);
    std::partial_sort_copy(
        all.begin(), all.end(),
        out.begin(), out.end(),
        [](const VectorSearchResult& a, const VectorSearchResult& b) {
            return a.score > b.score;
        });
    return out;
}

core::Status ExactVectorIndex::Remove(std::span<const std::int64_t> ids) {
    if (ids.empty()) {
        return core::Status::Ok();
    }
    std::unordered_set<std::int64_t> target(ids.begin(), ids.end());

    std::size_t write = 0;
    for (std::size_t read = 0; read < ids_.size(); ++read) {
        if (target.contains(ids_[read])) {
            continue;
        }
        if (write != read) {
            ids_[write] = ids_[read];
            std::memcpy(
                data_.data() + write * dimension_,
                data_.data() + read * dimension_,
                dimension_ * sizeof(float));
        }
        ++write;
    }
    ids_.resize(write);
    data_.resize(write * dimension_);
    return core::Status::Ok();
}

core::Status ExactVectorIndex::Save(const std::filesystem::path& path) const {
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) {
        return core::Status(core::ErrorCode::Unavailable,
                            "ExactVectorIndex::Save: failed to open output file");
    }

    if (!os.write(kExactIndexMagic, sizeof(kExactIndexMagic))) {
        return core::Status(core::ErrorCode::InternalError, "write magic failed");
    }
    if (!WritePod<std::uint32_t>(os, kExactIndexVersion)) {
        return core::Status(core::ErrorCode::InternalError, "write version failed");
    }
    const std::uint64_t dim = dimension_;
    if (!WritePod<std::uint64_t>(os, dim)) {
        return core::Status(core::ErrorCode::InternalError, "write dimension failed");
    }
    const std::uint64_t count = ids_.size();
    if (!WritePod<std::uint64_t>(os, count)) {
        return core::Status(core::ErrorCode::InternalError, "write count failed");
    }

    if (count > 0) {
        if (!os.write(
                reinterpret_cast<const char*>(ids_.data()),
                static_cast<std::streamsize>(count * sizeof(std::int64_t)))) {
            return core::Status(core::ErrorCode::InternalError, "write ids failed");
        }
        if (!os.write(
                reinterpret_cast<const char*>(data_.data()),
                static_cast<std::streamsize>(count * dimension_ * sizeof(float)))) {
            return core::Status(core::ErrorCode::InternalError, "write vectors failed");
        }
    }
    return core::Status::Ok();
}

core::Result<std::unique_ptr<ExactVectorIndex>> ExactVectorIndex::LoadFromFile(
    const std::filesystem::path& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        return core::Status(core::ErrorCode::NotFound,
                            "ExactVectorIndex::LoadFromFile: file not found");
    }

    char magic[sizeof(kExactIndexMagic)] = {};
    if (!is.read(magic, sizeof(magic)) ||
        std::memcmp(magic, kExactIndexMagic, sizeof(kExactIndexMagic)) != 0) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "ExactVectorIndex::LoadFromFile: bad magic");
    }

    std::uint32_t version = 0;
    if (!ReadPod(is, version)) {
        return core::Status(core::ErrorCode::FailedPrecondition, "read version failed");
    }
    if (version != kExactIndexVersion) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "ExactVectorIndex::LoadFromFile: unsupported version");
    }

    std::uint64_t dim = 0;
    if (!ReadPod(is, dim) || dim == 0) {
        return core::Status(core::ErrorCode::FailedPrecondition, "read dimension failed");
    }
    std::uint64_t count = 0;
    if (!ReadPod(is, count)) {
        return core::Status(core::ErrorCode::FailedPrecondition, "read count failed");
    }

    std::unique_ptr<ExactVectorIndex> index(
        new ExactVectorIndex(static_cast<std::size_t>(dim)));

    if (count > 0) {
        index->ids_.resize(static_cast<std::size_t>(count));
        index->data_.resize(static_cast<std::size_t>(count * dim));
        if (!is.read(
                reinterpret_cast<char*>(index->ids_.data()),
                static_cast<std::streamsize>(count * sizeof(std::int64_t)))) {
            return core::Status(core::ErrorCode::FailedPrecondition, "read ids failed");
        }
        if (!is.read(
                reinterpret_cast<char*>(index->data_.data()),
                static_cast<std::streamsize>(count * dim * sizeof(float)))) {
            return core::Status(core::ErrorCode::FailedPrecondition, "read vectors failed");
        }
    }

    return index;
}

core::Status ExactVectorIndex::Reset() {
    ids_.clear();
    data_.clear();
    return core::Status::Ok();
}

} // namespace vector
