#include "faiss_vector_index.h"

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>

#include <exception>
#include <utility>

namespace vector {

namespace {

core::Status TranslateFaissError(const std::exception& e, std::string_view context) {
    std::string msg;
    msg.reserve(context.size() + 64);
    msg.append(context);
    msg.append(": ");
    msg.append(e.what());
    return core::Status(core::ErrorCode::InternalError, std::move(msg));
}

} // namespace

struct FaissVectorIndex::Impl {
    std::size_t dimension = 0;
    std::unique_ptr<faiss::IndexIDMap2> index;
};

FaissVectorIndex::FaissVectorIndex() noexcept : impl_(std::make_unique<Impl>()) {}
FaissVectorIndex::~FaissVectorIndex() = default;
FaissVectorIndex::FaissVectorIndex(FaissVectorIndex&&) noexcept = default;
FaissVectorIndex& FaissVectorIndex::operator=(FaissVectorIndex&&) noexcept = default;

std::size_t FaissVectorIndex::Dimension() const noexcept {
    return impl_ ? impl_->dimension : 0;
}

std::size_t FaissVectorIndex::Size() const noexcept {
    if (!impl_ || !impl_->index) {
        return 0;
    }
    return static_cast<std::size_t>(impl_->index->ntotal);
}

std::string FaissVectorIndex::BackendName() const {
    return "faiss_flat_ip";
}

core::Result<std::unique_ptr<FaissVectorIndex>> FaissVectorIndex::Create(
    VectorIndexOptions options) {
    if (options.dimension == 0) {
        return core::Status(core::ErrorCode::InvalidArgument,
                            "FaissVectorIndex::Create: dimension must be > 0");
    }

    try {
        auto flat = std::make_unique<faiss::IndexFlatIP>(
            static_cast<int>(options.dimension));
        auto idmap = std::make_unique<faiss::IndexIDMap2>(flat.get());
        // IndexIDMap2 takes ownership of the underlying Index when own_fields
        // is set; we transfer ownership explicitly to make the model unambiguous.
        idmap->own_fields = true;
        (void)flat.release();

        std::unique_ptr<FaissVectorIndex> wrapper(new FaissVectorIndex());
        wrapper->impl_->dimension = options.dimension;
        wrapper->impl_->index = std::move(idmap);
        return wrapper;
    } catch (const faiss::FaissException& e) {
        return TranslateFaissError(e, "FaissVectorIndex::Create");
    } catch (const std::exception& e) {
        return TranslateFaissError(e, "FaissVectorIndex::Create");
    }
}

core::Status FaissVectorIndex::Add(
    std::span<const float> vectors,
    std::span<const std::int64_t> ids) {
    if (!impl_ || !impl_->index) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "FaissVectorIndex::Add: index not initialized");
    }
    if (ids.empty()) {
        return core::Status::Ok();
    }
    if (vectors.size() != ids.size() * impl_->dimension) {
        return core::Status(core::ErrorCode::InvalidArgument,
                            "FaissVectorIndex::Add: vectors.size() must equal ids.size() * dimension");
    }

    try {
        static_assert(sizeof(faiss::idx_t) == sizeof(std::int64_t),
                      "faiss::idx_t must be 64-bit to match our ID model");
        impl_->index->add_with_ids(
            static_cast<faiss::idx_t>(ids.size()),
            vectors.data(),
            reinterpret_cast<const faiss::idx_t*>(ids.data()));
        return core::Status::Ok();
    } catch (const faiss::FaissException& e) {
        return TranslateFaissError(e, "FaissVectorIndex::Add");
    } catch (const std::exception& e) {
        return TranslateFaissError(e, "FaissVectorIndex::Add");
    }
}

core::Result<std::vector<VectorSearchResult>> FaissVectorIndex::Search(
    std::span<const float> query,
    std::size_t top_k) const {
    if (!impl_ || !impl_->index) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "FaissVectorIndex::Search: index not initialized");
    }
    if (query.size() != impl_->dimension) {
        return core::Status(core::ErrorCode::InvalidArgument,
                            "FaissVectorIndex::Search: query dimension mismatch");
    }
    if (top_k == 0) {
        return std::vector<VectorSearchResult>{};
    }
    if (impl_->index->ntotal == 0) {
        return std::vector<VectorSearchResult>{};
    }

    const std::size_t k = std::min<std::size_t>(
        top_k, static_cast<std::size_t>(impl_->index->ntotal));

    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);

    try {
        impl_->index->search(
            /*n=*/1,
            query.data(),
            static_cast<faiss::idx_t>(k),
            distances.data(),
            labels.data());
    } catch (const faiss::FaissException& e) {
        return TranslateFaissError(e, "FaissVectorIndex::Search");
    } catch (const std::exception& e) {
        return TranslateFaissError(e, "FaissVectorIndex::Search");
    }

    std::vector<VectorSearchResult> out;
    out.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        if (labels[i] < 0) {
            continue;
        }
        out.push_back({static_cast<std::int64_t>(labels[i]), distances[i]});
    }
    return out;
}

core::Status FaissVectorIndex::Remove(std::span<const std::int64_t> ids) {
    if (!impl_ || !impl_->index) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "FaissVectorIndex::Remove: index not initialized");
    }
    if (ids.empty()) {
        return core::Status::Ok();
    }
    try {
        faiss::IDSelectorBatch selector(
            ids.size(),
            reinterpret_cast<const faiss::idx_t*>(ids.data()));
        impl_->index->remove_ids(selector);
        return core::Status::Ok();
    } catch (const faiss::FaissException& e) {
        return TranslateFaissError(e, "FaissVectorIndex::Remove");
    } catch (const std::exception& e) {
        return TranslateFaissError(e, "FaissVectorIndex::Remove");
    }
}

core::Status FaissVectorIndex::Save(const std::filesystem::path& path) const {
    if (!impl_ || !impl_->index) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "FaissVectorIndex::Save: index not initialized");
    }
    try {
        const auto utf8 = path.u8string();
        std::string narrow(utf8.begin(), utf8.end());
        faiss::write_index(impl_->index.get(), narrow.c_str());
        return core::Status::Ok();
    } catch (const faiss::FaissException& e) {
        return TranslateFaissError(e, "FaissVectorIndex::Save");
    } catch (const std::exception& e) {
        return TranslateFaissError(e, "FaissVectorIndex::Save");
    }
}

core::Result<std::unique_ptr<FaissVectorIndex>> FaissVectorIndex::LoadFromFile(
    const std::filesystem::path& path,
    std::size_t expected_dimension) {
    faiss::Index* raw_loaded = nullptr;
    try {
        const auto utf8 = path.u8string();
        std::string narrow(utf8.begin(), utf8.end());
        raw_loaded = faiss::read_index(narrow.c_str());
    } catch (const faiss::FaissException& e) {
        return TranslateFaissError(e, "FaissVectorIndex::LoadFromFile");
    } catch (const std::exception& e) {
        return TranslateFaissError(e, "FaissVectorIndex::LoadFromFile");
    }

    if (raw_loaded == nullptr) {
        return core::Status(core::ErrorCode::InternalError,
                            "FaissVectorIndex::LoadFromFile: read_index returned null");
    }

    auto* idmap = dynamic_cast<faiss::IndexIDMap2*>(raw_loaded);
    if (idmap == nullptr) {
        delete raw_loaded;
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "FaissVectorIndex::LoadFromFile: file is not an IndexIDMap2");
    }
    std::unique_ptr<faiss::IndexIDMap2> owned(idmap);

    const std::size_t dim = static_cast<std::size_t>(owned->d);
    if (expected_dimension != 0 && expected_dimension != dim) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "FaissVectorIndex::LoadFromFile: dimension mismatch");
    }

    std::unique_ptr<FaissVectorIndex> wrapper(new FaissVectorIndex());
    wrapper->impl_->dimension = dim;
    wrapper->impl_->index = std::move(owned);
    return wrapper;
}

core::Status FaissVectorIndex::Reset() {
    if (!impl_ || !impl_->index) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "FaissVectorIndex::Reset: index not initialized");
    }
    try {
        impl_->index->reset();
        return core::Status::Ok();
    } catch (const faiss::FaissException& e) {
        return TranslateFaissError(e, "FaissVectorIndex::Reset");
    } catch (const std::exception& e) {
        return TranslateFaissError(e, "FaissVectorIndex::Reset");
    }
}

} // namespace vector
