#include "vector_partition_registry.h"

namespace agent::vector_storage {

PartitionRegistry::PartitionRegistry(std::shared_ptr<IVectorRepository> repo)
    : repo_(std::move(repo)) {}

core::Result<std::int64_t> PartitionRegistry::Resolve(const PartitionKey& key) {
    {
        std::shared_lock read_lock(lock_);
        auto it = id_cache_.find(key);
        if (it != id_cache_.end()) {
            return it->second;
        }
    }

    // Cache miss — hit the repository (creates if missing) and populate cache.
    // Note: repo->EnsurePartition is idempotent under UNIQUE constraint, so
    // a concurrent Resolve on the same key will safely converge to one id.
    auto repo_r = repo_->EnsurePartition(key);
    if (!repo_r) return repo_r.status();
    std::int64_t id = std::move(repo_r).value();

    {
        std::unique_lock write_lock(lock_);
        id_cache_[key] = id;
    }
    return id;
}

core::Result<std::optional<std::int64_t>> PartitionRegistry::Lookup(const PartitionKey& key) const {
    {
        std::shared_lock read_lock(lock_);
        auto it = id_cache_.find(key);
        if (it != id_cache_.end()) {
            return std::optional<std::int64_t>{it->second};
        }
    }

    // Not cached — ask the repository without creating.
    auto repo_r = repo_->LookupPartition(key);
    if (!repo_r) return repo_r.status();
    auto opt = std::move(repo_r).value();
    if (opt.has_value()) {
        std::unique_lock write_lock(lock_);
        id_cache_[key] = static_cast<std::int64_t>(opt.value());
    }
    return opt;
}

void PartitionRegistry::NotifyModified(std::int64_t partition_id, std::int64_t now_ms) {
    std::unique_lock write_lock(lock_);
    auto& slot = modified_ms_cache_[partition_id];
    if (now_ms > slot) {
        slot = now_ms;
    }
}

std::int64_t PartitionRegistry::LastModifiedAtMs(std::int64_t partition_id) const {
    std::shared_lock read_lock(lock_);
    auto it = modified_ms_cache_.find(partition_id);
    return it != modified_ms_cache_.end() ? it->second : 0;
}

void PartitionRegistry::Clear() {
    std::unique_lock write_lock(lock_);
    id_cache_.clear();
    modified_ms_cache_.clear();
}

std::size_t PartitionRegistry::CacheSize() const {
    std::shared_lock read_lock(lock_);
    return id_cache_.size();
}

}  // namespace agent::vector_storage
