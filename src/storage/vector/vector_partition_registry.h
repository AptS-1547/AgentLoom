#pragma once

#include "vector_repository.h"

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace agent::vector_storage {

/// In-memory cache for PartitionKey <-> partition_id resolution plus last_modified
/// tracking used by VectorIndexManager to detect stale hydrated Faiss indices.
///
/// Reads are lock-free via shared_lock; writes take exclusive lock.
/// Repository remains the source of truth — this cache only avoids the round-trip
/// for frequently-resolved keys and centralizes modification notifications.
class PartitionRegistry {
public:
    explicit PartitionRegistry(std::shared_ptr<IVectorRepository> repo);

    /// Resolve a partition key to its id, creating it via repo->EnsurePartition
    /// if missing. Subsequent calls with the same key return the cached id.
    core::Result<std::int64_t> Resolve(const PartitionKey& key);

    /// Look up a partition without creating it.  Returns nullopt if the key has
    /// never been resolved or registered.
    core::Result<std::optional<std::int64_t>> Lookup(const PartitionKey& key) const;

    /// Repository writers call this after a successful insert/delete to record
    /// the latest modification time for the partition.  IndexManager compares
    /// against this value to detect that its hydrated index is stale.
    void NotifyModified(std::int64_t partition_id, std::int64_t now_ms);

    /// Get the last recorded modification time. Returns 0 if no NotifyModified
    /// has been received in this process.
    std::int64_t LastModifiedAtMs(std::int64_t partition_id) const;

    /// Drop all cached entries (testing / collection rebuild).
    void Clear();

    /// Approximate cache size for diagnostics.
    std::size_t CacheSize() const;

private:
    std::shared_ptr<IVectorRepository> repo_;
    mutable std::shared_mutex lock_;
    mutable std::unordered_map<PartitionKey, std::int64_t, PartitionKeyHash> id_cache_;
    std::unordered_map<std::int64_t, std::int64_t> modified_ms_cache_;
};

}  // namespace agent::vector_storage
