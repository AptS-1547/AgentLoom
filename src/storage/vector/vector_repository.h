#pragma once

#include "vector_metadata.h"
#include "result.h"

#include <span>
#include <vector>

namespace agent::vector_storage {

/// Vector repository interface: SQLite as single source of truth
/// Does NOT expose sqlite3* handles (pImpl isolation)
class IVectorRepository {
public:
    virtual ~IVectorRepository() = default;

    // Collection / Partition management
    virtual core::Result<std::int64_t> EnsureCollection(const CollectionDescriptor& desc) = 0;
    virtual core::Result<CollectionDescriptor> GetCollection(std::int64_t id) const = 0;
    virtual core::Result<std::int64_t> EnsurePartition(const PartitionKey& key) = 0;
    virtual core::Result<std::optional<std::int64_t>> LookupPartition(const PartitionKey& key) const = 0;
    virtual core::Result<PartitionSnapshot> GetPartitionSnapshot(std::int64_t partition_id) const = 0;

    // Vector entry CRUD
    virtual core::Result<std::int64_t> InsertEntry(EntryRecord entry) = 0;
    virtual core::Status InsertEntries(std::span<EntryRecord> entries) = 0;  // single transaction batch
    virtual core::Result<EntryRecord> GetEntry(std::int64_t entry_id) const = 0;
    virtual core::Result<std::vector<EntryRecord>> LookupEntries(std::span<const std::int64_t> ids) const = 0;
    virtual core::Result<std::vector<EntryRecord>> ListEntries(
        std::int64_t partition_id, bool include_forgotten) const = 0;
    virtual core::Result<std::optional<std::int64_t>> FindEntryIdByMemoryHash(
        std::int64_t partition_id, std::string_view memory_hash) const = 0;

    // Lifecycle updates
    virtual core::Status MarkRecalled(std::span<const std::int64_t> ids, std::int64_t now_ms) = 0;
    virtual core::Status MarkForgotten(std::span<const std::int64_t> ids,
                                       std::int64_t now_ms, std::int64_t deleted_after_ms) = 0;
    virtual core::Status Reactivate(std::span<const std::int64_t> ids, std::int64_t now_ms) = 0;
    virtual core::Status DeleteEntry(std::int64_t id) = 0;
};

}  // namespace agent::vector_storage
