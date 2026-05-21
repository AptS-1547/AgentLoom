#pragma once

#include "vector_repository.h"

#include <memory>

namespace storage::sqlite {
class SqliteConnectionPool;
}

namespace agent::vector_storage {

/// SQLite-backed vector repository implementation
/// Uses pImpl to hide sqlite3* handles from public interface
class SqliteVectorRepository : public IVectorRepository {
public:
    explicit SqliteVectorRepository(std::shared_ptr<storage::sqlite::SqliteConnectionPool> pool);
    ~SqliteVectorRepository() override;

    SqliteVectorRepository(const SqliteVectorRepository&) = delete;
    SqliteVectorRepository& operator=(const SqliteVectorRepository&) = delete;
    SqliteVectorRepository(SqliteVectorRepository&&) = delete;
    SqliteVectorRepository& operator=(SqliteVectorRepository&&) = delete;

    /// Schema initialization (idempotent, CREATE TABLE IF NOT EXISTS)
    core::Status EnsureSchema();

    // IVectorRepository interface
    core::Result<std::int64_t> EnsureCollection(const CollectionDescriptor& desc) override;
    core::Result<CollectionDescriptor> GetCollection(std::int64_t id) const override;
    core::Result<std::int64_t> EnsurePartition(const PartitionKey& key) override;
    core::Result<std::optional<std::int64_t>> LookupPartition(const PartitionKey& key) const override;
    core::Result<PartitionSnapshot> GetPartitionSnapshot(std::int64_t partition_id) const override;

    core::Result<std::int64_t> InsertEntry(EntryRecord entry) override;
    core::Status InsertEntries(std::span<EntryRecord> entries) override;
    core::Result<EntryRecord> GetEntry(std::int64_t entry_id) const override;
    core::Result<std::vector<EntryRecord>> LookupEntries(std::span<const std::int64_t> ids) const override;
    core::Result<std::vector<EntryRecord>> ListEntries(
        std::int64_t partition_id, bool include_forgotten) const override;

    core::Status MarkRecalled(std::span<const std::int64_t> ids, std::int64_t now_ms) override;
    core::Status MarkForgotten(std::span<const std::int64_t> ids,
                               std::int64_t now_ms, std::int64_t deleted_after_ms) override;
    core::Status Reactivate(std::span<const std::int64_t> ids, std::int64_t now_ms) override;
    core::Status DeleteEntry(std::int64_t id) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace agent::vector_storage
