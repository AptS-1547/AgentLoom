#include "sqlite_vector_repository.h"
#include "sqlite/sqlite_connection_pool.h"
#include "sqlite/sqlite_connection.h"
#include "sqlite/sqlite_statement.h"
#include "sqlite/sqlite_transaction.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <cstring>
#include <array>

namespace agent::vector_storage {

namespace {

using storage::sqlite::SqliteConnectionLease;
using storage::sqlite::SqliteConnectionPool;
using storage::sqlite::SqliteStatement;
using storage::sqlite::SqliteStepResult;
using storage::sqlite::SqliteTransaction;

constexpr std::chrono::milliseconds kAcquireTimeout{5000};

std::string LoadSchemaSQL() {
    std::filesystem::path schema_path = std::filesystem::path(__FILE__).parent_path() / "schema.sql";
    std::ifstream file(schema_path);
    if (!file) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

core::Status ApplyVectorSchemaV1(storage::sqlite::SqliteConnection& connection) {
    const auto schema_sql = LoadSchemaSQL();
    if (schema_sql.empty()) {
        return core::Status::Error(
            core::ErrorCode::InternalError,
            "failed to load vector schema.sql");
    }
    return connection.Execute(schema_sql);
}

const std::array<storage::sqlite::SqliteMigrationStep, 1> kVectorMigrations{{
    {
        .version = 1,
        .name = "create_vector_repository_schema",
        .checksum = "vector_repository_v1_schema_sql",
        .apply = ApplyVectorSchemaV1,
    },
}};

std::size_t HashCombine(std::size_t seed, std::size_t value) noexcept {
    return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

std::int64_t NowMs() noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Convert std::vector<float> to byte span (native byte order, single-machine only)
std::span<const std::byte> VectorToBytes(const std::vector<float>& v) noexcept {
    return std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(v.data()),
        v.size() * sizeof(float));
}

std::vector<float> BytesToVector(const std::vector<std::byte>& bytes) {
    if (bytes.size() % sizeof(float) != 0) {
        return {};
    }
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

core::Status BindOptInt64(SqliteStatement& stmt, int index, const std::optional<std::int64_t>& v) {
    if (v.has_value()) {
        return stmt.BindInt64(index, *v);
    }
    return stmt.BindNull(index);
}

std::optional<std::int64_t> ColumnOptInt64(const SqliteStatement& stmt, int index) {
    if (stmt.ColumnIsNull(index)) {
        return std::nullopt;
    }
    return stmt.ColumnInt64(index);
}

}

std::size_t PartitionKeyHash::operator()(const PartitionKey& key) const noexcept {
    std::size_t seed = 0;
    seed = HashCombine(seed, static_cast<std::size_t>(key.collection_id));
    seed = HashCombine(seed, std::hash<std::string>{}(key.tenant_id));
    seed = HashCombine(seed, std::hash<std::string>{}(key.user_id));
    seed = HashCombine(seed, std::hash<std::string>{}(key.memory_level));
    seed = HashCombine(seed, std::hash<std::string>{}(key.scope_extras_json));
    return seed;
}

struct SqliteVectorRepository::Impl {
    std::shared_ptr<SqliteConnectionPool> pool;

    explicit Impl(std::shared_ptr<SqliteConnectionPool> p) : pool(std::move(p)) {}

    core::Result<SqliteConnectionLease> AcquireRead() {
        return pool->WaitAcquireReadFor(kAcquireTimeout);
    }

    core::Result<SqliteConnectionLease> AcquireWrite() {
        return pool->WaitAcquireWriteFor(kAcquireTimeout);
    }

    static core::Status BindEntryFields(SqliteStatement& stmt, const EntryRecord& e) {
        int i = 1;
        auto rc = stmt.BindInt64(i++, e.partition_id);  if (!rc.ok()) return rc;
        rc = stmt.BindText(i++, e.cache_key);          if (!rc.ok()) return rc;
        rc = stmt.BindText(i++, e.text_hash);          if (!rc.ok()) return rc;
        rc = stmt.BindText(i++, e.content_hash);       if (!rc.ok()) return rc;
        rc = stmt.BindText(i++, e.memory_hash);        if (!rc.ok()) return rc;
        rc = stmt.BindBlob(i++, VectorToBytes(e.vector)); if (!rc.ok()) return rc;
        rc = stmt.BindInt64(i++, e.recall_count);       if (!rc.ok()) return rc;
        rc = BindOptInt64(stmt, i++, e.last_recalled_at_ms); if (!rc.ok()) return rc;
        rc = stmt.BindInt(i++, e.forgotten ? 1 : 0);    if (!rc.ok()) return rc;
        rc = BindOptInt64(stmt, i++, e.forgotten_at_ms); if (!rc.ok()) return rc;
        rc = BindOptInt64(stmt, i++, e.deleted_after_ms); if (!rc.ok()) return rc;
        rc = stmt.BindInt64(i++, e.forget_epoch);       if (!rc.ok()) return rc;
        rc = stmt.BindText(i++, e.memory_type);         if (!rc.ok()) return rc;
        rc = stmt.BindText(i++, e.emotion);             if (!rc.ok()) return rc;
        rc = stmt.BindDouble(i++, e.emotion_intensity); if (!rc.ok()) return rc;
        rc = stmt.BindDouble(i++, e.state_arousal);     if (!rc.ok()) return rc;
        rc = stmt.BindText(i++, e.answer_type);         if (!rc.ok()) return rc;
        rc = stmt.BindText(i++, e.payload);             if (!rc.ok()) return rc;
        rc = stmt.BindText(i++, e.extra_metadata_json); if (!rc.ok()) return rc;
        rc = stmt.BindInt64(i++, e.created_at_ms);      if (!rc.ok()) return rc;
        rc = BindOptInt64(stmt, i++, e.expires_at_ms);  if (!rc.ok()) return rc;
        return core::Status::Ok();
    }

    static EntryRecord ReadEntry(const SqliteStatement& stmt) {
        EntryRecord e;
        int i = 0;
        e.entry_id            = stmt.ColumnInt64(i++);
        e.partition_id        = stmt.ColumnInt64(i++);
        e.cache_key           = stmt.ColumnText(i++);
        e.text_hash           = stmt.ColumnText(i++);
        e.content_hash        = stmt.ColumnText(i++);
        e.memory_hash         = stmt.ColumnText(i++);
        e.vector              = BytesToVector(stmt.ColumnBlob(i++));
        e.recall_count        = stmt.ColumnInt64(i++);
        e.last_recalled_at_ms = ColumnOptInt64(stmt, i++);
        e.forgotten           = stmt.ColumnInt(i++) != 0;
        e.forgotten_at_ms     = ColumnOptInt64(stmt, i++);
        e.deleted_after_ms    = ColumnOptInt64(stmt, i++);
        e.forget_epoch        = stmt.ColumnInt64(i++);
        e.memory_type         = stmt.ColumnText(i++);
        e.emotion             = stmt.ColumnText(i++);
        e.emotion_intensity   = static_cast<float>(stmt.ColumnDouble(i++));
        e.state_arousal       = static_cast<float>(stmt.ColumnDouble(i++));
        e.answer_type         = stmt.ColumnText(i++);
        e.payload             = stmt.ColumnText(i++);
        e.extra_metadata_json = stmt.ColumnText(i++);
        e.created_at_ms       = stmt.ColumnInt64(i++);
        e.expires_at_ms       = ColumnOptInt64(stmt, i++);
        return e;
    }
};

// Column list constants for SELECT/INSERT statements.
namespace {

constexpr const char* kEntryColumns =
    "entry_id, partition_id, cache_key, text_hash, content_hash, memory_hash, "
    "vector, recall_count, last_recalled_at_ms, forgotten, forgotten_at_ms, "
    "deleted_after_ms, forget_epoch, memory_type, emotion, emotion_intensity, "
    "state_arousal, answer_type, payload, extra_metadata, created_at_ms, expires_at_ms";

constexpr const char* kInsertEntrySQL =
    "INSERT INTO vector_entries ("
    "partition_id, cache_key, text_hash, content_hash, memory_hash, vector, "
    "recall_count, last_recalled_at_ms, forgotten, forgotten_at_ms, "
    "deleted_after_ms, forget_epoch, memory_type, emotion, emotion_intensity, "
    "state_arousal, answer_type, payload, extra_metadata, created_at_ms, expires_at_ms"
    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";

}

SqliteVectorRepository::SqliteVectorRepository(std::shared_ptr<SqliteConnectionPool> pool)
    : impl_(std::make_unique<Impl>(std::move(pool))) {}

SqliteVectorRepository::~SqliteVectorRepository() = default;

core::Status SqliteVectorRepository::EnsureSchema() {
    if (!impl_ || !impl_->pool) {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "vector repository pool is not configured");
    }
    storage::sqlite::SqliteMigrationRunner runner(impl_->pool);
    std::array<storage::sqlite::ISqliteMigrationSource*, 1> sources{this};
    return runner.ApplyAll(sources);
}

std::string_view SqliteVectorRepository::MigrationNamespace() const noexcept {
    return "vector_repository";
}

std::span<const storage::sqlite::SqliteMigrationStep>
SqliteVectorRepository::MigrationSteps() const noexcept {
    return kVectorMigrations;
}

core::Result<std::int64_t> SqliteVectorRepository::EnsureCollection(const CollectionDescriptor& desc) {
    auto lease_r = impl_->AcquireWrite();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();
    auto& conn = lease.connection();

    // Try lookup first
    {
        auto stmt_r = conn.Prepare("SELECT collection_id FROM vector_collections WHERE name = ?");
        if (!stmt_r) return stmt_r.status();
        auto stmt = std::move(stmt_r).value();
        if (auto rc = stmt.BindText(1, desc.name); !rc.ok()) return rc;
        auto step_r = stmt.Step();
        if (!step_r) return step_r.status();
        if (std::move(step_r).value() == SqliteStepResult::Row) {
            return stmt.ColumnInt64(0);
        }
    }

    // Insert new
    auto stmt_r = conn.Prepare(
        "INSERT INTO vector_collections ("
        "name, embedding_model_fingerprint, tokenizer_fingerprint, "
        "pooling_strategy, normalization, dimension, corpus_version, "
        "policy_version, created_at_ms) VALUES (?,?,?,?,?,?,?,?,?)");
    if (!stmt_r) return stmt_r.status();
    auto stmt = std::move(stmt_r).value();

    if (auto rc = stmt.BindText(1, desc.name); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(2, desc.embedding_model_fingerprint); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(3, desc.tokenizer_fingerprint); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(4, desc.pooling_strategy); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(5, desc.normalization); !rc.ok()) return rc;
    if (auto rc = stmt.BindInt64(6, static_cast<std::int64_t>(desc.dimension)); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(7, desc.corpus_version); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(8, desc.policy_version); !rc.ok()) return rc;
    if (auto rc = stmt.BindInt64(9, NowMs()); !rc.ok()) return rc;

    auto step_r = stmt.Step();
    if (!step_r) return step_r.status();
    return conn.LastInsertRowId();
}

core::Result<CollectionDescriptor> SqliteVectorRepository::GetCollection(std::int64_t id) const {
    auto lease_r = impl_->AcquireRead();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();

    auto stmt_r = lease.connection().Prepare(
        "SELECT name, embedding_model_fingerprint, tokenizer_fingerprint, "
        "pooling_strategy, normalization, dimension, corpus_version, policy_version "
        "FROM vector_collections WHERE collection_id = ?");
    if (!stmt_r) return stmt_r.status();
    auto stmt = std::move(stmt_r).value();

    if (auto rc = stmt.BindInt64(1, id); !rc.ok()) return rc;
    auto step_r = stmt.Step();
    if (!step_r) return step_r.status();
    if (std::move(step_r).value() != SqliteStepResult::Row) {
        return core::Status::Error(core::ErrorCode::NotFound, "Collection not found");
    }

    CollectionDescriptor d;
    d.name = stmt.ColumnText(0);
    d.embedding_model_fingerprint = stmt.ColumnText(1);
    d.tokenizer_fingerprint = stmt.ColumnText(2);
    d.pooling_strategy = stmt.ColumnText(3);
    d.normalization = stmt.ColumnText(4);
    d.dimension = static_cast<std::size_t>(stmt.ColumnInt64(5));
    d.corpus_version = stmt.ColumnText(6);
    d.policy_version = stmt.ColumnText(7);
    return d;
}

core::Result<std::int64_t> SqliteVectorRepository::EnsurePartition(const PartitionKey& key) {
    auto lease_r = impl_->AcquireWrite();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();
    auto& conn = lease.connection();

    // Lookup first
    {
        auto stmt_r = conn.Prepare(
            "SELECT partition_id FROM vector_partitions "
            "WHERE collection_id = ? AND tenant_id = ? AND user_id = ? AND memory_level = ?");
        if (!stmt_r) return stmt_r.status();
        auto stmt = std::move(stmt_r).value();
        if (auto rc = stmt.BindInt64(1, key.collection_id); !rc.ok()) return rc;
        if (auto rc = stmt.BindText(2, key.tenant_id); !rc.ok()) return rc;
        if (auto rc = stmt.BindText(3, key.user_id); !rc.ok()) return rc;
        if (auto rc = stmt.BindText(4, key.memory_level); !rc.ok()) return rc;
        auto step_r = stmt.Step();
        if (!step_r) return step_r.status();
        if (std::move(step_r).value() == SqliteStepResult::Row) {
            return stmt.ColumnInt64(0);
        }
    }

    // Insert
    auto stmt_r = conn.Prepare(
        "INSERT INTO vector_partitions ("
        "collection_id, tenant_id, user_id, memory_level, scope_extras, "
        "vector_count, last_modified_at_ms, created_at_ms) "
        "VALUES (?,?,?,?,?,0,?,?)");
    if (!stmt_r) return stmt_r.status();
    auto stmt = std::move(stmt_r).value();
    const auto now = NowMs();
    if (auto rc = stmt.BindInt64(1, key.collection_id); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(2, key.tenant_id); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(3, key.user_id); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(4, key.memory_level); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(5, key.scope_extras_json); !rc.ok()) return rc;
    if (auto rc = stmt.BindInt64(6, now); !rc.ok()) return rc;
    if (auto rc = stmt.BindInt64(7, now); !rc.ok()) return rc;
    auto step_r = stmt.Step();
    if (!step_r) return step_r.status();
    return conn.LastInsertRowId();
}

core::Result<std::optional<std::int64_t>> SqliteVectorRepository::LookupPartition(const PartitionKey& key) const {
    auto lease_r = impl_->AcquireRead();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();

    auto stmt_r = lease.connection().Prepare(
        "SELECT partition_id FROM vector_partitions "
        "WHERE collection_id = ? AND tenant_id = ? AND user_id = ? AND memory_level = ?");
    if (!stmt_r) return stmt_r.status();
    auto stmt = std::move(stmt_r).value();
    if (auto rc = stmt.BindInt64(1, key.collection_id); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(2, key.tenant_id); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(3, key.user_id); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(4, key.memory_level); !rc.ok()) return rc;
    auto step_r = stmt.Step();
    if (!step_r) return step_r.status();
    if (std::move(step_r).value() != SqliteStepResult::Row) {
        return std::optional<std::int64_t>{};
    }
    return std::optional<std::int64_t>{stmt.ColumnInt64(0)};
}

core::Result<PartitionSnapshot> SqliteVectorRepository::GetPartitionSnapshot(std::int64_t partition_id) const {
    auto lease_r = impl_->AcquireRead();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();

    auto stmt_r = lease.connection().Prepare(
        "SELECT vector_count, last_modified_at_ms FROM vector_partitions WHERE partition_id = ?");
    if (!stmt_r) return stmt_r.status();
    auto stmt = std::move(stmt_r).value();
    if (auto rc = stmt.BindInt64(1, partition_id); !rc.ok()) return rc;
    auto step_r = stmt.Step();
    if (!step_r) return step_r.status();
    if (std::move(step_r).value() != SqliteStepResult::Row) {
        return core::Status::Error(core::ErrorCode::NotFound, "Partition not found");
    }
    PartitionSnapshot s;
    s.partition_id = partition_id;
    s.vector_count = stmt.ColumnInt64(0);
    s.last_modified_at_ms = stmt.ColumnInt64(1);
    return s;
}

core::Result<std::int64_t> SqliteVectorRepository::InsertEntry(EntryRecord entry) {
    if (entry.created_at_ms == 0) {
        entry.created_at_ms = NowMs();
    }
    auto lease_r = impl_->AcquireWrite();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();
    auto& conn = lease.connection();

    SqliteTransaction tx(conn);
    if (auto rc = tx.Begin(); !rc.ok()) return rc;

    // Insert entry
    {
        auto stmt_r = conn.Prepare(kInsertEntrySQL);
        if (!stmt_r) return stmt_r.status();
        auto stmt = std::move(stmt_r).value();
        if (auto rc = Impl::BindEntryFields(stmt, entry); !rc.ok()) return rc;
        auto step_r = stmt.Step();
        if (!step_r) return step_r.status();
    }
    std::int64_t new_id = conn.LastInsertRowId();

    // Update partition counter
    {
        auto stmt_r = conn.Prepare(
            "UPDATE vector_partitions SET vector_count = vector_count + 1, "
            "last_modified_at_ms = ? WHERE partition_id = ?");
        if (!stmt_r) return stmt_r.status();
        auto stmt = std::move(stmt_r).value();
        if (auto rc = stmt.BindInt64(1, entry.created_at_ms); !rc.ok()) return rc;
        if (auto rc = stmt.BindInt64(2, entry.partition_id); !rc.ok()) return rc;
        auto step_r = stmt.Step();
        if (!step_r) return step_r.status();
    }

    if (auto rc = tx.Commit(); !rc.ok()) return rc;
    return new_id;
}

core::Status SqliteVectorRepository::InsertEntries(std::span<EntryRecord> entries) {
    if (entries.empty()) return core::Status::Ok();

    auto lease_r = impl_->AcquireWrite();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();
    auto& conn = lease.connection();

    SqliteTransaction tx(conn);
    if (auto rc = tx.Begin(); !rc.ok()) return rc;

    const auto now = NowMs();

    auto insert_stmt_r = conn.Prepare(kInsertEntrySQL);
    if (!insert_stmt_r) return insert_stmt_r.status();
    auto insert_stmt = std::move(insert_stmt_r).value();

    auto count_stmt_r = conn.Prepare(
        "UPDATE vector_partitions SET vector_count = vector_count + 1, "
        "last_modified_at_ms = ? WHERE partition_id = ?");
    if (!count_stmt_r) return count_stmt_r.status();
    auto count_stmt = std::move(count_stmt_r).value();

    for (auto& entry : entries) {
        if (entry.created_at_ms == 0) entry.created_at_ms = now;

        if (auto rc = insert_stmt.Reset(); !rc.ok()) return rc;
        if (auto rc = insert_stmt.ClearBindings(); !rc.ok()) return rc;
        if (auto rc = Impl::BindEntryFields(insert_stmt, entry); !rc.ok()) return rc;
        auto step_r = insert_stmt.Step();
        if (!step_r) return step_r.status();
        entry.entry_id = conn.LastInsertRowId();

        if (auto rc = count_stmt.Reset(); !rc.ok()) return rc;
        if (auto rc = count_stmt.ClearBindings(); !rc.ok()) return rc;
        if (auto rc = count_stmt.BindInt64(1, entry.created_at_ms); !rc.ok()) return rc;
        if (auto rc = count_stmt.BindInt64(2, entry.partition_id); !rc.ok()) return rc;
        auto step2_r = count_stmt.Step();
        if (!step2_r) return step2_r.status();
    }

    return tx.Commit();
}

core::Result<EntryRecord> SqliteVectorRepository::GetEntry(std::int64_t entry_id) const {
    auto lease_r = impl_->AcquireRead();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();

    std::string sql = "SELECT ";
    sql += kEntryColumns;
    sql += " FROM vector_entries WHERE entry_id = ?";

    auto stmt_r = lease.connection().Prepare(sql);
    if (!stmt_r) return stmt_r.status();
    auto stmt = std::move(stmt_r).value();
    if (auto rc = stmt.BindInt64(1, entry_id); !rc.ok()) return rc;
    auto step_r = stmt.Step();
    if (!step_r) return step_r.status();
    if (std::move(step_r).value() != SqliteStepResult::Row) {
        return core::Status::Error(core::ErrorCode::NotFound, "Entry not found");
    }
    return Impl::ReadEntry(stmt);
}

core::Result<std::optional<std::int64_t>> SqliteVectorRepository::FindEntryIdByMemoryHash(
    std::int64_t partition_id, std::string_view memory_hash) const {
    auto lease_r = impl_->AcquireRead();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();

    auto stmt_r = lease.connection().Prepare(
        "SELECT entry_id FROM vector_entries WHERE partition_id = ? AND memory_hash = ?");
    if (!stmt_r) return stmt_r.status();
    auto stmt = std::move(stmt_r).value();
    if (auto rc = stmt.BindInt64(1, partition_id); !rc.ok()) return rc;
    if (auto rc = stmt.BindText(2, std::string(memory_hash)); !rc.ok()) return rc;
    auto step_r = stmt.Step();
    if (!step_r) return step_r.status();
    if (std::move(step_r).value() != SqliteStepResult::Row) {
        return std::optional<std::int64_t>{};
    }
    return std::optional<std::int64_t>{stmt.ColumnInt64(0)};
}

core::Result<std::vector<EntryRecord>> SqliteVectorRepository::LookupEntries(
    std::span<const std::int64_t> ids) const {
    std::vector<EntryRecord> out;
    if (ids.empty()) return out;

    auto lease_r = impl_->AcquireRead();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();

    std::string sql = "SELECT ";
    sql += kEntryColumns;
    sql += " FROM vector_entries WHERE entry_id = ?";

    auto stmt_r = lease.connection().Prepare(sql);
    if (!stmt_r) return stmt_r.status();
    auto stmt = std::move(stmt_r).value();

    out.reserve(ids.size());
    for (auto id : ids) {
        if (auto rc = stmt.Reset(); !rc.ok()) return rc;
        if (auto rc = stmt.ClearBindings(); !rc.ok()) return rc;
        if (auto rc = stmt.BindInt64(1, id); !rc.ok()) return rc;
        auto step_r = stmt.Step();
        if (!step_r) return step_r.status();
        if (std::move(step_r).value() == SqliteStepResult::Row) {
            out.push_back(Impl::ReadEntry(stmt));
        }
    }
    return out;
}

core::Result<std::vector<EntryRecord>> SqliteVectorRepository::ListEntries(
    std::int64_t partition_id, bool include_forgotten) const {
    std::vector<EntryRecord> out;

    auto lease_r = impl_->AcquireRead();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();

    std::string sql = "SELECT ";
    sql += kEntryColumns;
    sql += " FROM vector_entries WHERE partition_id = ?";
    if (!include_forgotten) sql += " AND forgotten = 0";
    sql += " ORDER BY entry_id";

    auto stmt_r = lease.connection().Prepare(sql);
    if (!stmt_r) return stmt_r.status();
    auto stmt = std::move(stmt_r).value();
    if (auto rc = stmt.BindInt64(1, partition_id); !rc.ok()) return rc;

    while (true) {
        auto step_r = stmt.Step();
        if (!step_r) return step_r.status();
        if (std::move(step_r).value() == SqliteStepResult::Done) break;
        out.push_back(Impl::ReadEntry(stmt));
    }
    return out;
}

core::Status SqliteVectorRepository::MarkRecalled(std::span<const std::int64_t> ids, std::int64_t now_ms) {
    if (ids.empty()) return core::Status::Ok();
    auto lease_r = impl_->AcquireWrite();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();
    auto& conn = lease.connection();

    SqliteTransaction tx(conn);
    if (auto rc = tx.Begin(); !rc.ok()) return rc;

    auto stmt_r = conn.Prepare(
        "UPDATE vector_entries SET recall_count = recall_count + 1, "
        "last_recalled_at_ms = ? WHERE entry_id = ?");
    if (!stmt_r) return stmt_r.status();
    auto stmt = std::move(stmt_r).value();

    for (auto id : ids) {
        if (auto rc = stmt.Reset(); !rc.ok()) return rc;
        if (auto rc = stmt.ClearBindings(); !rc.ok()) return rc;
        if (auto rc = stmt.BindInt64(1, now_ms); !rc.ok()) return rc;
        if (auto rc = stmt.BindInt64(2, id); !rc.ok()) return rc;
        auto step_r = stmt.Step();
        if (!step_r) return step_r.status();
    }
    return tx.Commit();
}

core::Status SqliteVectorRepository::MarkForgotten(std::span<const std::int64_t> ids,
                                                    std::int64_t now_ms,
                                                    std::int64_t deleted_after_ms) {
    if (ids.empty()) return core::Status::Ok();
    auto lease_r = impl_->AcquireWrite();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();
    auto& conn = lease.connection();

    SqliteTransaction tx(conn);
    if (auto rc = tx.Begin(); !rc.ok()) return rc;

    auto stmt_r = conn.Prepare(
        "UPDATE vector_entries SET forgotten = 1, forgotten_at_ms = ?, "
        "deleted_after_ms = ?, forget_epoch = forget_epoch + 1 "
        "WHERE entry_id = ? AND forgotten = 0");
    if (!stmt_r) return stmt_r.status();
    auto stmt = std::move(stmt_r).value();

    for (auto id : ids) {
        if (auto rc = stmt.Reset(); !rc.ok()) return rc;
        if (auto rc = stmt.ClearBindings(); !rc.ok()) return rc;
        if (auto rc = stmt.BindInt64(1, now_ms); !rc.ok()) return rc;
        if (auto rc = stmt.BindInt64(2, deleted_after_ms); !rc.ok()) return rc;
        if (auto rc = stmt.BindInt64(3, id); !rc.ok()) return rc;
        auto step_r = stmt.Step();
        if (!step_r) return step_r.status();
    }
    return tx.Commit();
}

core::Status SqliteVectorRepository::Reactivate(std::span<const std::int64_t> ids, std::int64_t now_ms) {
    if (ids.empty()) return core::Status::Ok();
    auto lease_r = impl_->AcquireWrite();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();
    auto& conn = lease.connection();

    SqliteTransaction tx(conn);
    if (auto rc = tx.Begin(); !rc.ok()) return rc;

    auto stmt_r = conn.Prepare(
        "UPDATE vector_entries SET forgotten = 0, forgotten_at_ms = NULL, "
        "deleted_after_ms = NULL, recall_count = recall_count + 1, "
        "last_recalled_at_ms = ? WHERE entry_id = ?");
    if (!stmt_r) return stmt_r.status();
    auto stmt = std::move(stmt_r).value();

    for (auto id : ids) {
        if (auto rc = stmt.Reset(); !rc.ok()) return rc;
        if (auto rc = stmt.ClearBindings(); !rc.ok()) return rc;
        if (auto rc = stmt.BindInt64(1, now_ms); !rc.ok()) return rc;
        if (auto rc = stmt.BindInt64(2, id); !rc.ok()) return rc;
        auto step_r = stmt.Step();
        if (!step_r) return step_r.status();
    }
    return tx.Commit();
}

core::Status SqliteVectorRepository::DeleteEntry(std::int64_t id) {
    auto lease_r = impl_->AcquireWrite();
    if (!lease_r) return lease_r.status();
    auto lease = std::move(lease_r).value();
    auto& conn = lease.connection();

    SqliteTransaction tx(conn);
    if (auto rc = tx.Begin(); !rc.ok()) return rc;

    // Find partition_id first so we can decrement counter
    std::int64_t partition_id = 0;
    {
        auto stmt_r = conn.Prepare("SELECT partition_id FROM vector_entries WHERE entry_id = ?");
        if (!stmt_r) return stmt_r.status();
        auto stmt = std::move(stmt_r).value();
        if (auto rc = stmt.BindInt64(1, id); !rc.ok()) return rc;
        auto step_r = stmt.Step();
        if (!step_r) return step_r.status();
        if (std::move(step_r).value() != SqliteStepResult::Row) {
            return core::Status::Ok();  // already gone
        }
        partition_id = stmt.ColumnInt64(0);
    }

    {
        auto stmt_r = conn.Prepare("DELETE FROM vector_entries WHERE entry_id = ?");
        if (!stmt_r) return stmt_r.status();
        auto stmt = std::move(stmt_r).value();
        if (auto rc = stmt.BindInt64(1, id); !rc.ok()) return rc;
        auto step_r = stmt.Step();
        if (!step_r) return step_r.status();
    }
    {
        auto stmt_r = conn.Prepare(
            "UPDATE vector_partitions SET vector_count = MAX(0, vector_count - 1), "
            "last_modified_at_ms = ? WHERE partition_id = ?");
        if (!stmt_r) return stmt_r.status();
        auto stmt = std::move(stmt_r).value();
        if (auto rc = stmt.BindInt64(1, NowMs()); !rc.ok()) return rc;
        if (auto rc = stmt.BindInt64(2, partition_id); !rc.ok()) return rc;
        auto step_r = stmt.Step();
        if (!step_r) return step_r.status();
    }
    return tx.Commit();
}

}
