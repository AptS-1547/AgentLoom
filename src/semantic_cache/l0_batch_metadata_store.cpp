#include "l0_batch_metadata_store.h"

#include "sqlite/sqlite_statement.h"
#include "sqlite/sqlite_transaction.h"

#include <array>

namespace agent::semantic_cache {

namespace {

constexpr std::chrono::milliseconds kAcquireTimeout{5000};

core::Status ValidateKey(const L0SessionKey& key) {
    if (key.tenant_id.empty() || key.user_id.empty() || key.session_id.empty()) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "L0 session metadata key requires tenant, user and session");
    }
    return core::Status::Ok();
}

core::Status ApplyL0SessionMetadataV1(storage::sqlite::SqliteConnection& connection) {
    return connection.Execute(R"SQL(
CREATE TABLE IF NOT EXISTS l0_active_batch_v2 (
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (tenant_id, user_id, session_id)
);
CREATE TABLE IF NOT EXISTS l0_timestamp_index_v2 (
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    PRIMARY KEY (tenant_id, user_id, session_id, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_l0_timestamp_index_v2_session
    ON l0_timestamp_index_v2(tenant_id, user_id, session_id, timestamp);
)SQL");
}

const std::array<storage::sqlite::SqliteMigrationStep, 1> kL0SessionMetadataMigrations{{
    {
        .version = 1,
        .name = "create_l0_session_metadata_v2",
        .checksum = "l0_session_metadata_v2_tables",
        .apply = ApplyL0SessionMetadataV1,
    },
}};

}

SqliteL0SessionBatchMetadataStore::SqliteL0SessionBatchMetadataStore(
    std::shared_ptr<storage::sqlite::SqliteConnectionPool> pool)
    : pool_(std::move(pool)) {}

core::Status SqliteL0SessionBatchMetadataStore::EnsureSchema() {
    if (!pool_) {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "L0 SQLite metadata pool is not configured");
    }
    auto read_lease_result = pool_->WaitAcquireReadFor(kAcquireTimeout);
    if (!read_lease_result.ok()) return read_lease_result.status();
    auto read_lease = std::move(read_lease_result).value();
    auto active_probe = read_lease.connection().Prepare(
        "SELECT tenant_id,user_id,session_id,timestamp,count FROM l0_active_batch_v2 LIMIT 0");
    auto timestamp_probe = read_lease.connection().Prepare(
        "SELECT tenant_id,user_id,session_id,timestamp FROM l0_timestamp_index_v2 LIMIT 0");
    if (active_probe.ok() && timestamp_probe.ok()) {
        return core::Status::Ok();
    }

    storage::sqlite::SqliteMigrationRunner runner(pool_);
    std::array<storage::sqlite::ISqliteMigrationSource*, 1> sources{this};
    return runner.ApplyAll(sources);
}

core::Result<std::optional<L0ActiveBatchState>>
SqliteL0SessionBatchMetadataStore::LoadActiveBatch(const L0SessionKey& key) {
    if (auto status = ValidateKey(key); !status.ok()) return status;
    auto lease_result = pool_->WaitAcquireReadFor(kAcquireTimeout);
    if (!lease_result.ok()) return lease_result.status();
    auto lease = std::move(lease_result).value();
    auto statement_result = lease.connection().Prepare(
        "SELECT timestamp, count FROM l0_active_batch_v2 "
        "WHERE tenant_id=?1 AND user_id=?2 AND session_id=?3");
    if (!statement_result.ok()) return statement_result.status();
    auto statement = std::move(statement_result).value();
    if (auto status = statement.BindText(1, key.tenant_id); !status.ok()) return status;
    if (auto status = statement.BindText(2, key.user_id); !status.ok()) return status;
    if (auto status = statement.BindText(3, key.session_id); !status.ok()) return status;
    auto step = statement.Step();
    if (!step.ok()) return step.status();
    if (step.value() == storage::sqlite::SqliteStepResult::Done) {
        return std::optional<L0ActiveBatchState>{};
    }
    return std::optional<L0ActiveBatchState>{L0ActiveBatchState{
        .timestamp = statement.ColumnInt64(0),
        .count = static_cast<std::size_t>(statement.ColumnInt64(1)),
    }};
}

core::Status SqliteL0SessionBatchMetadataStore::ClearActiveBatch(
    const L0SessionKey& key) {
    if (auto status = ValidateKey(key); !status.ok()) return status;
    auto lease_result = pool_->WaitAcquireWriteFor(kAcquireTimeout);
    if (!lease_result.ok()) return lease_result.status();
    auto lease = std::move(lease_result).value();
    auto statement_result = lease.connection().Prepare(
        "DELETE FROM l0_active_batch_v2 "
        "WHERE tenant_id=?1 AND user_id=?2 AND session_id=?3");
    if (!statement_result.ok()) return statement_result.status();
    auto statement = std::move(statement_result).value();
    if (auto status = statement.BindText(1, key.tenant_id); !status.ok()) return status;
    if (auto status = statement.BindText(2, key.user_id); !status.ok()) return status;
    if (auto status = statement.BindText(3, key.session_id); !status.ok()) return status;
    auto step = statement.Step();
    return step.ok() ? core::Status::Ok() : step.status();
}

core::Status SqliteL0SessionBatchMetadataStore::SaveActiveBatch(
    const L0SessionKey& key,
    const L0ActiveBatchState& state) {
    if (auto status = ValidateKey(key); !status.ok()) return status;
    auto lease_result = pool_->WaitAcquireWriteFor(kAcquireTimeout);
    if (!lease_result.ok()) return lease_result.status();
    auto lease = std::move(lease_result).value();
    auto statement_result = lease.connection().Prepare(
        "INSERT INTO l0_active_batch_v2 "
        "(tenant_id,user_id,session_id,timestamp,count) VALUES (?1,?2,?3,?4,?5) "
        "ON CONFLICT(tenant_id,user_id,session_id) DO UPDATE SET "
        "timestamp=excluded.timestamp,count=excluded.count");
    if (!statement_result.ok()) return statement_result.status();
    auto statement = std::move(statement_result).value();
    if (auto status = statement.BindText(1, key.tenant_id); !status.ok()) return status;
    if (auto status = statement.BindText(2, key.user_id); !status.ok()) return status;
    if (auto status = statement.BindText(3, key.session_id); !status.ok()) return status;
    if (auto status = statement.BindInt64(4, state.timestamp); !status.ok()) return status;
    if (auto status = statement.BindInt64(5, static_cast<std::int64_t>(state.count)); !status.ok()) return status;
    auto step = statement.Step();
    return step.ok() ? core::Status::Ok() : step.status();
}

core::Result<std::vector<std::int64_t>>
SqliteL0SessionBatchMetadataStore::LoadTimestampIndex(const L0SessionKey& key) {
    if (auto status = ValidateKey(key); !status.ok()) return status;
    auto lease_result = pool_->WaitAcquireReadFor(kAcquireTimeout);
    if (!lease_result.ok()) return lease_result.status();
    auto lease = std::move(lease_result).value();
    auto statement_result = lease.connection().Prepare(
        "SELECT timestamp FROM l0_timestamp_index_v2 "
        "WHERE tenant_id=?1 AND user_id=?2 AND session_id=?3 ORDER BY timestamp ASC");
    if (!statement_result.ok()) return statement_result.status();
    auto statement = std::move(statement_result).value();
    if (auto status = statement.BindText(1, key.tenant_id); !status.ok()) return status;
    if (auto status = statement.BindText(2, key.user_id); !status.ok()) return status;
    if (auto status = statement.BindText(3, key.session_id); !status.ok()) return status;
    std::vector<std::int64_t> timestamps;
    while (true) {
        auto step = statement.Step();
        if (!step.ok()) return step.status();
        if (step.value() == storage::sqlite::SqliteStepResult::Done) break;
        timestamps.push_back(statement.ColumnInt64(0));
    }
    return timestamps;
}

core::Status SqliteL0SessionBatchMetadataStore::ReplaceTimestampIndex(
    const L0SessionKey& key,
    std::span<const std::int64_t> timestamps) {
    if (auto status = ValidateKey(key); !status.ok()) return status;
    auto lease_result = pool_->WaitAcquireWriteFor(kAcquireTimeout);
    if (!lease_result.ok()) return lease_result.status();
    auto lease = std::move(lease_result).value();
    auto& connection = lease.connection();
    storage::sqlite::SqliteTransaction transaction(connection);
    if (auto status = transaction.Begin(); !status.ok()) return status;

    auto delete_result = connection.Prepare(
        "DELETE FROM l0_timestamp_index_v2 "
        "WHERE tenant_id=?1 AND user_id=?2 AND session_id=?3");
    if (!delete_result.ok()) return delete_result.status();
    auto del = std::move(delete_result).value();
    if (auto status = del.BindText(1, key.tenant_id); !status.ok()) return status;
    if (auto status = del.BindText(2, key.user_id); !status.ok()) return status;
    if (auto status = del.BindText(3, key.session_id); !status.ok()) return status;
    if (auto status = del.Step(); !status.ok()) return status.status();

    auto insert_result = connection.Prepare(
        "INSERT OR IGNORE INTO l0_timestamp_index_v2 "
        "(tenant_id,user_id,session_id,timestamp) VALUES (?1,?2,?3,?4)");
    if (!insert_result.ok()) return insert_result.status();
    auto insert = std::move(insert_result).value();
    for (const auto timestamp : timestamps) {
        if (auto status = insert.BindText(1, key.tenant_id); !status.ok()) return status;
        if (auto status = insert.BindText(2, key.user_id); !status.ok()) return status;
        if (auto status = insert.BindText(3, key.session_id); !status.ok()) return status;
        if (auto status = insert.BindInt64(4, timestamp); !status.ok()) return status;
        if (auto status = insert.Step(); !status.ok()) return status.status();
        if (auto status = insert.Reset(); !status.ok()) return status;
        if (auto status = insert.ClearBindings(); !status.ok()) return status;
    }
    return transaction.Commit();
}

std::string_view SqliteL0SessionBatchMetadataStore::MigrationNamespace() const noexcept {
    return "l0_session_metadata";
}

std::span<const storage::sqlite::SqliteMigrationStep>
SqliteL0SessionBatchMetadataStore::MigrationSteps() const noexcept {
    return kL0SessionMetadataMigrations;
}

}
