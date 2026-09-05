#include "sqlite_migration.h"

#include "sqlite_statement.h"
#include "sqlite_transaction.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <string>
#include <unordered_set>

namespace storage::sqlite {

namespace {

constexpr const char* kLedgerTableSql = R"SQL(
CREATE TABLE IF NOT EXISTS agentloom_schema_migrations (
    component      TEXT NOT NULL,
    version        INTEGER NOT NULL,
    migration_name TEXT NOT NULL,
    checksum       TEXT NOT NULL,
    applied_at_ms  INTEGER NOT NULL,
    PRIMARY KEY(component, version)
);
CREATE INDEX IF NOT EXISTS idx_agentloom_schema_migrations_applied
    ON agentloom_schema_migrations(applied_at_ms);
)SQL";

std::int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

core::Status InvalidSource(std::string message) {
    return core::Status::Error(core::ErrorCode::InvalidArgument, std::move(message));
}

}

SqliteMigrationRunner::SqliteMigrationRunner(
    std::shared_ptr<SqliteConnectionPool> pool,
    SqliteMigrationRunnerOptions options,
    core::LoggerAdapter logger)
    : pool_(std::move(pool)),
      options_(std::move(options)),
      logger_(logger.valid() ? std::move(logger)
                             : core::LoggerAdapter::ForModule("sqlite-migration")) {}

core::Status SqliteMigrationRunner::ApplyAll(
    std::span<ISqliteMigrationSource* const> sources) {
    if (!pool_) {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "SQLite migration runner pool is not configured");
    }
    // 兼容旧 Repository 的 path 构造方式；显式启动过的 pool 重复 Start 是幂等的。
    if (auto status = pool_->Start(); !status.ok()) {
        return status;
    }
    if (options_.acquire_timeout <= std::chrono::milliseconds::zero()) {
        return InvalidSource("SQLite migration acquire timeout must be positive");
    }
    if (auto status = ValidateSources(sources); !status.ok()) {
        return status;
    }

    auto lease_result = pool_->WaitAcquireWriteFor(options_.acquire_timeout);
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    auto lease = std::move(lease_result).value();
    if (auto status = EnsureLedger(lease.connection()); !status.ok()) {
        logger_.error("[sqlite-migration] ensure ledger failed code={} message={}",
                      static_cast<int>(status.code()), status.message());
        return status;
    }
    lease.Release();

    for (auto* source : sources) {
        if (auto status = ApplySource(*source); !status.ok()) {
            return status;
        }
    }
    return core::Status::Ok();
}

core::Status SqliteMigrationRunner::EnsureLedger(SqliteConnection& connection) {
    return connection.Execute(kLedgerTableSql);
}

core::Status SqliteMigrationRunner::ValidateSources(
    std::span<ISqliteMigrationSource* const> sources) const {
    if (sources.empty()) {
        return InvalidSource("SQLite migration sources are empty");
    }
    std::unordered_set<std::string> namespaces;
    for (const auto* source : sources) {
        if (!source) {
            return InvalidSource("SQLite migration source is null");
        }
        const auto name = source->MigrationNamespace();
        if (name.empty() || !namespaces.emplace(name).second) {
            return InvalidSource("SQLite migration namespace is empty or duplicated");
        }
        const auto steps = source->MigrationSteps();
        std::uint32_t expected_version = 1;
        for (const auto& step : steps) {
            if (step.version != expected_version || step.name.empty() ||
                step.checksum.empty() || !step.apply) {
                return InvalidSource(
                    "SQLite migration steps must use contiguous versions and valid descriptors");
            }
            ++expected_version;
        }
        if (steps.empty()) {
            return InvalidSource("SQLite migration source has no steps");
        }
    }
    return core::Status::Ok();
}

core::Result<std::vector<SqliteMigrationRunner::AppliedMigration>>
SqliteMigrationRunner::LoadApplied(SqliteConnection& connection,
                                   std::string_view component) const {
    auto statement_result = connection.Prepare(
        "SELECT version, migration_name, checksum "
        "FROM agentloom_schema_migrations WHERE component = ?1 ORDER BY version ASC");
    if (!statement_result.ok()) {
        return statement_result.status();
    }
    auto statement = std::move(statement_result).value();
    if (auto status = statement.BindText(1, std::string(component)); !status.ok()) {
        return status;
    }

    std::vector<AppliedMigration> applied;
    while (true) {
        auto step_result = statement.Step();
        if (!step_result.ok()) {
            return step_result.status();
        }
        if (step_result.value() == SqliteStepResult::Done) {
            break;
        }
        applied.push_back({
            .version = static_cast<std::uint32_t>(statement.ColumnInt(0)),
            .name = statement.ColumnText(1),
            .checksum = statement.ColumnText(2),
        });
    }
    return applied;
}

core::Status SqliteMigrationRunner::ApplySource(ISqliteMigrationSource& source) {
    auto lease_result = pool_->WaitAcquireWriteFor(options_.acquire_timeout);
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    auto lease = std::move(lease_result).value();
    auto applied_result = LoadApplied(lease.connection(), source.MigrationNamespace());
    if (!applied_result.ok()) {
        return applied_result.status();
    }
    const auto applied = std::move(applied_result).value();
    const auto steps = source.MigrationSteps();
    if (options_.reject_newer_schema &&
        !applied.empty() && applied.back().version > steps.back().version) {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "SQLite database schema is newer than migration source");
    }

    std::uint32_t expected_applied_version = 1;
    for (const auto& applied_step : applied) {
        if (applied_step.version != expected_applied_version) {
            return core::Status::Error(
                core::ErrorCode::DataLoss,
                "SQLite migration history contains a version gap or duplicate");
        }
        ++expected_applied_version;
        if (applied_step.version == 0 || applied_step.version > steps.back().version) {
            continue;
        }
        const auto& declared = steps[applied_step.version - 1];
        if (applied_step.name != declared.name ||
            (options_.verify_checksums && applied_step.checksum != declared.checksum)) {
            logger_.error(
                "[sqlite-migration] history mismatch component={} version={}",
                source.MigrationNamespace(), applied_step.version);
            return core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "SQLite migration history checksum or name mismatch");
        }
    }
    lease.Release();

    for (const auto& step : steps) {
        if (auto status = ApplyStep(source, step); !status.ok()) {
            return status;
        }
    }
    return core::Status::Ok();
}

core::Status SqliteMigrationRunner::ApplyStep(
    ISqliteMigrationSource& source,
    const SqliteMigrationStep& step) {
    auto lease_result = pool_->WaitAcquireWriteFor(options_.acquire_timeout);
    if (!lease_result.ok()) {
        return lease_result.status();
    }
    auto lease = std::move(lease_result).value();
    auto& connection = lease.connection();
    // 迁移正文和 ledger 插入必须在同一个事务中完成，失败由 RAII transaction 回滚。
    SqliteTransaction transaction(connection);
    if (auto status = transaction.Begin(); !status.ok()) {
        return status;
    }

    auto existing_result = connection.Prepare(
        "SELECT migration_name, checksum FROM agentloom_schema_migrations "
        "WHERE component = ?1 AND version = ?2");
    if (!existing_result.ok()) {
        return existing_result.status();
    }
    auto existing = std::move(existing_result).value();
    if (auto status = existing.BindText(1, std::string(source.MigrationNamespace())); !status.ok()) {
        return status;
    }
    if (auto status = existing.BindInt64(2, step.version); !status.ok()) {
        return status;
    }
    auto existing_step = existing.Step();
    if (!existing_step.ok()) {
        return existing_step.status();
    }
    if (existing_step.value() == SqliteStepResult::Row) {
        const auto name = existing.ColumnText(0);
        const auto checksum = existing.ColumnText(1);
        if (name != step.name || (options_.verify_checksums && checksum != step.checksum)) {
            return core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "SQLite migration history checksum or name mismatch");
        }
        return transaction.Commit();
    }

    try {
        if (auto status = step.apply(connection); !status.ok()) {
            logger_.error(
                "[sqlite-migration] step failed component={} version={} name={} code={} message={}",
                source.MigrationNamespace(), step.version, step.name,
                static_cast<int>(status.code()), status.message());
            return status;
        }
    } catch (const std::exception& exception) {
        return core::Status::Error(
            core::ErrorCode::InternalError,
            std::string("SQLite migration threw: ") + exception.what());
    } catch (...) {
        return core::Status::Error(
            core::ErrorCode::InternalError,
            "SQLite migration threw an unknown exception");
    }

    auto insert_result = connection.Prepare(
        "INSERT INTO agentloom_schema_migrations "
        "(component, version, migration_name, checksum, applied_at_ms) "
        "VALUES (?1, ?2, ?3, ?4, ?5)");
    if (!insert_result.ok()) {
        return insert_result.status();
    }
    auto insert = std::move(insert_result).value();
    if (auto status = insert.BindText(1, std::string(source.MigrationNamespace())); !status.ok()) return status;
    if (auto status = insert.BindInt64(2, step.version); !status.ok()) return status;
    if (auto status = insert.BindText(3, std::string(step.name)); !status.ok()) return status;
    if (auto status = insert.BindText(4, std::string(step.checksum)); !status.ok()) return status;
    if (auto status = insert.BindInt64(5, NowMs()); !status.ok()) return status;
    auto insert_step = insert.Step();
    if (!insert_step.ok()) {
        return insert_step.status();
    }
    return transaction.Commit();
}

}
