#pragma once

#include "logger_adapter.h"
#include "sqlite_connection_pool.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace storage::sqlite {

using SqliteMigrationApply = std::function<core::Status(SqliteConnection& connection)>;

struct SqliteMigrationStep {
    std::uint32_t version = 0;
    std::string_view name;
    std::string_view checksum;
    SqliteMigrationApply apply = nullptr;
};

class ISqliteMigrationSource {
public:
    virtual ~ISqliteMigrationSource() = default;

    virtual std::string_view MigrationNamespace() const noexcept = 0;
    virtual std::span<const SqliteMigrationStep> MigrationSteps() const noexcept = 0;
};

struct SqliteMigrationRunnerOptions {
    std::chrono::milliseconds acquire_timeout{5000};
    bool reject_newer_schema = true;
    bool verify_checksums = true;
};

class SqliteMigrationRunner final {
public:
    SqliteMigrationRunner(
        std::shared_ptr<SqliteConnectionPool> pool,
        SqliteMigrationRunnerOptions options = {},
        core::LoggerAdapter logger = core::LoggerAdapter::ForModule("sqlite-migration"));

    // 在业务线程池启动前串行应用所有模块迁移，确保 schema 与账本原子推进。
    // 在业务线程池启动前串行应用所有模块迁移，确保 schema 与账本原子推进。
    // 在业务线程池启动前串行应用所有模块迁移，确保 schema 与账本原子推进。
    // 在业务线程池启动前串行应用所有模块迁移，确保 schema 与账本原子推进。
    core::Status ApplyAll(std::span<ISqliteMigrationSource* const> sources);

private:
    struct AppliedMigration {
        std::uint32_t version = 0;
        std::string name;
        std::string checksum;
    };

    core::Status EnsureLedger(SqliteConnection& connection);
    core::Status ValidateSources(std::span<ISqliteMigrationSource* const> sources) const;
    core::Result<std::vector<AppliedMigration>> LoadApplied(
        SqliteConnection& connection,
        std::string_view component) const;
    core::Status ApplySource(ISqliteMigrationSource& source);
    core::Status ApplyStep(
        ISqliteMigrationSource& source,
        const SqliteMigrationStep& step);

    std::shared_ptr<SqliteConnectionPool> pool_;
    SqliteMigrationRunnerOptions options_;
    core::LoggerAdapter logger_;
};

}
