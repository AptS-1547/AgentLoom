#include "sqlite/sqlite_connection_pool.h"
#include "sqlite/sqlite_migration.h"
#include "sqlite/sqlite_statement.h"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <memory>
#include <string>

namespace {

using storage::sqlite::ISqliteMigrationSource;
using storage::sqlite::SqliteConnectionPool;
using storage::sqlite::SqliteConnectionPoolOptions;
using storage::sqlite::SqliteMigrationRunner;
using storage::sqlite::SqliteMigrationStep;
using storage::sqlite::SqliteStepResult;

std::filesystem::path TestPath(std::string_view name) {
    auto path = std::filesystem::temp_directory_path() / std::string(name);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + "-wal", ec);
    std::filesystem::remove(path.string() + "-shm", ec);
    return path;
}

class TestSource final : public ISqliteMigrationSource {
public:
    TestSource(std::string name, std::span<const SqliteMigrationStep> steps)
        : name_(std::move(name)), steps_(steps) {}

    std::string_view MigrationNamespace() const noexcept override { return name_; }
    std::span<const SqliteMigrationStep> MigrationSteps() const noexcept override { return steps_; }

private:
    std::string name_;
    std::span<const SqliteMigrationStep> steps_;
};

bool TableExists(SqliteConnectionPool& pool, std::string_view table) {
    auto lease_result = pool.WaitAcquireRead();
    if (!lease_result.ok()) {
        return false;
    }
    auto lease = std::move(lease_result).value();
    auto statement_result = lease.connection().Prepare(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1");
    if (!statement_result.ok()) {
        return false;
    }
    auto statement = std::move(statement_result).value();
    if (!statement.BindText(1, std::string(table)).ok()) {
        return false;
    }
    auto step = statement.Step();
    return step.ok() && step.value() == SqliteStepResult::Row;
}

int CountLedgerRows(SqliteConnectionPool& pool, std::string_view component) {
    auto lease_result = pool.WaitAcquireRead();
    EXPECT_TRUE(lease_result.ok()) << lease_result.status().message();
    if (!lease_result.ok()) {
        return -1;
    }
    auto lease = std::move(lease_result).value();
    auto statement_result = lease.connection().Prepare(
        "SELECT COUNT(*) FROM agentloom_schema_migrations WHERE component=?1");
    EXPECT_TRUE(statement_result.ok()) << statement_result.status().message();
    if (!statement_result.ok()) {
        return -1;
    }
    auto statement = std::move(statement_result).value();
    EXPECT_TRUE(statement.BindText(1, std::string(component)).ok());
    auto step = statement.Step();
    EXPECT_TRUE(step.ok());
    EXPECT_EQ(step.value(), SqliteStepResult::Row);
    return statement.ColumnInt(0);
}

TEST(SqliteMigrationTest, AppliesFreshSchemaAndIsIdempotent) {
    const auto path = TestPath("agent_sqlite_migration_fresh.db");
    auto pool = std::make_shared<SqliteConnectionPool>(SqliteConnectionPoolOptions{
        .path = path.string(), .read_connection_count = 1, .write_connection_count = 1,
        .busy_timeout_ms = 1000, .enable_wal = true});
    ASSERT_TRUE(pool->Start().ok());

    const std::array<SqliteMigrationStep, 2> steps{{
        {.version = 1, .name = "create_items", .checksum = "items-v1",
         .apply = [](auto& connection) { return connection.Execute("CREATE TABLE items(id INTEGER PRIMARY KEY)"); }},
        {.version = 2, .name = "add_value", .checksum = "items-v2",
         .apply = [](auto& connection) { return connection.Execute("ALTER TABLE items ADD COLUMN value TEXT NOT NULL DEFAULT ''"); }},
    }};
    TestSource source("items", steps);
    std::array<ISqliteMigrationSource*, 1> sources{&source};
    SqliteMigrationRunner runner(pool);

    ASSERT_TRUE(runner.ApplyAll(sources).ok());
    ASSERT_TRUE(runner.ApplyAll(sources).ok());
    EXPECT_TRUE(TableExists(*pool, "items"));
    EXPECT_EQ(CountLedgerRows(*pool, "items"), 2);
}

TEST(SqliteMigrationTest, KeepsComponentVersionsIndependent) {
    const auto path = TestPath("agent_sqlite_migration_components.db");
    auto pool = std::make_shared<SqliteConnectionPool>(SqliteConnectionPoolOptions{
        .path = path.string(), .read_connection_count = 1, .write_connection_count = 1,
        .busy_timeout_ms = 1000, .enable_wal = true});
    ASSERT_TRUE(pool->Start().ok());

    const std::array<SqliteMigrationStep, 1> first_steps{{
        {.version = 1, .name = "create_first", .checksum = "first-v1",
         .apply = [](auto& connection) { return connection.Execute("CREATE TABLE first_component(id INTEGER)"); }},
    }};
    const std::array<SqliteMigrationStep, 1> second_steps{{
        {.version = 1, .name = "create_second", .checksum = "second-v1",
         .apply = [](auto& connection) { return connection.Execute("CREATE TABLE second_component(id INTEGER)"); }},
    }};
    TestSource first("first", first_steps);
    TestSource second("second", second_steps);
    std::array<ISqliteMigrationSource*, 2> sources{&first, &second};
    SqliteMigrationRunner runner(pool);

    ASSERT_TRUE(runner.ApplyAll(sources).ok());
    EXPECT_TRUE(TableExists(*pool, "first_component"));
    EXPECT_TRUE(TableExists(*pool, "second_component"));
    EXPECT_EQ(CountLedgerRows(*pool, "first"), 1);
    EXPECT_EQ(CountLedgerRows(*pool, "second"), 1);
}

TEST(SqliteMigrationTest, RollsBackFailedStepAndDoesNotWriteLedger) {
    const auto path = TestPath("agent_sqlite_migration_rollback.db");
    auto pool = std::make_shared<SqliteConnectionPool>(SqliteConnectionPoolOptions{
        .path = path.string(), .read_connection_count = 1, .write_connection_count = 1,
        .busy_timeout_ms = 1000, .enable_wal = true});
    ASSERT_TRUE(pool->Start().ok());

    const std::array<SqliteMigrationStep, 1> steps{{
        {.version = 1, .name = "failed_step", .checksum = "failed-v1",
         .apply = [](auto& connection) {
             if (auto status = connection.Execute("CREATE TABLE rollback_target(id INTEGER)"); !status.ok()) {
                 return status;
             }
             return core::Status::Error(core::ErrorCode::InternalError, "scripted migration failure");
         }},
    }};
    TestSource source("rollback", steps);
    std::array<ISqliteMigrationSource*, 1> sources{&source};
    SqliteMigrationRunner runner(pool);

    auto status = runner.ApplyAll(sources);
    ASSERT_FALSE(status.ok());
    EXPECT_FALSE(TableExists(*pool, "rollback_target"));
    EXPECT_EQ(CountLedgerRows(*pool, "rollback"), 0);
}

TEST(SqliteMigrationTest, RejectsChangedChecksum) {
    const auto path = TestPath("agent_sqlite_migration_checksum.db");
    auto pool = std::make_shared<SqliteConnectionPool>(SqliteConnectionPoolOptions{
        .path = path.string(), .read_connection_count = 1, .write_connection_count = 1,
        .busy_timeout_ms = 1000, .enable_wal = true});
    ASSERT_TRUE(pool->Start().ok());

    const std::array<SqliteMigrationStep, 1> original_steps{{
        {.version = 1, .name = "create_checksum", .checksum = "checksum-v1",
         .apply = [](auto& connection) { return connection.Execute("CREATE TABLE checksum_target(id INTEGER)"); }},
    }};
    TestSource original("checksum", original_steps);
    std::array<ISqliteMigrationSource*, 1> sources{&original};
    SqliteMigrationRunner runner(pool);
    ASSERT_TRUE(runner.ApplyAll(sources).ok());

    const std::array<SqliteMigrationStep, 1> changed_steps{{
        {.version = 1, .name = "create_checksum", .checksum = "checksum-changed",
         .apply = [](auto&) { return core::Status::Ok(); }},
    }};
    TestSource changed("checksum", changed_steps);
    sources[0] = &changed;
    auto status = runner.ApplyAll(sources);
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), core::ErrorCode::FailedPrecondition);
}

}
