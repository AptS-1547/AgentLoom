#pragma once

#include "result.h"
#include "sqlite/sqlite_connection_pool.h"
#include "sqlite/sqlite_migration.h"

#include <cstdint>
#include <chrono>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <concepts>
#include <type_traits>

namespace agent::semantic_cache {

struct L0SessionKey {
    std::string tenant_id = "default";
    std::string user_id;
    std::string session_id;
};

struct L0ActiveBatchState {
    std::int64_t timestamp = 0;
    std::size_t count = 0;
};

class IL0SessionBatchMetadataStore {
public:
    virtual ~IL0SessionBatchMetadataStore() = default;

    // 元数据严格以 SessionKey 隔离，不能退化为 user/tenant 级共享状态。
    virtual core::Result<std::optional<L0ActiveBatchState>> LoadActiveBatch(
        const L0SessionKey& key) = 0;

    virtual core::Status SaveActiveBatch(
        const L0SessionKey& key,
        const L0ActiveBatchState& state) = 0;

    virtual core::Status ClearActiveBatch(
        const L0SessionKey& key) = 0;

    virtual core::Result<std::vector<std::int64_t>> LoadTimestampIndex(
        const L0SessionKey& key) = 0;

    virtual core::Status ReplaceTimestampIndex(
        const L0SessionKey& key,
        std::span<const std::int64_t> timestamps) = 0;
};

template <typename Provider>
concept L0SessionMetadataProvider =
    std::derived_from<std::remove_cvref_t<Provider>, IL0SessionBatchMetadataStore>;

template <L0SessionMetadataProvider Provider>
core::Status EnsureL0MetadataReady(Provider& provider) {
    if constexpr (requires { provider.EnsureSchema(); }) {
        return provider.EnsureSchema();
    }
    return core::Status::Ok();
}

class SqliteL0SessionBatchMetadataStore final
    : public IL0SessionBatchMetadataStore,
      public storage::sqlite::ISqliteMigrationSource {
public:
    explicit SqliteL0SessionBatchMetadataStore(
        std::shared_ptr<storage::sqlite::SqliteConnectionPool> pool);

    core::Status EnsureSchema();

    core::Result<std::optional<L0ActiveBatchState>> LoadActiveBatch(
        const L0SessionKey& key) override;
    core::Status SaveActiveBatch(
        const L0SessionKey& key,
        const L0ActiveBatchState& state) override;
    core::Status ClearActiveBatch(const L0SessionKey& key) override;
    core::Result<std::vector<std::int64_t>> LoadTimestampIndex(
        const L0SessionKey& key) override;
    core::Status ReplaceTimestampIndex(
        const L0SessionKey& key,
        std::span<const std::int64_t> timestamps) override;

    std::string_view MigrationNamespace() const noexcept override;
    std::span<const storage::sqlite::SqliteMigrationStep>
    MigrationSteps() const noexcept override;

private:
    std::shared_ptr<storage::sqlite::SqliteConnectionPool> pool_;
};

}
