#include "vector/sqlite_vector_repository.h"
#include "vector/vector_fingerprint.h"
#include "vector/vector_partition_registry.h"
#include "sqlite/sqlite_connection_pool.h"
#include "sqlite/sqlite_connection.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

using agent::vector_storage::CollectionDescriptor;
using agent::vector_storage::EntryRecord;
using agent::vector_storage::PartitionKey;
using agent::vector_storage::PartitionKeyHash;
using agent::vector_storage::SqliteVectorRepository;
using storage::sqlite::SqliteConnection;
using storage::sqlite::SqliteConnectionPool;
using storage::sqlite::SqliteConnectionPoolOptions;

namespace {

std::filesystem::path MakeTempDbPath(const std::string& name) {
    auto path = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + "-wal", ec);
    std::filesystem::remove(path.string() + "-shm", ec);
    return path;
}

void PreCreateDatabaseFile(const std::filesystem::path& path) {
    auto result = SqliteConnection::Open(path.string());
    ASSERT_TRUE(result.ok()) << result.status().message();
    auto conn = std::move(result).value();
    ASSERT_TRUE(conn.EnableWal().ok());
}

std::int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

CollectionDescriptor MakeCollection(const std::string& name = "test_collection") {
    CollectionDescriptor d;
    d.name = name;
    d.embedding_model_fingerprint = "embed_fp_v1";
    d.tokenizer_fingerprint = "tok_fp_v1";
    d.pooling_strategy = "mean";
    d.normalization = "l2";
    d.dimension = 4;
    d.corpus_version = "v1";
    d.policy_version = "v1";
    return d;
}

PartitionKey MakeKey(std::int64_t cid, const std::string& user, const std::string& level) {
    PartitionKey k;
    k.collection_id = cid;
    k.tenant_id = "";
    k.user_id = user;
    k.memory_level = level;
    k.scope_extras_json = "{}";
    return k;
}

EntryRecord MakeEntry(std::int64_t partition_id,
                      const std::string& memory_hash,
                      const std::vector<float>& vec,
                      const std::string& memory_type = "fact") {
    EntryRecord e;
    e.partition_id = partition_id;
    e.cache_key = "ck_" + memory_hash;
    e.text_hash = "th_" + memory_hash;
    e.content_hash = "ch_" + memory_hash;
    e.memory_hash = memory_hash;
    e.vector = vec;
    e.memory_type = memory_type;
    e.emotion = "neutral";
    e.emotion_intensity = 0.5f;
    e.state_arousal = 0.1f;
    e.payload = "{\"text\":\"" + memory_hash + "\"}";
    e.created_at_ms = NowMs();
    return e;
}

}  // namespace

class VectorRepoTest : public ::testing::Test {
protected:
    void SetUp() override {
        db_path_ = MakeTempDbPath("agent_vector_repo_test.db");
        PreCreateDatabaseFile(db_path_);

        SqliteConnectionPoolOptions options;
        options.path = db_path_.string();
        options.read_connection_count = 2;
        options.write_connection_count = 1;
        options.busy_timeout_ms = 1000;
        options.enable_wal = true;

        pool_ = std::make_shared<SqliteConnectionPool>(options);
        ASSERT_TRUE(pool_->Start().ok());
        repo_ = std::make_unique<SqliteVectorRepository>(pool_);
        ASSERT_TRUE(repo_->EnsureSchema().ok());
    }

    void TearDown() override {
        repo_.reset();
        if (pool_) pool_->Close();
        pool_.reset();
        std::error_code ec;
        std::filesystem::remove(db_path_, ec);
        std::filesystem::remove(db_path_.string() + "-wal", ec);
        std::filesystem::remove(db_path_.string() + "-shm", ec);
    }

    std::filesystem::path db_path_;
    std::shared_ptr<SqliteConnectionPool> pool_;
    std::unique_ptr<SqliteVectorRepository> repo_;
};

// ───── Collection ─────

TEST_F(VectorRepoTest, EnsureCollectionInsertsThenLooksUp) {
    auto r1 = repo_->EnsureCollection(MakeCollection());
    ASSERT_TRUE(r1.ok());
    auto id1 = std::move(r1).value();
    EXPECT_GT(id1, 0);

    auto r2 = repo_->EnsureCollection(MakeCollection());
    ASSERT_TRUE(r2.ok());
    EXPECT_EQ(std::move(r2).value(), id1);  // same name -> same id
}

TEST_F(VectorRepoTest, GetCollectionRoundTrip) {
    auto in = MakeCollection("c1");
    auto id_r = repo_->EnsureCollection(in);
    ASSERT_TRUE(id_r.ok());

    auto out_r = repo_->GetCollection(std::move(id_r).value());
    ASSERT_TRUE(out_r.ok());
    auto out = std::move(out_r).value();
    EXPECT_EQ(out.name, in.name);
    EXPECT_EQ(out.embedding_model_fingerprint, in.embedding_model_fingerprint);
    EXPECT_EQ(out.dimension, in.dimension);
    EXPECT_EQ(out.pooling_strategy, in.pooling_strategy);
}

TEST_F(VectorRepoTest, GetCollectionNotFound) {
    auto r = repo_->GetCollection(9999);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), core::ErrorCode::NotFound);
}

// ───── Partition ─────

TEST_F(VectorRepoTest, EnsurePartitionIdempotentAndIsolated) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();

    auto p1 = repo_->EnsurePartition(MakeKey(cid, "u1", "L2"));
    ASSERT_TRUE(p1.ok());
    auto p1_id = std::move(p1).value();

    auto p2 = repo_->EnsurePartition(MakeKey(cid, "u1", "L2"));
    ASSERT_TRUE(p2.ok());
    EXPECT_EQ(std::move(p2).value(), p1_id);

    // Different memory_level -> different partition
    auto p3 = repo_->EnsurePartition(MakeKey(cid, "u1", "L3"));
    ASSERT_TRUE(p3.ok());
    EXPECT_NE(std::move(p3).value(), p1_id);

    // Different user_id -> different partition
    auto p4 = repo_->EnsurePartition(MakeKey(cid, "u2", "L2"));
    ASSERT_TRUE(p4.ok());
    EXPECT_NE(std::move(p4).value(), p1_id);
}

TEST_F(VectorRepoTest, LookupPartitionReturnsNullopt) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();
    auto r = repo_->LookupPartition(MakeKey(cid, "u_missing", "L1"));
    ASSERT_TRUE(r.ok());
    EXPECT_FALSE(std::move(r).value().has_value());
}

TEST_F(VectorRepoTest, GetPartitionSnapshotTracksInserts) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();
    auto pid = std::move(repo_->EnsurePartition(MakeKey(cid, "u1", "L2"))).value();

    auto s0 = repo_->GetPartitionSnapshot(pid);
    ASSERT_TRUE(s0.ok());
    EXPECT_EQ(std::move(s0).value().vector_count, 0);

    auto e_r = repo_->InsertEntry(MakeEntry(pid, "h1", {1.0f, 0.0f, 0.0f, 0.0f}));
    ASSERT_TRUE(e_r.ok());

    auto s1 = repo_->GetPartitionSnapshot(pid);
    ASSERT_TRUE(s1.ok());
    auto snap1 = std::move(s1).value();
    EXPECT_EQ(snap1.vector_count, 1);
    EXPECT_GT(snap1.last_modified_at_ms, 0);
}

// ───── Entry ─────

TEST_F(VectorRepoTest, InsertEntryRoundTrip) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();
    auto pid = std::move(repo_->EnsurePartition(MakeKey(cid, "u1", "L2"))).value();

    std::vector<float> vec{0.1f, 0.2f, 0.3f, 0.4f};
    auto in = MakeEntry(pid, "h1", vec);
    in.expires_at_ms = NowMs() + 86400000;

    auto id_r = repo_->InsertEntry(in);
    ASSERT_TRUE(id_r.ok());
    auto id = std::move(id_r).value();

    auto out_r = repo_->GetEntry(id);
    ASSERT_TRUE(out_r.ok());
    auto out = std::move(out_r).value();

    EXPECT_EQ(out.entry_id, id);
    EXPECT_EQ(out.partition_id, pid);
    EXPECT_EQ(out.memory_hash, "h1");
    EXPECT_EQ(out.payload, in.payload);
    EXPECT_EQ(out.memory_type, "fact");
    ASSERT_EQ(out.vector.size(), 4u);
    EXPECT_FLOAT_EQ(out.vector[0], 0.1f);
    EXPECT_FLOAT_EQ(out.vector[3], 0.4f);
    EXPECT_FLOAT_EQ(out.emotion_intensity, 0.5f);
    ASSERT_TRUE(out.expires_at_ms.has_value());
    EXPECT_EQ(*out.expires_at_ms, *in.expires_at_ms);
}

TEST_F(VectorRepoTest, InsertEntriesBatchAndListByPartition) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();
    auto pid_l2 = std::move(repo_->EnsurePartition(MakeKey(cid, "u1", "L2"))).value();
    auto pid_l3 = std::move(repo_->EnsurePartition(MakeKey(cid, "u1", "L3"))).value();

    std::vector<EntryRecord> entries;
    entries.push_back(MakeEntry(pid_l2, "l2_a", {1, 0, 0, 0}));
    entries.push_back(MakeEntry(pid_l2, "l2_b", {0, 1, 0, 0}));
    entries.push_back(MakeEntry(pid_l3, "l3_a", {0, 0, 1, 0}));
    entries.push_back(MakeEntry(pid_l3, "l3_b", {0, 0, 0, 1}));

    ASSERT_TRUE(repo_->InsertEntries(entries).ok());
    for (const auto& e : entries) {
        EXPECT_GT(e.entry_id, 0);
    }

    auto l2_r = repo_->ListEntries(pid_l2, false);
    ASSERT_TRUE(l2_r.ok());
    auto l2 = std::move(l2_r).value();
    ASSERT_EQ(l2.size(), 2u);
    EXPECT_EQ(l2[0].memory_hash, "l2_a");
    EXPECT_EQ(l2[1].memory_hash, "l2_b");

    auto l3_r = repo_->ListEntries(pid_l3, false);
    ASSERT_TRUE(l3_r.ok());
    auto l3 = std::move(l3_r).value();
    ASSERT_EQ(l3.size(), 2u);
    EXPECT_EQ(l3[0].memory_hash, "l3_a");

    auto s_r = repo_->GetPartitionSnapshot(pid_l2);
    ASSERT_TRUE(s_r.ok());
    EXPECT_EQ(std::move(s_r).value().vector_count, 2);
}

TEST_F(VectorRepoTest, LookupEntriesByIds) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();
    auto pid = std::move(repo_->EnsurePartition(MakeKey(cid, "u1", "L2"))).value();

    auto id1 = std::move(repo_->InsertEntry(MakeEntry(pid, "a", {1, 0, 0, 0}))).value();
    auto id2 = std::move(repo_->InsertEntry(MakeEntry(pid, "b", {0, 1, 0, 0}))).value();
    auto id3 = std::move(repo_->InsertEntry(MakeEntry(pid, "c", {0, 0, 1, 0}))).value();

    std::vector<std::int64_t> ids = {id1, id3, 99999};  // 99999 missing
    auto r = repo_->LookupEntries(ids);
    ASSERT_TRUE(r.ok());
    auto entries = std::move(r).value();
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_EQ(entries[0].memory_hash, "a");
    EXPECT_EQ(entries[1].memory_hash, "c");
    (void)id2;
}

// ───── Lifecycle ─────

TEST_F(VectorRepoTest, MarkRecalledIncrementsCounter) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();
    auto pid = std::move(repo_->EnsurePartition(MakeKey(cid, "u1", "L2"))).value();
    auto id = std::move(repo_->InsertEntry(MakeEntry(pid, "a", {1, 0, 0, 0}))).value();

    auto now1 = NowMs();
    std::vector<std::int64_t> ids = {id};
    ASSERT_TRUE(repo_->MarkRecalled(ids, now1).ok());
    ASSERT_TRUE(repo_->MarkRecalled(ids, now1 + 100).ok());

    auto e = std::move(repo_->GetEntry(id)).value();
    EXPECT_EQ(e.recall_count, 2);
    ASSERT_TRUE(e.last_recalled_at_ms.has_value());
    EXPECT_EQ(*e.last_recalled_at_ms, now1 + 100);
}

TEST_F(VectorRepoTest, MarkForgottenAndListFiltering) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();
    auto pid = std::move(repo_->EnsurePartition(MakeKey(cid, "u1", "L2"))).value();

    auto id1 = std::move(repo_->InsertEntry(MakeEntry(pid, "alive", {1, 0, 0, 0}))).value();
    auto id2 = std::move(repo_->InsertEntry(MakeEntry(pid, "dead",  {0, 1, 0, 0}))).value();

    auto now = NowMs();
    std::vector<std::int64_t> dead_ids = {id2};
    ASSERT_TRUE(repo_->MarkForgotten(dead_ids, now, now + 7 * 86400000).ok());

    auto active_r = repo_->ListEntries(pid, false);
    ASSERT_TRUE(active_r.ok());
    auto active = std::move(active_r).value();
    ASSERT_EQ(active.size(), 1u);
    EXPECT_EQ(active[0].memory_hash, "alive");

    auto all_r = repo_->ListEntries(pid, true);
    ASSERT_TRUE(all_r.ok());
    EXPECT_EQ(std::move(all_r).value().size(), 2u);

    auto dead_entry = std::move(repo_->GetEntry(id2)).value();
    EXPECT_TRUE(dead_entry.forgotten);
    EXPECT_EQ(dead_entry.forget_epoch, 1);
    ASSERT_TRUE(dead_entry.forgotten_at_ms.has_value());
    (void)id1;
}

TEST_F(VectorRepoTest, ReactivateRestoresEntry) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();
    auto pid = std::move(repo_->EnsurePartition(MakeKey(cid, "u1", "L2"))).value();
    auto id = std::move(repo_->InsertEntry(MakeEntry(pid, "x", {1, 0, 0, 0}))).value();

    std::vector<std::int64_t> ids = {id};
    auto now = NowMs();
    ASSERT_TRUE(repo_->MarkForgotten(ids, now, now + 86400000).ok());
    ASSERT_TRUE(repo_->Reactivate(ids, now + 1000).ok());

    auto e = std::move(repo_->GetEntry(id)).value();
    EXPECT_FALSE(e.forgotten);
    EXPECT_FALSE(e.forgotten_at_ms.has_value());
    EXPECT_FALSE(e.deleted_after_ms.has_value());
    EXPECT_EQ(e.recall_count, 1);
}

TEST_F(VectorRepoTest, DeleteEntryRemovesAndDecrementsCount) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();
    auto pid = std::move(repo_->EnsurePartition(MakeKey(cid, "u1", "L2"))).value();
    auto id = std::move(repo_->InsertEntry(MakeEntry(pid, "x", {1, 0, 0, 0}))).value();

    auto s1 = std::move(repo_->GetPartitionSnapshot(pid)).value();
    EXPECT_EQ(s1.vector_count, 1);

    ASSERT_TRUE(repo_->DeleteEntry(id).ok());

    auto e = repo_->GetEntry(id);
    EXPECT_FALSE(e.ok());
    EXPECT_EQ(e.status().code(), core::ErrorCode::NotFound);

    auto s2 = std::move(repo_->GetPartitionSnapshot(pid)).value();
    EXPECT_EQ(s2.vector_count, 0);
}

// ───── 跨分区物理隔离（细粒度分区核心验证）─────

TEST_F(VectorRepoTest, CrossPartitionPhysicalIsolation) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();

    struct PartitionData {
        std::int64_t partition_id;
        std::string user;
        std::string level;
        std::vector<std::int64_t> entry_ids;
    };

    std::vector<PartitionData> partitions = {
        {0, "alice", "L1", {}}, {0, "alice", "L2", {}},
        {0, "alice", "L3", {}}, {0, "bob",   "L2", {}},
    };

    for (auto& p : partitions) {
        p.partition_id = std::move(repo_->EnsurePartition(MakeKey(cid, p.user, p.level))).value();
        for (int i = 0; i < 4; ++i) {
            std::string hash = p.user + "_" + p.level + "_" + std::to_string(i);
            auto id = std::move(repo_->InsertEntry(MakeEntry(p.partition_id, hash, {0, 0, 0, 0}))).value();
            p.entry_ids.push_back(id);
        }
    }

    // Each partition should see exactly its own 4 entries
    for (const auto& p : partitions) {
        auto list_r = repo_->ListEntries(p.partition_id, false);
        ASSERT_TRUE(list_r.ok());
        auto list = std::move(list_r).value();
        ASSERT_EQ(list.size(), 4u) << "Partition " << p.user << "/" << p.level << " leaked";
        std::string expected_prefix = p.user + "_" + p.level + "_";
        for (const auto& e : list) {
            EXPECT_TRUE(e.memory_hash.starts_with(expected_prefix))
                << "Found foreign entry: " << e.memory_hash;
            EXPECT_EQ(e.partition_id, p.partition_id);
        }
    }
}

// ───── 唯一约束 ─────

TEST_F(VectorRepoTest, DuplicateMemoryHashInSamePartitionFails) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();
    auto pid = std::move(repo_->EnsurePartition(MakeKey(cid, "u1", "L2"))).value();

    auto r1 = repo_->InsertEntry(MakeEntry(pid, "dup", {1, 0, 0, 0}));
    ASSERT_TRUE(r1.ok());

    // Same memory_hash in same partition -> UNIQUE violation
    auto r2 = repo_->InsertEntry(MakeEntry(pid, "dup", {0, 1, 0, 0}));
    EXPECT_FALSE(r2.ok());
}

TEST_F(VectorRepoTest, SameMemoryHashAllowedAcrossPartitions) {
    auto cid = std::move(repo_->EnsureCollection(MakeCollection())).value();
    auto p1 = std::move(repo_->EnsurePartition(MakeKey(cid, "u1", "L2"))).value();
    auto p2 = std::move(repo_->EnsurePartition(MakeKey(cid, "u1", "L3"))).value();

    ASSERT_TRUE(repo_->InsertEntry(MakeEntry(p1, "shared", {1, 0, 0, 0})).ok());
    ASSERT_TRUE(repo_->InsertEntry(MakeEntry(p2, "shared", {0, 1, 0, 0})).ok());
}

// ───── PartitionKey hash/equality ─────

TEST(PartitionKeyTest, HashConsistency) {
    PartitionKey key1;
    key1.collection_id = 1;
    key1.tenant_id = "tenant1";
    key1.user_id = "user1";
    key1.memory_level = "L2";

    PartitionKey key2 = key1;
    PartitionKeyHash hasher;
    EXPECT_EQ(hasher(key1), hasher(key2));

    PartitionKey key3 = key1;
    key3.memory_level = "L3";
    EXPECT_NE(hasher(key1), hasher(key3));
}

TEST(PartitionKeyTest, Equality) {
    PartitionKey key1;
    key1.collection_id = 1;
    key1.user_id = "user1";
    key1.memory_level = "L2";

    PartitionKey key2 = key1;
    EXPECT_EQ(key1, key2);

    key2.memory_level = "L3";
    EXPECT_NE(key1, key2);
}

// ───── Fingerprint policy ─────

namespace VFP = agent::vector_storage::VectorFingerprint;

TEST(FingerprintTest, EmbeddingFingerprintStableForSameInputs) {
    auto a = VFP::ComputeEmbeddingFingerprint("model/path.onnx", 768, "mean", "l2");
    auto b = VFP::ComputeEmbeddingFingerprint("model/path.onnx", 768, "mean", "l2");
    EXPECT_FALSE(a.empty());
    EXPECT_EQ(a, b);
    EXPECT_EQ(a.size(), 64u);  // sha256 hex = 64 chars
}

TEST(FingerprintTest, EmbeddingFingerprintChangesPerField) {
    auto base = VFP::ComputeEmbeddingFingerprint("model.onnx", 768, "mean", "l2");
    EXPECT_NE(base, VFP::ComputeEmbeddingFingerprint("other.onnx", 768, "mean", "l2"));
    EXPECT_NE(base, VFP::ComputeEmbeddingFingerprint("model.onnx", 384, "mean", "l2"));
    EXPECT_NE(base, VFP::ComputeEmbeddingFingerprint("model.onnx", 768, "cls",  "l2"));
    EXPECT_NE(base, VFP::ComputeEmbeddingFingerprint("model.onnx", 768, "mean", "none"));
}

TEST(FingerprintTest, TokenizerFingerprintOfFile) {
    auto tmp = MakeTempDbPath("agent_fp_tokenizer.json");
    {
        std::ofstream out(tmp);
        out << R"({"version":"1.0","model":"bert"})";
    }
    auto fp = VFP::ComputeTokenizerFingerprint(tmp.string());
    EXPECT_EQ(fp.size(), 64u);

    // Same contents → same fingerprint
    auto tmp2 = MakeTempDbPath("agent_fp_tokenizer2.json");
    {
        std::ofstream out(tmp2);
        out << R"({"version":"1.0","model":"bert"})";
    }
    EXPECT_EQ(fp, VFP::ComputeTokenizerFingerprint(tmp2.string()));

    // Missing file → empty
    EXPECT_TRUE(VFP::ComputeTokenizerFingerprint("/no/such/path.json").empty());

    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    std::filesystem::remove(tmp2, ec);
}

// ───── PartitionRegistry ─────

using agent::vector_storage::PartitionRegistry;

class PartitionRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        db_path_ = MakeTempDbPath("agent_partition_registry_test.db");
        PreCreateDatabaseFile(db_path_);

        SqliteConnectionPoolOptions options;
        options.path = db_path_.string();
        options.read_connection_count = 2;
        options.write_connection_count = 1;
        options.enable_wal = true;
        pool_ = std::make_shared<SqliteConnectionPool>(options);
        ASSERT_TRUE(pool_->Start().ok());

        repo_ = std::make_shared<SqliteVectorRepository>(pool_);
        ASSERT_TRUE(repo_->EnsureSchema().ok());

        auto cid_r = repo_->EnsureCollection(MakeCollection("reg_test"));
        ASSERT_TRUE(cid_r.ok());
        cid_ = std::move(cid_r).value();

        registry_ = std::make_unique<PartitionRegistry>(repo_);
    }
    void TearDown() override {
        registry_.reset();
        repo_.reset();
        if (pool_) pool_->Close();
        pool_.reset();
        std::error_code ec;
        std::filesystem::remove(db_path_, ec);
        std::filesystem::remove(db_path_.string() + "-wal", ec);
        std::filesystem::remove(db_path_.string() + "-shm", ec);
    }

    std::filesystem::path db_path_;
    std::shared_ptr<SqliteConnectionPool> pool_;
    std::shared_ptr<SqliteVectorRepository> repo_;
    std::int64_t cid_ = 0;
    std::unique_ptr<PartitionRegistry> registry_;
};

TEST_F(PartitionRegistryTest, ResolveCreatesThenCaches) {
    auto key = MakeKey(cid_, "u1", "L2");
    auto r1 = registry_->Resolve(key);
    ASSERT_TRUE(r1.ok());
    auto id1 = std::move(r1).value();
    EXPECT_EQ(registry_->CacheSize(), 1u);

    // Second call hits cache, returns same id
    auto r2 = registry_->Resolve(key);
    ASSERT_TRUE(r2.ok());
    EXPECT_EQ(std::move(r2).value(), id1);
    EXPECT_EQ(registry_->CacheSize(), 1u);
}

TEST_F(PartitionRegistryTest, LookupReturnsNulloptForMissing) {
    auto r = registry_->Lookup(MakeKey(cid_, "ghost", "L4"));
    ASSERT_TRUE(r.ok());
    EXPECT_FALSE(std::move(r).value().has_value());
    EXPECT_EQ(registry_->CacheSize(), 0u);  // missing keys not cached
}

TEST_F(PartitionRegistryTest, LookupHydratesCacheForExistingPartition) {
    auto key = MakeKey(cid_, "u1", "L3");
    auto eid = std::move(repo_->EnsurePartition(key)).value();

    // First Lookup hits repo
    auto r1 = registry_->Lookup(key);
    ASSERT_TRUE(r1.ok());
    auto v1 = std::move(r1).value();
    ASSERT_TRUE(v1.has_value());
    EXPECT_EQ(*v1, eid);
    EXPECT_EQ(registry_->CacheSize(), 1u);

    // Now cached
    auto r2 = registry_->Lookup(key);
    ASSERT_TRUE(r2.ok());
    EXPECT_EQ(*std::move(r2).value(), eid);
}

TEST_F(PartitionRegistryTest, NotifyModifiedTracksMonotonicLatest) {
    auto pid = std::move(registry_->Resolve(MakeKey(cid_, "u1", "L2"))).value();

    EXPECT_EQ(registry_->LastModifiedAtMs(pid), 0);

    registry_->NotifyModified(pid, 1000);
    EXPECT_EQ(registry_->LastModifiedAtMs(pid), 1000);

    // Older timestamp shouldn't move it backward
    registry_->NotifyModified(pid, 500);
    EXPECT_EQ(registry_->LastModifiedAtMs(pid), 1000);

    // Newer timestamp advances it
    registry_->NotifyModified(pid, 2000);
    EXPECT_EQ(registry_->LastModifiedAtMs(pid), 2000);
}

TEST_F(PartitionRegistryTest, ClearWipesEverything) {
    auto pid = std::move(registry_->Resolve(MakeKey(cid_, "u1", "L2"))).value();
    registry_->NotifyModified(pid, 1000);

    registry_->Clear();
    EXPECT_EQ(registry_->CacheSize(), 0u);
    EXPECT_EQ(registry_->LastModifiedAtMs(pid), 0);
}
