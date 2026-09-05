#include "session_persistence_contracts.h"

#include <gtest/gtest.h>

#include <map>
#include <mutex>

namespace {

using namespace agent::service::persona;

std::string KeyOf(const SessionPersistenceKey& key) {
    return key.tenant_id + "\n" + key.session_id;
}

class InMemoryLeaseStore final : public IRuntimeSessionLeaseStore {
public:
    core::Result<RuntimeSessionLease> Acquire(RuntimeSessionLeaseRequest request) override {
        if (request.key.session_id.empty() || request.runtime_instance_id.empty() ||
            request.ttl_ms <= 0) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "invalid lease request");
        }
        std::lock_guard lock(mutex_);
        auto& lease = leases_[KeyOf(request.key)];
        if (!lease.runtime_instance_id.empty() && lease.expires_at_ms > request.now_ms &&
            lease.runtime_instance_id != request.runtime_instance_id) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "session lease is owned");
        }
        lease.key = request.key;
        lease.runtime_instance_id = request.runtime_instance_id;
        lease.fencing_token = std::max<std::uint64_t>(lease.fencing_token + 1, 1);
        lease.lease_revision += 1;
        lease.acquired_at_ms = request.now_ms;
        lease.expires_at_ms = request.now_ms + request.ttl_ms;
        return lease;
    }

    core::Result<RuntimeSessionLease> Renew(RuntimeSessionLeaseRenewal renewal) override {
        if (renewal.ttl_ms <= 0) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "invalid lease ttl");
        }
        std::lock_guard lock(mutex_);
        auto found = leases_.find(KeyOf(renewal.lease.key));
        if (found == leases_.end() || found->second.runtime_instance_id != renewal.lease.runtime_instance_id ||
            found->second.fencing_token != renewal.lease.fencing_token ||
            found->second.expires_at_ms <= renewal.now_ms) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "stale session lease");
        }
        found->second.lease_revision += 1;
        found->second.expires_at_ms = renewal.now_ms + renewal.ttl_ms;
        return found->second;
    }

    core::Status Release(const RuntimeSessionLease& lease,
                         std::string_view,
                         std::string_view) override {
        std::lock_guard lock(mutex_);
        auto found = leases_.find(KeyOf(lease.key));
        if (found == leases_.end()) {
            return core::Status::Ok();
        }
        if (found->second.runtime_instance_id != lease.runtime_instance_id ||
            found->second.fencing_token != lease.fencing_token) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "stale session lease");
        }
        leases_.erase(found);
        return core::Status::Ok();
    }

    core::Result<std::optional<RuntimeSessionLease>> Resolve(
        const SessionPersistenceKey& key) override {
        std::lock_guard lock(mutex_);
        auto found = leases_.find(KeyOf(key));
        if (found == leases_.end()) {
            return std::optional<RuntimeSessionLease>{};
        }
        return std::optional<RuntimeSessionLease>{found->second};
    }

private:
    std::mutex mutex_;
    std::map<std::string, RuntimeSessionLease> leases_;
};

class InMemoryCheckpointStore final : public IRuntimeSessionCheckpointStore {
public:
    core::Result<std::optional<RuntimeSessionCheckpointRecord>> LoadLatest(
        const SessionPersistenceKey& key) override {
        std::lock_guard lock(mutex_);
        auto found = records_.find(KeyOf(key));
        if (found == records_.end()) {
            return std::optional<RuntimeSessionCheckpointRecord>{};
        }
        return std::optional<RuntimeSessionCheckpointRecord>{found->second};
    }

    core::Status Save(RuntimeSessionCheckpointRecord record,
                      std::uint64_t expected_checkpoint_revision) override {
        if (record.identity.key.session_id.empty() || record.identity.runtime_instance_id.empty() ||
            record.identity.fencing_token == 0) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "invalid checkpoint identity");
        }
        std::lock_guard lock(mutex_);
        auto& current = records_[KeyOf(record.identity.key)];
        if (current.checkpoint_revision != expected_checkpoint_revision) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "checkpoint revision conflict");
        }
        if (current.identity.fencing_token > record.identity.fencing_token) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "stale checkpoint fencing token");
        }
        record.checkpoint_revision = expected_checkpoint_revision + 1;
        records_[KeyOf(record.identity.key)] = std::move(record);
        return core::Status::Ok();
    }

    core::Status DeleteForClosedSession(const SessionPersistenceKey& key,
                                        std::string_view runtime_instance_id,
                                        std::uint64_t fencing_token) override {
        std::lock_guard lock(mutex_);
        auto found = records_.find(KeyOf(key));
        if (found == records_.end()) {
            return core::Status::Ok();
        }
        if (found->second.identity.runtime_instance_id != runtime_instance_id ||
            found->second.identity.fencing_token != fencing_token) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "stale checkpoint owner");
        }
        records_.erase(found);
        return core::Status::Ok();
    }

private:
    std::mutex mutex_;
    std::map<std::string, RuntimeSessionCheckpointRecord> records_;
};

RuntimeSessionLeaseRequest MakeRequest(std::string instance, std::int64_t now) {
    return {
        .key = {.tenant_id = "tenant-a", .session_id = "session-a"},
        .runtime_instance_id = std::move(instance),
        .now_ms = now,
        .ttl_ms = 1000,
        .request_id = "request-1",
        .trace_id = "trace-1",
    };
}

RuntimeSessionCheckpointRecord MakeCheckpoint(const RuntimeSessionLease& lease) {
    RuntimeSessionCheckpointRecord record;
    record.identity.key = lease.key;
    record.identity.user_uuid = "user-a";
    record.identity.persona_id = "persona-a";
    record.identity.runtime_instance_id = lease.runtime_instance_id;
    record.identity.fencing_token = lease.fencing_token;
    record.checkpoint.turn_sequence = 1;
    record.checkpoint.recent_turns_json = "[]";
    return record;
}

TEST(SessionPersistenceContractTest, LeaseFencingRejectsConcurrentAndStaleOwners) {
    InMemoryLeaseStore store;
    auto first_result = store.Acquire(MakeRequest("runtime-a", 100));
    ASSERT_TRUE(first_result.ok()) << first_result.status().message();
    const auto first = first_result.value();

    auto conflict = store.Acquire(MakeRequest("runtime-b", 200));
    ASSERT_FALSE(conflict.ok());
    EXPECT_EQ(conflict.status().code(), core::ErrorCode::ResourceExhausted);

    auto renewed = store.Renew({
        .lease = first,
        .now_ms = 200,
        .ttl_ms = 1000,
        .request_id = "renew-1",
        .trace_id = "trace-2",
    });
    ASSERT_TRUE(renewed.ok()) << renewed.status().message();

    auto stale = first;
    stale.runtime_instance_id = "runtime-old";
    auto stale_release = store.Release(stale, "release-stale", "trace-3");
    EXPECT_EQ(stale_release.code(), core::ErrorCode::FailedPrecondition);
    auto resolved = store.Resolve(first.key);
    ASSERT_TRUE(resolved.ok());
    ASSERT_TRUE(resolved.value().has_value());
    EXPECT_EQ(resolved.value()->fencing_token, first.fencing_token);
}

TEST(SessionPersistenceContractTest, CheckpointUsesRevisionCasAndFencing) {
    InMemoryLeaseStore leases;
    auto lease_result = leases.Acquire(MakeRequest("runtime-a", 100));
    ASSERT_TRUE(lease_result.ok());
    const auto lease = lease_result.value();
    InMemoryCheckpointStore checkpoints;

    auto record = MakeCheckpoint(lease);
    ASSERT_TRUE(checkpoints.Save(record, 0).ok());
    auto loaded = checkpoints.LoadLatest(lease.key);
    ASSERT_TRUE(loaded.ok());
    ASSERT_TRUE(loaded.value().has_value());
    EXPECT_EQ(loaded.value()->checkpoint_revision, 1u);

    record.checkpoint.turn_sequence = 2;
    EXPECT_EQ(checkpoints.Save(record, 0).code(), core::ErrorCode::FailedPrecondition);
    ASSERT_TRUE(checkpoints.Save(record, 1).ok());

    record.identity.fencing_token = lease.fencing_token + 1;
    ASSERT_TRUE(checkpoints.Save(record, 2).ok());
    auto stale = record;
    stale.identity.fencing_token = lease.fencing_token;
    EXPECT_EQ(checkpoints.Save(stale, 3).code(), core::ErrorCode::FailedPrecondition);
}

}
