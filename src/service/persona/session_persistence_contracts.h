#pragma once

#include "result.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace agent::service::persona {

struct SessionPersistenceKey {
    std::string tenant_id = "default";
    std::string session_id;
};

struct BusinessSessionProjection {
    SessionPersistenceKey key;
    std::string user_uuid;
    std::string persona_id;
    std::string business_status;
    std::uint64_t business_revision = 0;
    std::uint64_t bootstrap_version = 0;
    std::uint64_t runtime_policy_version = 0;
    std::int64_t updated_at_ms = 0;
};

enum class RuntimeSessionState {
    Creating,
    Active,
    Closing,
    Closed,
    RecoveryRequired,
};

struct RuntimeSessionIdentity {
    SessionPersistenceKey key;
    std::string user_uuid;
    std::string persona_id;
    std::string runtime_instance_id;
    std::uint64_t runtime_revision = 0;
    std::uint64_t fencing_token = 0;
    std::uint32_t schema_version = 1;
};

struct RuntimeSessionLifecycleRecord {
    RuntimeSessionIdentity identity;
    RuntimeSessionState state = RuntimeSessionState::Creating;
    std::string close_reason;
    std::int64_t created_at_ms = 0;
    std::int64_t last_active_at_ms = 0;
    std::int64_t updated_at_ms = 0;
};

struct RuntimeSessionCheckpoint {
    std::uint32_t codec_version = 1;
    std::uint64_t turn_sequence = 0;
    std::string emotion_state_json;
    std::string recent_turns_json;
    std::string persona_config_version;
    std::string memory_cursor;
    std::int64_t checkpoint_at_ms = 0;
    std::string checksum;
};

struct RuntimeSessionCheckpointRecord {
    RuntimeSessionIdentity identity;
    RuntimeSessionCheckpoint checkpoint;
    std::uint64_t checkpoint_revision = 0;
};

struct RuntimeSessionLease {
    SessionPersistenceKey key;
    std::string runtime_instance_id;
    std::uint64_t fencing_token = 0;
    std::int64_t acquired_at_ms = 0;
    std::int64_t expires_at_ms = 0;
    std::uint64_t lease_revision = 0;
};

struct RuntimeSessionLeaseRequest {
    SessionPersistenceKey key;
    std::string runtime_instance_id;
    std::int64_t now_ms = 0;
    std::int64_t ttl_ms = 0;
    std::string request_id;
    std::string trace_id;
};

struct RuntimeSessionLeaseRenewal {
    RuntimeSessionLease lease;
    std::int64_t now_ms = 0;
    std::int64_t ttl_ms = 0;
    std::string request_id;
    std::string trace_id;
};

struct RuntimeSessionOwnerHint {
    SessionPersistenceKey key;
    std::string owner_node_id;
    std::string runtime_instance_id;
    std::uint64_t fencing_token = 0;
    std::uint64_t route_revision = 0;
    std::int64_t expires_at_ms = 0;
};

struct RuntimeSessionMaintenanceQuery {
    std::size_t limit = 0;
    std::int64_t before_ms = 0;
    std::string runtime_instance_id;
};

class IBusinessSessionProjectionReader {
public:
    virtual ~IBusinessSessionProjectionReader() = default;

    virtual core::Result<BusinessSessionProjection> Get(
        const SessionPersistenceKey& key,
        std::optional<std::uint64_t> minimum_revision = std::nullopt) = 0;
};

class IRuntimeSessionLifecycleStore {
public:
    virtual ~IRuntimeSessionLifecycleStore() = default;

    virtual core::Result<RuntimeSessionLifecycleRecord> Load(
        const SessionPersistenceKey& key) = 0;

    virtual core::Status Create(
        const RuntimeSessionLifecycleRecord& record) = 0;

    virtual core::Status MarkActive(
        const SessionPersistenceKey& key,
        std::string_view runtime_instance_id,
        std::uint64_t fencing_token,
        std::uint64_t expected_revision) = 0;

    virtual core::Status MarkClosing(
        const SessionPersistenceKey& key,
        std::string_view runtime_instance_id,
        std::uint64_t fencing_token,
        std::string_view reason) = 0;

    virtual core::Status MarkClosed(
        const SessionPersistenceKey& key,
        std::string_view runtime_instance_id,
        std::uint64_t fencing_token,
        std::string_view reason) = 0;
};

class IRuntimeSessionLeaseStore {
public:
    virtual ~IRuntimeSessionLeaseStore() = default;

    // Acquire/Renew/Release 必须由 backend 原子校验 owner 和 fencing token。
    virtual core::Result<RuntimeSessionLease> Acquire(
        RuntimeSessionLeaseRequest request) = 0;

    virtual core::Result<RuntimeSessionLease> Renew(
        RuntimeSessionLeaseRenewal renewal) = 0;

    virtual core::Status Release(
        const RuntimeSessionLease& lease,
        std::string_view request_id,
        std::string_view trace_id) = 0;

    virtual core::Result<std::optional<RuntimeSessionLease>> Resolve(
        const SessionPersistenceKey& key) = 0;
};

class IRuntimeSessionCheckpointStore {
public:
    virtual ~IRuntimeSessionCheckpointStore() = default;

    virtual core::Result<std::optional<RuntimeSessionCheckpointRecord>> LoadLatest(
        const SessionPersistenceKey& key) = 0;

    virtual core::Status Save(
        RuntimeSessionCheckpointRecord record,
        std::uint64_t expected_checkpoint_revision) = 0;

    virtual core::Status DeleteForClosedSession(
        const SessionPersistenceKey& key,
        std::string_view runtime_instance_id,
        std::uint64_t fencing_token) = 0;
};

class IRuntimeSessionRouteStore {
public:
    virtual ~IRuntimeSessionRouteStore() = default;

    virtual core::Result<std::optional<RuntimeSessionOwnerHint>> Resolve(
        const SessionPersistenceKey& key) = 0;

    virtual core::Status Publish(RuntimeSessionOwnerHint hint) = 0;

    virtual core::Status Remove(const RuntimeSessionOwnerHint& hint) = 0;
};

class IRuntimeSessionMaintenanceStore {
public:
    virtual ~IRuntimeSessionMaintenanceStore() = default;

    virtual core::Result<std::vector<RuntimeSessionLifecycleRecord>>
    ListRecoveryCandidates(RuntimeSessionMaintenanceQuery query) = 0;

    virtual core::Result<std::size_t> PurgeClosed(
        RuntimeSessionMaintenanceQuery query) = 0;
};

}
