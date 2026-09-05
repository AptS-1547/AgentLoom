# Session Persistence Contracts (2026-08)

> 状态：当前架构契约
>
> 适用范围：AgentLoom Persona Session 的 SQLite、本地关系型 backend、PostgreSQL Runtime 和未来远端 Repository adapter
>
> 本文只定义 Runtime Session 这一业务模块所需的领域接口和数据边界，不规定下游 Application 的业务表结构，也不把 HTTP/SSE transport 引入 Session persistence 层。Auth、Persona Metadata、Document、Vector 等模块应分别定义自己的 Repository contract；本文不试图替代这些模块接口。

## 1. 设计结论

Session 全流程涉及的关系型数据不使用一个笼统的 `ISessionDatabase` 或“大而全”的 `IRuntimeSessionRepository`。作为 Runtime Session 模块内部的拆分，每一种数据库职责都通过单独的 capability interface 暴露：

```text
SessionEnsure / BusinessProjectionReader
        -> 只读业务投影，确认 Session 是否允许进入 Runtime

RuntimeSessionLifecycleStore
        -> Runtime instance 创建、加载、状态转换、关闭

RuntimeSessionLeaseStore
        -> owner lease、续租、释放、fencing token

RuntimeSessionCheckpointStore
        -> immutable checkpoint、revision CAS、恢复读取

RuntimeSessionRouteStore（可选）
        -> owner hint、route revision、节点发现缓存

RuntimeSessionMaintenanceStore（可选）
        -> 过期清理、恢复扫描、归档和诊断
```

这些接口可以由同一个 PostgreSQL connection pool 和 transaction runtime 实现，也可以由不同 adapter 实现。接口拆分是为了替换能力，而不是强制每个接口使用独立数据库或独立连接池。

## 2. 不变量

1. `SessionManager`、`PersonaRuntime` 和 Gateway 业务代码只依赖领域 contract，不依赖 SQL、SQLite statement、PostgreSQL driver 或连接池类型。
2. Business projection 是 Application 业务事实的只读投影；AgentLoom 不通过这些接口修改业务 Session、权限、账单或 usage。
3. Runtime lifecycle、lease、checkpoint 属于 AgentLoom Runtime 控制面，可以由 Runtime 自己写入专属 schema。
4. 所有持久化时间使用 UTC epoch milliseconds 或数据库 timestamptz；`steady_clock` 只用于进程内 deadline 和耗时，不能持久化。
5. 所有 mutation 都必须携带 `request_id`、`trace_id`、`runtime_instance_id`；会影响 owner 的 mutation 还必须携带 `fencing_token`。
6. 旧 owner 的 checkpoint、close、release 或异步结果发布必须被拒绝，不能静默覆盖新 owner 的状态。
7. 正常 Turn 不访问关系型数据库。数据库只用于 Ensure/cache miss、恢复、lease heartbeat、checkpoint、Close、shutdown drain 和维护任务。
8. 每个接口的错误都映射为 `core::Status`；重试策略由 DB adapter 控制，业务层不能自由重试 SQL。
9. 接口 DTO 不包含 `sqlite3*`、`PGconn*`、裸指针、mutex、future、worker index、HTTP connection 或 callback。

## 3. 共享标识和 DTO

### 3.1 Session key

```cpp
struct SessionPersistenceKey {
    std::string tenant_id;
    std::string session_id;
};
```

`session_id` 在部署范围内唯一时，`tenant_id` 仍然必须保留在 contract 和查询条件中，防止未来多租户迁移时发生隐式放宽。

### 3.2 Business projection

```cpp
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
```

该 DTO 只描述 Runtime 需要的最小业务投影，不复制 Application 的完整业务 Session。

### 3.3 Runtime identity and lifecycle

```cpp
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
```

`runtime_instance_id` 标识一次 Runtime 进程/节点 ownership，不能用业务 `session_id` 代替。

### 3.4 Checkpoint

Checkpoint 应是有版本的 immutable DTO。第一版只保存 Session 重建所需的最小状态：

```cpp
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
```

不能写入在途 callback/future、worker 状态、锁、HTTP connection、token 或大型 Memory payload。`recent_turns_json` 需要有明确大小上限，超限应返回 `ResourceExhausted`，不能无界扩展数据库记录。

### 3.5 Lease and fencing

```cpp
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
```

Fencing token 只递增，不回收、不复用。数据库 adapter 必须在同一条原子语句或短事务中完成 owner 条件检查和 token 更新。

## 4. 细粒度接口

### 4.1 `IBusinessSessionProjectionReader`

职责：读取 Application 拥有的只读 Session 投影。

```cpp
class IBusinessSessionProjectionReader {
public:
    virtual ~IBusinessSessionProjectionReader() = default;

    virtual core::Result<BusinessSessionProjection> Get(
        const SessionPersistenceKey& key,
        std::optional<std::uint64_t> minimum_revision = std::nullopt) = 0;
};
```

约束：

- 只读，不提供 `Upsert`、`Patch` 或 `Delete`；
- 查询必须带 tenant/session 条件；
- `minimum_revision` 不满足时返回 `FailedPrecondition` 或明确的 stale 状态；
- 可实现为 PostgreSQL read-only adapter、SQLite local projection 或 gRPC remote reader；
- 不参与正常每 Turn 路径，优先由 bootstrap/route cache 调用。

### 4.2 `IRuntimeSessionLifecycleStore`

职责：Runtime Session 的创建、加载和 lifecycle state transition。

```cpp
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
```

生命周期接口不负责 lease acquire，不负责 checkpoint 写入，也不负责业务 projection 更新。`MarkActive/Closing/Closed` 必须验证 instance 和 fencing token。

### 4.3 `IRuntimeSessionLeaseStore`

职责：owner lease 和 fencing token。

```cpp
class IRuntimeSessionLeaseStore {
public:
    virtual ~IRuntimeSessionLeaseStore() = default;

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
```

Lease store 不隐式创建 Runtime lifecycle record，也不写 checkpoint。Acquire 冲突返回 `ResourceExhausted`、`FailedPrecondition` 或专用 stale-owner status；不能返回空 lease 并让上层自行猜测。

### 4.4 `IRuntimeSessionCheckpointStore`

职责：checkpoint 的保存、读取和 revision CAS。

```cpp
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
```

`Save` 必须同时校验 `runtime_instance_id`、`fencing_token` 和 expected checkpoint revision。旧 revision 或旧 owner 只能返回冲突，不得覆盖新记录。

### 4.5 `IRuntimeSessionRouteStore`（可选）

职责：保存跨节点粘性路由的最小 owner hint，不保存完整 Session。

```cpp
struct RuntimeSessionOwnerHint {
    SessionPersistenceKey key;
    std::string owner_node_id;
    std::string runtime_instance_id;
    std::uint64_t fencing_token = 0;
    std::uint64_t route_revision = 0;
    std::int64_t expires_at_ms = 0;
};

class IRuntimeSessionRouteStore {
public:
    virtual ~IRuntimeSessionRouteStore() = default;

    virtual core::Result<std::optional<RuntimeSessionOwnerHint>> Resolve(
        const SessionPersistenceKey& key) = 0;

    virtual core::Status Publish(RuntimeSessionOwnerHint hint) = 0;

    virtual core::Status Remove(
        const RuntimeSessionOwnerHint& hint) = 0;
};
```

该接口可以先由 Redis 实现；如果用 PostgreSQL 实现，也必须保持 owner hint 与 Runtime lease 的 fencing 语义一致。它不是 Session 的权威状态源。

### 4.6 `IRuntimeSessionMaintenanceStore`（可选）

职责：恢复扫描、过期清理、归档和只读诊断，不进入请求 critical path。

```cpp
struct RuntimeSessionMaintenanceQuery {
    std::size_t limit = 0;
    std::int64_t before_ms = 0;
    std::string runtime_instance_id;
};

class IRuntimeSessionMaintenanceStore {
public:
    virtual ~IRuntimeSessionMaintenanceStore() = default;

    virtual core::Result<std::vector<RuntimeSessionLifecycleRecord>>
    ListRecoveryCandidates(RuntimeSessionMaintenanceQuery query) = 0;

    virtual core::Result<std::size_t> PurgeClosed(
        RuntimeSessionMaintenanceQuery query) = 0;
};
```

维护接口必须有界，不能用无条件 `ListAll` 把整个数据库加载到内存。

## 5. Session 全流程映射

```text
Ensure / Create
  -> IBusinessSessionProjectionReader::Get
  -> IRuntimeSessionRouteStore::Resolve（可选）
  -> IRuntimeSessionLeaseStore::Acquire
  -> IRuntimeSessionLifecycleStore::Create / MarkActive
  -> IRuntimeSessionCheckpointStore::LoadLatest（有记录时）
  -> 构造进程内 SessionState

Normal Turn
  -> 只使用 SessionManager 内存状态和 fencing snapshot
  -> 不访问数据库

Lease heartbeat
  -> IRuntimeSessionLeaseStore::Renew
  -> 更新进程内 lease snapshot

Checkpoint
  -> 生成 immutable RuntimeSessionCheckpointRecord
  -> 有界 DB executor
  -> IRuntimeSessionCheckpointStore::Save(CAS)

Close / Finalize
  -> 停止新 admission
  -> drain 在途 Turn
  -> IRuntimeSessionCheckpointStore::Save（需要时）
  -> IRuntimeSessionLifecycleStore::MarkClosing / MarkClosed
  -> IRuntimeSessionLeaseStore::Release
  -> IRuntimeSessionRouteStore::Remove（可选）

Recovery
  -> IRuntimeSessionMaintenanceStore::ListRecoveryCandidates
  -> IRuntimeSessionLeaseStore::Acquire
  -> IRuntimeSessionCheckpointStore::LoadLatest
  -> IRuntimeSessionLifecycleStore::MarkActive
```

顺序不是简单的数据库事务拼接。Lease、checkpoint、lifecycle 和 route 可能由不同 backend 实现，跨 backend 不做伪原子双写；使用 fencing、revision、幂等 request_id 和可恢复状态处理部分成功。

## 6. Adapter 组合方式

### 6.1 SQLite local profile

```text
SqliteBusinessSessionProjectionReader
SqliteRuntimeSessionLifecycleStore
SqliteRuntimeSessionLeaseStore
SqliteRuntimeSessionCheckpointStore
SqliteRuntimeSessionMaintenanceStore
        -> shared SqliteConnectionPool
        -> SqliteMigrationRunner
```

第一版可以由一个 `SqliteSessionPersistenceBundle` 持有同一个 pool，但对外仍按单独接口注入。业务代码不得接收 bundle 后绕过接口直接拿 pool。

### 6.2 PostgreSQL production profile

```text
PostgresBusinessSessionProjectionReader  # read-only role
PostgresRuntimeSessionLifecycleStore
PostgresRuntimeSessionLeaseStore
PostgresRuntimeSessionCheckpointStore
PostgresRuntimeSessionMaintenanceStore
        -> shared RAII connection pool
        -> independent DB executor
```

PostgreSQL runtime schema 和 Application business schema 分开管理。Go/Application 拥有 business migration；AgentLoom Runtime migration 只拥有 runtime instance、lease、checkpoint 和 owner hint schema。

### 6.3 Remote adapter

未来可以把单个接口替换成 gRPC/远端服务，例如只替换 `IBusinessSessionProjectionReader`，而 Runtime lease/checkpoint 仍然由本地 PostgreSQL 处理。接口拆分的价值就是允许这种局部替换。

## 7. 第一批实现边界

第一批不直接改 `SessionManager` 的全部生命周期，而先完成可独立测试的 contract 和 SQLite adapter：

1. 新建稳定 DTO 和 6 个接口头文件；
2. 为 SQLite Runtime schema 注册 migration namespace；
3. 实现 lifecycle、lease、checkpoint 三个最小 SQLite adapter；
4. 建立 contract test fixture，覆盖 fresh、upgrade、CAS 冲突、fencing 冲突、重复 close；
5. 以可选依赖方式把 persistence bundle 接到 composition root；
6. SessionManager 先通过 observer/checkpoint sink 接入，不在第一批把数据库调用塞进每个 Turn；
7. PostgreSQL 实现复用同一 DTO/contract suite，后续只替换 adapter 和 SQL。

第一批明确不做：

- 修改 Application business schema；
- C++ 直接管理 Go 的 PostgreSQL migration；
- 把 Auth、Document、Vector 一次性迁移到 PostgreSQL；
- 每 Turn 同步写 checkpoint；
- 将完整 `SessionState` 序列化进数据库；
- HTTP SSE 或 WebSocket transport 改造。

## 8. 测试与错误标准

每个接口至少需要：

- SQLite fresh database -> latest schema；
- 已有旧版本 -> latest schema；
- 重复 migration 不重复执行；
- 两个启动器竞争 migration；
- lease 竞争和过期接管；
- 旧 fencing token 写 checkpoint/close 被拒绝；
- checkpoint revision CAS 冲突；
- close 幂等和 shutdown drain；
- 数据库不可用、连接池耗尽和 deadline 超时；
- SQLite/PostgreSQL contract suite 共享语义断言。

错误统一映射：

```text
NotFound
AlreadyExists
FailedPrecondition      # revision/fencing/business revision 不满足
ResourceExhausted       # pool、maintenance 或 checkpoint queue 有界资源耗尽
Unavailable              # DB/节点暂不可用
Timeout
Cancelled
DataLoss                # checksum、codec 或不可恢复数据损坏
```

业务层在返回错误前使用 logger 输出 trace、tenant/session、接口名、revision/fencing 结果和耗时；禁止记录 SQL 参数、密码、token 和完整业务 payload。

## 9. 后续实现顺序

```text
1. contract DTO/header + in-memory fake
2. SQLite migration runner + runtime schema
3. SQLite lifecycle/lease/checkpoint adapters
4. SessionManager checkpoint observer / close integration
5. PostgreSQL pool + DB executor + adapters
6. SQLite/PostgreSQL shared contract and recovery E2E
7. route hint / maintenance / multi-node handoff
```

这条顺序保证每一步都能独立测试和回滚。任何 backend 替换都只影响 adapter/composition root，不改变 SessionManager 的业务调度接口。
