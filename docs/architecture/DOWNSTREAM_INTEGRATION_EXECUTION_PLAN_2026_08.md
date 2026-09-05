# Downstream Integration Execution Plan (2026-08)

> 状态：当前计划
>
> 适用范围：CloudTask 云端并行任务、关系型数据库重构、PostgreSQL Runtime 和后续 HTTP SSE
>
> 本计划面向 AgentLoom 与下游 Application 的近期集成，不把下游产品业务模型固化到 Runtime。

## 1. 背景和目标

AgentLoom 已经完成普通 Persona Turn 的主要异步边界和性能基线。下一阶段优先完成与下游业务集成直接相关的通用能力：

1. CloudTask 云端并行任务采用真正的异步 Provider 调用；
2. CloudTask 支持有序增量结果、取消、终态和有界并发；
3. SQLite schema 生命周期从业务 Repository 中收敛到 migration runtime；
4. 建立可选 PostgreSQL Runtime Session 最小纵向闭环；
5. 在异步任务协议稳定后，再共同确定 HTTP SSE 的传输协议和 Gateway 接线方式。

当前性能框架不再扩展为主线工作。必要的延迟和资源指标随上述业务接口同步补充，但不阻塞协议闭环。

## 2. 总体依赖

```text
CloudTask async operation protocol
  -> ordered incremental result sink
  -> cancellation / terminal state / bounded admission
  -> HTTP SSE transport（后续讨论）

Repository domain contracts
  -> SQLite migration runtime
  -> PostgreSQL connection/transaction runtime
  -> Runtime Session lease/fencing/checkpoint
  -> downstream Application bootstrap and recovery
```

CloudTask 与 SSE 共享增量事件、取消和终态语义，但 CloudTask 不依赖 HTTP。数据库 Repository contract 与具体 SQL backend 分离，业务层不得感知 SQLite、PostgreSQL driver 或 SQL placeholder 类型。

## 3. Phase A：CloudTask 异步任务协议

### 3.1 第一阶段范围

第一阶段只改造 CloudTask 核心协议和单元测试：

- 增加 RAII 异步 operation handle；
- 增加 ordered incremental result sink；
- 每个 chunk 使用 `IAsyncLlmClient::CompleteAsync()`；
- Provider 回调只写入状态并投递 continuation；
- 同一 task 按 `chunk_sequence` 向 sink 交付；
- 不同 task 可以并行执行；
- task 和 chunk 并发均有明确上限；
- `Cancel()` 幂等，并向所有在途 Provider operation 传播；
- `OnCompleted` 恰好调用一次；
- sink 拒绝结果时停止继续调度，并以结构化 `core::Status` 结束任务；
- 保留现有同步 `Execute()` 和 future `ExecuteAsync()` 兼容入口。

第一阶段不实现 Provider token streaming，也不修改 HTTP route。

### 3.2 协议边界

异步调用方通过以下角色协作：

```text
ICloudTaskCoordinator
  -> StartAsync(chunks, sink)
  -> ICloudTaskOperation

ICloudTaskOperation
  -> Cancel()
  -> Snapshot()

ICloudTaskResultSink
  -> OnChunk(result)
  -> OnCompleted(summary)
```

`OnChunk` 只按 chunk 终态交付。单个 chunk 的 Provider 失败同样形成一个有序的 `CloudTaskChunkResult`，不会堵塞后续序列。Task 级协议错误、取消、sink 错误和 Coordinator shutdown 通过 `CloudTaskSummary::status` 表示。

### 3.3 生命周期

```text
Pending
  -> Running
  -> Cancelling（可选）
  -> Completed | Failed | Cancelled
```

终态后必须满足：

- 不再调度新 chunk；
- 所有在途 `IAsyncLlmOperation` 已完成或收到取消；
- 不再调用 `OnChunk`；
- `OnCompleted` 恰好一次；
- Coordinator active task 计数已释放；
- operation handle 可以安全晚于 Coordinator 调用方释放。

### 3.4 背压和有序交付

第一版继续复用 `OrderedBitmapWindow<CloudTaskChunkResult>`。每个 Provider callback 将完成结果写入 window，再由 Coordinator continuation executor 排序 drain。

`ICloudTaskResultSink::OnChunk()` 是有界同步交付点：

- 返回 `Ok` 表示消费方已经接收该结果；
- 返回失败表示消费方拒绝继续接收，Coordinator 取消剩余 Provider operation；
- sink 不得无限阻塞；需要跨线程或网络缓冲的 consumer 应自行使用有界队列；
- 后续 SSE adapter 必须把 HTTP 写队列容量映射为该背压语义。

### 3.5 兼容入口

现有接口暂时保留：

```text
Execute()
  -> 同步兼容路径

ExecuteAsync() -> future
  -> 收集 ordered incremental sink 的兼容适配器
```

新业务代码应使用 operation + sink 接口。兼容入口用于现有算法流水线、测试和迁移期 consumer，不作为新的扩展基础。

### 3.6 测试闸门

- 异步提交立即返回；
- 多 chunk Provider 调用真实并发；
- Provider 乱序完成但 sink 严格有序；
- 单 chunk 失败不阻塞后续 chunk；
- 多个 task 可同时在途；
- 超过 active task 上限返回 `ResourceExhausted`；
- operation 取消幂等并传播到全部在途请求；
- late callback 不造成重复 chunk 或重复终态；
- sink 失败取消剩余工作；
- Coordinator 析构期间任务以 `Cancelled` 收口；
- 同步和 future 兼容入口行为不回归。

## 4. Phase B：数据库重构

### 4.1 模块级 SQLite Repository contract 与 backend adapter

本阶段只重构 SQLite。Redis 等 KV backend 不进入 SQL migration，也不为了形式统一实现 SQL Repository。数据库重构的核心单位是使用 SQLite 的业务模块，而不是某一张表或某一个 Session 类型。每个需要 SQLite 持久化的模块都应先定义不包含 SQL 类型的领域接口，再提供 SQLite adapter；未来 PostgreSQL/MySQL adapter 按下游实际需要单独实现：

```text
IAuthSessionStore
  -> SqliteAuthSessionStore
  -> RedisAuthSessionStore（现有 KV adapter，保持不动）
  -> PostgresAuthSessionStore（按需）

IPersonaMetadataStore
  -> SqlitePersonaMetadataStore
  -> RedisPersonaMetadataCache（现有 KV cache，保持不动）
  -> PostgresPersonaMetadataStore（按需）

IDocumentMetadataRepository
  -> SqliteDocumentMetadataRepository
  -> PostgresDocumentMetadataRepository（按需）

IVectorRepository
  -> SqliteVectorRepository
  -> Postgres/pgvector adapter（按需）
```

模块领域接口负责 CRUD、查询、状态转换和领域错误；SQLite adapter 负责 SQL、事务、连接池、placeholder 和 `core::Status` 映射。业务服务只能依赖模块领域接口，不能接收 `SqliteConnectionPool` 或数据库 path。Redis 继续使用现有 KV 接口、TTL、key namespace 和连接池，不与 SQLite migration runner 建立继承或统一 `IDatabase` 关系。

Auth 和 Persona Metadata 已经提供了可复用样板；Vector 已经有 `IVectorRepository`，下一步重点是把 schema 生命周期与其 CRUD 分离；Document Metadata 先补齐 `IDocumentMetadataRepository`，再把 SQLite 实现显式命名为 SQLite adapter，同时保留旧类型别名。

### 4.2 SQLite migration runtime

- 使用 RAII connection、statement 和 transaction；
- 建立 namespaced migration ledger；
- Repository 只声明领域能力和 migration source；
- 迁移执行、版本校验、checksum、事务和日志由统一 runner 负责；
- 迁移失败以 `core::Status` 返回并阻止服务进入 readiness；
- 现有 `EnsureSchema()` 在兼容期只允许转发，不继续散落 DDL。

截至 2026-08-25，migration runner 已用于 Auth、Persona Metadata、Document Metadata、Document Analysis、Vector、Long-term Memory 和 L0 Session metadata。Semantic Cache 的 legacy repair 仍保持独立，尚未把 Redis SCAN/rebuild 逻辑并入 SQLite migration。

不设计统一 SQLite、Redis 和 PostgreSQL 的 `IDatabase`。不同 backend 只通过领域 Repository contract 对齐。

### 4.3 SQLite Repository contract 与 Session 控制面

优先定义 Runtime Session 所需的领域接口：

```text
IBusinessSessionMetadataReader
IRuntimeSessionRepository
IRuntimeSessionLeaseStore
```

接口使用稳定 DTO、revision、fencing token 和 `core::Status`，不暴露 SQL result、connection、transaction 或 driver 专有类型。

## 5. Phase C：PostgreSQL（后续按需）

PostgreSQL 不属于当前 SQLite 重构阶段。未来如果下游需要替换 SQLite，再按模块实现 PostgreSQL adapter；它不是 AgentLoom 编译、安装和本地测试的强制依赖。

未来 PostgreSQL 接入按模块逐步替换，不要求一次性迁移全部 SQLite Store。当前阶段只要求 SQLite contract 和 migration source 稳定：

- 只读 business session projection；
- Runtime Session instance/snapshot/checkpoint；
- lease acquire/renew/release；
- fencing token 和 checkpoint CAS；
- 独立 DB executor；
- RAII pool/lease/transaction；
- driver error 到 `core::Status` 的统一映射；
- SQLite contract tests；PostgreSQL contract tests 在 adapter 开始实现时复用。

正常 Turn 在 route/bootstrap cache 命中时不得访问 PostgreSQL。数据库只参与首次 Ensure、恢复、节点切换、批量 heartbeat、checkpoint 和关闭收口。

## 6. Phase D：HTTP SSE（待共同确认）

SSE 会影响 HTTP response writer、连接生命周期、写队列、断连取消和公共协议，因此不在 CloudTask 第一阶段顺带实现。

进入实现前需要共同确认：

1. SSE endpoint 是现有 Chat route 的 transport variant，还是独立 CloudTask endpoint；
2. event type、event id、resume/`Last-Event-ID` 和终态 schema；
3. chunk result 与 Provider token delta 是否使用同一协议；
4. 客户端断连是否立即取消 CloudTask，还是允许后台完成；
5. HTTP 写队列容量、慢消费者策略和最大连接时长；
6. heartbeat、代理缓冲和 idle timeout 策略；
7. 认证过期、Session close 和 Gateway shutdown 的终态行为；
8. 是否需要断线重连后的持久化 replay。

在这些问题确定前，CloudTask 只提供 transport-neutral sink 和 operation，不引入 SSE 专有字段。

## 7. 实施顺序

```text
A1  CloudTask operation/sink/summary contract
A2  IAsyncLlmClient fan-out + ordered incremental drain
A3  cancellation/shutdown/multi-task admission tests
A4  existing Execute/future compatibility verification

B1  模块级 SQLite Repository contract 盘点与缺口补齐
B2  SQLite migration runner and component sources
B3  Auth/Persona/Document/Vector SQLite adapter 逐模块接线
B4  Runtime Session repository DTO/contracts

C1  SQLite Semantic Cache schema/repair 分离
C2  SQLite registry/checkpoint vertical E2E

D1  SSE protocol review
D2  HTTP streaming writer and CloudTask adapter
```

每个阶段都先通过单元测试，再补对应 E2E。性能验证以防止明显回归和确认资源上限为主，不在当前阶段扩展新的 benchmark framework。

## 8. 完成标准

- CloudTask 新异步路径不使用同步 LLM 等待 worker；
- ordered incremental sink、取消、终态和多 task admission 全部有测试；
- SQLite migration 与业务 Repository DDL 职责分离；
- Auth、Persona、Document、Vector 等 SQL 模块均可通过领域接口替换 backend；
- PostgreSQL Runtime Session 仍是后续可选 adapter，不作为当前完成条件；
- L0 必须保持 Session-owned `VectorIndexManager` 生命周期，不允许提升为 User/Tenant 共享状态；
- SSE 在协议评审完成前不侵入 CloudTask 领域模型；
- 所有异常映射为 `core::Status`，业务失败输出结构化日志；
- 所有文件、配置、测试输入和报告默认使用 UTF-8。

## 9. 相关入口

- `src/llm/cloud_task_coordinator.h`
- `src/llm/openai_llm_client.h`
- `src/core/ordered_bitmap_window.h`
- `src/storage/sqlite/`
- `docs/performance/BENCHMARK_ARCHITECTURE_DECISION_2026_08.md`
