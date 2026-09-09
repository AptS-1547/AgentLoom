# 记忆遗忘算法迁移前设计（2026-09）

> 日期：2026-09-09
>
> 状态：迁移前评审，不表示遗忘维护已在 AgentLoom 启用
>
> 来源：早期 Python 记忆系统的遗忘、恢复和阈值实验

## 1. 目标

早期探索实现已经验证了一套比固定 TTL 更完整的记忆生命周期：

```text
active -> forgotten -> recovery window -> reactivated
                              \-> deleted（尚未完成）
```

当前目标不是重新发明遗忘算法，而是判断哪些语义值得迁入 AgentLoom，
并适配当前 C++ 的 tenant-aware repository、异步 Runtime、maintenance 和可观测性。

本设计坚持先 dry-run、后逻辑遗忘、最后物理删除。任何阶段都不得因为已有字段和 CRUD
就推断生产遗忘功能已经启用。

## 2. 早期实践中值得保留的部分

### 2.1 生命周期元数据

早期实现显式保存：

```text
memory_level / memory_type
lifecycle_created_at
last_recalled_at / recall_count
forgotten / forgotten_at / deleted_after / forget_epoch
emotion / emotion_intensity
state_valence / state_arousal
behavior / context_id
memory_hash / content_hash / backend id
```

其中最重要的经验是：只有最终进入 Prompt 的 L2/L3 记忆才算有效召回，向量检索候选、
阈值淘汰项和重复 recovery 不得增加 `recall_count`。

### 2.2 保留权重

早期基线公式为：

```text
W = W_base
    * exp(-lambda * delta_days)
    * (1 + alpha * log1p(recall_count))
    * M_encode
    * M_scan
```

- `delta_days` 从最后有效召回计算；从未召回时使用创建时间；
- L2 的基础权重更低、时间衰减更快；
- `log1p(recall_count)` 防止高频记忆永久固化；
- `M_encode` 反映写入时的情绪强度和 arousal；
- `M_scan` 只做维护时的小幅全局调制，不能支配单条记忆；
- 所有倍率和最终概率均有上下界。

### 2.3 硬保护和可恢复遗忘

早期实现默认保护：

- L4；
- `capability`、`profile`、`safety`、`active_task`；
- 最小保留期内的新记忆。

遗忘先写逻辑状态，普通召回过滤 `forgotten=true`。在 `deleted_after` 之前，
显式历史查询或 active 召回不足可以进入独立 recovery search；成功恢复后撤销 forgotten 状态、
清空删除期限并增加一次有效召回。

### 2.4 Dry-run 和分布测试

早期实现将维护决策输出为逐条 plan，并验证：

- 遗忘率随年龄区间整体单调上升；
- 新记忆和硬保护类型不被误伤；
- forgotten 在普通召回中不可见；
- recovery 在悔过期内有效；
- 生命周期 metadata 能与向量后端往返同步。

这些不变量比具体默认参数更值得迁移。

## 3. 当前 AgentLoom 已具备的基础

当前 C++ vector repository 已有：

```text
recall_count / last_recalled_at_ms
forgotten / forgotten_at_ms
deleted_after_ms / forget_epoch
expires_at_ms
memory_type / emotion / emotion_intensity / state_arousal
```

并提供：

```text
MarkRecalled
MarkForgotten
Reactivate
DeleteEntry
```

`VectorIndexManager` 普通查询也能排除 forgotten entry。因此迁移重点不是 schema CRUD，
而是生命周期策略、有效召回反馈、维护租约、索引一致性、恢复入口和测试门禁。

2026-09-09 的 owner 修复已经要求 L3 使用 `(tenant_id, user_id)` 分区；遗忘扫描、
恢复和审计必须沿用完整 owner，不能退回早期仅 `user_id` 的假设。

## 4. 不能直接复制的架构差异

| 早期探索实现 | 当前 AgentLoom | 迁移要求 |
| --- | --- | --- |
| Python manager 集中编排 | Persona、repository、index、maintenance 分层 | 算法做纯策略，I/O 留给维护任务 |
| Mem0/Qdrant payload 同步 | SQLite repository 为事实源，内存索引可重建 | 不新增旁路 lifecycle JSON store |
| 主要按 `user_id` 隔离 | tenant-aware `(tenant_id,user_id)` | 所有 API 和日志必须保留 tenant |
| 进程内锁 | 可能多进程/多节点维护 | 需要 repository lease 或 CAS |
| 同步调用较多 | callback/线程池异步 Runtime | maintenance 不占 Turn worker |
| L2/L3/L4 分层 | 当前文档收敛为 L0/L3/L4 | 先重新映射层级语义 |
| Python random | C++ 可重放审计 | seed、版本和输入快照必须持久化 |

### 4.1 层级映射不能机械替换

早期 L1 是当前窗口，L2 是会话状态快照，L3 是长期事实，L4 是稳定知识/画像。
当前 AgentLoom 的 L0 同时承担最近会话和短期语义上下文，不能简单把旧 L2 参数套给 L0。

建议候选边界：

- L0 使用确定性 Session retention、TTL 和物理容量治理；
- L3 使用 retention weight、逻辑遗忘和 recovery；
- L4 不参加随机遗忘，仅允许显式 supersede、版本失效和策略下线；
- 若未来重新引入独立 state snapshot 类型，再评估是否使用旧 L2 衰减参数。

该边界需要基准数据确认，当前不写死为实现常量。

## 5. 建议模块边界

遗忘计算应是无 I/O 的可测试策略：

```cpp
class IMemoryForgettingPolicy {
public:
    virtual ~IMemoryForgettingPolicy() = default;

    virtual core::Result<ForgettingPlan> BuildPlan(
        std::span<const MemoryLifecycleRecord> records,
        const ForgettingContext& context) const = 0;
};
```

维护任务负责：

```text
获取 tenant/user lease
  -> 读取一致的 active snapshot
  -> BuildPlan
  -> 写 dry-run artifact / metrics
  -> policy gate
  -> repository MarkForgotten
  -> 通知或失效 resident index
  -> 提交 maintenance checkpoint
```

策略对象不得持有 SQLite connection、Redis client、线程池或全局 registry。

## 6. 分阶段迁移

### Phase 0：语义冻结

- 固定 `MemoryLifecycleRecord` 和 `ForgettingDecision` schema；
- 明确 Unix 毫秒、业务时区和扫描窗口；
- 从早期实现导出固定输入、seed 和期望决策；
- 冻结 hard-keep、权重、桶边界和概率公式版本；
- 给算法和配置计算 hash。

退出门槛：同一输入、seed 和版本在 Windows/Linux 得到相同 plan。

### Phase 1：C++ 纯策略与 dry-run

- 实现 `IMemoryForgettingPolicy`；
- maintenance 只读取 snapshot 和输出 plan，不写 forgotten；
- 记录每个 owner 的候选量、权重分布、选择量和 hard-keep 原因；
- 运行至少一个完整 retention 窗口并与 Python 基线对比。

退出门槛：年龄分布、保护率和遗忘候选率满足预设容差，且无跨 tenant 数据。

### Phase 2：有效召回反馈

- 仅对最终进入 Prompt 的 L3 entry 调用 `MarkRecalled`；
- 一个 Turn 内重复命中只计一次；
- recovery 自己计一次并防止普通路径重复计数；
- 写失败不能阻塞已成功响应，但必须重试或形成可恢复事件。

退出门槛：并发、重试、late callback 和进程重启下计数不重复。

### Phase 3：逻辑遗忘

- 默认关闭，通过配置和 tenant allowlist 灰度；
- 使用 repository CAS/lease 防止重复扫描；
- `MarkForgotten` 与 maintenance checkpoint 保持可恢复顺序；
- resident index 必须同步移除或因 snapshot 版本变化重新 hydrate；
- 所有决策写入可审计 artifact。

退出门槛：forgotten 普通召回不可见，任务重试幂等，关闭开关可立即停止新扫描。

### Phase 4：Recovery

- 独立查询 forgotten entry，不放宽普通召回；
- 同时满足历史意图、独立阈值和 `deleted_after`；
- 增加 tenant/user rate limit；
- `Reactivate`、索引通知和 recall count 保持一致；
- 对用户是否表达“想起来了”由 Persona policy 决定，不由存储层生成。

退出门槛：无法跨 tenant、无法恢复过期项、并发恢复幂等。

### Phase 5：延迟物理删除

物理删除是单独项目，不随逻辑遗忘自动开启。至少需要：

- deletion audit log；
- legal hold / safety / active policy 硬保护；
- repository 批量删除和 index `NotifyDeleted`；
- 删除前 snapshot/备份与恢复演练；
- 多节点 lease；
- 明确数据保留和隐私政策。

## 7. 测试矩阵

### 单元测试

- retention weight 各因子和 clamp；
- `last_recalled_at` 与创建时间 anchor；
- `log1p(recall_count)`；
- hard-keep；
- 固定 seed 决策；
- 空候选、非法时间和极端数值。

### 分布与属性测试

- 年龄越高，分桶遗忘率整体不下降；
- recall count 增加不会降低保留权重；
- L4 和硬保护类型选择率恒为零；
- 相同输入/seed/version 可重放；
- tenant 重命名不改变算法结果，但改变存储 owner。

### 集成与 E2E

- SQLite lifecycle 往返；
- exact/Faiss resident index 在 forgotten/reactivate 后一致；
- maintenance crash/restart；
- 同 owner 双任务 lease 冲突；
- 不同 tenant 同 user ID 隔离；
- recovery window 和 rate limit；
- Application Session/Persona 真实召回只更新已注入 entry。

### 压测

- 10k/100k/1M entry 扫描时间和内存；
- 多 tenant 公平性；
- repository writer 饱和时的 batch 和 backpressure；
- maintenance 与在线 L3 search 并行时的 P95/P99。

## 8. 审计字段

每次扫描至少记录：

```text
scan_id / policy_version / random_seed
tenant_id / user_id
snapshot_revision
started_at_ms / finished_at_ms
candidate_count / protected_count / selected_count
avg_weight / min_weight / max_weight
dry_run / applied_count / conflict_count / failure_count
```

每条决策至少记录：

```text
memory_hash / entry_id
memory_level / memory_type
created_at_ms / last_recalled_at_ms / recall_count
retention_weight / forget_pressure / tree_depth / prune_probability
hard_keep_reason / final_action
```

日志不得包含完整敏感 memory payload。

## 9. 待确认决策

1. 当前 L0 是否完全排除随机遗忘，只采用 TTL/容量治理；
2. 旧 L2 状态快照在当前架构中是否需要独立 memory type；
3. `M_scan` 是否仍使用当前 Persona 情绪，还是只保留写入时显著性；
4. 随机桶剪枝是否保留，或改为确定性 threshold + 配额；
5. maintenance lease 落 SQLite、Redis 还是 Runtime persistence contract；
6. dry-run 观察窗口和允许的误伤率；
7. recovery 的用户意图识别由规则、Skill 还是 Persona Runtime 负责；
8. physical deletion 的审批、保留期和回滚要求。

## 10. 当前禁止事项

- 未经 dry-run 直接开启 `MarkForgotten`；
- 将 Python 配置默认值直接作为生产常量；
- 用 `user_id` 代替 `(tenant_id,user_id)`；
- 新建旁路 lifecycle JSON store；
- 将 L4 capability、safety 或 active policy 纳入随机遗忘；
- 在没有 lease/CAS 和审计 artifact 时运行多节点维护；
- 把逻辑遗忘等同于立即物理删除。

