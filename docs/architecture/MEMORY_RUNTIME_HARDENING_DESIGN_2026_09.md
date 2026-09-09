# Memory Runtime 加固设计（2026-09）

> 日期：2026-09-09
>
> 状态：讨论基线，读取降级与 Prompt 信任边界尚未实现
>
> 范围：Persona Runtime 的 L0/L3 读取、上下文注入、失败语义和可观测性

## 1. 背景与当前边界

当前在线链路为：

```text
Session recent history
  -> L0 session-scoped semantic lookup
  -> L3 tenant/user fact lookup
  -> memory context
  -> Persona system prompt
  -> LLM
  -> L0 turn admission
```

2026-09-09 已优先处理以下数据契约问题：

- L0 Redis v2 batch key 与 Unix 毫秒时间单位统一；
- L0 进程内索引和 release 使用完整 `(tenant_id, user_id, session_id)`；
- L0 Redis repair 使用对应的 v2 Session namespace；
- L0 到 L3 通过 `IL0RecordSource` 解耦，不再由 compressor 自行解释 Redis key；
- L3 API、registry 和 vector partition 使用 `(tenant_id, user_id)`；
- L0 admission 成功后登记可维护的 owner；
- L3 压缩输出采用严格 JSON string array 门禁、有限长度和幂等写入；
- L3 检索使用 fact 类型和最低相关性门槛；
- 通用 runtime maintenance 下沉到 gateway foundation，Auth cleanup 保留为 reference Gateway 私有实现；
- Application 在显式启用 `l3_flush_scheduler` 时复用同一 L3 maintenance task。

本设计不重新讨论上述 owner 和存储契约。本轮暂停并留待评审的是：

1. 记忆读取失败时，当前 Turn 应硬失败还是降级继续；
2. L0/L3 内容进入 system prompt 时采用什么信任模型。

## 2. 读取失败降级

### 2.1 当前行为

当前 `SemanticMemoryContextProvider` 的 L0 或 L3 查询失败会返回失败 `core::Status`，
`PersonaRuntime` 随后结束当前 Turn。该行为简单、可观察，并避免在依赖记忆的业务中无声改变语义；
代价是 Redis、SQLite、embedding 或 L3 index 的短暂故障会扩大为对话不可用。

文档中曾出现“记忆失败时跳过 L3、继续对话”的目标描述，但它尚不是当前代码行为。

### 2.2 不采用全量软降级

不能把所有错误统一转换为空记忆：

- owner、tenant、schema 或 fingerprint 不一致可能表示隔离或数据契约错误；
- `DataLoss` 可能表示持久化数据损坏；
- 训练任务可能声明某些策略或事实为本轮必需上下文；
- 静默降级会让上层误以为生成结果使用了完整记忆。

因此更合适的候选是“按层级、错误类型和业务要求分级”，而不是 catch-all。

### 2.3 候选决策矩阵

| 条件 | 候选行为 | 说明 |
| --- | --- | --- |
| `NotFound` / 合法空结果 | 正常继续 | 不属于故障 |
| L3 `Timeout` / `Unavailable` | 跳过 L3，保留 recent history 与可用 L0 | L3 是增强事实层 |
| L0 `Timeout` / `Unavailable` | 保留 Session recent history，标记 degraded | L0 与 recent history 不是同一来源 |
| `ResourceExhausted` | 有界降级或显式拒绝 | 需结合队列压力和 tenant quota |
| `InvalidArgument` / owner 不一致 | 硬失败 | 调用契约或隔离错误 |
| `FailedPrecondition` | 默认硬失败 | 常表示装配或生命周期错误 |
| `PermissionDenied` | 硬失败并安全审计 | 不得降级绕过权限 |
| `DataLoss` | 隔离数据并硬失败或受控降级 | 需先定义损坏记录隔离能力 |
| `InternalError` / `Unknown` | 默认硬失败 | 未分类前不应悄悄放行 |

最终策略还应受一次 Turn 的 `MemoryRequirement` 控制：

```text
Optional      记忆是增强项，允许按矩阵降级
RequiredL0    缺少 L0 时拒绝生成
RequiredL3    缺少指定版本 L3 时拒绝生成
RequiredAll   任一要求的记忆层失败即拒绝
```

该字段必须来自可信 Runtime/业务配置，不能由用户文本或 LLM 参数决定。

### 2.4 降级结果必须可见

若采用降级，至少需要在内部结果和指标中携带：

```text
memory_degraded
degraded_layers
degradation_reason_code
memory_snapshot_version
trace_id / tenant_id / session_id
```

外部是否展示降级提示由产品场景决定，但内部 usage/runtime event 不得丢失该事实。
降级响应不应进入通用答案缓存，避免把缺少关键上下文的回答复用为正常结果。

### 2.5 实施前测试门禁

- L0/L3 `NotFound` 不应失败；
- Redis/SQLite/embedding timeout 的各层降级矩阵；
- `PermissionDenied`、owner mismatch 和 schema error 始终硬失败；
- degraded response 不进入正常 answer cache；
- 同 Session 顺序、shutdown、late callback 和 exactly-once completion 不改变；
- 指标能区分“无记忆命中”和“记忆服务故障”。

## 3. Prompt 信任边界

### 3.1 当前风险

当前 L0 payload 和 L3 fact 会拼入 `memory.system_context`，随后成为 system prompt 的一部分。
这些内容可能来源于历史用户文本、模型回复、压缩模型输出或未来的 Skill result。

XML 风格标签只能帮助模型理解结构，不能建立安全边界。若历史文本包含“忽略之前要求”等指令，
它在召回后可能获得比原始 user message 更高的提示位置，形成持久化 Prompt injection。

### 3.2 记忆是数据，不是指令

建议冻结以下原则：

- L0/L3 召回内容默认是不可信历史数据；
- 记忆内容只能提供事实候选、上下文和 provenance，不能改变 Persona、安全或工具权限；
- 只有受控的 L4 capability/policy artifact 可以携带指令语义；
- 即使是 L4，也必须先通过版本、审批状态和 scope 校验；
- Skill result 进入 L0 后仍是工具事实，不自动成为运行指令。

### 3.3 候选消息布局

不建议继续把原始记忆字符串直接拼入长期 system instruction。候选结构为：

```text
system: 稳定 Persona、安全策略、记忆使用规则
developer/runtime context: 可信策略版本和业务约束
memory context: 类型化、有限、转义后的不可信数据
user: 当前输入
```

若 Provider 只支持 system/user/assistant/tool 四类角色，则应至少使用结构化 JSON envelope，
并在稳定 system policy 中明确声明 envelope 内字符串不得作为指令执行。

```json
{
  "kind": "recalled_memory",
  "trust": "untrusted_data",
  "source": "L0|L3|skill_result",
  "items": [
    {
      "content": "...",
      "score": 0.82,
      "created_at_ms": 0,
      "provenance": {}
    }
  ]
}
```

### 3.4 必要限制

- 对每层、每条和整批内容设 UTF-8 字节/token 上限；
- 使用结构化 JSON serializer，不手工拼接未转义字段；
- 不把检索分数、内部 ID 或敏感 provenance 无条件暴露给模型；
- 对客户原文、CRM 和 Skill result 做字段级脱敏；
- 保留 source、版本、时间和 stale 状态；
- Prompt 日志只记录 hash、计数和安全摘要，不记录完整敏感内容；
- 被召回不等于被采纳，最终进入 Prompt 后才更新有效召回计数。

### 3.5 实施前实验

需要覆盖直接注入、间接注入和跨 Turn 持久化：

- 历史用户文本要求忽略 system policy；
- 工具结果中包含伪造指令；
- L3 压缩结果重复攻击字符串；
- XML/JSON 闭合和转义攻击；
- 超长内容截断后结构仍合法；
- 正常事实召回率和回答质量不因防护显著下降。

## 4. 待确认决策

1. 哪些训练场景需要 `RequiredL3`，由 Go Bootstrap 还是 Application 配置声明；
2. L0 故障时仅使用 Session recent history 是否满足产品语义；
3. `InternalError` 是否允许按已知 provider 白名单降级；
4. memory context 使用独立 role、结构化 content part，还是兼容 system suffix；
5. Skill result、CRM 事实和策略卡是否使用不同 trust level；
6. degraded response 是否向学员展示，以及如何避免泄漏内部架构；
7. Prompt injection 防护的离线金标和允许的误拒绝率。
8. L3 flush 应压缩“已结束的前一业务日”还是允许当天增量快照；当前
   `flushed_dates_` 仅按日期去重，无法表达新 owner 或 per-owner checkpoint，启用前需改为
   `(tenant_id,user_id,date,source_snapshot)` 级幂等状态。

## 5. 暂缓项

在上述决策和测试门禁冻结前，不实施：

- catch-all 记忆读取软降级；
- 将任意记忆原文提升为可信 system 指令；
- 根据模型输出切换 `MemoryRequirement`；
- 把降级生成结果当作完整记忆结果缓存；
- 以关键词过滤替代结构化信任边界。
