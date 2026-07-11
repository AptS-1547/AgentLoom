# Skill 有限媒体流 Drain 与映射缓存设计

## 1. 文档目的

本文档定义 `vision.observe` 等有限媒体 Skill 的 selected-frame 保留、推理端映射缓存、流结束 drain、事件聚合和 Skill 完成协议。

设计前提是：

- 所有多模态媒体管线都绑定一次有限的 Skill Session；
- `SkillSessionOptions::max_duration` 或请求级 `max_duration` 为硬生命周期边界；
- 不支持无限实时监控，也不允许用异常大的 `max_duration` 等价构造无限流；
- sampler 命中后的帧是高价值稀疏输入，正常路径不再使用 `RejectNewest` 静默丢弃；
- shared-memory IPC 是短生命周期数据面，不承担长期积压；
- 推理端 private mapped spool 承担 Skill 生命周期内尚未处理的 encoded frame；
- Skill 只有在输入 sealed、全部 accepted frame 获得 terminal outcome、事件聚合完成后才能正常 `Closed`。

本文档是设计约束，不表示相关接口和 mmap spool 已经完成实现。

## 2. 与当前 Skill 实现的校准

当前 Skill Session 代码位于：

```text
src/service/persona/skill_session_manager.h
src/service/persona/skill_session_manager.cpp
src/service/persona/skill_vision_event_sink.h
src/service/persona/skill_vision_event_sink.cpp
src/service/gateway/persona_gateway_agent_routes.cpp
src/service/gateway/runtime_maintenance_service.cpp
```

当前已经实现：

| 能力 | 当前行为 |
|------|----------|
| Session key | 使用 `(session_id, skill_id)` 组合键 |
| Start | 新 session 进入 `Starting`；活动 session 重复 Start 为幂等刷新 |
| Ready | `MarkReady()` 将非 terminal session 设置为 `Ready` |
| Observation | `RecordObservation()` 保存最近 observation，并将状态设置为 `Running` |
| Stop | `Stop()` 当前直接设置为 `Closed` |
| Failure | `MarkFailed()` 直接进入 `Failed` |
| Expiration | maintenance 根据 startup/max-duration/idle/closing timeout 设置 `Expired` |
| Gateway control | HTTP/WS 已提供 start、stop、status |
| Prompt injection | Persona Runtime 可注入 Skill status 或最近 observation |
| Vision bridge | `SkillVisionEventSink` 可把 `VisionEvent` 写为 `SkillObservation` |

当前状态枚举已经包含：

```text
Idle
Starting
Ready
Running
WaitingInput
Closing
Closed
Failed
Expired
```

但当前实现仍有以下缺口：

1. `Stop()` 没有进入 `Closing`，而是直接进入 `Closed`。
2. 没有异步 close participant、drain handle 或 completion fence。
3. `(session_id, skill_id)` 没有 execution generation；terminal session 重新 Start 时会复用同一 key。
4. `RecordObservation()` 在 `Closing` 状态仍会把 session 改回 `Running`。
5. `MarkReady()` 在 `Closing` 状态仍可能把 session 改回 `Ready`。
6. `summarize` 和 `write_l3` 已存在于 stop request，但 manager 当前没有执行对应的 close workflow。
7. `closing_timeout` 已配置并被 maintenance 检查，但正常 Stop 不会留下 `Closing` 状态，因此该超时目前主要是预留能力。
8. 当前 `SessionInferenceFrameResultTable` 是内存结果表，不适合在较长 Skill 内无限保存完整 VLM result。

因此，本设计不改变现有代码事实，而是定义实现 mapped spool 前需要补齐的最小 Skill 生命周期扩展。

## 3. 核心不变量

### 3.1 有限流

每次媒体处理必须属于一个明确的 Skill execution：

```text
skill_id
session_id
execution_id
max_duration
```

创建时必须能确定资源预算。超过 `max_duration`、spool hard limit 或 drain deadline 时，Skill 应明确进入 `Failed` 或 `Expired`，不能转换为无期限后台任务。

### 3.2 Selected frame 无静默丢失

对 sampler 已选中的帧，目标语义为：

```text
每个被 Gateway 接受的 selected frame
-> 被推理端 mapped spool 接纳
-> 获得一次 terminal success/failure
```

必须满足：

```text
silent_drop_count == 0
terminal_frames == admitted_frames
```

VLM 业务失败、deadline、解析失败可以形成 terminal failure，但不能无记录消失。

### 3.3 Skill 正常完成屏障

正常 `Closed` 必须满足：

```text
input_sealed
&& admitted_frames == expected_selected_frames
&& ready_frames == 0
&& claimed_frames == 0
&& terminal_frames == expected_selected_frames
&& aggregation_pending == 0
&& final_result_published
```

### 3.4 Shared-memory slot 不承载 backlog

shared-memory IPC 只负责跨进程短时传输：

```text
Gateway encoded frame
-> shared slot
-> inference mapped spool
-> ack shared slot
```

VLM worker、事件聚合器和 Skill manager 不得持有 shared slot span，也不得覆盖 claimed/in-flight slot。

## 4. Execution Identity

当前 `(session_id, skill_id)` 只能标识逻辑 Skill，不能区分同一会话中先后两次执行。mapped spool 和异步回调必须增加：

```cpp
std::string execution_id;
```

建议由 backend 在 `Start()` 创建新执行时生成，并返回到 Gateway、media pipeline 和 inference control plane。

完整 identity：

```text
SkillExecutionKey = session_id + skill_id + execution_id
```

每个 selected frame 还应增加连续序号：

```cpp
std::uint64_t selected_sequence;
```

要求：

```text
selected_sequence = 1..N
```

`frame_id` 继续表示媒体帧身份；`selected_sequence` 只用于本次 Skill 的 admission、Seal 和 drain 完整性校验。

迟到的旧 execution frame、result 或 drain notification 必须根据 `execution_id` 被拒绝，不能写入重新启动后的 Skill。

## 5. 生命周期扩展

### 5.1 当前状态与计划语义

```text
Starting
  创建 Skill execution，初始化 media/VLM 资源。

Ready
  资源就绪，可以接受媒体输入。

Running
  正在接受帧并持续产生 provisional observation。

Closing
  已停止接受新的 selected frame，正在 flush publish、Seal、drain 和 finalize。

Closed
  全部 accepted frame 已 terminal，final result 已发布，资源已释放。

Failed
  关键错误导致无法形成正常完整结果。

Expired
  startup/max-duration/idle/closing timeout 触发强制终止。
```

`WaitingInput` 保留给需要用户补充参数的通用 Skill，不用于媒体 drain。

### 5.2 正常状态转换

```text
Starting -> Ready
Ready -> Running
Running -> Closing
Closing -> Closed
```

允许没有 observation 的短 Skill：

```text
Ready -> Closing -> Closed
```

### 5.3 Stop 迁移

当前 `ISkillSessionManager::Stop()` 直接进入 `Closed`。迁移建议：

1. 增加显式 `BeginClosing()` 与 `CompleteClosing()`。
2. Gateway 的媒体 Skill stop route 先调用 `BeginClosing()`。
3. media close participant 异步完成 flush、Seal、drain 和 aggregation。
4. participant 成功后调用 `CompleteClosing()`。
5. 简单、无异步资源的 Skill 可以继续由兼容 `Stop()` 执行 `BeginClosing() + CompleteClosing()`。

建议接口草案：

```cpp
struct SkillSessionCloseRequest {
    std::string skill_id;
    std::string session_id;
    std::string execution_id;
    std::string authenticated_user_uuid;
    std::string trace_id;
    std::string source;
    std::string reason;
    bool summarize = true;
    bool write_l3 = false;
};

class ISkillSessionManager {
public:
    virtual core::Result<SkillSessionSnapshot> BeginClosing(
        const SkillSessionCloseRequest& request) = 0;

    virtual core::Result<SkillSessionSnapshot> CompleteClosing(
        std::string_view session_id,
        std::string_view skill_id,
        std::string_view execution_id,
        std::string status_text,
        std::string_view trace_id) = 0;
};
```

具体命名可在实现时调整，但必须保留“请求关闭”和“关闭完成”两个不同动作。

### 5.4 Closing 期间的 observation

关闭期间允许 coordinator/aggregator 发布最后一批 terminal observation 或 final result，但不得让状态退回 `Running`。

计划规则：

```text
RecordObservation(Closing):
  接受 observation
  更新 recent_observations/last_activity
  保持 Closing

MarkReady(Closing):
  FailedPrecondition
```

## 6. 数据面

### 6.1 完整路径

```text
RTC decoded RGB
-> OpenCvFrameSampler
-> selected_sequence allocation
-> JPEG/PNG encoder
-> Gateway pending publish queue
-> shared-memory IPC
-> InferenceFrameIpcReceiver
-> MappedInferenceFrameStore
-> StoredFrameHandle ready queue
-> InferenceFrameCoordinator
-> IVlmVisionClient
-> incremental event aggregator
-> Skill final result
```

### 6.2 唯一一次私有拷贝

当前 receiver 执行：

```text
shared slot -> core::MemoryBlock -> private backlog
```

mapped spool 落地后应调整为：

```text
shared slot -> mapped record payload
```

mmap record 成为推理端 private input buffer。写入完成并发布 record 后立即 ack shared slot。

### 6.3 Gateway publish backpressure

shared ring 临时满时，`InferenceFrameGatewayProducer::Publish()` 当前返回 `ResourceExhausted`。有限 Skill 的正常策略应改成：

```text
publish ResourceExhausted
-> encoded frame 保留在 per-execution pending queue
-> 等待 slot credit/wake
-> 重新调度 IO task
```

背压应传递给 Gateway IO 调度和 Skill 状态，但不能让 IO worker 同步阻塞到整个 Skill drain 完成。推荐使用 delayed completion、wake signal 或重新提交任务。

在发送 Seal 前必须保证：

```text
所有 selected_sequence 1..N 均成功 publish
```

如果 inference 不可用、重试超时或 Gateway pending queue 达到 hard limit，应让 Skill 明确失败，而不是丢帧后继续返回成功。

## 7. Mapped Inference Frame Store

### 7.1 接口边界

建议新增：

```cpp
class IInferenceFrameStore {
public:
    virtual ~IInferenceFrameStore() = default;

    virtual core::Result<StoredFrameHandle> Admit(
        const StoredFrameMetadata& metadata,
        std::span<const std::byte> encoded_payload) = 0;

    virtual core::Result<StoredFrameLease> WaitClaim(
        std::chrono::milliseconds timeout) = 0;

    virtual core::Status Complete(
        StoredFrameLease lease,
        const core::Status& terminal_status) = 0;

    virtual core::Status SealExecution(
        const SkillMediaSealRequest& request) = 0;

    virtual core::Result<SkillMediaDrainSnapshot> Snapshot(
        const SkillExecutionKey& key) const = 0;

    virtual core::Status CancelExecution(
        const SkillExecutionKey& key) = 0;

    virtual void Shutdown() = 0;
};
```

业务接口不暴露裸 mmap 地址。`StoredFrameLease` 必须 move-only RAII，并只提供同步 `Analyze()` 期间有效的只读 `std::span<const std::byte>`。

### 7.2 分段文件

使用固定大小 file-backed mapped segments：

```text
media_spool_root/
  <execution_id>/
    manifest
    segment-000000.bin
    segment-000001.bin
```

推荐 segment 大小通过配置确定，例如 64-256 MiB。不要使用单个持续扩容并反复 remap 的大文件。

每个 segment：

```text
SegmentHeader
RecordHeader[N]
EncodedPayloadArea
```

record metadata 至少包含：

```text
execution_id
session_id
skill_id
selected_sequence
frame_id
timestamp_us
trace_id
format
saliency
payload_offset
payload_size
checksum
state
```

record 状态：

```text
Writing -> Ready -> Claimed -> Terminal
```

一个 segment 的所有 record 都进入 `Terminal` 后：

```text
unmap -> close -> delete/recycle
```

Windows 必须先解除 mapping 和文件句柄，再删除文件。映射与文件生命周期必须由 RAII 封装；实现应使用 Boost 跨平台能力，不直接依赖单一平台 API。

### 7.3 临时性与恢复

第一版可以把 spool 定义为 Skill execution 临时存储：

- 正常完成后立即清理；
- Cancel/Failed/Expired 后清理；
- inference 进程崩溃时，由 Gateway/Skill supervisor 将 execution 标记失败；
- 服务启动时清理不属于活动 execution 的残留目录。

如果后续要求 inference 重启后继续 drain，再增加 versioned manifest、checksum 和 `Claimed -> Ready` 恢复；第一版不应同时承担完整 crash-recovery journal。

## 8. Seal 与 Drain 控制协议

### 8.1 为什么需要 expected count

shared-memory data plane 与 gRPC control plane 独立运行。最后一帧可能仍在 ring 或 receiver 中时，Seal RPC 已经到达 inference。

因此 Seal 不能解释为“当前队列已经完整”，而只能解释为“Gateway 不会再产生新的 selected frame”。

```cpp
struct SkillMediaSealRequest {
    std::string skill_id;
    std::string session_id;
    std::string execution_id;
    std::uint64_t expected_selected_frames = 0;
    std::uint64_t last_selected_sequence = 0;
    std::string trace_id;
};
```

要求：

```text
expected_selected_frames == last_selected_sequence == N
```

inference 收到 Seal 后继续接收尚在数据面的 record，直到 admitted count 达到 N。

### 8.2 Drain snapshot

```cpp
struct SkillMediaDrainSnapshot {
    std::uint64_t expected_frames = 0;
    std::uint64_t admitted_frames = 0;
    std::uint64_t ready_frames = 0;
    std::uint64_t claimed_frames = 0;
    std::uint64_t successful_frames = 0;
    std::uint64_t failed_frames = 0;
    std::uint64_t terminal_frames = 0;
    std::uint64_t spool_bytes = 0;
    std::uint64_t peak_spool_bytes = 0;
    bool sealed = false;
    bool aggregation_pending = false;
    bool final_result_published = false;
};
```

### 8.3 gRPC 职责

现有 `MultimodalInference` proto 只有逐请求 `GenerateVLM` / `GenerateVLMSync`，尚无媒体 Skill 控制 RPC。计划新增控制面，帧 payload 继续留在 shared-memory IPC。

控制操作至少包括：

```text
OpenSkillMediaExecution
SealSkillMediaInput
GetSkillMediaDrainStatus
CancelSkillMediaExecution
```

完成通知可以使用异步 server-streaming、双向 control stream，或 Gateway 定时查询。第一版可以先使用 unary Seal + status polling，降低实现复杂度；不能让 Gateway IO worker 阻塞等待 RPC 长连接完成。

### 8.4 完成顺序

```text
Gateway BeginClosing
-> 停止 sampler admission
-> flush compute/encode tasks
-> flush pending IPC publish queue
-> SealSkillMediaInput(N)
-> inference admitted == N
-> coordinator terminal == N
-> event aggregator finalize
-> final observation/result publish
-> CompleteClosing
-> Skill Closed
```

## 9. Coordinator、结果表与事件聚合

当前 `InferenceFrameCoordinator` 已能：

- 从 private backlog 并发取帧；
- 同步调用 `IVlmVisionClient::Analyze()`；
- 把成功、业务失败和异常清洗为 terminal record；
- 统计 processed/success/failure/result-publish-failure。

接入 mapped store 后：

```text
OwnedInferenceFrame
-> StoredFrameLease
```

VLM worker 在同步 `Analyze()` 期间持有 lease，完成后调用 store `Complete()`。

事件聚合应增量消费 terminal result：

```text
OnFrameTerminal(result)
-> 更新 execution 内事件窗口
-> 释放不再需要的完整 result payload
```

Seal drain 完成后：

```text
FinalizeExecution(execution_id)
-> 最终有序 VisionEvent/SkillObservation
```

当前 `SessionInferenceFrameResultTable` 可继续用于单元测试、短 session 和排序验证，但不应成为所有有限媒体 Skill 的长期完整结果仓库。实现 mapped spool 时应增加 incremental result consumer，或为结果提供独立有界 spool。

## 10. 取消、失败与超时

### 10.1 正常 Stop

正常 Stop 表示输入结束并要求完整 drain，不等价于 Cancel：

```text
Running -> Closing -> Closed
```

### 10.2 Cancel

第一版建议只实现明确的立即取消：

```text
停止新 admission
取消尚未 claim 的 record
等待 claimed Analyze 安全返回或达到 deadline
不生成伪完整 final result
清理 spool
-> Failed 或 Expired
```

如果未来支持 partial result，必须显式携带：

```text
complete = false
terminal_frames < admitted_frames
```

### 10.3 Timeout 对齐

现有 Skill timeout 映射：

| 配置 | 媒体含义 |
|------|----------|
| `startup_timeout` | media pipeline、IPC 和 inference execution 初始化超时 |
| `max_duration` | Skill 接受媒体输入的总时长硬上限 |
| `idle_timeout` | 无输入、无 observation、无 drain progress 的空闲超时 |
| `closing_timeout` | flush、Seal、VLM drain、aggregation 和 cleanup 总超时 |

进入 `Closing` 后不再使用 `max_duration` 判断；由 `closing_timeout` 约束 drain。每次有效 drain progress 可以更新 activity，但不能无限延长 `closing_timeout` 的绝对 deadline。

### 10.4 资源耗尽

有限流仍需配置 hard limits：

```text
max_selected_frames_per_execution
max_spool_bytes_per_execution
max_spool_bytes_total
min_free_disk_bytes
max_gateway_pending_publish_bytes
```

这些限制用于 admission 和异常保护，不是正常丢帧阈值。资源无法保证时应在 Start/Open 阶段拒绝 execution；运行中触发 hard limit 时应让 Skill 明确失败。

## 11. Skill Snapshot 与 Gateway 状态

通用 `SkillSessionSnapshot` 不应直接塞入大量 media-only 字段。建议增加 profile progress JSON 或独立结构，并由 Gateway status envelope 输出：

```json
{
  "state": "closing",
  "statusText": "vision frames draining",
  "executionId": "...",
  "progress": {
    "stage": "draining",
    "expectedFrames": 147,
    "admittedFrames": 147,
    "terminalFrames": 96,
    "spoolBytes": 18273412,
    "peakSpoolBytes": 41781231
  }
}
```

Persona Prompt 在 `Closing` 时只能说明 Skill 正在完成处理，不能把 provisional observation 伪装为最终结果。`Closed` 后再注入 final observation/result。

## 12. 配置建议

新增配置应继续进入 `skill_session` 或独立 `media_skill` section：

```text
media_skill.spool_root
media_skill.segment_bytes
media_skill.max_selected_frames_per_execution
media_skill.max_spool_bytes_per_execution
media_skill.max_spool_bytes_total
media_skill.min_free_disk_bytes
media_skill.max_gateway_pending_publish_bytes
media_skill.publish_retry_timeout_ms
media_skill.drain_poll_interval_ms
```

路径按 UTF-8 处理，日志展示不得依赖系统本地编码。`spool_root` 必须经过规范化和 root 约束，execution ID 不能直接作为未校验路径片段。

## 13. 可观测性与验收指标

核心计数：

```text
rtc_generated
sampler_selected
gateway_publish_pending
ipc_published
spool_admitted
vlm_started
vlm_succeeded
vlm_failed
terminal_frames
silent_drop_count
```

容量与延迟：

```text
gateway_pending_publish_high_water
spool_peak_bytes
spool_segment_count
spool_write p50/p95/p99
oldest_pending_age
selected_to_vlm_start p50/p95/p99
input_seal_to_drained latency
event_finalize latency
total Skill completion latency
segment_reclaim latency
```

正常完成必须满足：

```text
sampler_selected == spool_admitted
spool_admitted == terminal_frames
silent_drop_count == 0
selected_event_coverage == 100%
```

## 14. 实施顺序

### Phase A：Skill 生命周期

1. 增加 `execution_id`。
2. 增加 `BeginClosing()` / `CompleteClosing()`。
3. 修正 Closing 状态下 `RecordObservation()` 和 `MarkReady()` 的转移规则。
4. 为 stop 幂等、重复 close、旧 execution 回调和 closing timeout 增加测试。

### Phase B：Mapped store

1. 定义 `IInferenceFrameStore`、`StoredFrameHandle` 和 `StoredFrameLease`。
2. 实现固定 segment file-backed mmap store。
3. receiver 改为 `shared slot -> mapped record`。
4. 实现 terminal reclaim、execution cleanup 和残留目录清理。

### Phase C：Seal 与 coordinator

1. metadata 增加 `execution_id` 和 `selected_sequence`。
2. coordinator 消费 `StoredFrameLease`。
3. 增加 Seal expected-count fence。
4. 增加 drain snapshot 和 final aggregation barrier。

### Phase D：Gateway control

1. 增加 per-execution pending publish queue。
2. ring 满时异步重试，不阻塞 IO worker。
3. 增加 inference gRPC media control RPC。
4. Skill stop route 接入 BeginClosing -> Seal -> CompleteClosing。

### Phase E：测试

1. finite-stream zero-silent-drop 准 E2E。
2. shared ring 满与 publish retry。
3. gRPC Seal 先于最后一帧到达。
4. 多 session/execution 隔离。
5. spool hard limit、磁盘写失败和 closing timeout。
6. Cancel、重复 Stop、重复 Seal、迟到 frame/result。
7. segment reclaim 与 Windows unmap-before-delete。
8. 真实双进程 Gateway/inference finite Skill drain E2E。

## 15. 非目标

第一版不实现：

- 无限媒体流；
- 等价超长有限流；
- shared-memory claimed slot 覆盖；
- sampler-selected frame 的普通 RejectNewest；
- inference 崩溃后的完整 mmap journal 恢复；
- 在 Gateway IO worker 中同步等待整个 Skill drain；
- 未完整 drain 时伪装成正常 `Closed`；
- 将单帧 VLM 输出直接写入情绪状态机或 L3 长期事实。

## 16. 当前结论

基于现有 Skill 生命周期和准 E2E 压测，正式方向调整为：

```text
有限 Skill input
-> selected frame 无静默丢失
-> inference private mapped spool
-> input Seal
-> VLM drain
-> incremental event aggregation
-> final result
-> Skill Closed
```

现有 `SegmentedInferenceFrameBacklog::RejectNewest` 保留为当前实现和 benchmark 基线，但不作为有限媒体 Skill 的目标生产策略。实现工作应先补齐 Skill `Closing` 生命周期，再替换 private backlog payload ownership，最后接入 gRPC Seal/Drain 控制面。

---

**文档状态**: 设计完成，代码待实现  
**最后更新**: 2026-07-11
