# Visual Tool Session Protocol

本文档定义视觉工具接入主链路时的信令、状态机、Prompt 注入、资源回收和记忆写入边界。`vision.observe` 是通用 Skill Session Protocol 的第一个复杂 profile；通用协议见 `docs/architecture/SKILL_SESSION_PROTOCOL.md`。

目标是在复用现有 WebRTC / OpenCV / VLM / L4 Tool Memory 基础设施的同时，避免不成熟的多模态结果污染主对话链路。

## 1. 设计目标

视觉能力不是一次性函数调用，而是一个有生命周期的工具会话。它会占用 RTC、解码、抽帧、VLM IO、缓存和事件队列资源，因此必须具备明确的启动、就绪、运行、关闭、错误和超时回收协议。

核心目标：

- 视觉工具默认不常驻注入主链路。
- 用户显式请求、前端信令、本地正则/L4 向量近邻和 LLM 主动调用都可以触发视觉会话。
- 视觉链路未就绪前，不允许 LLM 编造画面内容。
- 视觉事件只以低置信工具观察形式注入 Prompt。
- 关闭时默认写 L4 工具会话摘要，不默认写 L3 长期记忆。
- 任何关键错误或超过极限处理时间时，守护线程必须关闭会话并释放资源。

## 2. 非目标

第一阶段不做以下内容：

- 不把 VLM 结果作为默认长期事实写入 L3。
- 不让视觉事件持续常驻主链路 Prompt。
- 不允许 LLM 无状态地直接“看见”画面。
- 不要求 DeepSeek 或其他 LLM 原生支持 Tool API。
- 不把 WebRTC session 和 chat session 混为同一个生命周期对象。

DeepSeek 没有原生 Skill API 时，通过结构化 Prompt 和可解析标签模拟工具调用。

## 3. 触发优先级

视觉工具会话的控制信号按优先级处理：

1. 前端显式信令
2. 用户输入的本地正则和 L4 向量近邻触发
3. LLM 主动结构化工具调用
4. 守护线程超时或错误兜底

显式前端信令优先级最高，因为它最接近用户授权和 UI 状态。本地正则/L4 向量近邻用于低成本、可解释触发。LLM 主动调用只作为补充，必须受工具会话状态机和策略约束。

## 4. 状态机

视觉工具会话状态：

```text
Idle
  未启动。

Starting
  已收到启动请求，正在准备 RTC / sampler / VLM 上下文。

Ready
  视觉链路已就绪，但尚未捕获可注入的稳定视觉事件。

Running
  正在接收并处理视觉事件，可能存在可注入观察结果。

Closing
  已收到关闭请求，停止接收新事件，正在 summary / 写入 L4 / 释放资源。

Closed
  正常关闭。

Failed
  出现关键错误，无法继续运行。

Expired
  被守护线程因超时、空闲或卡死状态强制关闭。
```

推荐枚举：

```cpp
enum class VisualToolSessionState {
    Idle,
    Starting,
    Ready,
    Running,
    Closing,
    Closed,
    Failed,
    Expired
};
```

主要状态转换：

```text
Idle -> Starting
  start signal / regex hit / L4 vector hit / LLM tool call

Starting -> Ready
  RTC/signaling/media pipeline ready

Ready -> Running
  first stable visual event or VLM observation accepted

Running -> Closing
  stop signal / user close request / LLM stop call

Closing -> Closed
  summary and cleanup complete

Starting|Ready|Running|Closing -> Failed
  critical error

Starting|Ready|Running|Closing -> Expired
  maintenance timeout
```

## 5. 信令协议

### 5.1 启动

```json
{
  "type": "vision.session.start",
  "session_id": "chat-session-001",
  "trace_id": "trace-xxx",
  "payload": {
    "source": "frontend|regex|vector|llm",
    "reason": "用户请求观察当前画面",
    "mode": "observe",
    "max_duration_ms": 120000
  }
}
```

处理规则：

- 如果会话不存在，创建 `VisualToolSession`。
- 如果会话已经 `Ready` 或 `Running`，刷新 `last_activity_at` 并返回当前状态。
- 如果会话处于 `Closing`，拒绝或等待关闭完成后重新启动。
- 如果会话处于 `Failed` / `Expired`，必须先完成清理再重新创建。

### 5.2 状态

```json
{
  "type": "vision.session.status",
  "session_id": "chat-session-001",
  "trace_id": "trace-xxx",
  "payload": {
    "state": "starting|ready|running|closing|closed|failed|expired",
    "can_observe": false,
    "message": "视觉链路正在启动",
    "error": ""
  }
}
```

### 5.3 观察结果

```json
{
  "type": "vision.observation",
  "session_id": "chat-session-001",
  "trace_id": "trace-xxx",
  "payload": {
    "summary": "画面中检测到明显移动，但无法确认具体对象。",
    "confidence": 0.62,
    "event_count": 2,
    "stale": false,
    "source": "sampler|monitor|vlm",
    "should_inject_prompt": true
  }
}
```

观察结果必须经过 gate：

- 过期结果不注入。
- 低置信结果只作为弱观察。
- 相似度过高的重复事件应去重。
- 涉及身份、性别、情绪、敏感属性的结果需要更高阈值或用户确认。

### 5.4 关闭

```json
{
  "type": "vision.session.stop",
  "session_id": "chat-session-001",
  "trace_id": "trace-xxx",
  "payload": {
    "source": "frontend|user|llm|daemon",
    "reason": "用户要求停止视觉观察",
    "summarize": true,
    "write_l3": false
  }
}
```

处理规则：

- 状态转为 `Closing`。
- 停止接收新视觉事件。
- 等待当前可取消 IO 任务尽快结束。
- 生成会话 summary。
- 默认写入 L4 工具会话状态，不默认写入 L3。
- 完成资源释放后转为 `Closed`。

### 5.5 错误

```json
{
  "type": "vision.session.error",
  "session_id": "chat-session-001",
  "trace_id": "trace-xxx",
  "payload": {
    "code": "TIMEOUT|RTC_FAILED|VLM_FAILED|NO_FRAME|INTERNAL",
    "message": "视觉链路超时，已自动关闭",
    "closed": true
  }
}
```

非关键错误只更新 `last_error`，不一定关闭会话。关键错误必须关闭并释放资源。

## 6. L4 Tool Memory 查询

L4 是全局工具/技能记忆，不按用户隔离。它通过 `memory_level="L4"` 与 L3 隔离：

```cpp
PartitionKey key;
key.collection_id = collection_id;
key.tenant_id = tenant_id;
key.user_id = "";        // global L4
key.memory_level = "L4";
```

L4 capability entry 示例：

```json
{
  "memory_type": "capability",
  "memory_hash": "global:l4:vision.observe:intent.camera",
  "payload": "用户想让助手观察画面、摄像头、屏幕、周围环境或当前动作时，使用 vision.observe 工具。",
  "extra_metadata_json": {
    "tool_id": "vision.observe",
    "priority": 100,
    "instruction": "若需要观察画面，输出 <agent_tool_call>{...}</agent_tool_call>。没有工具结果前不要编造画面内容。",
    "schema": "{\"tool\":\"vision.session.start\",\"arguments\":{\"reason\":\"用户请求观察画面\"}}"
  }
}
```

触发规则：

- 正则命中明确视觉请求时，直接生成 `vision.observe` L4 hit。
- 向量近邻命中 L4 capability 时，生成对应 tool instruction。
- 命中后只注入短 `<tool_memory_l4>` block，不注入长工具说明。

## 7. Prompt 注入策略

### 7.1 L4 命中但视觉会话未启动

```text
<tool_memory_l4>
- tool: vision.observe
  instruction: 用户可能正在请求视觉观察。若需要启动视觉工具，输出
  <agent_tool_call>{"tool":"vision.session.start","arguments":{"reason":"简短原因"}}</agent_tool_call>
没有工具结果前，不要编造画面内容。
</tool_memory_l4>
```

### 7.2 Starting

```text
<tool_status tool="vision.observe" state="starting">
视觉观察链路正在启动。当前还没有可靠视觉结果，不要描述画面。
</tool_status>
```

### 7.3 Ready

```text
<tool_status tool="vision.observe" state="ready">
视觉观察链路已就绪，但尚未捕获显著视觉事件。
</tool_status>
```

### 7.4 Running

```text
<vision_observation>
summary: 最近画面中检测到明显移动，可能有人靠近摄像头，但无法确认身份。
confidence: 0.62
source: vision.observe
注意：这是视觉工具的不确定观察，不是绝对事实。
</vision_observation>
```

### 7.5 Failed / Expired

```text
<tool_status tool="vision.observe" state="failed">
视觉观察工具暂时不可用：摄像头连接失败。请不要声称看到了画面。
</tool_status>
```

## 8. 主链路处理顺序

推荐顺序：

```text
1. 收到用户输入或前端信令。
2. 处理前端 vision.start / vision.stop。
3. 对用户输入执行本地 regex 和 L4 vector 查询。
4. 如触发视觉能力：
   - 若没有活跃视觉会话，调用 VisualToolSessionManager.Start。
   - 本轮 Prompt 注入工具状态，而不是编造视觉内容。
5. 如果已有 Ready/Running 会话：
   - 查询最近 observation。
   - 经过 gate 后注入 Prompt。
6. LLM 如输出结构化 tool_call：
   - 解析并校验 schema。
   - 转为 VisualToolSessionManager.Start/Stop。
   - 返回工具状态或进行二次生成。
7. 响应结束后，更新 session activity。
```

## 9. 线程池边界

视觉链路应复用现有三池架构：

```text
net pool:
  WebSocket / HTTP / WebRTC signaling / frontend status message

compute_pool:
  regex/L4 lightweight routing
  OpenCV 抽帧
  FFT / saliency / event monitor
  image resize / encode
  vector dedup

io_pool:
  VLM API
  Redis checkpoint
  L3/L4 storage write
  summary generation
```

禁止在 GStreamer callback 或 WS 协议处理线程内直接执行 VLM 请求。

## 10. 守护线程策略

必须通过 `RuntimeMaintenanceService` 注册视觉会话维护任务。

建议超时：

```text
startup_timeout:
  Starting 最长 15s。

max_duration:
  Ready/Running 最长 2-5min，按配置调整。

idle_timeout:
  Ready/Running 在无用户相关输入、无前端连接、无有效事件时最多 60s。

closing_timeout:
  Closing 最长 10s，超过后强制释放。
```

扫描逻辑：

```text
if state == Starting and now - started_at > startup_timeout:
    Expire("startup_timeout")

if state in {Ready, Running} and now - started_at > max_duration:
    Expire("max_duration")

if state in {Ready, Running} and now - last_activity_at > idle_timeout:
    Expire("idle_timeout")

if state == Closing and now - closing_started_at > closing_timeout:
    ForceClose("closing_timeout")
```

清理必须释放：

- WebRTC session / connection mapping
- GStreamer pipeline
- appsink callback 引用
- pending VLM task 的可取消状态
- recent observation buffer
- Redis checkpoint / reconnect state

## 11. 错误分级

非关键错误：

- 单帧解码失败
- 单次 VLM 请求失败
- 单次 embedding 去重失败
- 短时间没有显著事件

处理：

- 记录日志和 `last_error`
- 更新状态消息
- 不立即关闭会话

关键错误：

- GStreamer fatal bus error
- WebRTC pipeline failed
- startup timeout
- closing timeout
- 连续 N 次 VLM 失败
- 资源耗尽
- 前端连接断开且超过 reconnect grace

处理：

- 状态转 `Failed` 或 `Expired`
- 通知前端
- Prompt 注入工具不可用状态
- 释放资源

## 12. L3 / L4 写入边界

关闭视觉会话时默认写 L4，不默认写 L3。

L4 可记录：

- tool_id
- session_id
- state
- last_summary
- last_success_at
- last_failure_at
- last_error
- event_count
- provider
- cooldown_until

L3 只在以下条件之一满足时写入：

- 用户显式确认视觉结果。
- 多轮、多帧、多来源一致，且内容不涉及敏感判断。
- 上层业务明确要求保存视觉事实。

L3 写入必须带来源：

```json
{
  "source": "vision.observe",
  "confidence": 0.82,
  "confirmed_by_user": true,
  "session_id": "chat-session-001"
}
```

未确认的 VLM 结果不得作为长期事实写入。

## 13. 推荐实现拆分

### Step 1: VisualToolSessionManager

- `Start`
- `Stop`
- `Get`
- `RecordObservation`
- `MarkReady`
- `MarkFailed`
- `CleanupExpired`

先不接真实 RTC，只实现状态机和单测。

### Step 2: 主链路接入

- L4 命中后调用 `Start`
- 根据状态注入 `<tool_status>`
- 根据 recent observation 注入 `<vision_observation>`
- `ChatResponse` 暴露视觉工具状态

### Step 3: WebRTC / VisionEvent 接入

- WebRTC ready -> `MarkReady`
- VisionEvent / VLM result -> `RecordObservation`
- GStreamer fatal bus error -> `MarkFailed`
- close callback -> `Stop` 或 `Expire`

### Step 4: 结构化 tool_call parser

- 解析 `<agent_tool_call>{...}</agent_tool_call>`
- 校验 tool id 和 schema
- 转为 `Start` / `Stop`
- 支持二次生成或下一轮反馈

## 14. 当前落地状态

已完成：

- L4 Tool Memory Provider 基础接口
- 正则触发视觉工具
- 全局 L4 向量近邻查询
- 主链路短 `<tool_memory_l4>` 注入

待实现：

- `VisualToolSessionManager`
- 视觉工具会话维护任务
- 前端 start/stop 信令桥接
- tool_call parser
- observation prompt 注入
- close summary 和 L4/L3 写入策略
