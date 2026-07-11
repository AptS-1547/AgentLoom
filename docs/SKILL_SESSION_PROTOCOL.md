# Skill Session Protocol

本文档定义 AgentLoom 中可复用的 Skill 会话协议。视觉观察 `vision.observe` 是第一个复杂 profile，但协议本身应适用于任意具有触发、执行、状态反馈、错误处理和资源回收需求的 Skill。

## 1. 背景

DeepSeek 等 OpenAI-compatible LLM 不一定提供原生 Skill / Tool API。系统仍然可以通过 L4 Tool Memory、结构化 Prompt、可解析 tool call 标签和本地路由器模拟 Skill 调用。

Skill 不应被设计成简单 Prompt 描述。更稳妥的方式是：

- L4 存储工具能力、触发语义和调用协议。
- 正则和向量近邻把用户输入映射到候选 Skill。
- 主链路只在命中后注入短工具协议。
- Skill 执行由后端状态机管理，而不是完全交给 LLM。
- Skill 结果作为 observation 注入，而不是让 LLM 假装自己直接拥有外部感官或工具能力。

## 2. 核心概念

### 2.1 Skill

Skill 是 Agent 可调用的外部能力，例如：

- `vision.observe`：视觉观察
- `document.analyze`：文档解析
- `memory.recall`：主动记忆检索
- `web.search`：联网检索
- `code.execute`：代码执行
- `classroom.monitor`：课堂状态监控
- `emotion.calibrate`：情绪样本采集或标定

Skill 必须具备：

- 唯一 `skill_id`
- 触发规则
- 调用 schema
- 状态机
- Prompt 注入策略
- 超时和错误处理策略
- L4 / L3 写入边界

### 2.2 Skill Session

Skill Session 是一次有生命周期的 Skill 执行实例。它不同于一次性函数调用。

一次性 Skill 可以在很短时间内 `Starting -> Running -> Closed`。复杂 Skill，例如视觉观察，可能长时间处于 `Ready` 或 `Running`，并持续产出 observation。

仓库当前所有媒体 Skill 都必须是有限流，并受 `max_duration` 约束；不支持无限媒体监控，也不允许通过异常大的 duration 等价构造无限流。媒体 Skill 的 selected-frame 映射缓存、Seal、drain 和关闭屏障设计见：

```text
docs/SKILL_MEDIA_DRAIN_SPOOL_DESIGN.md
```

### 2.3 L4 Tool Memory

L4 是全局工具/技能记忆，不按用户隔离，只通过 `memory_level="L4"` 和 L3 隔离。

L4 负责存储：

- Skill 能力说明
- 触发语义
- 调用 schema
- 使用约束
- 最近工具状态
- 最近错误
- cooldown / availability

L4 不应存储未经确认的长期用户事实。长期事实仍属于 L3。

## 3. 触发来源和优先级

Skill 会话可由以下来源触发：

1. 前端显式信令
2. 本地规则触发，例如 regex
3. L4 向量近邻触发
4. LLM 结构化 tool call
5. 守护线程恢复或兜底关闭

推荐优先级：

```text
frontend explicit signal
  > local regex / deterministic router
  > L4 vector capability hit
  > LLM tool_call
  > maintenance daemon
```

前端显式信令最可信。本地规则和 L4 近邻用于低成本、可解释触发。LLM tool call 灵活但不应直接获得资源控制权，必须经过后端 session manager 校验。

## 4. 通用状态机

推荐通用状态：

```text
Idle
  未启动。

Starting
  已收到启动请求，正在准备资源或上下文。

Ready
  Skill 已就绪，但尚未产出可注入结果。

Running
  Skill 正在执行，可能持续产出 observation / progress / result。

WaitingInput
  Skill 需要用户、前端或外部系统补充输入。

Closing
  已收到关闭请求，正在停止任务、写入 summary 或释放资源。

Closed
  正常关闭。

Failed
  关键错误导致无法继续。

Expired
  超时或守护线程强制关闭。
```

推荐枚举：

```cpp
enum class SkillSessionState {
    Idle,
    Starting,
    Ready,
    Running,
    WaitingInput,
    Closing,
    Closed,
    Failed,
    Expired
};
```

状态转换：

```text
Idle -> Starting
  start signal / regex hit / L4 vector hit / LLM tool_call

Starting -> Ready
  resource ready, no result yet

Starting -> Running
  resource ready and result generation starts immediately

Ready -> Running
  first accepted observation/progress/result

Running -> WaitingInput
  needs user confirmation or additional arguments

WaitingInput -> Running
  input received

Ready|Running|WaitingInput -> Closing
  stop signal

Closing -> Closed
  cleanup complete

Starting|Ready|Running|WaitingInput|Closing -> Failed
  critical error

Starting|Ready|Running|WaitingInput|Closing -> Expired
  maintenance timeout
```

当前代码尚未完整实现上述 `Closing` 转移：`SkillSessionManager::Stop()` 目前直接进入 `Closed`。需要异步资源 drain 的 Skill 必须先扩展为 `BeginClosing -> CompleteClosing`，不能把本节推荐状态机误认为当前实现行为。

## 5. 通用信令

### 5.1 Start

```json
{
  "type": "skill.session.start",
  "skill_id": "vision.observe",
  "session_id": "chat-session-001",
  "trace_id": "trace-xxx",
  "payload": {
    "source": "frontend|regex|vector|llm",
    "reason": "用户请求观察当前画面",
    "arguments": {},
    "max_duration_ms": 120000
  }
}
```

### 5.2 Status

```json
{
  "type": "skill.session.status",
  "skill_id": "vision.observe",
  "session_id": "chat-session-001",
  "trace_id": "trace-xxx",
  "payload": {
    "state": "starting|ready|running|waiting_input|closing|closed|failed|expired",
    "available": true,
    "message": "Skill 正在启动",
    "error": ""
  }
}
```

### 5.3 Observation

Observation 是 Skill 的中间观察结果。它可以被注入 Prompt，但通常不是最终结果。

```json
{
  "type": "skill.observation",
  "skill_id": "vision.observe",
  "session_id": "chat-session-001",
  "trace_id": "trace-xxx",
  "payload": {
    "summary": "画面中检测到明显移动，但无法确认具体对象。",
    "confidence": 0.62,
    "source": "vlm",
    "stale": false,
    "should_inject_prompt": true,
    "metadata": {}
  }
}
```

### 5.4 Progress

Progress 表示长任务进度，适用于文档解析、代码执行、批处理等 Skill。

```json
{
  "type": "skill.progress",
  "skill_id": "document.analyze",
  "session_id": "chat-session-001",
  "trace_id": "trace-xxx",
  "payload": {
    "stage": "chunking",
    "progress": 0.45,
    "message": "正在切分文档"
  }
}
```

### 5.5 Result

Result 是 Skill 的最终结果。

```json
{
  "type": "skill.result",
  "skill_id": "document.analyze",
  "session_id": "chat-session-001",
  "trace_id": "trace-xxx",
  "payload": {
    "summary": "文档主要讨论...",
    "confidence": 0.88,
    "metadata": {}
  }
}
```

### 5.6 Stop

```json
{
  "type": "skill.session.stop",
  "skill_id": "vision.observe",
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

### 5.7 Error

```json
{
  "type": "skill.session.error",
  "skill_id": "vision.observe",
  "session_id": "chat-session-001",
  "trace_id": "trace-xxx",
  "payload": {
    "code": "TIMEOUT|RESOURCE_FAILED|MODEL_FAILED|INTERNAL",
    "message": "Skill 执行超时，已自动关闭",
    "closed": true
  }
}
```

## 6. L4 Capability Entry

L4 使用全局向量分区：

```cpp
PartitionKey key;
key.collection_id = collection_id;
key.tenant_id = tenant_id;
key.user_id = "";        // global scope
key.memory_level = "L4";
```

Capability entry 示例：

```json
{
  "memory_type": "capability",
  "memory_hash": "global:l4:vision.observe:intent.camera",
  "payload": "用户想让助手观察画面、摄像头、屏幕、周围环境或当前动作时，使用 vision.observe Skill。",
  "extra_metadata_json": {
    "skill_id": "vision.observe",
    "tool_id": "vision.observe",
    "priority": 100,
    "instruction": "若需要观察画面，输出结构化 Skill 调用。没有工具结果前不要编造画面内容。",
    "schema": "{\"type\":\"skill.session.start\",\"skill_id\":\"vision.observe\",\"arguments\":{\"reason\":\"用户请求观察画面\"}}"
  }
}
```

规则：

- `payload` 用于 embedding 语义检索。
- `extra_metadata_json.instruction` 用于命中后短 Prompt 注入。
- `schema` 用于约束 LLM 输出结构。
- 不应把长说明常驻 system prompt。

## 7. Prompt 注入协议

### 7.1 L4 命中

```text
<skill_memory_l4>
- skill: vision.observe
  instruction: 用户可能正在请求视觉观察。若需要启动视觉 Skill，输出：
  <agent_tool_call>{"type":"skill.session.start","skill_id":"vision.observe","arguments":{"reason":"简短原因"}}</agent_tool_call>
没有工具结果前，不要编造外部观察。
</skill_memory_l4>
```

### 7.2 Skill 状态

```text
<skill_status skill="vision.observe" state="starting">
视觉观察链路正在启动。当前还没有可靠视觉结果，不要描述画面。
</skill_status>
```

### 7.3 Skill Observation

```text
<skill_observation skill="vision.observe">
summary: 最近画面中检测到明显移动，可能有人靠近摄像头，但无法确认身份。
confidence: 0.62
source: vision.observe
注意：这是工具的不确定观察，不是绝对事实。
</skill_observation>
```

### 7.4 Skill Error

```text
<skill_status skill="vision.observe" state="failed">
视觉观察工具暂时不可用：摄像头连接失败。请不要声称看到了画面。
</skill_status>
```

## 8. 主链路处理顺序

推荐处理顺序：

```text
1. 收到用户输入或前端信令。
2. 处理前端 skill.session.start / skill.session.stop。
3. 对用户输入执行本地 regex 和 L4 vector 查询。
4. 如命中 Skill：
   - 查询 SkillSessionManager 当前状态。
   - 必要时 Start。
   - 本轮 Prompt 注入 skill status 或短调用协议。
5. 如果已有 Ready/Running session：
   - 查询 recent observation/result。
   - 经过 gate 后注入 Prompt。
6. LLM 输出 agent_tool_call 时：
   - 解析并校验 schema。
   - 转为 SkillSessionManager.Start/Stop。
   - 返回状态或触发二次生成。
7. 响应结束后更新 activity 和 L4 状态。
```

## 9. 通用接口建议

```cpp
struct SkillSessionStartRequest {
    std::string skill_id;
    std::string session_id;
    std::string user_uuid;
    std::string persona_id;
    std::string trace_id;
    std::string source;
    std::string reason;
    std::string arguments_json;
    std::chrono::milliseconds max_duration{120000};
};

struct SkillSessionStopRequest {
    std::string skill_id;
    std::string session_id;
    std::string trace_id;
    std::string source;
    std::string reason;
    bool summarize = true;
    bool write_l3 = false;
};

struct SkillObservation {
    std::string skill_id;
    std::string session_id;
    std::string trace_id;
    std::string summary;
    double confidence = 0.0;
    bool stale = false;
    bool should_inject_prompt = false;
    std::string metadata_json = "{}";
};

struct SkillSessionSnapshot {
    std::string skill_id;
    std::string session_id;
    SkillSessionState state = SkillSessionState::Idle;
    std::string status_text;
    std::string last_observation;
    std::string last_error;
    std::chrono::system_clock::time_point started_at;
    std::chrono::system_clock::time_point last_activity_at;
};

class ISkillSessionManager {
public:
    virtual ~ISkillSessionManager() = default;

    virtual core::Result<SkillSessionSnapshot> Start(
        const SkillSessionStartRequest& request) = 0;

    virtual core::Result<SkillSessionSnapshot> Stop(
        const SkillSessionStopRequest& request) = 0;

    virtual core::Result<std::optional<SkillSessionSnapshot>> Get(
        std::string_view session_id,
        std::string_view skill_id) = 0;

    virtual core::Status RecordObservation(
        const SkillObservation& observation) = 0;

    virtual std::size_t CleanupExpired(std::stop_token stop_token) = 0;
};
```

## 10. 守护线程策略

Skill Session 必须支持维护任务兜底：

```text
startup_timeout:
  Starting 状态最长等待时间。

max_duration:
  整个 session 最长存在时间。

idle_timeout:
  无用户输入、无前端连接、无有效 observation 的最长空闲时间。

closing_timeout:
  Closing 状态最长等待时间。
```

维护任务必须能将 session 转为 `Expired`，并释放底层资源。

## 11. 错误分级

非关键错误：

- 单次模型调用失败
- 单个输入片段解析失败
- 单次缓存或 embedding 查询失败
- 短暂无有效 observation

关键错误：

- 资源初始化失败
- pipeline fatal error
- startup timeout
- closing timeout
- 连续 N 次模型失败
- 资源耗尽

关键错误必须关闭 session 并反馈给前端和主链路 Prompt。

## 12. L3 / L4 写入边界

默认写 L4：

- skill availability
- last status
- last summary
- last error
- cooldown
- recent result hash
- provider / backend

谨慎写 L3：

- 用户确认后的事实
- 高置信且业务要求长期保存的结果
- 多来源一致且不涉及敏感属性的结论

未经确认的 observation 不得作为长期事实直接写入 L3。

## 13. Vision Profile

`vision.observe` 是本协议的第一个复杂 Skill profile。

它的特殊点：

- 资源重：WebRTC、GStreamer、OpenCV、VLM。
- 有持续事件流。
- 结果容易幻觉。
- 关闭和超时回收非常重要。
- 默认只写 L4，不写 L3。

视觉 profile 的详细协议见：

```text
docs/VISUAL_TOOL_SESSION_PROTOCOL.md
```

## 14. 当前实现状态

已完成：

- `ISkillSessionManager`、`SkillSessionManager` 与通用状态枚举。
- Start/Get/MarkReady/MarkFailed/RecordObservation/CleanupExpired。
- HTTP/WS `skill.session.start`、`skill.session.stop` 和 status 路由。
- startup/max-duration/idle/closing timeout 配置与 maintenance task。
- `SkillVisionEventSink` 将 `VisionEvent` 转换为 `SkillObservation`。
- Persona Runtime 注入 Skill status 或最近 observation。
- L4 Tool Memory Provider、正则触发视觉 capability、全局 L4 向量近邻查询和短 Prompt 注入。

待实现：

- tool_call parser。
- L4 状态写回与 L3 确认写入。
- execution generation，避免迟到回调污染重新启动的同名 Skill。
- `Closing` 的 Begin/Complete 两阶段关闭；当前 Stop 直接 Closed。
- Closing 期间 observation 不回退到 Running 的状态保护。
- 通用 progress/result 与异步 close participant。
- 媒体 Skill mapped spool、Seal/Drain gRPC 控制面和 final aggregation barrier。

当前实现校准：

- session key 是 `(session_id, skill_id)`，尚无 `execution_id`；
- 活动 session 重复 Start 为幂等刷新；terminal session 再次 Start 会重置同一 key；
- Stop 当前同步进入 `Closed`，`summarize/write_l3` 尚未驱动 close workflow；
- `WaitingInput` 和 `Closing` 已定义，但尚无完整业务转移接口；
- `closing_timeout` 已配置和检查，但正常 Stop 当前不会停留在 Closing。

---

**最后更新**: 2026-07-11
