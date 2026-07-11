# Vision System Design

本文档定义 AgentLoom 后续视觉系统的整体实现与优化架构。核心方案是：

```text
VisualSkill 生命周期控制
  + ViT 视觉表征校正
  + VLM 语义解释
  + Gate/Fusion 可信观察生成
  + L4/L3 分层记忆写入
```

视觉能力不应被设计为“摄像头持续输入 -> VLM 描述 -> 直接塞入主 Prompt”。更合理的架构是将视觉作为可触发、可关闭、可回收、可审计的 Skill Session，同时用专业视觉表征模型降低 VLM 幻觉风险。

相关协议：

- `docs/SKILL_SESSION_PROTOCOL.md`
- `docs/VISUAL_TOOL_SESSION_PROTOCOL.md`

## 1. 背景与问题

当前系统已经具备多模态工程基础：

- WebRTC signaling
- GStreamer `webrtcbin`
- appsink 获取真实视频帧
- OpenCV 帧差法、MOG2、连通域、边缘变化、FFT 自适应抽帧
- VisionEvent / VisionEventMonitor
- OpenAI-compatible VLM client
- L4 Tool Memory 触发式 Prompt 注入
- 主链路三池架构

但直接将 VLM 常驻挂载到主链路存在明显风险：

- 小 VLM / 通用 VLM 在人脸、性别、主体和客体连续性上容易幻觉。
- 单帧描述无法保证时序一致性。
- VLM 结果如果直接进入 Prompt，会污染主链路上下文。
- 错误视觉事实如果写入 L3，会造成长期记忆污染。
- RTC / VLM / 解码链路是资源型对象，必须有生命周期管理。
- 用户没有明确请求视觉时，系统不应默认声称自己看到了画面。

因此，视觉系统需要同时解决两个层面的问题：

```text
系统层：
  什么时候看、谁能触发、如何启动、如何关闭、如何注入主链路、如何回收资源。

模型层：
  如何降低 VLM 幻觉，如何验证视觉事件连续性，如何判断观察结果是否可信。
```

## 2. 总体架构

整体链路：

```text
User / Frontend / LLM
  -> Trigger Router
      -> regex
      -> L4 vector capability memory
      -> structured tool_call
  -> Skill Session Manager
      -> start / status / stop / error
      -> timeout cleanup
  -> WebRTC / Media Pipeline
      -> GStreamer
      -> appsink
      -> frame buffer RAII
  -> Frame Sampler
      -> OpenCV motion detection
      -> FFT adaptive sampling
      -> event monitor
  -> Visual Representation Layer
      -> ViT / visual encoder
      -> frame embedding
      -> similarity / continuity
      -> object/person/scene hints
  -> VLM Semantic Layer
      -> OpenAI-compatible VLM
      -> structured visual interpretation
  -> Gate / Fusion
      -> confidence
      -> temporal consistency
      -> vector similarity
      -> sensitive-attribute policy
  -> Main Chain Injection
      -> skill_status
      -> skill_observation
      -> weak context only
  -> Memory
      -> L4 tool state / session summary
      -> L3 only after confirmation or strict policy
```

分层视图：

```text
Layer 0: Trigger
Layer 1: Skill Session
Layer 2: Stream / Frame
Layer 3: Visual Representation
Layer 4: VLM Interpretation
Layer 5: Gate / Fusion
Layer 6: Main Chain Injection
Layer 7: L4 / L3 Memory
```

## 3. VisualSkill 架构

`vision.observe` 是通用 Skill Session Protocol 的第一个复杂 profile。

它不是一次性函数调用，而是一个有生命周期的视觉会话：

```text
Idle
  -> Starting
  -> Ready
  -> Running
  -> Closing
  -> Closed

任意运行阶段:
  -> Failed
  -> Expired
```

### 3.1 触发来源

视觉 Skill 可由以下来源触发：

```text
1. 前端显式 start/stop 信令
2. 用户输入本地 regex 命中
3. L4 全局工具记忆向量近邻命中
4. LLM 输出结构化 tool_call
5. 守护线程超时或错误兜底
```

优先级：

```text
frontend explicit signal
  > regex
  > L4 vector
  > LLM tool_call
  > daemon cleanup
```

前端显式信令代表 UI 和用户授权，优先级最高。本地 regex 和 L4 向量近邻用于可解释触发。LLM tool_call 只能作为建议，必须由后端状态机校验。

### 3.2 L4 触发模式

L4 是全局工具/技能记忆，不按用户隔离：

```cpp
PartitionKey key;
key.collection_id = collection_id;
key.tenant_id = tenant_id;
key.user_id = "";
key.memory_level = "L4";
```

L4 capability entry 的职责：

- 表示系统具备某个 Skill。
- 表示该 Skill 的触发语义。
- 提供短调用协议。
- 提供工具使用约束。

命中后只注入短 Prompt：

```text
<tool_memory_l4>
- tool: vision.observe
  instruction: 用户可能正在请求视觉观察。若需要启动视觉工具，输出结构化工具调用。
没有工具结果前，不要编造画面内容。
</tool_memory_l4>
```

不应将完整工具说明常驻注入 system prompt。

### 3.3 生命周期管理

视觉会话必须由 `SkillSessionManager` 或 `VisualToolSessionManager` 管理：

- `Start`
- `Stop`
- `Get`
- `RecordObservation`
- `MarkReady`
- `MarkFailed`
- `CleanupExpired`

所有状态转换必须产生日志，并携带 trace id。

### 3.4 守护线程兜底

视觉 Skill 是资源型会话，必须注册维护任务。

建议超时：

```text
startup_timeout: 15s
max_duration: 2-5min
idle_timeout: 60s
closing_timeout: 10s
```

触发超时后：

- 状态转 `Expired`
- 停止接收新事件
- 释放 WebRTC / GStreamer / pending VLM task
- 通知前端
- 向主链路注入工具不可用状态

## 4. Stream / Frame 层

当前工程基础：

- `WebRtcBin`
- `WebRtcMediaPipeline`
- `GstMappedFrameBuffer`
- `OpenCvFrameSampler`
- `VisionEventMonitor`
- `WebRtcSessionRegistry`
- `RuntimeMaintenanceService`

该层职责：

- 建立 WebRTC 媒体流。
- 通过 GStreamer appsink 获取真实帧。
- 用 RAII 包装 GStreamer buffer，避免大对象复制。
- 在 compute_pool 中执行抽帧和视觉事件生成。
- 将显著视觉事件交给下一层。

该层不负责高层语义判断，也不直接调用 VLM。

线程边界：

```text
GStreamer callback:
  只做最小工作，提交 frame view。

compute_pool:
  OpenCV 抽帧、FFT、事件聚合、图像预处理。

io_pool:
  VLM 请求、Redis/L4/L3 写入。
```

## 5. ViT 视觉表征校正层

专业 ViT / 视觉塔用于降低 VLM 幻觉，提供稳定视觉证据。

它不是必须替代 VLM，而是作为 VLM 的前置和旁路校正层：

```text
Frame
  -> ViT / visual encoder
      -> frame embedding
      -> similarity score
      -> continuity score
      -> object/person/scene hints
  -> VLM semantic interpretation
  -> Gate / Fusion
```

### 5.1 主要用途

```text
1. 帧间去重
   高相似帧不重复调用 VLM。

2. 事件连续性
   判断某个视觉事件是否跨多帧稳定存在。

3. 主体稳定性
   降低 VLM 将背景误认为主体的概率。

4. 低层视觉 hint
   提供 person/object/scene/action 的弱标签。

5. VLM 结果校验
   VLM 输出与视觉 embedding 或专门 detector 不一致时降权。

6. 敏感判断 gate
   对身份、性别、表情、情绪等高风险结论提高阈值。
```

### 5.2 推荐模型位置

可选实现：

- CLIP / SigLIP / EVA-CLIP 视觉塔
- 专门 person detector
- lightweight object detector
- action / pose detector
- 本地 ViT embedding ONNX

第一阶段可优先只接视觉 embedding：

```text
frame -> visual embedding -> similarity / continuity
```

再逐步增加 object/person/scene hints。

## 6. VLM 语义解释层

VLM 负责高层语义解释：

- 描述画面
- 提取用户请求相关视觉事实
- 将视觉事件转为自然语言 observation
- 输出结构化 JSON

VLM 不应独立决定长期事实，也不应直接写 L3。

推荐 VLM 请求约束：

```text
1. 只描述可见内容。
2. 不推断身份、性别、年龄、情绪等敏感属性，除非非常明确且用户要求。
3. 不确定时输出 unknown / uncertain。
4. 输出结构化 JSON。
5. 给出 confidence。
6. 区分 fact、weak_interpretation 和 speculation。
```

推荐输出：

```json
{
  "summary": "画面中有人靠近摄像头，但无法确认身份。",
  "confidence": 0.66,
  "facts": [
    "画面中出现明显移动"
  ],
  "weak_interpretations": [
    "可能有人靠近摄像头"
  ],
  "unsafe_or_uncertain": [
    "无法确认身份、性别或情绪"
  ]
}
```

## 7. Gate / Fusion 层

Gate / Fusion 负责生成最终可注入主链路的可信观察。

输入证据：

```text
1. OpenCV sampler saliency
2. VisionEventMonitor 聚合结果
3. ViT embedding similarity / continuity
4. VLM confidence
5. L4 Skill session state
6. 历史 observation 去重结果
7. 敏感属性策略
```

输出：

```text
accepted_observation
weak_observation
discarded_hallucination
needs_confirmation
```

### 7.1 推荐规则

```text
高置信注入:
  VLM confidence 高
  多帧连续
  ViT similarity 支持
  不涉及敏感属性

弱注入:
  有视觉事件
  但 VLM confidence 中等
  或语义解释不完全确定

丢弃:
  VLM 与视觉表征冲突
  单帧异常
  重复事件
  敏感结论低置信

需要确认:
  涉及用户身份、外貌、情绪、长期事实
```

### 7.2 Prompt 注入等级

```text
tool_status:
  只有工具状态，不含视觉内容。

weak_observation:
  明确标注不确定性。

accepted_observation:
  可作为本轮上下文，但仍标注 source=vision.observe。

confirmed_fact:
  用户确认后才允许写 L3。
```

## 8. 主链路注入

视觉内容只能通过 Skill 协议进入主链路。

### 8.1 未就绪

```text
<skill_status skill="vision.observe" state="starting">
视觉观察链路正在启动。当前还没有可靠视觉结果，不要描述画面。
</skill_status>
```

### 8.2 就绪但无事件

```text
<skill_status skill="vision.observe" state="ready">
视觉观察链路已就绪，但尚未捕获显著视觉事件。
</skill_status>
```

### 8.3 弱观察

```text
<skill_observation skill="vision.observe" confidence="0.62" strength="weak">
最近画面中检测到明显移动，可能有人靠近摄像头，但无法确认身份。
注意：这是视觉工具的不确定观察，不是绝对事实。
</skill_observation>
```

### 8.4 工具不可用

```text
<skill_status skill="vision.observe" state="failed">
视觉观察工具暂时不可用：摄像头连接失败。请不要声称看到了画面。
</skill_status>
```

LLM 回复必须遵守：

- 没有 observation 时不得描述画面。
- observation 是工具结果，不是 LLM 自身感官。
- 弱观察需要保留不确定性。
- 不应主动写入长期事实。

## 9. L4 / L3 记忆策略

### 9.1 L4

L4 默认记录：

- skill_id
- availability
- session state
- last_summary
- last_error
- last_success_at
- last_failure_at
- provider
- cooldown
- recent event count

L4 是工具状态和能力记忆，不是用户长期事实记忆。

### 9.2 L3

L3 只在严格条件下写入：

- 用户显式确认视觉结果。
- 多帧、多来源、高置信一致。
- 不涉及敏感属性。
- 业务明确要求持久化。

写入必须包含来源和置信度：

```json
{
  "source": "vision.observe",
  "confidence": 0.82,
  "confirmed_by_user": true,
  "session_id": "chat-session-001"
}
```

未经确认的 VLM observation 不得直接写入 L3。

## 10. 前端交互

前端需要能接收：

- `skill.session.status`
- `skill.observation`
- `skill.progress`
- `skill.session.error`
- `skill.session.stop`

前端可发送：

- `skill.session.start`
- `skill.session.stop`

视觉启动时，前端应展示工具状态，而不是假定模型已经能看到画面。

示例 UI 状态：

```text
正在启动视觉观察...
视觉观察已就绪
正在分析画面变化
视觉观察已关闭
视觉工具不可用
```

## 11. 部署与资源边界

视觉系统依赖较重：

- GStreamer
- OpenCV
- VLM API 或本地 VLM runtime
- 未来 ViT / visual encoder

当前推荐：

- 保持常规链接目标。
- 用配置控制视觉能力是否启用。
- 不做 DLL 热加载。
- 后续如本地 VLM 变重，可考虑独立 vision runtime 进程或容器。

## 12. 实现路线

### Phase 1: VisualSkill Runtime

目标：先把生命周期和主链路边界做稳。

- Skill / Visual session manager
- start / stop / status / error
- maintenance cleanup
- Prompt status injection
- L4 session state writeback
- 单测状态机

### Phase 2: Controlled VLM Integration

目标：让 VLM 以受控 observation 形式进入主链路。

- VLM async analyzer
- observation buffer
- confidence gate
- weak / accepted observation prompt
- close summary
- 前端状态反馈

### Phase 3: ViT-based Perception Correction

目标：降低 VLM 幻觉和重复调用成本。

- visual embedding
- frame similarity dedup
- temporal consistency
- object/person/scene hints
- VLM result verification
- sensitive attribute policy

### Phase 4: Memory Integration

目标：建立可靠的 L4/L3 写入闭环。

- L4 tool state summary
- user-confirmed L3 write
- visual observation audit metadata
- memory cleanup and retention policy

## 13. 当前实现状态

已完成：

- WebRTC signaling 基础链路
- GStreamer webrtcbin / appsink frame 获取
- OpenCV 抽帧与 FFT 自适应算法
- VisionEvent / VisionEventMonitor
- OpenAI-compatible VLM client 基础
- L4 ToolMemoryProvider
- 全局 L4 向量近邻查询
- 视觉 regex trigger
- 主链路短 `<tool_memory_l4>` 注入

待实现：

- Skill / Visual session manager
- 视觉 session maintenance task
- 前端 Skill start/stop 桥接
- LLM tool_call parser
- VLM observation 注入
- ViT 视觉表征校正层
- close summary 和 L4/L3 写入策略

## 14. 总结

最终视觉系统应由四个核心部分组成：

```text
VisualSkill:
  负责视觉能力的触发、生命周期、状态反馈和资源回收。

ViT / Visual Encoder:
  负责低层视觉表征、相似度、连续性和幻觉校正。

VLM:
  负责高层语义解释和自然语言 observation。

Gate / Memory:
  负责可信度判断、Prompt 注入等级、L4/L3 写入边界。
```

这一路线比直接把 VLM 输出注入主链路更安全，也更适合长期扩展为通用 Skill 系统。
