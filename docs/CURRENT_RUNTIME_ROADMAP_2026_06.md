# 当前 Runtime 后续路线图

> 日期：2026-06-20  
> 目的：在正式 Server、E2E 测试、文档解析链路、L3 长期记忆、语义缓存基础设施、情绪感知融合层基本完成后，重新整理下一阶段最值得投入的工程方向。

## 1. 当前阶段判断

当前系统已经基本具备旧 MVP 功能的完整 C++ 后端能力，并且已经形成较完整的 E2E Server 与正式 Server 入口。P0 级别的“能跑通主链路并交付”的能力已经基本完成，后续 P0 不再作为主要开发任务，而是作为每轮重要改动后的回归门槛。

已经具备的关键能力：

- 正式 Server 入口已经集成主链路、文档解析链路、长期记忆 L3、守护线程等核心接口。
- E2E Server 已有较完整测试基础，可作为正式 Server 的回归参照。
- Redis、SQLite、语义缓存、文档解析缓存、user registry 等基础设施已经接入。
- BERT gRPC 情绪推理链路可用，支持 ONNX / CUDA / CPU 与 batch 推理。
- 情绪融合层已经从简单 softmax evidence 融合演进为 BERT/keyword/vector 快速融合 + LLM gate 仲裁。
- MAP 标定器已经能评估 mixed5000 与 SMP2020，并输出 LLM gate 阈值扫描。
- Docker 多容器部署方向已经明确，BERT/VLM 推理服务更适合作为独立容器。

当前最重要的阶段性结论：

```text
P0 已完成，后续作为回归门槛；
P1 进入语义缓存主链路接入；
P2 补完情绪层 domain/OOD pre-gate；
模型升级和 VLM 进入后续迭代。
```

## 2. 更新后的优先级

| 优先级 | 方向 | 当前定位 |
|--------|------|----------|
| P0 | 交付态回归验证 | 已基本完成，作为 release gate |
| P1 | 语义缓存主链路接入 | 当前最值得优先实现的功能 |
| P2 | 情绪层 domain/OOD pre-gate | 情绪感知工程闭环的最后关键结构 |
| P3 | BERT/情绪模型升级与数据闭环 | 赛后或下一阶段模型工程 |
| P4 | VLM 多模态实时链路 | 流处理与抽帧基本完成，待多路推流验证和 VLM 接入 |

## 3. P0：交付态回归验证

P0 已经不再是主要开发任务，而是保留为回归检查项。每次改动主链路、缓存、情绪感知或部署脚本后，应按需执行。

回归门槛：

- 正式 Server 能启动。
- 主链路 E2E 能过。
- 文档解析链路 E2E 能过。
- Redis 相关测试能过。
- BERT gRPC / emotion server 能跑。
- L3 长期记忆路径正常。
- SQLite user registry 路径正常。
- Docker / 多容器脚本在 Linux target 下可构建或至少具备明确构建路径。
- 配置文件不包含硬编码敏感 key。
- Windows/Linux 条件编译不互相污染。

P0 的核心目标是防止后续功能开发破坏已经具备的交付能力。

## 4. P1：语义缓存主链路接入

语义缓存是当前最值得优先推进的功能。原因是底层能力已经相对完备，接口清晰，文档解析缓存已经证明了缓存价值，且收益对主链路延迟、成本和一致性都比较确定。

已经具备：

- L0 语义搜索算法。
- Redis 语义缓存基础设施。
- `CacheRecord` 扩展。
- embedding/vector 基础设施。
- batch 检索能力。
- 文档解析缓存接入经验。

### 4.1 第一版策略：cached_reference 注入

主链路第一版不建议“命中即直接返回旧答案”。直接返回会带来明显重复感，也可能忽略当前 session、当前问题细节和用户表达差异。

建议第一版采用保守策略：

```text
semantic cache hit
  -> 生成 cached_reference
  -> 注入 LLM prompt / context
  -> LLM 基于当前问题、会话上下文和缓存参考生成最终回复
```

这将语义缓存定位为：

```text
cache-augmented generation
```

而不是：

```text
answer replacement
```

### 4.2 命中分层

建议按相似度和作用方式分层：

| 命中层级 | 策略 |
|----------|------|
| High similarity | 注入缓存答案摘要、结构、关键事实，允许 LLM 复用 |
| Mid similarity | 只注入相关历史答案或关键点，不要求复用表达 |
| Low similarity | 不使用缓存 |

### 4.3 抗重复策略

需要避免用户明显感知到“复制粘贴”：

- 不直接返回原缓存文本。
- 缓存记录维护 `hit_count`。
- 同一 session 连续命中同一 cache record 时降权。
- `hit_count` 高时提示 LLM 换表达或重组结构。
- 当前问题与缓存问题存在关键差异时，只作为参考，不作为答案。
- 置信不足、上下文冲突或用户显式要求新答案时回退完整 LLM。

### 4.4 后续进阶：selective bypass

当 reference 模式稳定后，可以再引入更激进的 bypass：

```text
极高相似度 + 低风险问题 + 非连续重复命中
  -> 允许模板化轻改写或直接复用部分答案
```

但 bypass 应该是第二阶段，不应作为第一版主链路策略。

## 5. P2：情绪层 domain/OOD pre-gate

情绪融合层经过本轮实验后，已经不适合继续投入大量时间调 softmax 权重或堆关键词。唯一还值得补的工程闭环是基于 embedding 域距离的 domain/OOD pre-gate。

实验结论详见：

- `docs/EMOTION_FUSION_GATE_EXPERIMENT.md`

关键观察：

| 数据集 | 数据域 | 最优 LLM 接管率 | 结论 |
|--------|--------|----------------:|------|
| mixed5000 | 训练分布/近训练分布 | 0.02% | LLM 基本不应接管 |
| SMP2020 | 外部分布 | 约 42% | LLM 有明显补偿作用 |

这说明 LLM 的主要价值不是常规融合，而是域外样本补偿。因此需要在调用 LLM 前先判断样本是否偏离训练/标定分布。

### 5.1 pre-gate 目标

当前已经有 post-gate：

```text
LLM 调用后，判断是否覆盖快速融合结果。
```

还缺 pre-gate：

```text
调用 LLM 前，判断是否值得调用。
```

pre-gate 的目标：

- 降低训练分布内不必要的 LLM 调用。
- 保留 SMP2020 这类域外样本上的 LLM 收益。
- 降低成本、延迟和 LLM 噪声。
- 在域外或冲突样本上降低状态机更新强度。

### 5.2 第一版 domain score

建议先实现轻量版本：

```text
domain_score = max cosine_similarity(text_embedding, in_domain_centroids)
```

或者按标签维护：

```text
domain_score[label] = cosine_similarity(text_embedding, label_domain_centroid[label])
```

第一版可以使用少量原型或 centroid：

- 训练域正常样本 centroid。
- mixed5000 / 弱项标签 centroid。
- 教育问答风格 centroid。
- SMP/社交媒体风格 centroid。
- BERT 弱项标签 centroid。

### 5.3 策略

当 `domain_score` 较高：

```text
信任 BERT/MAP 快速融合层；
收紧 LLM post-gate；
甚至跳过 LLM 调用。
```

当 `domain_score` 较低：

```text
降低 BERT margin 信任；
放宽 LLM fallback；
必要时降低情绪状态机更新强度。
```

### 5.4 分工边界

当前情绪感知层的职责应保持清晰：

```text
上下文靠记忆和状态；
域漂移靠 gate；
主观性靠概率和平滑；
显著性能突破靠模型和数据。
```

上下文缺失不应主要交给 gate 解决，因为 message array、LLM context、短期会话、长期记忆 L3 和情绪状态机已经能承载上下文。gate 的核心职责是跨域泛化风险识别。

## 6. P3：BERT/情绪模型升级与数据闭环

这不是当前最紧急的工程交付任务，但它是情绪感知层后续显著提升的主要来源。

当前实验表明：

- 域内数据上 BERT/感知层已经接近先前统计诊断提示的经验上限。
- SMP2020 上存在明显域漂移。
- 正则/vector/MAP 只能小幅修正域外表现。
- LLM gate 能补偿域外样本，但成本和调用率较高。

后续模型方向：

- 使用更专业的中文情绪识别模型。
- 继续微调当前 `hfl/chinese-roberta-wwm-ext` 派生模型。
- 使用教育对话目标域数据。
- 构建真实 Agent 高冲突样本回流集。
- 尝试多标签情绪建模。
- 做更强模型到小 ONNX 模型的蒸馏。
- 重新标定 per-label reliability。

这一阶段需要数据、训练和重新评估，不建议在当前主链路交付阶段打断工程收敛。

## 7. P4：VLM 多模态实时链路

VLM 仍然是高价值方向，但当前瓶颈已经不再是“有没有视频输入层”。代码现状显示，WebRTC signaling、GStreamer `webrtcbin` media pipeline、`appsink` decoded frame 输出、OpenCV 动态抽帧、视觉事件聚合和 smoke 工具已经基本落地。下一阶段重点应从“搭链路”转为“验证多路推流、正式接入 VLM、做资源隔离和稳定性回归”。

已经具备：

- `src/media/webrtc_signaling_handler.*`：复用现有 WebSocket runtime，支持 `/ws/vision/signaling` 的 offer/answer/ICE/close/resume/config。
- `src/media/webrtc_session_registry.*`：维护 WebRTC session、connection、reconnect token、frame checkpoint，并支持 Redis checkpoint store。
- `src/media/webrtc_media_pipeline.*`：创建 GStreamer pipeline，接入 `webrtcbin`，通过 `decodebin -> videoconvert -> appsink` 输出 RGB frame。
- `src/media/opencv_frame_sampler.*`：按 session 维护 MOG2、直方图、边缘变化、EMA、cooldown 和可选自适应 FPS 抽帧状态。
- `src/media/vision_runtime_interfaces.*`：定义 `IFrameSampler`、`IVlmVisionClient`、`VisionEvent`、`VisionEventMonitor`、dedup sink 和 context provider。
- `tools/playwright_webrtc_signaling_smoke.py` 与 `tools/webrtc_signaling_smoke_server.cpp`：已具备浏览器 WebRTC smoke、多 session 参数、解码帧落盘和 sampler 验证入口。
- `tests/media/*`：覆盖 signaling、session checkpoint、pipeline 创建、OpenCV sampler、事件聚合和 dedup。

仍需验证或补齐：

- 多路浏览器推流下的 CPU、内存、线程池队列、GStreamer pipeline 生命周期和关闭回收。
- `IVlmVisionClient` 当前还是接口位，media pipeline 尚未把关键帧编码后正式提交给 VLM 服务。
- VLM 请求需要放入 IO pool 或独立 worker，不能在 GStreamer callback、WebSocket 线程或主对话线程中执行。
- 视觉事件进入 persona runtime 的链路已有 `skill_vision_event_sink` 基础，但 VLM observation 的正式字段、置信度、缓存与降权策略仍需收敛。
- NVIDIA/VAAPI/D3D11 等硬件解码策略仍需按平台验证，CPU-only fallback 必须保留。

### 7.1 当前链路定位

当前链路应视为“实时视觉输入与抽帧底座已基本完成”，而不是早期的“待实现 WebRTC 输入层”。更准确的当前目标是：

```text
Browser camera
  -> WebRTC signaling over /ws/vision/signaling
  -> GStreamer webrtcbin media pipeline
  -> decodebin / videoconvert / appsink
  -> OpenCV dynamic frame sampler
  -> VisionEvent / monitor / dedup
  -> async VLM request/cache/event queue
  -> controlled vision.observe observation
```

这一阶段仍不建议让 VLM 直接闭环到情绪状态机。更稳的定位是：先把多路推流、抽帧、关键帧事件、VLM observation 和主链路注入做成可控工具会话。

### 7.2 信令与会话层现状

项目现有 WebSocket/Boost.Beast/Asio 封装已经被 WebRTC signaling 复用，内部端点为：

```text
/ws/vision/signaling
```

该端点负责：

- 交换 SDP offer/answer。
- 交换 ICE candidate。
- 返回 ICE server config。
- 维护 connection id、session id、trace id 和 reconnect token。
- 支持 resume / close。
- 在 pipeline 创建失败、未知消息、断线和重连失败时返回明确错误。

媒体数据仍通过 WebRTC RTP/RTCP 进入 GStreamer 管线，不通过 WebSocket 传输大体积帧。后续重点不是重新设计信令协议，而是验证多 session 并发、断线重连、刷新、重复 close、pipeline fatal error 这些生产路径。

### 7.3 GStreamer/webrtcbin 与抽帧现状

当前 media pipeline 已经按如下路径构建：

```text
webrtcbin
  -> incoming pad
  -> queue
  -> decodebin
  -> videoconvert
  -> appsink(RGB)
  -> compute_pool frame sampler
```

`appsink` 输出帧后通过 `core::ThreadPool` 投递给 `IFrameSampler`，避免在 GStreamer callback 中执行重计算。OpenCV sampler 已具备显著度、时间窗口、cooldown 和自适应采样能力；后续更应关注多路推流下的采样公平性、队列背压、关键帧编码成本和事件去重，而不是继续重写输入层。

硬件解码建议保持为渐进能力：

- NVIDIA 环境优先验证现有 decoder preference 与 GStreamer 插件可用性。
- Linux VAAPI 和 Windows D3D11 需要分别验证，不应让平台单一 API 成为默认路径。
- CPU-only 软件解码 fallback 必须作为可运行基线。
- VLM 推理服务仍建议独立容器或独立进程，Gateway 只负责输入、采样、事件和请求编排。
- 推理模块迁移按已编译 gRPC server、protobuf/gRPC 头文件和运行时依赖交付；不再作为 Gateway/service 可复用库继续拆分。

### 7.4 多容器边界

多模态链路仍建议保持多容器架构：

| 容器 | 职责 |
|------|------|
| Gateway | HTTP/WebSocket/WebRTC signaling、session、记忆、缓存、主链路编排 |
| Vision Ingest / VLM Adapter | GStreamer/WebRTC media、帧抽样、VLM 请求与视觉事件生成 |
| BERT Emotion Server | 文本情绪推理 |
| Redis | 缓存、会话元数据、语义/视觉事件存储 |

如果第一版为了开发便利把 Gateway 与 Vision Ingest 放在同一进程，也应保持接口边界清晰，后续可以拆成独立容器。VLM 推理本身不建议和 Gateway 强耦合，因为模型依赖、GPU/CUDA/驱动兼容性和资源占用都明显不同。

BERT/VLM 推理服务的项目间迁移边界是 gRPC 进程和协议头文件。可迁移库目标应集中在 Persona Runtime、Session、Classroom、Gateway route/core/helper、文档与记忆编排等业务复用层；推理服务内部模型封装和 server 入口直接随 `emotion_inference_server` / `multimodal_inference_server` 产物迁移。

### 7.5 VLM 幻觉治理策略

多模态第一阶段不应把 VLM 输出当作强事实源，尤其不能直接用单帧 VLM 描述覆盖用户身份、性别、表情或情绪判断。此前 E2E 观察已经显示，Qwen VL 3B 对人脸、身份、性别和客体持续性存在幻觉，场景识别相对更可用。

因此 VLM 输出需要分层使用：

| 输出类型 | 第一阶段策略 |
|----------|--------------|
| 场景/环境 | 可作为低风险上下文注入 |
| 动作变化 | 可作为事件提示，但需跨帧确认 |
| 物体存在 | 需要连续帧或缓存一致性确认 |
| 身份/性别/人脸属性 | 默认不作为可靠事实 |
| 表情/情绪 | 不直接写入情绪状态机 |

建议维护视觉事件而不是直接维护“结论”：

```text
visual_event = {
  scene_hint,
  action_hint,
  object_hint,
  confidence,
  source_frame_id,
  temporal_consistency
}
```

这些事件可以进入 LLM context 或语义/视觉缓存，但进入情绪状态机前必须经过更严格的门控。

### 7.6 下一版验收标准

WebRTC/VLM 下一版建议以“多路稳定推流 + 受控 VLM observation”为验收目标，而不是以单路链路能否跑通为目标：

- Playwright smoke 能以 `--sessions > 1` 建立多路 WebRTC 推流，并输出每路 frame/session 统计。
- 多路推流下 session registry、reconnect checkpoint、close/cleanup 不泄漏 session 或 pipeline。
- GStreamer fatal bus error、浏览器刷新、重复 ICE、重复 close、pipeline 创建失败不会拖垮主 Server。
- compute pool / IO pool 有明确队列容量、丢帧或降采样策略，不能无限堆积关键帧和 VLM 请求。
- sampler 产生的关键帧能编码为 VLM 请求输入，并通过 `IVlmVisionClient` 或 adapter 异步提交。
- VLM 返回的 `VisionInferenceResult` 能转换为 `VisionEvent.analysis` / `vision.observe` observation。
- VLM cache、视觉事件 dedup、session 连续命中降权和 rate limit 至少覆盖基础路径。
- 视觉 observation 可进入 LLM context，但不将单帧 VLM 输出直接写入情绪状态机或 L3 长期事实。
- Windows 本地 smoke 与 Linux/container 构建路径互不污染。

建议推进顺序：

```text
1. 使用现有 Playwright smoke 验证多路推流、断线重连、close/cleanup。
2. 为 media pipeline 增加关键帧编码与 VLM 请求调度，不在 GStreamer callback 内调用 VLM。
3. 实现正式 IVlmVisionClient adapter，对接本地 multimodal_inference_server 或 OpenAI-compatible VLM。
4. 接入 VLM cache、事件 dedup、rate limit 与 session 级连续命中降权。
5. 将 VLM result 转成受控 vision.observe observation，并注入 persona runtime。
6. 做多路长稳压测，记录 CPU、内存、队列长度、VLM 调用频率和失败率。
7. 最后再评估是否让高置信、多帧一致的视觉事件影响情绪状态机。
```

VLM 当前阶段定位：

```text
场景/动作辅助感知，而不是直接决定用户情绪。
```

## 8. 当前建议执行顺序

短期建议：

```text
1. 语义缓存主链路保守接入。
2. 情绪 domain/OOD pre-gate。
3. 赛前或交付前做一次全量 E2E 回归。
4. 源码交付清单整理。
5. 可复用 service targets 与推理 gRPC server 产物迁移清单整理。
```

赛后或下一阶段：

```text
1. BERT/情绪模型升级。
2. 真实 Agent 数据闭环。
3. 多模态多路推流验证与 VLM 正式接入。
4. 语义缓存从 reference 模式升级到 selective bypass。
```

不建议继续作为主方向投入：

```text
1. 继续让 LLM 参与普通 softmax logits 平均。
2. 无限制增加关键词试图全局超过 BERT。
3. 在没有 domain gate 的情况下扩大 LLM fallback 调用。
4. 让 VLM 直接接管情绪判断。
```

## 9. 与旧 Roadmap 的关系

`NEXT_RUNTIME_ROADMAP.md` 记录了从 tokenizer、embedding、vector storage、semantic cache、document pipeline 到 runtime foundation 的历史路线图。该文档仍保留历史价值。

本文档是 2026-06-20 之后的当前执行路线图，用于指导后续最值得投入的工程任务。
