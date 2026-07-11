# 当前 Runtime 后续路线图

> 日期：2026-06-20  
> 实现核对：2026-07-11
> 目的：在正式 Server、E2E 测试、文档解析链路、L3 长期记忆、语义缓存基础设施、情绪感知融合层基本完成后，重新整理下一阶段最值得投入的工程方向。

## 1. 当前阶段判断

当前系统已经基本具备旧 MVP 功能的完整 C++ 后端能力，并且已经形成较完整的 E2E Server 与正式 Server 入口。P0 级别的“能跑通主链路并交付”的能力已经基本完成，后续 P0 不再作为主要开发任务，而是作为每轮重要改动后的回归门槛。

已经具备的关键能力：

- 正式 Server 入口已经集成主链路、文档解析链路、长期记忆 L3、守护线程等核心接口。
- E2E Server 已有较完整测试基础，可作为正式 Server 的回归参照。
- Redis、SQLite、语义缓存、文档解析缓存、user registry 等基础设施已经接入。
- BERT gRPC 情绪推理链路可用，支持 ONNX / CUDA / CPU 与 batch 推理。
- BERT/VLM 推理端 gRPC 已统一 RPC 边界：业务失败与异常统一映射为 `core::Status`/gRPC status，失败日志携带 trace/method/request 上下文，trace ID 通过 metadata 回传，统计覆盖校验、认证、业务失败和未知异常；进程内真实 gRPC 测试已覆盖主要错误路径与流式失败。
- 情绪融合层已经从简单 softmax evidence 融合演进为 BERT/keyword/vector 快速融合 + LLM gate 仲裁。
- MAP 标定器已经能评估 mixed5000 与 SMP2020，并输出 LLM gate 阈值扫描。
- Docker 多容器部署方向已经明确，BERT/VLM 推理服务更适合作为独立容器。
- 开源迁移边界已经建立：项目公开名称为 AgentLoom，专有教学评估实现和指标配置从当前源码树移除，Gateway 通过 `IReportEvaluator` 接受下游领域 provider；本机 `.clangd`、CMake cache 和 E2E 模型配置保留在本地并由 `.example` 文件替代公开版本。
- 构建树已提供 `AgentLoom::*` CMake alias，便于下游通过 `add_subdirectory()` 依赖核心库；安装式 `find_package(AgentLoom)` 尚待公共头文件和 export 边界稳定后补充。

当前最重要的阶段性结论：

```text
P0 已完成，后续作为回归门槛；
P1 的语义缓存参考注入基础版已经接入，进入策略与回归收口；
P2 补完情绪层 domain/OOD pre-gate；
VLM 输入底座、Prompt KV Cache 和 runner pool 已推进，但正式 Gateway/VLM adapter 尚未闭环。
开源源码边界和基础文档导航已完成，后续按普通 release gate 维护。
```

## 2. 更新后的优先级

| 优先级 | 方向 | 当前定位 |
|--------|------|----------|
| P0 | 交付态回归验证 | 已基本完成，作为 release gate |
| P1 | 语义缓存主链路接入 | 基础参考注入已完成，待命中分层、抗重复和正式回归 |
| P2 | 情绪层 domain/OOD pre-gate | 情绪感知工程闭环的最后关键结构 |
| P3 | BERT/情绪模型升级与数据闭环 | 赛后或下一阶段模型工程 |
| P4 | VLM 多模态实时链路 | 组件与 smoke 底座已完成，待正式 Gateway 接线、VLM adapter 和多路验证 |

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

## 4. P1：语义缓存主链路接入（基础版已完成）

语义缓存已经不再只是底层基础设施。`SemanticMemoryContextProvider` 会按 user/session/persona 构造 L0 查询，将命中 payload 作为 `<memory_l0>` 参考上下文注入 system message；每轮成功回复后，`AdmitTurn` 会把用户输入、回复和情绪状态写回缓存。`tests/service/persona_runtime_test.cpp` 已覆盖命中注入、最近十轮原始对话和回写路径。

已经具备：

- L0 语义搜索算法。
- Redis 语义缓存基础设施。
- `CacheRecord` 扩展。
- embedding/vector 基础设施。
- batch 检索能力。
- 文档解析缓存接入经验。
- Persona 主链路 L0 lookup、参考上下文注入和 turn admission。
- scope、answer type、fingerprint、quality、expiry 与上下文风险策略接口。

### 4.1 当前策略：L0 reference 注入

当前主链路采用保守策略：

```text
semantic cache hit
  -> 生成 <memory_l0> reference block
  -> 注入 LLM system context
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

### 4.2 命中分层（待补）

当前 `CacheLookupResult` 已返回 `similarity_score`，但 Persona 注入层尚未按相似度改变 payload 形态或注入强度。下一步建议按相似度和作用方式分层：

| 命中层级 | 策略 |
|----------|------|
| High similarity | 注入缓存答案摘要、结构、关键事实，允许 LLM 复用 |
| Mid similarity | 只注入相关历史答案或关键点，不要求复用表达 |
| Low similarity | 不使用缓存 |

### 4.3 抗重复策略（待补）

当前缓存记录与 Persona 注入层尚未维护 session 连续命中、`hit_count` 或表达去重状态。需要避免用户明显感知到“复制粘贴”：

- 不直接返回原缓存文本。
- 缓存记录维护 `hit_count`。
- 同一 session 连续命中同一 cache record 时降权。
- `hit_count` 高时提示 LLM 换表达或重组结构。
- 当前问题与缓存问题存在关键差异时，只作为参考，不作为答案。
- 置信不足、上下文冲突或用户显式要求新答案时回退完整 LLM。

### 4.4 后续进阶：selective bypass

`PersonaRuntime` 已预留 `IAnswerCacheProvider`，并具备命中后跳过 LLM 的执行分支；但仓库中尚无正式 `IAnswerCacheProvider` 实现，也未在正式 Server 中接线。因此当前生产路径仍应以 `<memory_l0>` reference 注入为准，不能把接口位视为 bypass 已交付。

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

VLM 仍然是高价值方向。WebRTC signaling、GStreamer `webrtcbin` media pipeline、`appsink` decoded frame 输出、OpenCV 动态抽帧、视觉事件聚合和 smoke 工具已经形成可测试组件，但它们目前主要由 `webrtc_signaling_smoke_server` 组装，正式 `agent_gateway_server` 尚未挂载 `/ws/vision/signaling`。因此当前状态应定义为“组件与 smoke 底座完成，生产链路尚未闭环”，而不是正式实时视觉输入已经交付。

已经具备：

- `src/media/webrtc_signaling_handler.*`：实现 offer/answer/ICE/close/resume/config；当前由 smoke server 挂载到 `/ws/vision/signaling`，正式 Gateway 尚未接线。
- `src/media/webrtc_session_registry.*`：维护 WebRTC session、connection、reconnect token、frame checkpoint，并支持 Redis checkpoint store。
- `src/media/webrtc_media_pipeline.*`：创建 GStreamer pipeline，接入 `webrtcbin`，通过 `decodebin -> videoconvert -> appsink` 输出 RGB frame。
- `src/media/opencv_frame_sampler.*`：按 session 维护 MOG2、直方图、边缘变化、EMA、cooldown 和可选自适应 FPS 抽帧状态。
- `src/media/frame_encoding.*`：定义编码帧协议，并通过 GStreamer appsrc/appsink 将 RGB/BGR 关键帧编码为独立 JPEG/PNG；支持 Auto/Software/NVIDIA/D3D11/VAAPI/QSV image-encoder preference、插件探测和软件回退。
- `src/media/inference_frame_gateway_producer.*`：把编码帧元数据和 payload 映射到共享内存 IPC producer，不向 WebRTC pipeline 泄漏 slot 生命周期。
- `src/media/vision_runtime_interfaces.*`：定义 `IFrameSampler`、`IVlmVisionClient`、`VisionEvent`、`VisionEventMonitor`、dedup sink 和 context provider。
- `src/service/persona/skill_vision_event_sink.*` 与 `PersonaRuntime`：将视觉事件记录为 `vision.observe` observation，并按置信度、duplicate/rate-limited 状态控制 prompt 注入。
- `tools/playwright_webrtc_signaling_smoke.py` 与 `tools/webrtc_signaling_smoke_server.cpp`：已具备浏览器 WebRTC smoke、多 session 参数、解码帧落盘和 sampler 验证入口。
- `tests/media/*`：覆盖 signaling、session checkpoint、pipeline 创建、OpenCV sampler、事件聚合和 dedup。
- `LlamaRunnerPool`、memory/Redis `IPromptKvCache`：共享只读模型和 mmproj，每个 runner 使用独立 context，并支持 exact image-prefix sequence state 恢复。

仍需验证或补齐：

- 正式 `agent_gateway_server` 挂载 signaling handler、session registry、维护任务和视觉 event sink。
- 多路浏览器推流下的 CPU、内存、线程池队列、GStreamer pipeline 生命周期和关闭回收。
- `IVlmVisionClient` 当前只有接口，没有 gRPC/OpenAI-compatible 具体 adapter；关键帧已经可以编码并进入共享内存 IPC，推理端 backlog -> coordinator -> result table 核心已完成，但正式 Gateway signaling 路由和真实 VLM adapter 尚未完成生产组装。
- VLM 请求需要放入 IO pool 或独立 worker，不能在 GStreamer callback、WebSocket 线程或主对话线程中执行。
- 视觉事件进入 persona runtime 的受控注入已实现，但目前事件主要来自 sampler 元数据；VLM analysis 字段尚无生产数据源。
- `DetectSaliency` 仍返回固定零值；`vit_model` 配置尚未对应实际 ViT 人脸/情绪推理实现。
- Prompt KV Cache 与 runner pool 缺少专门的单元测试、Redis 跨进程恢复测试和可复现性能基线。
- NVIDIA/VAAPI/D3D11 等硬件解码策略仍需按平台验证，CPU-only fallback 必须保留。

### 7.1 当前链路定位

当前链路应视为“可由 smoke server 组装验证的实时视觉输入与抽帧组件已经具备”，而不是正式 Gateway 链路已经完成。更准确的实现状态是：

```text
Browser camera
  -> /ws/vision/signaling (smoke server 已挂载，正式 Gateway 待接线)
  -> GStreamer webrtcbin media pipeline (已实现)
  -> decodebin / videoconvert / appsink (已实现)
  -> OpenCV dynamic frame sampler (已实现)
  -> VisionEvent / monitor / dedup (已实现)
  -> JPEG/PNG frame encoder -> IO pool -> shared-memory producer (组件与双进程 E2E 已实现，正式 Gateway 接线待完成)
  -> inference receiver -> private backlog -> VLM coordinator -> result table（核心已实现，生产组装待完成）
  -> VisionAnalysis -> controlled vision.observe observation (注入端已实现，数据源待接通)
```

这一阶段仍不建议让 VLM 直接闭环到情绪状态机。更稳的定位是：先把多路推流、抽帧、关键帧事件、VLM observation 和主链路注入做成可控工具会话。

### 7.2 信令与会话层现状

项目现有 WebSocket/Boost.Beast/Asio 封装已经被 WebRTC signaling 组件复用。当前端点由 `tools/webrtc_signaling_smoke_server.cpp` 挂载为：

```text
/ws/vision/signaling
```

handler 负责：

- 交换 SDP offer/answer。
- 交换 ICE candidate。
- 返回 ICE server config。
- 维护 connection id、session id、trace id 和 reconnect token。
- 支持 resume / close。
- 在 pipeline 创建失败、未知消息、断线和重连失败时返回明确错误。

媒体数据仍通过 WebRTC RTP/RTCP 进入 GStreamer 管线，不通过 WebSocket 传输大体积帧。正式 Gateway 还需要注册该 handler，并接入 registry 生命周期和维护任务；完成接线后，再验证多 session 并发、断线重连、刷新、重复 close、pipeline fatal error 等生产路径。

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

`appsink` 输出帧后通过 `core::ThreadPool` 投递给 `IFrameSampler`，避免在 GStreamer callback 中执行重计算。`VideoFrameView` 现包含 `row_stride_bytes`，sampler 和 encoder 均支持带行 padding 的 RGB/BGR buffer。sampler 命中后可同时发布 saliency `VisionEvent`，并调用 `IVideoFrameEncoder` 生成 `EncodedVideoFrame`；编码完成后通过 IO pool 调用 `IEncodedVideoFrameSink`，因此共享内存 publish 不运行在 GStreamer callback。

编码器只选择适合独立关键帧的 JPEG/PNG image encoder。Auto 模式按 `nvjpegenc -> qsvjpegenc -> vaapijpegenc/vajpegenc -> d3d11jpegenc -> jpegenc/avenc_mjpeg` 探测；强制硬件 preference 在目标插件缺失时返回 `Unavailable`，Auto 才回退软件。本机 GStreamer 具备 `nvh264enc`/`mfh264enc`，但不具备 `nvjpegenc`，因此实测正确选择 `jpegenc`。有状态 H.264/NVENC 没有被用于 VLM 单关键帧协议，避免在推理端引入额外视频解码状态。

硬件解码建议保持为渐进能力：

- NVIDIA 环境优先验证现有 decoder preference 与 GStreamer 插件可用性。
- Linux VAAPI 和 Windows D3D11 需要分别验证，不应让平台单一 API 成为默认路径。
- CPU-only 软件解码 fallback 必须作为可运行基线。
- VLM 推理服务仍建议独立容器或独立进程，Gateway 只负责输入、采样、事件和请求编排。
- 推理模块迁移按已编译 gRPC server、protobuf/gRPC 头文件和运行时依赖交付；不再作为 Gateway/service 可复用库继续拆分。

### 7.4 多容器边界

多模态链路目标上仍建议保持多容器架构。下表描述目标职责，而不是当前正式 Gateway 已完成的接线：

| 容器 | 职责 |
|------|------|
| Gateway | HTTP/WebSocket/WebRTC signaling、session、记忆、缓存、主链路编排、 GStreamer/WebRTC media、帧抽样、VLM 请求与视觉事件生成 |
| BERT Emotion Server | 文本情绪推理 |
| Redis | 缓存、会话元数据、语义/视觉事件存储 |

VLM 推理本身不建议和 Gateway 强耦合，因为模型依赖、GPU/CUDA/驱动兼容性和资源占用都明显不同。

BERT/VLM 推理服务的项目间迁移边界是 gRPC 进程和协议头文件。可迁移库目标应集中在 Persona Runtime、Session、Classroom、Gateway route/core/helper、文档与记忆编排等业务复用层；推理服务内部模型封装和 server 入口直接随 `emotion_inference_server` / `multimodal_inference_server` 产物迁移。

### 7.5 VLM 幻觉治理策略

多模态第一阶段不应把 VLM 输出当作强事实源，尤其不能直接用单帧 VLM 描述覆盖用户身份、性别、表情或情绪判断。此前 E2E 观察已经显示，Qwen VL 3B 对人脸、身份、性别和客体持续性存在幻觉，场景识别相对更可用。

可采用专业的人脸识别+情绪训练头的VIT代替VLM直接识别人脸特征，可能更有效，且相当轻量。保留GLM 4V作为最终回退，但依然不建议大量依赖云API。

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
2. 将现有关键帧编码、IO-pool IPC producer 和 signaling handler 组装进正式 Gateway 配置与生命周期。
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

### 7.7 帧传输层优化：零拷贝共享内存 IPC

VLM 推理服务与 Gateway 出于**故障隔离**保持进程分离（VLM 崩溃/挂起不能拖垮主 Gateway，现有 gRPC deadline、默认 health check，以及统一异常/状态/trace/统计边界已具备基础故障检测与诊断能力）。当前 VLMRequest.image_data 使用 gRPC bytes 传输 JPEG/PNG 图像，vlm_frame_dir_quality_test.py、vlm_grpc_memory_stress.py 和视觉 E2E 已实际走 GenerateVLMSync 发送帧；尚未完成的是 WebRTC media pipeline 到该 gRPC 协议的正式 IVlmVisionClient adapter。图像 payload 单帧通常为几十至上百 KB，若传未压缩 1080p RGB 则可达数 MB；protobuf 序列化、发送缓冲、loopback 和反序列化开销会随帧率与多路会话放大。

结论：**保留进程隔离与 gRPC，但把帧本体从 gRPC payload 中剥离**。

```text
数据面：Boost.Interprocess 共享内存 slot，帧 payload 不进入 protobuf/gRPC
控制面：slot 内固定宽度元数据 + per-slot sequence；后续可增加轻量 wake/health 信号
```

当前第一版没有让 gRPC 承载逐帧描述符，而是把固定宽度描述符直接放在共享 slot header 中。这样 publish/claim/ack 不依赖 RPC 往返；gRPC 继续承担服务控制、health、异常诊断和不适合共享内存的数据。该模式与 ROS2/Triton 的 shared-memory data plane 思路一致，但具体 slot 协议由本项目实现并测试。

#### 共享内存布局

```text
[RegionHeader]
  magic/version/header_size/region_size
  slot_count/payload_capacity
  epoch
  enqueue_position/dequeue_position
  published/claimed/acknowledged/rejected counters

[SlotHeader × N]
  sequence + epoch
  session_id/trace_id/frame_id/timestamp_us
  width/height/format/saliency/payload_size

[Payload × N]
  固定容量帧字节区
```

共享区只包含固定布局标量、定长字符数组和 `std::atomic<T>`，不放进程地址、裸指针或 STL 容器。slot 数至少为 2；当前默认 64 slots，每槽默认 4 MiB，部署时应按编码帧上限和会话数调小或调大。

#### 生命周期与竞态收敛

- **有界 MPMC sequence ring**：每个 slot 使用独立 `sequence`；producer 在 payload/metadata 写完后 `store(release)` 发布，consumer 以 `load(acquire)` claim，ack 后推进到下一轮可写 sequence。多个 Gateway 投递线程和多个 receiver 线程可并行使用不同 slot。
- **claim 使用 RAII**：`ClaimedSharedFrame` 是 move-only；显式 `Acknowledge()` 与析构都会且只会释放一次 slot，异常路径不会遗留普通未 ack claim。
- **只在 receiver 持有共享视图**：receiver 同步执行 `shared slot -> core::MemoryBlock` 一次 memcpy，随后立即 ack；backlog、worker、VLM/API 永远只拿私有 `OwnedInferenceFrame`。
- **当前满策略为 RejectNewest**：返回 `ResourceExhausted`，由上游采样/投递策略决定重试或丢帧；尚未实现覆盖 claimed/in-flight slot，也不允许直接覆盖最旧 slot。
- **session 隔离发生在私有 backlog**：共享 ring 负责高吞吐传输，复制后的帧再按 session 分区调度和按时间戳重组。

#### 跨平台

- 共享内存创建、打开、映射和移除使用 **Boost.Interprocess 1.85** RAII；不直接依赖 `CreateFileMapping`、`shm_open` 或 `mmap`。
- 数据面同步只使用 `std::atomic<uint64_t/uint32_t>` acquire/release/CAS，并通过 `static_assert(...is_always_lock_free)` 拒绝不满足跨进程原子前提的平台/ABI。
- region/slot 使用 cache-line 对齐，协议包含 magic、version、layout offsets 和 epoch；打开端会验证完整布局后才访问 slot。

#### 故障隔离衔接

region `epoch` 用于隔离 remove/recreate 后的新旧实例，打开端拒绝不兼容布局。当前已保证正常返回、校验失败、内存池失败、backlog 拒绝和 C++ 异常路径都会 ack。`RecoverableInferenceFrameIpcSink` 与 `ReconnectableInferenceFrameIpcSource` 提供稳定业务接口下的 producer recreate / consumer reconnect；真实双进程测试会让 consumer 在持有 claim 时直接 `_Exit`，确认旧 ring 因 stale slot 进入背压，再执行 remove/recreate、验证 epoch 变化，并由新 consumer 恢复读取。

当前恢复仍是**协调式 region 重建**，而不是旧 region 内原地回收：正式服务需要由 gRPC health/进程监督器触发 producer `Recreate()` 和 inference source `Reconnect()`。自动判断 peer owner、在不重建 region 的情况下回收 writing/claimed slot，仍需要 heartbeat/lease 或独立 generation owner 协议；目前没有伪装成已完成。

#### 渐进落地路径

当前已完成 **RTC 关键帧编码 + Gateway producer adapter + 共享内存传输核心 + 推理端 receiver + 私有调度与 coordinator 核心**：`GStreamerVideoFrameEncoder` 完成 image encoder capability probe、JPEG/PNG 编码和 stride 处理；`InferenceFrameGatewayProducer` 把编码帧发布到 IPC；`agent_ipc` 提供 Boost.Interprocess region、MPMC sequence publish/claim/ack、epoch/layout 校验、RAII claim 和协调式 recreate/reconnect；`InferenceFrameIpcReceiver` 完成格式校验、唯一一次 pooled private copy、立即 ack 和 backlog submit；`SegmentedInferenceFrameBacklog` 完成 session 隔离和并发取帧；`InferenceFrameCoordinator` 使用现有 `core::ThreadPool` 调用同步 `IVlmVisionClient`，把成功、业务失败和清洗后的异常都发布为 terminal result；结果表在 session 内按 `timestamp_us + frame_id` 重组。

```text
已完成：RGB/BGR stride-aware JPEG/PNG 编码、硬件 image encoder 探测与软件回退、
        Gateway producer adapter、共享内存 MPMC publish/claim/ack、receiver 私有拷贝、
        backlog/coordinator/result table、真实双进程 JPEG E2E、claim 中途强退故障注入、
        协调 recreate/reconnect、RTC->IPC->FakeVLM 准 E2E 分阶段压测。
下一步：正式 Gateway `/ws/vision/signaling` 配置与生命周期组装、真实 IVlmVisionClient adapter、
        gRPC health/进程监督器自动触发 recreate/reconnect、轻量 wake/control。
随后：真实多路有限 Skill RTC 长稳压测，验证 mapped spool 零静默丢失、Seal-to-drain 延迟、
      segment 回收和 closing timeout；不覆盖 Reading/claimed slot，如确有必要再实现旧 region 内 stale-slot 原地回收。
```

该实现已经越过原路线中的 mutex 原型，直接采用经过独立映射、多线程 exactly-once 测试的原子 sequence ring；仍需以真实双进程故障注入和视频负载压测作为生产启用门槛。

#### RTC -> IPC -> FakeVLM 准 E2E 压测基线（2026-07-11）

`media_rtc_ipc_fake_vlm_stress` 已把真实 `OpenCvFrameSampler`、`GStreamerVideoFrameEncoder`、Gateway producer、shared-memory IPC、receiver、private backlog、coordinator 和 result table 串成可归因链路。输入端使用 synthetic decoded RGB，VLM 使用可配置延迟 FakeVLM；IPC 使用同机独立 create/open 端点。因此它用于定位队列和 admission 拐点，不替代真实浏览器 RTP/WebRTC decoder、真实双进程长稳或真实模型压测。

首轮 320x180 JPEG、3 秒场景结果：

| 场景 | 输入 | selected | receiver/backlog 拒绝 | post-sampler 丢失率 | 高 saliency 丢失率 | 事件覆盖 |
|------|------|----------|-----------------------|-------------------|---------------------|----------|
| balanced | 4 session x 15 FPS，4 VLM worker x 20 ms | 147 | 0 | 0% | 0% | 8/8，100% |
| moderate | 8 session x 30 FPS，2 VLM worker x 100 ms | 656 | 508 | 77.4% | 86.5% | 16/16，100% |
| vlm_overload | 8 session x 30 FPS，2 VLM worker x 300 ms | 656 | 590 | 89.9% | 98.7% | 6/16，37.5% |
| synchronized_burst | 8 session x 30 FPS，2 VLM worker x 250 ms，小 IPC/backlog | 656 | 602 | 91.8% | 97.4% | 8/16，50% |

这批数据的首要结论不是“共享内存吞吐不足”，而是当前 `RejectNewest` private backlog admission 在 VLM 服务率低于 selected frame 到达率时会直接丢弃绝大多数已经抽样的高价值帧：四组场景的 compute queue、编码、IO queue 和 IPC publish 均为零丢失，过载损失全部出现在 receiver 向 private backlog 提交阶段。`moderate` 虽然 frame-level 丢失已达 77.4%，但每个事件仍至少保留一帧；继续增加 VLM 延迟后，事件唯一代表帧开始被一起丢弃，事件覆盖迅速跌至 37.5%-50%。

由于项目的媒体处理始终绑定有限 `SkillSession`，下一版目标不再是为 selected frame 选择常规丢弃算法，而是使用推理端 private file-backed mapped spool 保存未处理帧，在 Skill input Seal 后完成 VLM drain 和事件 finalize，再把 Skill 从 `Closing` 转为 `Closed`。shared-memory claimed/in-flight slot 继续禁止覆盖；`RejectNewest`、`DropOldestReady`、latest-wins 和 event-aware replacement 只保留为 benchmark 或异常资源耗尽策略，不作为正常有限 Skill 路径。详细设计见 `docs/SKILL_MEDIA_DRAIN_SPOOL_DESIGN.md`。当前 sampler 选中率在合成高变化输入上约 82%-91%，该比例用于制造压力，不代表真实摄像头分布。

### 7.8 VLM Prompt Cache 与并行化（核心实现完成，回归待补）

当前代码已经完成基于 llama.cpp sequence state 的严格前缀 Prompt Cache，并与共享模型、独立 context 的 runner pool 结合。已有本地测试观察到两者组合相对未优化基线约 **70%-80% 的整体效率提升**；但仓库尚缺固定输入、固定硬件条件下的自动化 benchmark，因此该数字应视为阶段性观测，不是稳定回归指标。

当前实现只复用可证明完全一致的前缀：

```text
cache key = cache schema version
          + model/mmproj fingerprint
          + session_id
          + exact SHA-256(image bytes)

cached state = 固定 chat-template 前导文本
             + image marker
             + 完全相同图像的 image tokens
```

命中后恢复该 sequence state，跳过已经缓存的固定前导文本与图像 chunk，但当前请求中图像之后的 prompt suffix 和所有生成 token 仍然重新推理。生成完成后删除 prefix 之后的 KV，只保留严格可复用前缀；恢复、tokenize、decode 或生成失败时清理对应 context，避免跨请求状态污染。

缓存后端支持进程内 memory cache 和带 TTL 的 Redis cache。runner pool 共享只读模型与 mmproj，每个 runner slot 持有独立 llama context 和 sampler；因此并行请求不会共享可变 KV context。

经过讨论和测试，以下两类方案风险高于当前收益，**不纳入正式实现**：

- 按 prompt template 长期绑定 `seq_id` 并跨 runner/context 复用状态。该方案会增加 sequence 生命周期、并发调度、卸载重载和异常清理复杂度，而现有序列化严格前缀方案已经取得足够收益。
- Vector Cache 命中后加载相似图像或历史请求的 KV state，再只重推末尾 token。近似图像的 image embeddings 与历史 KV 不具备严格位置和 attention 对齐保证，可能造成跨帧语义污染，不能用 cosine/SSIM 阈值替代一致性证明。

Exist Cache 与 Vector Cache 继续用于结果级缓存和检索；Prompt Cache 只负责 exact image-prefix KV 复用，两类缓存不互相转换。后续工作以命中率、prompt eval、吞吐量、runner queue、恢复失败率和跨进程 Redis 恢复测试为主，不再扩展近似 KV 复用范围。

当前缺口：

- 为 sequence state store/restore、失败清理和同 runner 热命中增加单元测试。
- 为 memory/Redis backend 增加 TTL、容量淘汰、损坏 payload 和跨进程恢复测试。
- 为 runner pool 增加并发 acquire/release、Unload 等待和异常请求隔离测试。
- 将 `prompt_kv_cache_hit` 暴露到 gRPC response/统计指标，建立可复现 benchmark。

## 8. 当前建议执行顺序

短期建议：

```text
1. 对语义缓存 reference 注入、Prompt KV Cache、runner pool 和 persona 配置改动做定向与全量回归。
2. 补语义缓存命中分层、连续命中降权和可观测性。
3. 实现情绪 domain/OOD pre-gate。
4. 为 Prompt KV/runner pool 建立专门测试与可复现性能基线。
5. 在公共 header 边界稳定后补 `install(EXPORT)`、`AgentLoomConfig.cmake` 和独立 consumer build test。
```

赛后或下一阶段：

```text
1. BERT/情绪模型升级。
2. 真实 Agent 数据闭环。
3. 正式 Gateway 挂载 WebRTC signaling，接入现有关键帧编码/IPC producer，并实现 VLM coordinator/analysis adapter。
4. 多模态多路推流、断线恢复与长稳压测。
5. 语义缓存从 reference 模式升级到 selective bypass。
```

不建议继续作为主方向投入：

```text
1. 继续让 LLM 参与普通 softmax logits 平均。
2. 无限制增加关键词试图全局超过 BERT。
3. 在没有 domain gate 的情况下扩大 LLM fallback 调用。
4. 让 VLM 直接接管情绪判断。
5. 基于相似图像或向量命中加载历史 KV state。
6. 为 prompt template 维护跨 runner/context 的长期 `seq_id` 状态。
```

## 9. 与旧 Roadmap 的关系

`NEXT_RUNTIME_ROADMAP.md` 记录了从 tokenizer、embedding、vector storage、semantic cache、document pipeline 到 runtime foundation 的历史路线图。该文档仍保留历史价值。

本文档是 2026-06-20 之后的当前执行路线图，用于指导后续最值得投入的工程任务。

Written By:Orange & Claude & Codex
