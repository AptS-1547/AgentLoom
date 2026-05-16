# 视觉链路完整分析

**文档版本**: v0.0.1beta1  
**分析日期**: 2026-05-05  
**涉及项目**:
- C++ 推理端: `AgentBackendPredict`
- Python 主项目: `MyNeuroLikeSystem`
- 比赛特化子项目: `EducationalAgentProject`

---

## 1. 架构概览

视觉链路是一个跨语言、多层次的实时视觉感知系统，从底层 C++ 推理到高层 Python 语义理解，完整流程如下：

```
┌─────────────────────────────────────────────────────────────────┐
│                      视觉输入源                                   │
│  (摄像头 / 视频文件 / 屏幕捕获)                                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Python 视觉感知层 (MyNeuroLikeSystem)                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ VisualPerceptionPipeline                                 │   │
│  │  - 帧采样 (AdaptiveFrameSampler)                         │   │
│  │  - 显著度检测 (MOG2 + 多特征融合)                         │   │
│  │  - 峰值选择 (TemporalPeakSelector)                       │   │
│  │  - 事件聚类 (VisualEventMonitor)                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              VLM 推理层 (C++ + Python)                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Python LLMClient (EducationalAgentProject)               │   │
│  │  - 图片预处理 (image_utils.py)                           │   │
│  │  - 缓存管理 (URL SHA256)                                 │   │
│  │  - API 调用 (Anthropic Vision API)                       │   │
│  └──────────────────┬───────────────────────────────────────┘   │
│                     │ (可选路由)                                 │
│                     ▼                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ C++ gRPC 推理服务 (AgentBackendPredict)                  │   │
│  │  - multimodal_inference_server                           │   │
│  │  - VLM 缓存 (vlm_cache)                                  │   │
│  │  - llama.cpp + mtmd                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              语义理解层 (MyNeuroLikeSystem)                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LLMVisualEventAnalyzer                                   │   │
│  │  - 结构化分析 (scene/facts/interpretations)              │   │
│  │  - 情绪信号提取 (visual_semantics.py)                    │   │
│  │  - 记忆候选生成                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Agent 集成层 (EducationalAgentProject)               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ AgentLoop                                                │   │
│  │  - 视觉事件转 AgentEvent                                 │   │
│  │  - 情绪状态注入                                          │   │
│  │  - 记忆持久化                                            │   │
│  │  - 对话生成                                              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心模块详解

### 2.1 Python 视觉感知层 (MyNeuroLikeSystem)

#### 2.1.1 自适应帧采样器 (AdaptiveFrameSampler)

**位置**: `src/vision/adaptive_sampler.py`

**功能**: 基于 FFT 频域分析 + 瞬时突变触发的动态采样率控制

**三层控制机制**:
1. **稳态层**: STFT → 高频能量比 → 目标 fps (EMA 平滑)
2. **反馈层**: detector 回传 change_score 超阈值 → spike boost
3. **预检测层**: producer 对跳过帧做轻量 absdiff → 超阈值强制采样

**关键参数**:
```python
stft_window_size: 32          # STFT 窗口大小
fps_min: 4.0                  # 最低采样率
fps_max: 15.0                 # 最高采样率
spike_threshold: 0.4          # 突变触发阈值
precheck_diff_threshold: 15.0 # 预检测帧差阈值
```

**性能**: 预检测约 0.1ms/帧，有效避免 FFT 窗口滞后问题

#### 2.1.2 显著度检测器 (VisualPerceptionPipeline)

**位置**: `src/vision/visual_pipeline.py`

**功能**: 多特征融合的实时显著度检测

**检测流程**:
```
原始帧 → 缩放(320px) → 高斯模糊 → MOG2 背景减除 → 形态学处理
                                    ↓
                    连通域分析 → 面积特征 (sigmoid 归一化)
                    直方图差分 → 颜色特征 (sigmoid 归一化)
                    Canny 边缘 → 边缘特征 (sigmoid 归一化)
                                    ↓
                    加权融合 (0.5/0.2/0.3) → EMA 平滑 → 显著度分数
```

**关键参数**:
```python
resize_width: 320                 # 处理分辨率
mog2_history: 500                 # MOG2 历史帧数
mog2_var_threshold: 16.0          # 方差阈值
area_weight: 0.5                  # 面积权重
histogram_weight: 0.2             # 直方图权重
edge_weight: 0.3                  # 边缘权重
ema_alpha: 0.3                    # EMA 平滑系数
```

**时序投票机制**: 3 帧窗口，至少 2 帧投票通过才认为是真实运动

#### 2.1.3 峰值选择器 (TemporalPeakSelector)

**位置**: `src/vision/visual_pipeline.py`

**功能**: 从显著度时序中识别局部峰值，生成视觉事件

**峰值条件**:
1. 局部最大值 (3 帧邻域)
2. 超过阈值 (默认 0.15)
3. 通过时序投票
4. 冷却时间 (默认 0.5s)

**Clip 模式**: 支持 `clip_half_duration` 参数，在峰值前后采集连续帧序列，用于动作分析

**关键参数**:
```python
peak_threshold: 0.15              # 峰值阈值
cooldown_seconds: 0.5             # 冷却时间
window_seconds: 1.0               # 事件窗口
peak_neighborhood_frames: 3       # 邻域大小
clip_half_duration: 0.0           # Clip 半长 (秒)
```

#### 2.1.4 事件监控器 (VisualEventMonitor)

**位置**: `src/vision/visual_monitor.py`

**功能**: 聚类相邻视觉事件，生成时间窗口摘要

**聚类策略**:
- 时间间隔 < `segment_merge_gap_seconds` (默认 2s) 的事件合并为一个 cluster
- 每个 cluster 选择峰值最高的事件作为代表
- 计算 salience_score = peak + 0.35 × accumulated + event_bonus

**窗口摘要**:
- 维护 `weak_monitor_buffer_seconds` (默认 30s) 的滑动窗口
- 周期性生成 `VisualWindowSummary`，包含该窗口内所有 cluster
- 用于长时间活动的概括性描述

**关键参数**:
```python
weak_monitor_buffer_seconds: 30.0  # 窗口大小
segment_merge_gap_seconds: 2.0     # 聚类间隔
```

---

### 2.2 VLM 推理层

#### 2.2.1 Python 图片处理 (EducationalAgentProject)

**位置**: `agent/src/media/image_utils.py`

**功能**: 为 Anthropic Vision API 提供图片预处理

**特性**:
- 基于 URL SHA256 哈希的文件缓存，24h TTL 自动过期
- 超过 `max_dimension` (推荐 ≤1568px) 自动等比缩放
- 支持 jpeg/png/gif/webp 格式
- Pillow 可选依赖，未安装时禁用图片功能

**缓存机制**:
```python
cache_dir = Path.home() / ".cache" / "agent_images"
cache_key = hashlib.sha256(url.encode()).hexdigest()
cache_ttl = 24 * 3600  # 24 小时
```

**API 格式**:
```python
ImageResult(
    base64_data: str,      # base64 编码
    media_type: str,       # "image/jpeg" 等
    original_url: str      # 原始 URL
)
```

#### 2.2.2 C++ gRPC 推理服务 (AgentBackendPredict)

**位置**: `src/server/main/multimodal_inference_server.cpp`, `src/server/grpc/`, `src/service/inference/`

**功能**: 统一多模态推理服务，整合 BERT + VLM + ViT

**VLM 推理接口**:
```protobuf
service MultimodalInference {
  rpc GenerateVLM(VLMRequest) returns (stream VLMToken);      // 流式
  rpc GenerateVLMSync(VLMRequest) returns (VLMResponse);      // 同步
}

message VLMRequest {
  oneof image_source {
    bytes image_data = 1;       // JPEG/PNG 二进制
    string image_path = 8;      // 本地路径（零拷贝）
  }
  string prompt = 2;
  int32 max_tokens = 3;
  float temperature = 4;
  string session_id = 9;
  bool allow_cache = 12;        // 启用缓存
  bool force_refresh = 13;      // 强制刷新
}
```

**VLM 缓存系统** (`src/cache/vlm_cache.cpp`):

**缓存键生成**:
```cpp
cache_key = SHA256(image_data) + SHA256(prompt)
```

**缓存策略**:
- TTL: 默认 3600s (1 小时)
- LRU 淘汰: max_entries (默认 512)
- 持久化: 可选磁盘持久化，重启后恢复
- 降级读取: 推理失败时返回过期缓存 (可配置)

**性能指标**:
- 缓存命中: 响应时间 < 10ms
- 首次推理: ~5s (Qwen2-VL-3B, RTX 5060)
- 吞吐量: ~126 tokens/s

**图片解码**:
- 使用 `mtmd_helper_bitmap_init_from_buf()` 从内存直接解码
- 支持 JPEG/PNG/GIF/WebP
- 无需临时文件，零拷贝

**VRAM 管理** (`src/config/server_options.h`, `src/service/inference/multimodal_service.cpp`):
```cpp
struct VramGuardOptions {
    int monitor_interval_seconds = 10;           // 监控间隔
    size_t warning_free_bytes = 1GB;             // 警告阈值
    size_t unload_free_bytes = 512MB;            // 卸载阈值
    bool unload_on_oom_error = true;             // OOM 自动恢复
};
```

---

### 2.3 语义理解层 (MyNeuroLikeSystem)

#### 2.3.1 视觉事件分析器 (LLMVisualEventAnalyzer)

**位置**: `src/vision/visual_analysis.py`

**功能**: 将视觉事件转换为结构化语义分析

**分析模式**:
1. **trigger**: 短时间片段的显著动作
2. **summary**: 长时间窗口的活动概括
3. **manual**: 手动触发的分析

**Prompt 策略**:
- 优先描述主体人物及其附近物体
- 过滤背景干扰（次要人物、轻微光照变化）
- Clip 模式下强调时序连续性

**输出结构** (`VisualAnalysis`):
```python
{
  "scene": "10字以内的场景概括",
  "facts": ["最多3条可直接观察到的事实"],
  "weak_interpretations": ["最多2条弱解释，使用'可能'等低强度表述"],
  "memory_candidate": "1条适合长期记忆的压缩观察",
  "agent_hint": "1条适合发给主 Agent 的简短描述"
}
```

**容错机制**:
- JSON 解析失败时使用正则提取
- 支持不完整 JSON 的字段恢复

#### 2.3.2 情绪信号提取 (visual_semantics.py)

**位置**: `src/vision/visual_semantics.py`

**功能**: 从视觉分析文本中提取情绪信号

**情绪映射**:
```python
_EMOTION_VA_BASES = {
    "joy": (0.8, 0.6),          # (valence, arousal)
    "sadness": (-0.7, 0.2),
    "anger": (-0.4, 0.8),
    "fear": (-0.6, 0.7),
    "surprise": (0.1, 0.8),
    # ...
}
```

**提取策略**:
1. **正则匹配**: 关键词匹配（"微笑"、"难过"等）
2. **原型相似度**: 与情绪原型句子计算向量相似度
3. **阈值判断**: vector_threshold = 0.62

**输出**: 情绪标签 + valence/arousal 坐标，用于注入 Agent 情绪状态

---

### 2.4 Agent 集成层 (EducationalAgentProject)

#### 2.4.1 视觉事件转换 (visual_agent.py)

**位置**: `MyNeuroLikeSystem/src/vision/visual_agent.py`

**功能**: 将 `VisualEvent` 转换为 `AgentEvent`

**转换流程**:
```python
VisualEvent → visual_event_to_agent_text() → AgentEvent
```

**AgentEvent 结构**:
```python
AgentEvent(
    type="visual",
    content="[视觉事件] ...",
    chat_mode=ChatMode.PRIVATE,
    is_mentioned=True,
    images=[...],              # 关键帧列表
    metadata={
        "event_id": ...,
        "peak_score": ...,
        "analysis": {...},     # 结构化分析
    }
)
```

#### 2.4.2 情绪状态注入

**配置**: `VisualPerceptionConfig.inject_to_emotion_state = True`

**流程**:
1. `derive_visual_emotion_signal()` 从分析文本提取情绪
2. 注入到 `EmotionState` 的 `visual_emotion_signal`
3. 影响后续对话的情绪融合

#### 2.4.3 记忆持久化

**配置**: `VisualPerceptionConfig.persist_to_memory = True`

**流程**:
1. `visual_event_memory_text()` 生成记忆文本
2. 使用 `memory_candidate` 字段（如果存在）
3. 存储到 Qdrant 向量数据库（如果启用）

---

## 3. 数据流分析

### 3.1 完整数据流

```
视频帧 (30fps)
    ↓
AdaptiveFrameSampler (动态 4-15fps)
    ↓
VisualPerceptionPipeline
    ├─ MOG2 背景减除
    ├─ 多特征融合 (面积/直方图/边缘)
    ├─ EMA 平滑
    └─ 时序投票
    ↓
FrameObservation (显著度分数 + 元数据)
    ↓
TemporalPeakSelector
    ├─ 局部峰值检测
    ├─ 冷却时间过滤
    └─ Clip 模式采集
    ↓
VisualEvent (峰值 + 关键帧)
    ↓
VisualEventMonitor
    ├─ 时间聚类
    ├─ 窗口摘要
    └─ 限流控制
    ↓
LLMVisualEventAnalyzer
    ├─ 图片预处理 (image_utils)
    ├─ VLM 推理 (Anthropic API / C++ gRPC)
    └─ 结构化解析
    ↓
VisualAnalysis (scene/facts/interpretations)
    ↓
visual_semantics.py
    └─ 情绪信号提取
    ↓
visual_agent.py
    └─ 转换为 AgentEvent
    ↓
AgentLoop
    ├─ 情绪状态注入
    ├─ 记忆持久化
    └─ 对话生成
```

### 3.2 性能瓶颈分析

| 阶段 | 耗时 | 瓶颈 | 优化方案 |
|------|------|------|----------|
| 帧采样 | ~0.1ms | 预检测帧差 | 已优化 (INTER_NEAREST) |
| 显著度检测 | ~5-10ms | MOG2 + 特征提取 | 已降分辨率 (320px) |
| 峰值选择 | <1ms | 滑动窗口 | 无瓶颈 |
| VLM 推理 (首次) | ~5s | llama.cpp 推理 | 缓存 + GPU offload |
| VLM 推理 (缓存命中) | <10ms | 内存查找 | 无瓶颈 |
| 语义解析 | <1ms | JSON 解析 | 无瓶颈 |

**总体延迟**:
- 无缓存: ~5-6s (VLM 推理主导)
- 缓存命中: ~20-30ms (显著度检测主导)

---

## 4. 当前落地状态

### 4.1 各模块在代码库中的实际状态

下表按"实际代码所在位置"区分三个仓库。"原型"指链路在代码中已经走通，但尚未经过规模化压测或线上对齐。

| 模块 | 位置 | 状态 | 备注 |
|------|------|------|------|
| AdaptiveFrameSampler | MyNeuroLikeSystem | 原型可用 | 参数仍在手工调优，缺少不同视频源的基线数据 |
| MOG2 + 多特征显著度检测 | MyNeuroLikeSystem | 原型可用 | 针对桌面/室内场景调过，室外/强光下未验证 |
| TemporalPeakSelector / Clip 模式 | MyNeuroLikeSystem | 原型可用 | Clip 模式刚接入，连续段落的选帧策略仍在迭代 |
| VisualEventMonitor 聚类与窗口摘要 | MyNeuroLikeSystem | 原型可用 | summary 模式产出的 prompt 还在调整 |
| LLMVisualEventAnalyzer | MyNeuroLikeSystem | 原型可用 | 直连云端 VLM（Anthropic Vision），JSON 解析有容错兜底 |
| visual_semantics 情绪提取 | MyNeuroLikeSystem | 原型可用 | 目前走正则 + 原型相似度，尚未与 BERT 情绪分类对齐 |
| image_utils 图片预处理 | EducationalAgentProject | 稳定 | 提供给 Anthropic Vision API 的 base64/缩放/本地缓存 |
| BERT gRPC 推理（小模型） | EducationalAgentProject + AgentBackendPredict | 已上线 | 比赛生产端已经在用，走 gRPC |
| multimodal_inference_server（含 VLM/VLM 缓存/VRAM 监控） | AgentBackendPredict | 初版可跑 | README 自标为 v0.0.1beta1，尚未在比赛项目内接入 |
| VLM 缓存（内存 + 可选持久化） | AgentBackendPredict | 初版可跑 | 单机测试过基本命中逻辑，缺少长期运行 / 淘汰 / 并发下的实测数据 |
| ViT / DetectSaliency | AgentBackendPredict | 仅占位 | proto 定义了接口，服务端无实现 |
| 视觉链路 → 比赛项目主链路 | EducationalAgentProject | 未接入 | 比赛主回路目前只处理 BERT 情绪推理结果，没有处理 VisualEvent，这部分属于实验性内容，由于缺乏一定量的实测验证，上线服务器前需要更多考虑 |

### 4.2 已经打通的链路

- **感知到事件**：帧 → 采样 → 显著度 → 峰值 → 事件聚类，这一段在 MyNeuroLikeSystem 内部是连续可运行的。
- **事件到语义**：VisualEvent → VLM 调用 → 结构化 JSON (`scene/facts/...`)，目前走的是 Anthropic Vision API（云端），不是本仓库的 C++ gRPC。
- **语义到 Agent**：VisualAnalysis → AgentEvent → AgentLoop，在主项目里有完整路径，但仅在离线/手动触发脚本里跑过。
- **C++ VLM 推理自身**：`multimodal_inference_server` 本地能加载 Qwen2-VL-3B + mmproj 并返回 token，缓存逻辑可触发。

### 4.3 尚未打通 / 尚未验证的链路

- **MyNeuroLikeSystem 视觉链路 ↔ C++ VLM 服务**：Python 端的 `LLMVisualEventAnalyzer` 目前直接调用 `LLMClient`（走云端 API），没有接入本仓库的 gRPC。换句话说，C++ 端目前也没在已实现的视频链路中进行过完整测试。
- **比赛项目（EducationalAgentProject）↔ 视觉链路**：比赛端目前只使用小模型（BERT）结果，没有订阅 `VisualEvent`，也没有任何地方 import `src.vision.*`。视觉相关的入口在比赛端是空的。
- **C++ VLM 缓存的命中率 / 淘汰 / 持久化回放**：代码实现了 TTL + LRU + 落盘，但只在开发机上做过单路径 smoke test，没有在实际数据流里跑过实时处理场景。
- **端到端延迟**：现在只有各段的局部测量，没有从"摄像头出帧"到"Agent 产生回复"的完整延迟数据。
- **VRAM 监控自动卸载**：逻辑写好了，但触发条件（free_bytes 阈值、OOM 回收）还没在真实长时间运行里复现过，阈值默认值没有实测支撑。

---

## 5. 为什么还没移植到比赛项目生产端

比赛子项目（EducationalAgentProject）当前的生产路径是：
`用户输入 → BERT 小模型（走 AgentBackendPredict 的 gRPC） → LLM 对话`。
视觉链路还停留在主项目 MyNeuroLikeSystem，没有并入比赛端。原因按从重到轻排列：

### 5.1 比赛端后端架构不具备处理流媒体的能力

视觉链路真正上线意味着后端要能持续接收和处理视频流，而比赛子项目最终是要部署到服务器的，这一点目前**在架构层面就没有铺路**：

- **现有后端是 Python + FastAPI 的请求-响应型服务**，处理的是单次文本对话请求。它没有：
  - 长连接 / 持续推流的入口（WebRTC、RTSP、RTMP、WebSocket 视频帧流都没有接入）。
  - 多路视频流的会话管理（每个学生一路视频，需要独立的解码线程、独立的显著度状态机、独立的 VLM 调用配额）。
  - 视频帧的零拷贝管线（FastAPI 的 multipart 或 base64 over HTTP 在持续高帧率下会被序列化开销和 GIL 拖死）。
- **MyNeuroLikeSystem 当前的视觉链路是单进程单源**：摄像头帧从本机读，事件直接 push 到本机的 AgentLoop。把这套结构原封不动放到服务器上，连"如何把帧送进来"这一步都不成立。
- **Python + FastAPI 的天花板**：即使加上 `uvicorn` 的多 worker、`asyncio` 的视频流端点，也难以在单台服务器上同时承载多路视频流的解码 + 显著度计算 + VLM 调度。Python 在帧级处理上的 GIL 和内存拷贝开销会成为瓶颈，OpenCV/NumPy 的 release-GIL 路径只能局部缓解。
- **结论**：要让视觉链路真正上线，后端的视频接入层、会话层、调度层基本上要**用更适合流媒体的栈重写**（候选包括 C++ / Rust 的媒体网关 + 现有的 C++ gRPC 推理服务 + 轻量 Python 业务编排）。这不是"加个接口"或"把模块迁移过去"的工作量，而是接近一次后端架构层级的重构。
- 在这个重构没有动工之前，**比赛端无论如何接收不到稳定的视频流**，视觉链路的"上线"无从谈起。当前阶段视觉链路只能停留在主项目里、在单机上做离线和半实时验证。

### 5.2 即使不考虑流媒体接入，端到端也未完整验证

- MyNeuroLikeSystem 的视觉链路目前只在开发机的**离线脚本 / 手动触发**下跑通过，没有在比赛端的长时间对话回路里持续运行过。
- 还缺少几类关键数据：
  - 不同输入源（摄像头、屏幕录制、视频文件）下显著度阈值是否需要重新标定。
  - 事件进入 `AgentLoop` 之后，是否会挤占正常对话的响应时序、是否会干扰情绪状态。
  - 长时间运行下的内存曲线、VLM 调用频次是否符合预期。
- 在这些数据补齐前，把视觉事件直接灌进比赛端的 AgentLoop 有明显的回归风险（比如"视觉事件频繁插话"、"情绪状态被视觉信号带偏"）。

### 5.3 C++ VLM 服务仍是初版

- `multimodal_inference_server` 当前是 **v0.0.1beta1**（见本仓库 README 更新日志），功能刚成型：
  - VLM 缓存、VRAM 自动卸载、请求验证这些都是最近一次大重构里加进来的，没有经历过压测。
  - 模型加载路径、mmproj 路径、`n_gpu_layers` 等配置项仍在根据机器情况手工调。
- 比赛端对稳定性敏感（评审现场、演示现场都是不可回滚的单次运行），把尚未经过长时间运行验证的组件接进去收益/风险比偏低。

### 5.4 上下游接口还没对齐

- Python 端的 `LLMVisualEventAnalyzer` 直接持有一个 `LLMClient`，输入是 `ImageResult`（base64 + media_type），和本仓库 `VLMRequest`（`bytes image_data` 或 `string image_path` + prompt + 生成参数）**只是语义上接近，没有适配层**。
- 要接上至少需要：
  - 一个 `GrpcVLMClient`，把 `ImageResult` 映射成 `VLMRequest`，把流式 token 聚合回 `VisualAnalysis`。
  - 缓存键的统一：Python 端目前是 `SHA256(URL)` 的文件缓存，C++ 端是 `SHA256(image_bytes) + SHA256(prompt)` 的内存/持久化缓存，这两层目前**不共享**，直接切过去会损失 Python 侧已有的下载缓存。
  - `visual_semantics` 的情绪提取目前是文本级启发式，如果要和 BERT 情绪分类统一，需要先定义"视觉文本 → BERT 输入"的规范。
- 这些适配层暂时没有写，换句话说：即使现在把 C++ 服务跑起来，比赛端也没法直接调用。

### 5.5 ViT 显著度检测只有占位

- proto 里有 `DetectSaliency`，但服务端没有实现，`vit_model` 字段在 `MultimodalServerOptions` 里保留，没有对应的推理代码路径。
- 目前的显著度检测仍然依赖 Python 侧的 MOG2 + 多特征融合，**没有深度模型参与**。对于"什么样的画面算值得触发 VLM"，现在是靠手工调阈值，缺少语义层的信号。
- 在显著度触发策略没有用深度模型加固之前，视觉链路在真实教学场景里容易出现两类问题：该触发时不触发（静态画面里的语义变化被漏掉）、不该触发时触发（背景抖动/光照变化被误判）。这两类问题在离线脚本里都见过，但修复路径（集成 ViT 或 CLIP）还没排期。

### 5.6 比赛项目自身还没给视觉留位置

- 翻 `EducationalAgentProject` 的源码，没有 `import src.vision`，`AgentLoop` 里也没有 `type="visual"` 的分支接入点。
- 要把视觉事件接进来，需要在比赛端加：事件订阅、对话中断/插入策略、对视觉事件的 rate limit、与既有情绪管线的融合规则。这是**比赛端侧的工作量**，不是只在推理端这边加个接口就能完成的。

### 5.7 一句话总结

> 视觉链路的各段（感知 / VLM / 语义 / Agent 转接）都能独立跑通，但**比赛端后端目前只能处理单次请求-响应型对话，没有流媒体接入层**，视频流在生产端连"进来"都做不到；即使抛开这一点，端到端也没有在比赛回路里完整跑过。要真正上线，比赛后端在视频接入 / 会话管理 / 调度这几层需要跨语言重写（大概率把 Python+FastAPI 降级为业务编排层，媒体和推理交给 C++/Rust），这是架构级工作量，不是"把模块迁移过去"的范畴。所以目前的策略是：让视觉链路继续在主项目里迭代，等后端流媒体能力具备之后再谈接入。
