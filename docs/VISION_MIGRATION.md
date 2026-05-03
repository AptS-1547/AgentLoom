# 视觉链路本地化迁移方案

## 1. 背景

当前视觉模块（`MyNeuroLikeSystem/project_src/src/vision`）运行在 Python 端，使用 GLM-4V 云端 API 做视觉语义分析。实测数据表明本地 VLM 推理显著优于云端：

| 指标 | GLM-4V 云端 | Qwen2.5-VL-3B 本地 (RTX 5060) |
|------|------------|-------------------------------|
| 单图 VQA 全文响应 | 18.7 s | 3.76 s |
| 图片编码 | N/A | 824 ms |
| 生成速度 | N/A | 123.88 tok/s |
| VRAM 占用 | N/A | ~3.6 GB |

本方案将整条视觉链路迁移到 C++ 推理端（AgentBackendPredict），利用 llama.cpp 的 mtmd 模块实现 ViT embedding 提取和 VLM 推理。

## 2. 目标架构

### 2.1 双层检测 + 本地 VLM

```
帧采集 → MOG2 预检测（~1ms，像素级，过滤静止帧）
              ↓ 有运动
         ViT embedding（~20ms，语义级变化检测）
              ↓ 帧间余弦相似度序列
         FFT 自适应采样（频域分析调控帧率）
              ↓ 峰值触发
         VLM 完整描述（~2-4s，本地 Qwen2.5-VL）
```

### 2.2 各层职责

| 层 | 实现 | 延迟 | 职责 |
|----|------|------|------|
| MOG2 预检测 | OpenCV C++ | ~1ms/帧 | 像素级运动过滤，跳过 90%+ 静止帧 |
| ViT embedding | llama.cpp mtmd | ~20ms/帧 | 语义级变化检测，帧间余弦相似度 |
| FFT 采样控制 | C++ (FFTW/自实现) | ~0.1ms | 频域分析 ViT 相似度序列，动态调控帧率 |
| 峰值检测 | C++ | ~0.1ms | 时序局部极大值 + 冷却期 + 清晰度优选 |
| VLM 描述 | llama.cpp Generate | ~2-4s | 关键帧语义分析，输出结构化 JSON |

### 2.3 与 Python 端的分工

```
C++ 推理端 (AgentBackendPredict)          Python 端 (MyNeuroLikeSystem)
┌─────────────────────────────┐          ┌──────────────────────────┐
│ gRPC MultimodalInference    │          │ AgentLoop                │
│ ├─ PredictEmotion (BERT)    │◄─────────│ ├─ 情绪融合              │
│ ├─ ProcessVideoStream (新)  │─────────►│ ├─ 视觉事件处理          │
│ │   ├─ MOG2 预检测          │  事件回调 │ ├─ 对话生成              │
│ │   ├─ ViT 变化检测         │          │ └─ 记忆写入              │
│ │   ├─ FFT 采样控制         │          │                          │
│ │   ├─ 峰值触发             │          │ 视频源管理               │
│ │   └─ VLM 描述生成         │          │ ├─ cv2.VideoCapture      │
│ └─ GenerateVLM (已有)       │          │ └─ gRPC 发帧             │
└─────────────────────────────┘          └──────────────────────────┘
```

Python 端只负责：
- 视频源管理（摄像头/文件）
- 帧采集 + JPEG 编码 + gRPC 发送
- 接收视觉事件 → 注入 AgentLoop

C++ 端负责所有计算密集型工作。

## 3. 关键技术点

### 3.1 ViT embedding 提取（不走 LLM）

llama.cpp 的 mtmd 模块支持独立 ViT 编码：

```cpp
// mtmd_encode_chunk() 内部只调用 clip_image_encode()
// 完全不触及 llama_decode()，可独立提取 ViT 特征

mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_buf(ctx, data, len);
mtmd_tokenize(ctx, chunks, &text, &bmp, 1);

// 找到 image chunk，只做 ViT 编码
mtmd_encode_chunk(ctx, image_chunk);
float* embd = mtmd_get_output_embd(ctx);
// embd 包含 [n_tokens × n_embd] 的 ViT embedding
// 可用于余弦相似度计算，不需要 LLM 推理
```

### 3.2 帧间变化检测

```cpp
// 对 ViT embedding 做全局平均池化，得到单个向量
// 计算相邻帧的余弦相似度
float cosine_sim = dot(embd_curr, embd_prev) / (norm(embd_curr) * norm(embd_prev));
float change_score = 1.0f - cosine_sim;  // 变化越大，分数越高
```

### 3.3 MOG2 作为预检测门控

```cpp
// OpenCV C++ MOG2，与 Python 版本等价
auto mog2 = cv2::createBackgroundSubtractorMOG2(500, 16.0, false);
cv::Mat foreground;
mog2->apply(frame, foreground);

// 形态学过滤 + 连通域面积检查
float foreground_ratio = countNonZero(foreground) / total_pixels;
if (foreground_ratio < min_threshold) {
    // 静止帧，跳过 ViT 编码
    continue;
}
```

### 3.4 FFT 自适应采样

输入信号从原来的三信号 change_score 替换为 ViT 余弦相似度序列：

```cpp
// 维护 ViT 相似度的滑动窗口（32 个样本）
// STFT 分析高频能量比
// 高频活跃 → 提高采样率（最高 15fps）
// 低频稳定 → 降低采样率（最低 2-4fps）
float r_high = high_freq_energy / total_energy;
float target_fps = fps_min + (fps_max - fps_min) * pow(r_high, gamma);
```

## 4. gRPC 接口设计（预留）

```protobuf
// 新增到 multimodal_inference.proto

// 视频流处理（server-side streaming）
rpc ProcessVideoStream(stream VideoFrame) returns (stream VisualEvent);

message VideoFrame {
  bytes frame_data = 1;       // JPEG 编码帧
  int64 timestamp_ms = 2;
  int32 width = 3;
  int32 height = 4;
}

message VisualEvent {
  int64 timestamp_ms = 1;
  float change_score = 2;     // ViT 变化分数
  float peak_score = 3;       // 峰值分数
  string description = 4;     // VLM 生成的描述（仅峰值触发时有值）
  string scene = 5;           // 场景概括
  repeated string facts = 6;  // 可观察事实
  string agent_hint = 7;      // Agent 提示（30 字内）
  string memory_candidate = 8; // 记忆候选
}
```

## 5. 新增 C++ 模块

```
src/
├── models/
│   ├── onnx_model.h/.cpp       # 已有，BERT 推理
│   ├── llama_runner.h/.cpp     # 已有，VLM 推理
│   └── vit_detector.h/.cpp     # 新增，ViT 变化检测
├── vision/                      # 新增，视觉管线
│   ├── mog2_prefilter.h/.cpp   # MOG2 预检测
│   ├── fft_sampler.h/.cpp      # FFT 自适应采样
│   ├── peak_selector.h/.cpp    # 时序峰值检测
│   └── visual_pipeline.h/.cpp  # 管线编排（多线程）
└── server/
    └── multimodal_inference_server.cpp  # 新增 ProcessVideoStream RPC
```

### 5.1 vit_detector.h 接口草案

```cpp
namespace vision {

struct ChangeDetectionResult {
    float cosine_similarity;    // 帧间余弦相似度
    float change_score;         // 1 - similarity
    bool significant_change;    // 是否超过阈值
};

class ViTDetector {
public:
    // 复用 llama.cpp 的 mtmd context
    bool Init(mtmd_context* mtmd_ctx, int n_embd);

    // 提取 ViT embedding 并与上一帧比较
    ChangeDetectionResult Detect(const std::vector<uint8_t>& frame_jpeg);

    // 重置参考帧（场景切换时）
    void ResetReference();

private:
    mtmd_context* mtmd_ctx_;
    std::vector<float> prev_embedding_;
    int n_embd_;
};

} // namespace vision
```

### 5.2 visual_pipeline.h 接口草案

```cpp
namespace vision {

struct VisualPipelineConfig {
    // MOG2 参数
    int mog2_history = 500;
    float mog2_var_threshold = 16.0f;
    float min_foreground_ratio = 0.005f;

    // ViT 参数
    float vit_change_threshold = 0.15f;

    // FFT 采样参数
    float fps_min = 2.0f;
    float fps_max = 15.0f;
    float fft_gamma = 0.7f;
    int fft_window_size = 32;

    // 峰值检测参数
    float peak_threshold = 0.15f;
    float peak_cooldown_seconds = 0.5f;
    float event_window_seconds = 1.0f;
};

class VisualPipeline {
public:
    using EventCallback = std::function<void(const VisualEvent&)>;

    bool Init(const VisualPipelineConfig& config,
              mtmd_context* mtmd_ctx,
              llm::LlamaRunner& llm_runner);

    // 处理单帧（由 gRPC 流式调用驱动）
    void ProcessFrame(const std::vector<uint8_t>& frame_jpeg,
                      int64_t timestamp_ms);

    void SetEventCallback(EventCallback callback);
    void Reset();

private:
    MOG2Prefilter mog2_;
    ViTDetector vit_;
    FFTSampler fft_;
    PeakSelector peak_;
    llm::LlamaRunner* llm_runner_;
    EventCallback callback_;
};

} // namespace vision
```

## 6. 依赖

| 依赖 | 用途 | 引入方式 |
|------|------|----------|
| OpenCV C++ | MOG2、形态学、图像预处理 | vcpkg 或系统安装 |
| llama.cpp mtmd | ViT embedding + VLM 推理 | 已有（预编译库） |
| FFTW（可选） | FFT 频域分析 | vcpkg，或用自实现的简单 FFT |

## 7. 实施阶段

### 阶段 1：ViT 变化检测原型
- 实现 `ViTDetector`，验证 mtmd_encode_chunk() 独立提取 embedding
- 测试帧间余弦相似度的区分度
- 基准测试：ViT 编码延迟、embedding 维度、相似度分布

### 阶段 2：MOG2 + FFT 管线
- 实现 `MOG2Prefilter`（OpenCV C++）
- 实现 `FFTSampler`（从 Python adaptive_sampler.py 移植）
- 实现 `PeakSelector`（从 Python TemporalPeakSelector 移植）
- 集成为 `VisualPipeline`

### 阶段 3：gRPC 集成
- 新增 `ProcessVideoStream` RPC
- Python 端视频源管理 + gRPC 发帧
- 事件回调 → AgentLoop 集成

### 阶段 4：LoRA 微调降低远端依赖
- 收集本地 VLM 的视觉描述数据
- LoRA 微调 Qwen2.5-VL-3B 提升特定场景准确率
- 逐步降低 GLM-4V 远端调用频率

## 8. 性能预期

| 场景 | 当前 Python + GLM-4V | 迁移后 C++ + 本地 VLM |
|------|---------------------|----------------------|
| 静止帧处理 | ~10ms (MOG2+三信号) | ~1ms (MOG2 预检测跳过) |
| 运动帧检测 | ~10ms | ~21ms (MOG2 1ms + ViT 20ms) |
| 事件描述生成 | 18.7s (GLM-4V API) | ~2-4s (本地 VLM) |
| 帧吞吐量 | ~100 fps (GIL 限制) | ~500+ fps (真并行) |
| 端到端事件响应 | ~19s | ~4s |
