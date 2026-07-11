# AgentLoom 推理服务端部署文档

## 概述

本文档主要说明推理服务部署。AgentLoom 当前提供三个正式 Server 边界：

| Server | 职责 |
|--------|------|
| `agent_gateway_server` | HTTP/WebSocket、Persona Runtime、记忆、文档和 Skill 编排 |
| `emotion_inference_server` | CPU/ONNX BERT 情绪推理 gRPC 服务 |
| `multimodal_inference_server` | BERT + llama.cpp/mtmd VLM 多模态 gRPC 服务 |

推理 Server 支持健康检查、统一 `core::Status`/gRPC 错误映射、trace metadata、运行时统计和受控关闭。VLM 与 Gateway 默认保持进程隔离；共享内存 IPC 用于同机帧数据面，gRPC 继续承担协议、健康检查和诊断边界。

> 文档状态：当前部署参考。具体字段以 `config/*.example.json`、`src/config/sections/` 和 Server `--help` 输出为准。

## 快速开始

### 1. 编译

```powershell
# Windows
& "C:\Program Files\CMake\bin\cmake.exe" -B build/x64-Release -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_CONFIGURATION_TYPES=Release `
  -DBERT_VCPKG_TRIPLET=x64-windows `
  -DBERT_USE_ONNXRUNTIME_GPU=OFF `
  -DLLAMA_CPP_ROOT="<path-to-llama.cpp>"

& "C:\Program Files\CMake\bin\cmake.exe" --build build/x64-Release `
  --config Release --parallel
```

```bash
# Linux
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DBERT_VCPKG_TRIPLET=x64-linux \
  -DBERT_USE_ONNXRUNTIME_GPU=AUTO

cmake --build build --parallel
```

### 2. 启动服务端

```powershell
cd build/x64-Release/Release

# 最小启动（配置文件）
./multimodal_inference_server.exe --config ../../../config/server.example.json

# 完整启动（BERT + VLM）
./multimodal_inference_server.exe `
  --llm "D:/path/to/qwen2-vl-7b.gguf" `
  --mmproj "D:/path/to/mmproj.gguf" `
  --bert "D:/path/to/joint_model.onnx" `
  --ngl 99 `
  --host 0.0.0.0 --port 50051
```

### 3. 健康检查

```bash
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
```

## 命令行参数

### 模型参数

| 参数 | 说明 | 必需 | 默认值 |
|------|------|------|--------|
| `--llm <path>` | LLM 模型 GGUF 路径 | ✓ | - |
| `--mmproj <path>` | 视觉投影层 GGUF 路径 | ✗ | - |
| `--bert <path>` | BERT 模型 ONNX 路径 | ✗ | - |
| `--vit <path>` | ViT 模型 ONNX 路径（预留） | ✗ | - |
| `--ngl <n>` | GPU offload 层数，-1 全部 | ✗ | -1 |
| `--provider <auto\|cpu\|cuda>` | ONNX 执行提供者 | ✗ | auto |
| `--cuda-device <id>` | CUDA 设备 ID | ✗ | 0 |

### 服务端参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host <ip>` | 监听地址 | 127.0.0.1 |
| `--port <port>` | 监听端口 | 50051 |
| `--log-dir <dir>` | 日志目录 | logs |
| `--grpc-num-cqs <n>` | gRPC 完成队列数 | auto (hw_threads/8) |
| `--grpc-min-pollers <n>` | 最小轮询线程数 | 1 |
| `--grpc-max-pollers <n>` | 最大轮询线程数 | auto (hw_threads/2) |
| `--max-recv-mb <n>` | 最大接收消息大小 MB | 100 |
| `--max-send-mb <n>` | 最大发送消息大小 MB | 10 |
| `--stats-log-interval-seconds <n>` | 统计日志间隔秒 | 30 |
| `--slow-request-ms <n>` | 慢请求阈值毫秒 | 250 |

## gRPC 接口

### 1. BERT 情绪分类

#### 单条推理

```protobuf
rpc PredictEmotion(EmotionRequest) returns (EmotionResponse);

message EmotionRequest {
  repeated int64 input_ids = 1;
  repeated int64 attention_mask = 2;
  repeated float personality = 3;  // 11 维人格向量
}

message EmotionResponse {
  repeated float emotion_logits = 1;      // 10 分类
  repeated float behavior_logits = 2;     // 12 分类
  repeated float tone_logits = 3;         // 8 分类
  float intensity = 4;
  repeated float response_length_logits = 5;  // 3 分类
  string error = 6;
}
```

#### 批量推理

```protobuf
rpc PredictEmotionBatch(EmotionBatchRequest) returns (EmotionBatchResponse);

message EmotionBatchRequest {
  uint32 batch_size = 1;
  uint32 seq_length = 2;
  repeated int64 input_ids = 3;          // 展平：batch_size * seq_length
  repeated int64 attention_mask = 4;
  repeated float personality = 5;        // 展平：batch_size * 11
}
```

### 2. VLM 视觉语言推理

#### 流式推理

```protobuf
rpc GenerateVLM(VLMRequest) returns (stream VLMToken);

message VLMRequest {
  string prompt = 1;
  oneof image_source {
    bytes image_data = 2;      // JPEG/PNG 二进制
    string image_path = 3;     // 本地文件路径（服务端侧）
  }
  int32 max_tokens = 4;
  float temperature = 5;
  int32 context_size = 6;
  float top_p = 7;
  int32 top_k = 8;
}

message VLMToken {
  string token = 1;
  bool is_final = 2;
}
```

#### 同步推理

```protobuf
rpc GenerateVLMSync(VLMRequest) returns (VLMResponse);

message VLMResponse {
  string text = 1;
  float image_encode_ms = 2;
  float prompt_eval_ms = 3;
  float eval_ms = 4;
  int32 prompt_tokens = 5;
  int32 generated_tokens = 6;
  string error = 7;
}
```

### 3. ViT 显著度检测（预留）

```protobuf
rpc DetectSaliency(SaliencyRequest) returns (SaliencyResponse);
```

## Python 客户端示例

### 安装依赖

```bash
pip install grpcio grpcio-tools pillow
```

### 生成 Python 代码

```bash
python -m grpc_tools.protoc \
  -I proto \
  --python_out=. \
  --grpc_python_out=. \
  proto/multimodal_inference.proto
```

### BERT 情绪分类

```python
import grpc
import multimodal_inference_pb2 as pb2
import multimodal_inference_pb2_grpc as pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = pb2_grpc.MultimodalInferenceStub(channel)

request = pb2.EmotionRequest(
    input_ids=[101, 2769, 3221, 1920, 102],
    attention_mask=[1, 1, 1, 1, 1],
    personality=[0.5] * 11
)

response = stub.PredictEmotion(request)
print(f"Emotion logits: {response.emotion_logits}")
```

### VLM 流式推理

```python
from PIL import Image
import io

# 加载图片
with open("image.jpg", "rb") as f:
    image_data = f.read()

request = pb2.VLMRequest(
    prompt="描述这张图片",
    image_data=image_data,
    max_tokens=256,
    temperature=0.7
)

# 流式接收
for token in stub.GenerateVLM(request):
    if not token.is_final:
        print(token.token, end='', flush=True)
print()
```

### VLM 同步推理

```python
request = pb2.VLMRequest(
    prompt="这张图片里有什么？",
    image_data=image_data,
    max_tokens=128
)

response = stub.GenerateVLMSync(request)
print(f"Generated: {response.text}")
print(f"Tokens: {response.generated_tokens}, Time: {response.eval_ms:.2f}ms")
```

## 性能优化

### 1. GPU Offload

```bash
# 全部层 offload 到 GPU
--ngl -1

# 部分层 offload（节省 VRAM）
--ngl 32
```

### 2. gRPC 线程池调优

```bash
# 高并发场景
--grpc-num-cqs 4 --grpc-max-pollers 16

# 低延迟场景
--grpc-num-cqs 1 --grpc-min-pollers 2 --grpc-max-pollers 4
```

### 3. LLM Lazy Load

LLM 模型采用 **lazy load** 策略：
- 首次请求时加载（~10-30 秒）
- 空闲 5 分钟后自动卸载释放 VRAM
- BERT 模型立即加载（模型小，~100ms）

### 4. 批量推理

BERT 支持批量推理，吞吐量提升 3-5 倍：

```python
request = pb2.EmotionBatchRequest(
    batch_size=32,
    seq_length=128,
    input_ids=[...],  # 32 * 128 = 4096 个元素
    attention_mask=[...],
    personality=[...]  # 32 * 11 = 352 个元素
)
response = stub.PredictEmotionBatch(request)
```

## 监控与日志

### 运行时统计

服务端每 30 秒输出统计日志：

```
[ServerStats] bind=0.0.0.0:50051 uptime_ms=120000 inflight=2 total_rpc=1523 
interval_rpc=48 total_samples=1523 interval_samples=48 total_errors=0 
interval_errors=0 avg_latency_ms=23.456 interval_avg_latency_ms=21.234 
max_latency_ms=156.789 max_batch_size=32 working_set_mb=2345.67 
private_usage_mb=2100.45 peak_working_set_mb=2456.78
```

### 慢请求日志

超过阈值（默认 250ms）的请求会记录：

```
[ServerSlowRequest] method=GenerateVLM samples=1 latency_ms=1234.567 success=true
```

### 日志文件

```
logs/
└── multimodal_inference_server.log
```

## 故障排查

### 1. LLM 加载失败

**症状**：`Failed to load LLM`

**原因**：
- 模型路径错误
- GGUF 格式不兼容
- VRAM 不足

**解决**：
```bash
# 检查模型文件
ls -lh /path/to/model.gguf

# 降低 GPU 层数
--ngl 20

# 查看详细日志
tail -f logs/multimodal_inference_server.log
```

### 2. 图片解码失败

**症状**：`Failed to decode image from memory`

**原因**：
- 图片格式不支持（仅支持 JPEG/PNG/GIF/WebP）
- 图片损坏
- 图片过大（超过 100MB）

**解决**：
```python
# 预处理图片
from PIL import Image
img = Image.open("input.jpg")
img = img.convert("RGB")
img.thumbnail((1568, 1568))  # 缩放
img.save("output.jpg", quality=85)
```

### 3. gRPC 消息过大

**症状**：`Received message larger than max`

**解决**：
```bash
# 增加消息大小限制
--max-recv-mb 200 --max-send-mb 50
```

### 4. CUDA Out of Memory

**症状**：`CUDA out of memory`

**解决**：
```bash
# 减少 GPU offload 层数
--ngl 16

# 或使用 CPU
--ngl 0
```

## 架构说明

### 主要源码边界

```
src/
├── core/                      # Status/Result、线程池、内存池和 RAII
├── config/                    # section registry、JSON/CLI 配置
├── models/                    # ONNX Runtime、llama.cpp/mtmd 和 runner pool
├── service/inference/         # 推理业务校验与服务接口
├── server/grpc/               # gRPC handler、错误和 trace 映射
├── server/main/               # Gateway、Emotion、Multimodal 进程入口
├── server/runtime/            # 日志和 Server 公共生命周期
├── media/                     # WebRTC、抽帧、编码和 VLM coordinator
└── ipc/                       # 共享内存帧数据面
```

### 关键特性

- **RAII 资源管理**：BitmapGuard/ChunksGuard/BatchGuard 消除内存泄漏
- **跨平台内存监控**：Windows (GetProcessMemoryInfo) / Linux (/proc/self/status)
- **生产级统计**：RuntimeStats + ScopedRequestStats + 周期性日志
- **健康检查**：gRPC 标准健康检查协议
- **优雅关闭**：std::jthread + stop_token 协作式取消

### 依赖版本

- **ONNX Runtime**: 1.17.1 (CPU) / 1.20.1 (GPU)
- **llama.cpp**: 主分支（需要 mtmd 支持）
- **gRPC**: 1.x (vcpkg)
- **Protobuf**: 3.x (vcpkg)
- **spdlog**: 1.x (vcpkg)

## 许可证

MIT License
