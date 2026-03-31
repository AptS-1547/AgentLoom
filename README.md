# AgentBackendPredict

`EducationalAgentProject` 的 C++ ONNX 推理后端，基于 gRPC + ONNX Runtime，为教育智能体仿真系统提供情绪/行为/语气预测服务。

主项目中本地 BERT 推理原本跑在 Python / PyTorch 上，多 persona 并发时成为瓶颈。此仓库将小模型推理服务化，解耦为独立可压测的链路：

```
Python tokenizer -> gRPC -> C++ ONNX Runtime -> Python 后处理
```

## 架构

```
客户端 (Python / C++ / 任意 gRPC 客户端)
  │
  │  gRPC (默认 127.0.0.1:50051)
  ▼
┌────────────────────────────────────────┐
│  BERTInference gRPC Server             │
│  ├─ Predict()      单条推理            │
│  └─ PredictBatch() 批量推理            │
│  ├─ gRPC Health Check                 │
│  ├─ 周期性运行统计                     │
│  └─ 慢请求日志                         │
│         │                              │
│         ▼                              │
│  OnnxBERTModel (ONNX Runtime 1.17.1)   │
│  ├─ CPU 全核 intra-op 并行             │
│  ├─ Extended 图优化                    │
│  └─ 线程安全，支持并发推理             │
└────────────────────────────────────────┘
```

## 服务职责

服务只接受已编码的张量，不负责 tokenizer、原始文本处理和上层业务逻辑：

**输入：**
- `input_ids` [batch, seq_len] — BERT token IDs（词表大小 21128）
- `attention_mask` [batch, seq_len] — 注意力掩码
- `personality` [batch, 11] — Big Five 人格 + 扩展维度

**输出：**
- `emotion_logits` [batch, 10] — 10 类情绪分类
- `behavior_logits` [batch, 12] — 12 类行为分类
- `tone_logits` [batch, 8] — 8 类语气分类
- `intensity` [batch, 1] — 情绪强度 0-1
- `response_length_logits` [batch, 3] — 回复长度（短/中/长）

协议定义在 [proto/bert_inference.proto](proto/bert_inference.proto)。

## 构建

CI 中的构建流程是当前仓库可用的参考流程。

### 依赖

- CMake 3.20+
- C++20 编译器
- vcpkg
- ONNX Runtime 预编译包
- Windows: Visual Studio 2022（v143 工具集）
- Linux: Ninja

`grpc` 和 `protobuf` 通过 vcpkg 管理，ONNX Runtime 使用预编译包。

### Windows

```powershell
.\vcpkg\vcpkg.exe install grpc:x64-windows protobuf:x64-windows

cmake -S . -B build\gha-windows `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCMAKE_CONFIGURATION_TYPES=Release `
  -DBERT_VCPKG_TRIPLET=x64-windows `
  -DBERT_USE_ONNXRUNTIME_GPU=OFF `
  -DONNXRUNTIME_CPU_ROOT="D:\path\to\onnxruntime-win-x64-1.17.1"

cmake --build build\gha-windows --config Release --parallel
```

> `CMakePresets.json` 中的 VS 路径为开发机本地值，换机器时建议显式传参。

### Linux

```bash
sudo apt-get update
sudo apt-get install -y ninja-build pkg-config curl tar unzip zip

./vcpkg/vcpkg install grpc:x64-linux protobuf:x64-linux

cmake -S . -B build/gha-linux \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBERT_VCPKG_TRIPLET=x64-linux \
  -DBERT_USE_ONNXRUNTIME_GPU=OFF \
  -DONNXRUNTIME_CPU_ROOT=/path/to/onnxruntime-linux-x64-1.17.1

cmake --build build/gha-linux --parallel
```

## 运行

### 启动服务

```bash
bert_inference_server <model.onnx> [port]
```

示例：

```bash
bert_inference_server ./joint_model.onnx 50051 --host 127.0.0.1 --provider cpu
```

参数列表：

| 参数 | 说明 |
|------|------|
| `--host` | 监听地址，默认 `127.0.0.1` |
| `--provider` | 推理后端：`auto` / `cpu` / `cuda` |
| `--cuda-device` | CUDA 设备编号 |
| `--intra-op` | intra-op 并行线程数 |
| `--inter-op` | inter-op 并行线程数 |
| `--grpc-num-cqs` | gRPC 完成队列数 |
| `--grpc-min-pollers` | gRPC 最小轮询线程数 |
| `--grpc-max-pollers` | gRPC 最大轮询线程数 |
| `--max-recv-mb` | 最大接收消息大小（MB） |
| `--max-send-mb` | 最大发送消息大小（MB） |
| `--stats-log-interval-seconds` | 统计日志间隔 |
| `--slow-request-ms` | 慢请求阈值（ms） |

### 健康检查

```bash
python tools/grpc_health_probe.py --target 127.0.0.1:50051
```

### 功能测试

```bash
bert_inference_client 127.0.0.1:50051
```

### 并发压测

```bash
bert_benchmark_client --target 127.0.0.1:50051 --concurrency 16 --requests 200 --seq-len 64
```

输出包括总请求数、成功/失败数、吞吐量、平均延迟和 p50/p95/p99 分位延迟。

## 与主项目对接

主项目 `EducationalAgentProject` 已支持对应的 gRPC 协议和 `onnx_grpc` 后端。切换配置：

```json
{
  "small_model": {
    "backend": "onnx_grpc",
    "tokenizer_path": "./onnx/joint",
    "onnx_target": "127.0.0.1:50051"
  }
}
```

主项目无需了解 C++ 推理细节，只需目标地址和协议。

## CI 与发布

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — Windows / Linux 双平台编译、下载 ONNX Runtime、生成 smoke test 模型、启动服务、health probe、客户端测试
- [`.github/workflows/release.yml`](.github/workflows/release.yml) — 产物自动打包发布

## 项目结构

```
AgentBackendPredict/
├── CMakeLists.txt               # 构建配置 (C++20, /utf-8)
├── CMakePresets.json            # VS2022 x64 预设
├── vcpkg.json                   # 依赖声明
├── triplets/
│   └── x64-windows-vs2022.cmake # vcpkg 自定义 triplet
├── proto/
│   └── bert_inference.proto     # gRPC 服务定义
├── src/
│   ├── onnx_model.h             # ONNX 模型封装 (PIMPL)
│   ├── onnx_model.cpp           # 推理实现
│   ├── bert_inference_server.cpp # gRPC 服务端
│   └── client_test.cpp          # 测试客户端
└── deps/
    └── onnxruntime-win-x64-1.17.1/  # 预编译 ONNX Runtime
```

## 注意事项

- 源文件使用 UTF-8 编码，CMake 已配置 `/utf-8` 编译选项
- vcpkg 依赖须用 VS2022 工具集编译（`x64-windows-vs2022` triplet），避免 ABI 不兼容
- ONNX Runtime 使用 `deps/` 下的预编译包，未通过 vcpkg 管理
- 服务默认绑定 `127.0.0.1`，适用于同机进程间通信场景，不直接暴露公网
