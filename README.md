# BERT ONNX Inference Service

基于 gRPC + ONNX Runtime 的 BERT 多任务推理后端，为教育智能体仿真系统提供高性能情绪/行为/语气预测服务。

## 架构

```
客户端 (Python / C++ / 任意 gRPC 客户端)
  │
  │  gRPC (port 50051)
  ▼
┌────────────────────────────────────────┐
│  BERTInference gRPC Server             │
│  ├─ Predict()      单条推理            │
│  └─ PredictBatch() 批量推理            │
│         │                              │
│         ▼                              │
│  OnnxBERTModel (ONNX Runtime 1.17.1)   │
│  ├─ CPU 全核 intra-op 并行             │
│  ├─ Extended 图优化                    │
│  └─ 线程安全，支持并发推理             │
└────────────────────────────────────────┘
```

## 模型输入输出

| 方向 | 名称 | 维度 | 说明 |
|------|------|------|------|
| 输入 | input_ids | [batch, seq_len] | BERT token IDs (词表大小 21128) |
| 输入 | attention_mask | [batch, seq_len] | 注意力掩码 |
| 输入 | personality | [batch, 11] | Big Five 人格 + 扩展维度 |
| 输出 | emotion_logits | [batch, 10] | 10 类情绪分类 |
| 输出 | behavior_logits | [batch, 12] | 12 类行为分类 |
| 输出 | tone_logits | [batch, 8] | 8 类语气分类 |
| 输出 | intensity | [batch, 1] | 情绪强度 0-1 |
| 输出 | response_length_logits | [batch, 3] | 回复长度 (短/中/长) |

## 构建

### 前置条件

- Visual Studio 2022 (v143 工具集)
- CMake 3.20+
- vcpkg

### 编译步骤

```bash
# 1. 安装依赖 (使用 VS2022 自定义 triplet)
vcpkg install grpc:x64-windows-vs2022 \
    --overlay-triplets=triplets \
    --vcpkg-root=<your-vcpkg-root>

# 2. 将编译产物拷贝到 vcpkg_installed/x64-windows/

# 3. CMake 配置 + 构建
cmake --preset x64-Release
cmake --build build/x64-Release --config Release
```

构建产物在 `build/x64-Release/Release/` 下：
- `bert_inference_server.exe` — 推理服务
- `bert_inference_client.exe` — 测试客户端
- `onnxruntime.dll` — 自动拷贝的运行时

## 运行

### 启动服务

```bash
bert_inference_server <model.onnx> [port]
```

- `model.onnx` — ONNX 格式的 BERT 联合模型
- `port` — 监听端口，默认 50051

### 测试

```bash
bert_inference_client [host:port]
```

执行三组测试：
1. **单条推理** — 测量单次延迟
2. **批量推理** — batch=4 和 batch=8，计算每样本延迟
3. **基准测试** — 5 次预热 + 100 次迭代，输出平均延迟和吞吐量

## gRPC 接口

```protobuf
service BERTInference {
  rpc Predict(PredictRequest) returns (PredictResponse);
  rpc PredictBatch(PredictBatchRequest) returns (PredictBatchResponse);
}
```

### 单条推理

```
PredictRequest {
  repeated int64 input_ids      // token IDs [seq_len]
  repeated int64 attention_mask  // 注意力掩码 [seq_len]
  repeated float personality     // 人格向量 [11]
}
```

### 批量推理

```
PredictBatchRequest {
  repeated int64 input_ids      // 展平 [batch * seq_len]
  repeated int64 attention_mask  // 展平 [batch * seq_len]
  repeated float personality     // 展平 [batch * 11]
  int32 batch_size
  int32 seq_length
}
```

## 项目结构

```
AgentBackendPredict/
├── CMakeLists.txt              # 构建配置 (C++20, /utf-8)
├── CMakePresets.json            # VS2022 x64 预设
├── vcpkg.json                   # 依赖声明
├── triplets/
│   └── x64-windows-vs2022.cmake # vcpkg 自定义 triplet (强制 VS2022)
├── proto/
│   └── bert_inference.proto     # gRPC 服务定义
├── src/
│   ├── onnx_model.h             # ONNX 模型封装 (PIMPL)
│   ├── onnx_model.cpp           # 推理实现
│   ├── bert_inference_server.cpp # gRPC 服务端
│   └── client_test.cpp          # 测试客户端 + 基准测试
└── deps/
    └── onnxruntime-win-x64-1.17.1/  # 预编译 ONNX Runtime
```

## 注意事项

- 源文件使用 UTF-8 编码，CMake 已配置 `/utf-8` 编译选项
- vcpkg 依赖必须用 VS2022 工具集编译（`x64-windows-vs2022` triplet），避免与 VS2022 的 ABI 不兼容
- ONNX Runtime 使用 `deps/` 下的预编译包，未通过 vcpkg 管理
