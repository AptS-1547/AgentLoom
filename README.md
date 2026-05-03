# AgentBackendPredict

统一多模态推理服务端，整合 BERT 情绪分类、VLM 视觉语言推理和 ViT 显著度检测。

## 特性

- **多模态推理**：BERT（ONNX Runtime）+ VLM（llama.cpp）+ ViT（预留）
- **流式/同步推理**：支持 token-by-token 流式输出和批量同步推理
- **生产级监控**：运行时统计、内存监控、慢请求追踪、健康检查
- **跨平台支持**：Windows / Linux，CPU / CUDA
- **Lazy Load**：LLM 首次请求时加载，空闲自动卸载释放 VRAM
- **图片解码**：内存直接解码 JPEG/PNG/GIF/WebP，无需临时文件

## 快速开始

### 编译

```powershell
# Windows
cmake -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CONFIGURATION_TYPES=Release `
  -DBERT_VCPKG_TRIPLET=x64-windows `
  -DBERT_USE_ONNXRUNTIME_GPU=OFF

cmake --build build --config Release --parallel
```

### 启动服务端

```powershell
cd build/Release

./multimodal_inference_server.exe `
  --llm "D:/path/to/qwen2-vl-7b.gguf" `
  --mmproj "D:/path/to/mmproj.gguf" `
  --bert "D:/path/to/joint_model.onnx" `
  --ngl 99 `
  --host 0.0.0.0 --port 50051
```

### Python 客户端

```python
import grpc
import multimodal_inference_pb2 as pb2
import multimodal_inference_pb2_grpc as pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = pb2_grpc.MultimodalInferenceStub(channel)

# BERT 情绪分类
request = pb2.EmotionRequest(
    input_ids=[101, 2769, 3221, 1920, 102],
    attention_mask=[1, 1, 1, 1, 1],
    personality=[0.5] * 11
)
response = stub.PredictEmotion(request)

# VLM 流式推理
with open("image.jpg", "rb") as f:
    image_data = f.read()

request = pb2.VLMRequest(
    prompt="描述这张图片",
    image_data=image_data,
    max_tokens=256
)

for token in stub.GenerateVLM(request):
    if not token.is_final:
        print(token.token, end='', flush=True)
```

## 文档

- **[部署文档](docs/DEPLOYMENT.md)** - 完整的部署指南、参数说明、性能优化
- **[API 文档](proto/multimodal_inference.proto)** - gRPC 接口定义

## 架构

```
src/
├── common/                    # 公共基础设施
│   ├── server_common.h/.cpp   # 统计、内存监控、参数解析
│   └── logger.h/.cpp          # 日志系统
├── models/                    # 模型封装层
│   ├── onnx_model.h/.cpp      # ONNX Runtime（BERT/ViT）
│   └── llama_runner.h/.cpp    # llama.cpp VLM 封装
├── server/                    # 服务端入口
│   └── multimodal_inference_server.cpp
└── client/                    # 测试客户端
    ├── client_test.cpp
    └── benchmark_client.cpp
```

## 性能

- **BERT 批量推理**：吞吐量提升 3-5 倍
- **VLM GPU Offload**：RTX 4090 约 50 tokens/s（Qwen2-VL-7B）
- **内存占用**：BERT ~500MB，VLM ~8GB（7B 模型 Q4_K_M）
- **启动时间**：BERT 立即加载（~100ms），VLM lazy load（首次 ~10-30s）

## 依赖

- **ONNX Runtime** 1.17.1 (CPU) / 1.20.1 (GPU)
- **llama.cpp** 主分支（需要 mtmd 支持）
- **gRPC** 1.x (vcpkg)
- **Protobuf** 3.x (vcpkg)
- **spdlog** 1.x (vcpkg)

## 许可证

MIT License

## 更新日志

### v2.0.0 (2026-05-04)

**重大重构**：
- 合并 BERT/LLM/Multimodal 三个服务端为单一 `multimodal_inference_server`
- 删除 `bert_inference_server` 和 `llm_inference_server`（功能已整合）
- 重构目录结构：按功能分为 `common/`、`models/`、`server/`、`client/`
- 提取公共基础设施到 `server_common` 库（统计、内存监控、参数解析）

**新特性**：
- 图片解码：使用 `mtmd_helper_bitmap_init_from_buf()` 从内存直接解码
- RAII 资源管理：BitmapGuard/ChunksGuard/BatchGuard 消除内存泄漏
- 简化 API：合并 `Generate()` 和 `GenerateStream()` 为单一接口
- 使用 `mtmd_helper_eval_chunks()` 简化 chunk 处理（减少 60+ 行代码）

**性能优化**：
- LLM lazy load + 空闲自动卸载（5 分钟）
- 跨平台内存监控（Windows/Linux）
- 周期性统计日志 + 慢请求追踪
- gRPC 健康检查 + 优雅关闭

### v1.0.0 (2024)

初始版本，独立的 BERT 和 LLM 服务端。
