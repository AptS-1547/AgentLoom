# AgentBackendPredict

统一多模态推理服务端，整合 BERT 情绪分类、VLM 视觉语言推理和 ViT 显著度检测。

## 特性

- **多模态推理**：BERT（ONNX Runtime）+ VLM（llama.cpp）+ ViT（预留）
- **流式/同步推理**：支持 token-by-token 流式输出和批量同步推理
- **智能缓存**：VLM 结果缓存（内存/持久化），支持过期策略和降级读取
- **生产级监控**：运行时统计、VRAM 监控、慢请求追踪、健康检查
- **跨平台支持**：Windows / Linux，CPU / CUDA
- **Lazy Load**：LLM 首次请求时加载，空闲自动卸载释放 VRAM
- **图片解码**：内存直接解码 JPEG/PNG/GIF/WebP，无需临时文件
- **配置文件支持**：JSON 配置文件 + 命令行参数混合配置
- **请求验证**：Token 认证、请求限流、参数校验

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

# 方式 1：命令行参数
./multimodal_inference_server.exe `
  --llm "D:/path/to/qwen2-vl-7b.gguf" `
  --mmproj "D:/path/to/mmproj.gguf" `
  --bert "D:/path/to/joint_model.onnx" `
  --ngl 99 `
  --host 0.0.0.0 --port 50051 `
  --cache-enabled --cache-persist

# 方式 2：配置文件（推荐）
./multimodal_inference_server.exe --config config/server.json
```

配置文件示例 `config/server.json`：
```json
{
  "grpc": {
    "host": "0.0.0.0",
    "port": 50051
  },
  "models": {
    "bert": "models/joint_model.onnx",
    "llm": "models/qwen2-vl-7b.gguf",
    "mmproj": "models/mmproj.gguf"
  },
  "llm": {
    "n_gpu_layers": 99
  },
  "vlm_cache": {
    "enabled": true,
    "persist": true,
    "cache_dir": "cache/vlm",
    "max_entries": 512,
    "ttl_seconds": 3600
  },
  "auth": {
    "enabled": true,
    "token_file": "config/auth_token.txt"
  }
}
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
    max_tokens=256,
    temperature=0.7,
    allow_cache=True  # 启用缓存
)

for token in stub.GenerateVLM(request):
    if token.cache_hit:
        print(f"[缓存命中: {token.cache_key}]")
    if not token.is_final:
        print(token.token, end='', flush=True)
    else:
        print(f"\n[生成完成，来源: {token.result_source}]")

# VLM 同步推理（适合批量处理）
response = stub.GenerateVLMSync(request)
print(response.text)
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
├── server/                    # 服务端核心
│   ├── multimodal_inference_server.cpp  # gRPC 服务实现
│   ├── server_config.h/.cpp             # 配置文件加载
│   ├── server_options.h                 # 配置结构定义
│   ├── request_validation.h/.cpp        # 请求验证和限流
│   └── vlm_cache.h/.cpp                 # VLM 结果缓存
└── client/                    # 测试客户端
    ├── client_test.cpp        # 功能测试
    └── benchmark_client.cpp   # 性能测试

config/                        # 配置文件目录
third_party/                   # 第三方库（nlohmann/json 等）
proto/                         # gRPC 协议定义
    ├── multimodal_inference.proto  # 统一多模态协议
    └── bert_inference.proto        # BERT 协议（向后兼容）
```

## 性能

- **BERT 批量推理**：吞吐量提升 3-5 倍
- **VLM GPU Offload**：RTX 5060 约 126 tokens/s（Qwen2-VL-3B）
- **VLM 缓存命中**：响应时间 < 10ms（vs 首次推理 ~5s）
- **内存占用**：BERT ~500MB，VLM ~4GB（3B 模型 Q4_K_M）
- **启动时间**：BERT 立即加载（~100ms），VLM lazy load（首次 ~10-30s）
- **VRAM 管理**：自动监控，低于阈值时卸载模型释放显存

## 依赖

- **ONNX Runtime** 1.17.1 (CPU) / 1.20.1 (GPU)
- **llama.cpp** 主分支（需要 mtmd 支持）
- **gRPC** 1.x (vcpkg)
- **Protobuf** 3.x (vcpkg)
- **spdlog** 1.x (vcpkg)
- **nlohmann/json** (header-only，已包含在 third_party/)

## 核心功能模块

### VLM 缓存系统

智能缓存 VLM 推理结果，显著降低重复请求延迟：

- **缓存键生成**：基于图片 SHA256 + Prompt SHA256
- **过期策略**：TTL（默认 1 小时）+ LRU 淘汰
- **持久化**：可选磁盘持久化，重启后恢复
- **降级读取**：推理失败时返回过期缓存（可配置）
- **统计信息**：命中率、缓存大小、热点查询追踪

### 请求验证与限流

生产级安全防护：

- **Token 认证**：支持文件/环境变量配置
- **参数校验**：自动验证 max_tokens、temperature 等参数范围
- **请求限流**：防止恶意大批量请求
- **慢请求追踪**：自动记录超时请求

### VRAM 监控

自动管理 GPU 显存：

- **周期性监控**：每 10 秒检查可用 VRAM
- **自动卸载**：低于阈值时卸载 LLM 释放显存
- **OOM 保护**：捕获 CUDA OOM 错误并自动恢复
- **跨平台支持**：Windows (NVML) / Linux (nvidia-smi)

## 许可证

MIT License

## 更新日志

### v0.0.1beta 1 (2026-05-04)

**重大重构**：
- 合并 BERT/LLM/Multimodal 三个服务端为单一 `multimodal_inference_server`
- 删除 `bert_inference_server` 和 `llm_inference_server`（功能已整合）
- 重构目录结构：按功能分为 `common/`、`models/`、`server/`、`client/`
- 提取公共基础设施到 `server_common` 库（统计、内存监控、参数解析）

**新特性**：
- **VLM 缓存系统**：内存/持久化缓存，支持 TTL、LRU 淘汰、降级读取
- **配置文件支持**：JSON 配置文件 + 命令行参数混合配置
- **请求验证**：Token 认证、参数校验、请求限流
- **VRAM 监控**：自动监控显存，低于阈值时卸载模型
- **图片解码**：使用 `mtmd_helper_bitmap_init_from_buf()` 从内存直接解码
- **RAII 资源管理**：BitmapGuard/ChunksGuard/BatchGuard 消除内存泄漏
- **简化 API**：合并 `Generate()` 和 `GenerateStream()` 为单一接口
- **使用 `mtmd_helper_eval_chunks()`** 简化 chunk 处理（减少 60+ 行代码）

**性能优化**：
- LLM lazy load + 空闲自动卸载（5 分钟）
- 跨平台内存监控（Windows/Linux）
- 周期性统计日志 + 慢请求追踪
- gRPC 健康检查 + 优雅关闭

**架构改进**：
- 模块化设计：`server_config`、`request_validation`、`vlm_cache` 独立模块
- 更清晰的职责分离：配置加载、请求验证、缓存管理各司其职
- 更好的可测试性和可维护性

### v0.0.1a (2026.3)

初始版本，独立的 BERT 和 LLM 服务端。
