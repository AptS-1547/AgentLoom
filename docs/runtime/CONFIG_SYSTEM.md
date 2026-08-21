# 配置系统扩展说明

## 概述

AgentLoom 使用 JSON 配置、CLI override 和 section registry 组合加载 Server 配置。配置解析集中在 `src/config/`，每个 section 独立负责字段加载、CLI 参数和校验，最终统一写入 `MultimodalServerOptions`。

当前已注册的主要 section：

| Section | 职责 |
|---------|------|
| `models` | BERT、VLM、mmproj、ViT 和运行时模型路径 |
| `grpc` | gRPC 地址、端口、消息限制和统计阈值 |
| `auth` | inference server metadata token |
| `limits` | 图片、prompt、token 和温度限制 |
| `vram_guard` | GPU 内存监控和 OOM 卸载策略 |
| `embedding` | tokenizer、ONNX embedding 和 pooling |
| `vlm_cache` | 结果缓存、vector cache 和 Prompt KV Cache |
| `llm` | OpenAI-compatible LLM backend 和凭据来源 |
| `http` | HTTP/WebSocket runtime |
| `gateway_auth` | Gateway JWT、cookie、SQLite/Redis session store |
| `persona_gateway` | Persona 默认配置、线程池、静态文件和文档存储 |
| `skill_session` | Skill Session timeout、容量和维护周期 |

> 文档状态：当前配置入口说明。新增字段时以相应 `src/config/sections/*_section.cpp` 为实现真值，并同步更新 example JSON 和本文件。

## HTTP 服务器配置

### JSON 配置

```json
{
  "http": {
    "address": "0.0.0.0",
    "port": 8080,
    "io_threads": 4,
    "request_timeout_seconds": 30,
    "websocket_idle_timeout_seconds": 300,
    "request_body_limit_mb": 16,
    "websocket_read_buffer_limit_mb": 16
  }
}
```

### 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `address` | string | `"0.0.0.0"` | 监听地址 |
| `port` | number | `8080` | 监听端口 (1-65535) |
| `io_threads` | number | `1` | IO 线程数（建议 CPU 核心数） |
| `request_timeout_seconds` | number | `30` | HTTP 请求超时（秒） |
| `websocket_idle_timeout_seconds` | number | `60` | WebSocket 空闲超时（秒） |
| `request_body_limit_mb` | number | `16` | HTTP 请求体大小限制（MB） |
| `websocket_read_buffer_limit_mb` | number | `16` | WebSocket 读缓冲区限制（MB） |

### CLI 参数

```bash
--http-address 0.0.0.0
--http-port 8080
--http-threads 4
--http-request-timeout 30
--http-websocket-timeout 300
--http-body-limit 16
--http-websocket-buffer-limit 16
```

## Embedding 模型配置

### JSON 配置

```json
{
  "embedding": {
    "tokenizer_path": "path/to/tokenizer.json",
    "onnx_model_path": "path/to/embedding-model.onnx",
    "execution_provider": "auto",
    "allow_cpu_fallback": true,
    "cuda_device_id": 0,
    "intra_op_num_threads": 0,
    "inter_op_num_threads": 0,
    "pooling_strategy": "mean",
    "normalize": true,
    "expected_dimension": 384,
    "require_token_type_ids": false
  }
}
```

### 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tokenizer_path` | string | `""` | HuggingFace tokenizer.json 路径 |
| `onnx_model_path` | string | `""` | ONNX embedding 模型路径 |
| `execution_provider` | string | `"auto"` | 执行提供者：`auto` / `cpu` / `cuda` |
| `allow_cpu_fallback` | bool | `true` | CUDA 不可用时回退到 CPU |
| `cuda_device_id` | number | `0` | CUDA 设备 ID |
| `intra_op_num_threads` | number | `0` | 算子内线程数（0=自动） |
| `inter_op_num_threads` | number | `0` | 算子间线程数（0=自动） |
| `pooling_strategy` | string | `"mean"` | Pooling 策略：`mean` / `cls` |
| `normalize` | bool | `true` | 是否 L2 归一化 |
| `expected_dimension` | number | `0` | 预期维度（0=不校验） |
| `require_token_type_ids` | bool | `false` | 是否需要 token_type_ids |

### CLI 参数

```bash
--embedding-tokenizer path/to/tokenizer.json
--embedding-model path/to/model.onnx
--embedding-provider auto
--embedding-cuda-device 0
--embedding-intra-threads 0
--embedding-inter-threads 0
--embedding-pooling mean
--embedding-normalize true
--embedding-dimension 384
```

### 验证规则

- **可选配置**：embedding section 完全可选，不配置则不加载模型
- **路径校验**：如果配置了任一路径，则两个路径都必须提供
- **Pooling 策略**：只支持 `mean` 和 `cls`
- **维度校验**：`expected_dimension > 0` 时，首次推理会校验实际维度

## 使用示例

### 完整配置文件

参考 `config.example.json`：

```json
{
  "models": {
    "llm": "models/llama-3.2-3b-instruct-q4_k_m.gguf",
    "bert": "models/bert-emotion.onnx"
  },
  "http": {
    "port": 8080,
    "io_threads": 4
  },
  "embedding": {
    "tokenizer_path": "models/tokenizer.json",
    "onnx_model_path": "models/minilm-l12-v2.onnx",
    "pooling_strategy": "mean",
    "normalize": true,
    "expected_dimension": 384,
    "batch_enabled": true,
    "batch_max_pending_requests": 1024,
    "batch_max_size": 16,
    "batch_max_wait_ms": 2,
    "batch_max_inflight": 1
  },
  "grpc": {
    "host": "127.0.0.1",
    "port": "50051",
    "max_pollers": 0
  }
}
```

### 启动命令

```bash
# 使用配置文件
./multimodal_inference_server --config config.json

# CLI 覆盖配置
./multimodal_inference_server --config config.json \
  --http-port 9090 \
  --embedding-model models/custom-embedding.onnx
```

### 代码中访问配置

```cpp
#include "server_options.h"

void InitializeEmbedding(const MultimodalServerOptions& options) {
    if (options.embedding.tokenizer_path.empty()) {
        // Embedding 未配置，跳过
        return;
    }

    // 加载 tokenizer
    auto tokenizer = vector::HfTokenizer::LoadFromFile(
        options.embedding.tokenizer_path
    ).value();

    // 加载 ONNX 模型
    vector::EmbeddingModelOptions opts;
    opts.model_path = options.embedding.onnx_model_path;
    opts.pooling = (options.embedding.pooling_strategy == "mean")
        ? vector::PoolingStrategy::Mean
        : vector::PoolingStrategy::Cls;
    opts.normalize = options.embedding.normalize;
    opts.expected_dimension = options.embedding.expected_dimension;
    opts.execution_provider = options.embedding.execution_provider;
    opts.cuda_device_id = options.embedding.cuda_device_id;

    auto model = vector::OnnxTextEmbeddingModel::Load(std::move(opts)).value();

    // 创建 pipeline
    auto pipeline = std::make_shared<vector::EmbeddingPipeline>(
        std::make_shared<vector::HfTokenizer>(std::move(tokenizer)),
        std::move(model)
    );
}

void InitializeHttpServer(const MultimodalServerOptions& options) {
    net::HttpServer server(options.http);
    // ...
}
```

## 配置优先级

1. **CLI 参数** — 最高优先级
2. **JSON 配置文件** — 中等优先级
3. **代码默认值** — 最低优先级

CLI 参数会覆盖 JSON 配置。

## 故障排查

### 问题：embedding 模型加载失败

**检查**：
1. 路径是否正确（绝对路径或相对于工作目录）
2. ONNX 模型 IR version 是否兼容（需要 IR ≤ 9）
3. tokenizer.json 是否存在

**解决**：
```bash
# 验证路径
ls -l path/to/tokenizer.json
ls -l path/to/model.onnx

# 检查 IR version
python -c "import onnx; m = onnx.load('model.onnx'); print(f'IR: {m.ir_version}')"

# 降级 IR version（如果需要）
python -c "import onnx; m = onnx.load('model.onnx'); m.ir_version = 9; onnx.save(m, 'model.onnx')"
```

### 问题：HTTP 端口被占用

**错误**：`bind: Address already in use`

**解决**：
```bash
# 检查端口占用
netstat -ano | findstr :8080

# 更换端口
--http-port 8081
```

### 问题：pooling_strategy 配置错误

**错误**：`embedding.pooling_strategy must be 'mean' or 'cls'`

**解决**：
```json
{
  "embedding": {
    "pooling_strategy": "mean"  // 只能是 "mean" 或 "cls"
  }
}
```

## 下一步

配置系统已就绪，可以：
1. **集成到主程序**：在 `main.cpp` 中加载配置并初始化 HTTP 服务器和 Embedding 模型
2. **添加更多 section**：如 Redis、Faiss、GStreamer 配置
3. **实现热重载**：监听配置文件变化并动态更新
