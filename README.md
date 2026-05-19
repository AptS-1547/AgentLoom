# AgentBackendPredict

AgentBackendPredict 是教育智能体项目的 C++ 后端基础设施与推理服务仓库。当前仓库已经从早期的单体推理服务，演进为以 `core / net / config / service / server / cache / vector / models` 分层的 C++20 基础设施工程。

本仓库当前承担两类职责：

1. **现有推理端职责**：提供 BERT/ONNX、VLM/llama.cpp、gRPC 推理服务、VLM 缓存、请求校验和运行时监控。
2. **下一阶段商业化基础设施职责**：沉淀 HTTP/WebSocket runtime、连接池、线程池、内存池、统一 RAII、配置系统、向量缓存和后续对话热路径缓存/RAG/推理引擎路由基础。

当前重点不是单纯增加业务逻辑，而是先建立稳定、可测试、可复用的服务端基础设施。

## 当前状态

已完成或正在使用的基础设施：

- `src/core`
  - `Status / Result`
  - `AppException`
  - raw memory pool
  - object pool
  - `SharedMemoryBlock`
  - trait-based `UniqueHandle`
  - blocking queue
  - thread pool
  - worker status monitoring
- `src/net`
  - Boost.Beast HTTP runtime
  - WebSocket runtime
  - `async_read_some` 分片读取路径
  - backpressure queue
  - bounded shared buffer
  - static file handler
  - connection pool
  - `IHttpRequest`
  - `IWebSocketStreamRequest`
- `src/config`
  - JSON config + CLI fallback
  - `IConfigSection`
  - section registry
  - macro-assisted section registration
  - distributed validation
- `src/service`
  - 推理业务编排
  - 请求校验
- `src/server`
  - gRPC adapter
  - process entry
  - runtime logger / common utilities
- `src/cache`
  - VLM cache
- `src/vector`
  - 轻量向量缓存/索引原型
- `tests`
  - core / net / config 单元测试

当前测试规模：

- `39` 个 CTest 用例
- 覆盖内存池、对象池、共享内存块、线程池、多生产者 submit、UniqueHandle、HTTP/WebSocket runtime、连接池、配置系统和 section validation。

## 架构

```text
src/
├── core/                      # Result/Status、异常、RAII、内存池、线程池、队列
├── net/                       # Boost.Beast HTTP/WebSocket、连接池、背压、请求接口
├── config/                    # 配置文件加载、CLI fallback、section registry
├── cache/                     # VLM 结果缓存
├── vector/                    # 向量缓存/索引基础，后续接 SIMD kernel / Faiss / SQLite
├── models/                    # ONNX Runtime、llama.cpp/mtmd 模型封装
├── service/                   # 业务编排层，目前包含 inference service
├── server/                    # gRPC adapter、runtime、main entry
│   ├── grpc/
│   ├── runtime/
│   └── main/
└── client/                    # BERT 兼容客户端和 benchmark client

tests/
├── core/
├── net/
└── config/

proto/
├── multimodal_inference.proto # 统一多模态协议
└── bert_inference.proto       # BERT 协议，保留向后兼容
```

主要 CMake target：

| Target | 类型 | 说明 |
|--------|------|------|
| `agent_core` | static library | core 基础设施 |
| `agent_net` | static library | HTTP/WebSocket/连接池/协议基础设施 |
| `server_runtime` | static library | 日志和 server common |
| `agent_models` | static library | ONNX Runtime + llama.cpp model wrapper |
| `agent_cache` | static library | VLM cache |
| `agent_vector` | static library | vector cache/index |
| `agent_config` | static library | config section system |
| `agent_service` | static library | inference service 编排 |
| `agent_server` | static library | gRPC service adapter |
| `multimodal_inference_server` | executable | 当前统一推理服务入口 |
| `bert_inference_client` | executable | BERT 兼容测试客户端 |
| `bert_benchmark_client` | executable | BERT benchmark client |
| `core_tests` | test executable | core 单元测试 |
| `net_tests` | test executable | net 单元测试 |
| `config_tests` | test executable | config 单元测试 |

## 构建

### Windows Release

```powershell
cmake -B build/x64-Release -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CONFIGURATION_TYPES=Release `
  -DBERT_VCPKG_TRIPLET=x64-windows `
  -DBERT_USE_ONNXRUNTIME_GPU=OFF

cmake --build build/x64-Release --target multimodal_inference_server --config Release --parallel
```

### Windows Release + Tests

```powershell
cmake -B build/x64-Release-Tests -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CONFIGURATION_TYPES=Release `
  -DBERT_BUILD_TESTS=ON `
  -DBERT_VCPKG_TRIPLET=x64-windows `
  -DBERT_USE_ONNXRUNTIME_GPU=OFF

cmake --build build/x64-Release-Tests --config Release --parallel
ctest --test-dir build/x64-Release-Tests -C Release --output-on-failure
```

常用增量构建：

```powershell
cmake --build build/x64-Release-Tests --target core_tests --config Release --parallel
cmake --build build/x64-Release-Tests --target net_tests --config Release --parallel
cmake --build build/x64-Release-Tests --target config_tests --config Release --parallel
```

## 运行服务

推荐使用配置文件启动：

```powershell
build\x64-Release\Release\multimodal_inference_server.exe --config config\server.example.json
```

也可以使用 CLI fallback 覆盖关键字段：

```powershell
build\x64-Release\Release\multimodal_inference_server.exe `
  --config config\server.example.json `
  --llm "D:/models/qwen2-vl.gguf" `
  --mmproj "D:/models/mmproj.gguf" `
  --bert "D:/models/joint_model.onnx" `
  --host 0.0.0.0 `
  --port 50051
```

当前配置入口由 `src/config` 的 section system 维护。新增配置字段时，优先新增或扩展对应 `IConfigSection` 实现，不建议把解析逻辑堆回 `main()` 或大型 if-else。

配置样例：

```json
{
  "models": {
    "llm": "D:/path/to/qwen2-vl.gguf",
    "mmproj": "D:/path/to/mmproj.gguf",
    "bert": "D:/path/to/joint_model.onnx",
    "vit": "",
    "n_gpu_layers": -1,
    "provider": "auto",
    "cuda_device": 0
  },
  "grpc": {
    "host": "127.0.0.1",
    "port": "50051",
    "log_dir": "logs",
    "num_cqs": 0,
    "min_pollers": 0,
    "max_pollers": 0,
    "max_receive_message_mb": 100,
    "max_send_message_mb": 10,
    "stats_log_interval_seconds": 30,
    "slow_request_ms": 250
  },
  "auth": {
    "metadata_key": "x-agent-auth",
    "token": "",
    "token_file": "",
    "token_env": "AGENT_BACKEND_AUTH_TOKEN"
  },
  "limits": {
    "max_image_mb": 20,
    "max_image_pixels": 16777216,
    "max_image_width": 8192,
    "max_image_height": 8192,
    "max_prompt_bytes": 8192,
    "min_context_size": 128,
    "max_context_size": 8192,
    "max_vlm_tokens": 2048,
    "max_temperature": 2.0,
    "max_top_k": 1000,
    "max_sequence_length": 512,
    "max_batch_size": 64,
    "max_token_id": 10000000,
    "max_abs_personality": 100.0
  },
  "vram_guard": {
    "monitor_interval_seconds": 10,
    "warning_free_mb": 1024,
    "unload_free_mb": 512,
    "min_free_before_load_mb": 0,
    "reload_after_unload": false,
    "unload_on_oom_error": true
  },
  "vlm_cache": {
    "enabled": false,
    "persist": false,
    "dir": "cache/vlm",
    "max_entries": 512,
    "max_mb": 1024,
    "ttl_seconds": 3600,
    "store_images": true,
    "store_prompts": true,
    "stale_on_failure": true,
    "default_allow_cache": true,
    "vector": {
      "enabled": false,
      "sim_threshold_high": 0.97,
      "sim_threshold_mid": 0.93,
      "max_saliency_for_mid": 0.15,
      "max_entries_per_bucket": 256,
      "ttl_seconds": 3600,
      "dir": "cache/vlm/vectors"
    }
  }
}
```

## 测试覆盖

当前测试覆盖重点：

- memory pool allocation / release / stats
- memory pool concurrent allocate / release
- object pool ownership transfer
- object pool construction failure rollback
- `SharedMemoryBlock` copy / move / reset reference counting
- blocking queue capacity and multi-producer/multi-consumer
- thread pool execution, shutdown drain, failure status, worker status
- concurrent submit from multiple producers
- `UniqueHandle` move / release / reset
- Beast HTTP request/response wrapper
- bounded `SharedBuffer`
- backpressure queue
- static file handler
- HTTP runtime handler dispatch
- typed `IHttpRequest`
- WebSocket upgrade / echo / fragmented read path
- typed `IWebSocketStreamRequest`
- WebSocket outbound backpressure
- connection pool lease lifetime and protocol limits
- config help detection
- config file + CLI fallback override
- auth token file resolving
- section-level validation

运行完整测试：

```powershell
ctest --test-dir build/x64-Release-Tests -C Release --output-on-failure
```

## 依赖

主要依赖：

- Visual Studio 2022 / MSVC
- C++20
- ONNX Runtime 1.17.1 CPU / 1.20.1 GPU
- llama.cpp with mtmd support
- OpenCV 4.10
- Boost 1.85 headers
- gRPC
- Protobuf
- spdlog
- OpenSSL
- SQLite prebuilt package
- Faiss CPU prebuilt package
- Eigen headers
- GTest for tests
- nlohmann/json, vendored in `third_party`

依赖策略：

- vcpkg 主要用于 gRPC、Protobuf、OpenSSL、spdlog、GTest 等基础库。
- ONNX Runtime、OpenCV、Boost、SQLite、Faiss、MKL、llama.cpp 当前按预编译/本地路径集成。
- 后续 Redis 计划通过 Boost.Redis 接入，Redis 只作为共享缓存和失效协调层，不作为唯一数据真源。

## 与 EducationalAgentProject 的关系

`EducationalAgentProject` 是教育智能体主项目，负责 persona、Prompt、记忆、OpenAI-compatible API、小模型前后处理、多 persona 调度和 Python 侧业务编排。

本仓库负责 C++ 后端基础设施、ONNX/gRPC 推理服务、协议层 runtime、缓存和后续高性能服务端能力。

当前设计方向：

- Python 侧保留快速迭代和复杂业务逻辑。
- C++ 侧负责热路径、资源治理、协议层、缓存、推理服务和稳定基础设施。
- 高 QPS 对话主路径后续以 vLLM 为主力 LLM serving。
- llama.cpp 保留为 VLM、低显存、边缘设备和主路径 OOM/过载备用推理路径。

## 缓存与推理路由规划

当前已实现：

- VLM 结果缓存
- 轻量 vector cache 原型
- HTTP/WebSocket connection pool
- typed request interfaces
- 基础 backpressure

下一阶段规划：

- `vector_similarity`
  - normalized dot product
  - fixed dim 384 / 768 kernels
  - dynamic fallback
  - Top-K / threshold scan
  - SIMD path after benchmark
- `LocalHotVectorIndex`
  - local hot embedding matrix
  - bucket-based lookup
  - Redis payload fetch
- `CuratedSemanticCache`
  - 离线中文教育对话缓存基底
  - DeepSeek / GLM / Qwen 生成
  - 多模型评审和质量分
  - answer_core + persona render
- `KnowledgeRagCache`
  - 教材、课标、经典题、常见误区
  - structured knowledge card
  - citation and corpus version
- `MemoryContextLayer`
  - memory scope validation
  - user/session scoped cache
  - long context fallback
- `LlmEngineRouter`
  - vLLM main path
  - llama.cpp fallback path
  - capacity-aware routing
  - OOM/overload degradation

详细讨论见：

- [对话热路径缓存与推理引擎路由策略](docs/CONVERSATION_CACHE_AND_INFERENCE_STRATEGY.md)
- [商业化架构方案](docs/COMMERCIAL_ARCHITECTURE.md)

## 文档

- [Infrastructure Development Plan](docs/INFRASTRUCTURE_PLAN.md)
- [Commercial Architecture](docs/COMMERCIAL_ARCHITECTURE.md)
- [Conversation Cache and Inference Strategy](docs/CONVERSATION_CACHE_AND_INFERENCE_STRATEGY.md)
- [Legacy Architecture Analysis](docs/LEGACY_ARCHITECTURE_ANALYSIS.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Architecture Diagram](docs/ARCHITECTURE_DIAGRAM.md)
- [Cache and Optimization](docs/CACHE_AND_OPTIMIZATION.md)
- [Net API Notes](docs/NET_API_NOTES.md)
- [GStreamer Windows Setup](docs/GSTREAMER_WINDOWS_SETUP.md)
- [Next Runtime Roadmap](docs/NEXT_RUNTIME_ROADMAP.md)
- [Vision Migration](docs/VISION_MIGRATION.md)
- [Vision Pipeline Analysis](docs/VISION_PIPELINE_ANALYSIS.md)
- [gRPC API](proto/multimodal_inference.proto)

## 当前工程原则

- core 层不依赖 gRPC、OpenCV、Faiss、SQLite、Redis。
- storage / vector / cache / net / service 应保持边界清晰。
- C API 资源优先通过 trait-based RAII 包装。
- 大对象、帧数据和网络 buffer 需要明确所有权和边界检查。
- HTTP/WebSocket 语义优先复用 Boost.Beast，不重复定义协议基础语义。
- Redis 是共享缓存，不是唯一真源。
- Faiss 是可替换向量索引实现，不向公共接口泄漏 Faiss 类型。
- SQLite 是本地持久化优先选项。
- 测试应覆盖失败路径、并发路径和资源释放路径。

## 更新日志

### 2026-05-16

- 重构为当前分层架构：`core / net / config / service / server / cache / vector / models`。
- 新增 C++20 core 基础设施：memory pool、object pool、`SharedMemoryBlock`、thread pool、blocking queue、`UniqueHandle`。
- 新增 Boost.Beast HTTP/WebSocket runtime。
- 新增 WebSocket `async_read_some` 分片读取路径和大消息边界限制。
- 新增 HTTP/WebSocket connection pool。
- 新增 `IHttpRequest` 和 `IWebSocketStreamRequest` 业务协议接口。
- 新增 section-based config system，替代旧 if-else CLI fallback。
- 新增 core / net / config 单元测试。
- 明确后续对话热路径缓存、RAG、Redis、vLLM/llama.cpp 路由方案。

### 2026-05-04

- 合并 BERT/LLM/Multimodal 三个服务端为单一 `multimodal_inference_server`。
- 整合 BERT ONNX Runtime、VLM llama.cpp/mtmd、VLM cache、请求验证和 VRAM guard。
- 引入 JSON 配置文件 + CLI fallback。
- 引入 gRPC 健康检查、运行时统计和慢请求追踪。

### 2026-03

- 初始版本，独立的 BERT 和 LLM 服务端。

## License

MIT License
