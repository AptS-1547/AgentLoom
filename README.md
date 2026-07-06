# AgentBackendPredict

> 教育智能体的 C++ 后端基础设施与推理服务

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20+-green.svg)](https://cmake.org/)
[![Tests](https://img.shields.io/badge/tests-59%2B%20passing-brightgreen.svg)](#测试覆盖)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📖 项目简介

AgentBackendPredict 是教育智能体项目的 C++ 后端基础设施，提供高性能推理服务、语义缓存、多级记忆管理和网关层服务编排能力。

从早期的单体推理服务演进为 **生产级 C++20 服务端基础设施**，以 `core / net / config / cache / vector / models / service / server` 分层架构，支撑教育智能体的热路径、资源管理和协议层能力。

### ✨ 核心特性

- 🚀 **高性能推理**：ONNX Runtime + llama.cpp/mtmd 多模态推理，支持 BERT/VLM
- 🧠 **多级记忆系统**：L0 上下文记忆 + L3 长期压缩记忆，支持向量化语义检索
- ❤️ **情感层融合**：BERT 主干 + 关键词/向量证据 + softmax 融合头 + V-A 情绪状态机
- 🎯 **语义缓存**：多层缓存策略（本地热缓存 + Redis 共享缓存 + Faiss 向量索引）
- 🌐 **完整网关层**：HTTP/WebSocket 服务器、连接池、背压控制、静态文件托管
- 📚 **文档分析**：DOCX/PPTX 解析、文档分块、元数据管理和 LLM 缓存
- 🎭 **课堂调度器**：多人格并发调度、主动发言状态机、Trace 链路追踪
- 🧩 **基础设施完善**：内存池、线程池、对象池、RAII 封装、统一错误处理

### 🎯 适用场景

- 教育智能体多模态推理后端
- 高性能对话服务热路径缓存
- 语义搜索与 RAG 检索增强
- 生产级 C++ 服务端基础设施参考

---

## 🚀 快速开始

### 前置依赖

- **编译器**：Visual Studio 2022 (MSVC) / GCC 11+ / Clang 14+
- **CMake**：3.20+
- **vcpkg**：用于管理 gRPC、Protobuf、OpenSSL、spdlog 等依赖
- **预编译库**：ONNX Runtime、OpenCV、Boost 1.85、llama.cpp

### 构建服务

**Windows Release 构建**

```powershell
cmake -B build/x64-Release -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CONFIGURATION_TYPES=Release `
  -DBERT_VCPKG_TRIPLET=x64-windows `
  -DBERT_USE_ONNXRUNTIME_GPU=OFF

cmake --build build/x64-Release --target multimodal_inference_server --config Release --parallel
```

**Linux Release 构建（WSL2）**

Linux 构建使用 WSL2 环境和自动化脚本：

```bash
# 1. 准备工具链和依赖
linux/scripts/bootstrap_toolchain.sh
linux/scripts/prepare_deps.sh

# 2. 配置构建
linux/scripts/configure.sh

# 3. 构建（不包含推理服务器）
linux/scripts/build.sh

# 4. 构建推理服务器（需要 CUDA llama.cpp）
linux/scripts/configure.sh --inference
linux/scripts/build.sh --inference

# 5. 运行测试
linux/scripts/test.sh

# 6. 打包产物
linux/scripts/package.sh
```

脚本会自动处理：
- Python venv 创建
- ONNX Runtime、Boost、Eigen、SQLite、Faiss、MKL 依赖下载
- vcpkg 依赖安装（gRPC、Protobuf、OpenSSL、spdlog 等）
- llama.cpp CUDA 构建（如需推理服务器）
- CMake 配置和 Ninja 构建

### 运行服务

**Agent Gateway Server（推荐生产部署）**

统一网关服务，集成 Persona Gateway、HTTP/WS、静态文件托管、文档分析：

```powershell
build\x64-Release\Release\agent_gateway_server.exe config\gateway.json
```

服务监听：
- `0.0.0.0:8080` - HTTP API + WebSocket + 静态文件
- 集成 Persona Runtime、L0/L3 记忆、语义缓存、课堂调度

**Multimodal Inference Server**

gRPC 多模态推理服务（BERT + VLM）：

```powershell
build\x64-Release\Release\multimodal_inference_server.exe --config config\server.example.json
```

服务监听：`127.0.0.1:50051`（gRPC）

**Emotion Inference Server**

仅 BERT 情绪推理的轻量级服务（CPU-only）：

```powershell
build\x64-Release\Release\emotion_inference_server.exe --config config\emotion.json
```

服务监听：`127.0.0.1:50052`（gRPC）

### 运行测试

```powershell
# 构建测试
cmake -B build/tests -G "Visual Studio 17 2022" -A x64 `
  -DBERT_BUILD_TESTS=ON `
  -DBERT_VCPKG_TRIPLET=x64-windows

cmake --build build/tests --config Release --parallel

# 运行所有测试
ctest --test-dir build/tests -C Release --output-on-failure
```

---

## 🏗️ 架构概览

### 模块分层

```text
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  Gateway · Persona Runtime · Classroom Scheduler · Session  │
└───────────────────────────────┬─────────────────────────────┘
┌───────────────────────────────┴─────────────────────────────┐
│                       Service Layer                          │
│  Inference · Document Analysis · Memory · Semantic Cache    │
└───────────────────────────────┬─────────────────────────────┘
┌───────────────────────────────┴─────────────────────────────┐
│                    Infrastructure Layer                      │
│  core · net · config · cache · vector · models · storage    │
└─────────────────────────────────────────────────────────────┘
```

### 核心模块

| 模块 | 职责 |
|------|------|
| **core** | Result/Status、异常、内存池、线程池、对象池、RAII、队列 |
| **net** | HTTP/WebSocket runtime、连接池、背压控制、请求接口 |
| **config** | 配置加载、CLI fallback、section registry、分布式校验 |
| **cache** | VLM 结果缓存、Redis 连接池、多级缓存策略 |
| **vector** | 向量索引（Exact/Faiss）、Embedding pipeline、HF Tokenizer FFI |
| **models** | ONNX Runtime 封装、llama.cpp/mtmd 封装、模型生命周期管理 |
| **service** | 推理编排、文档分析、记忆管理、语义缓存 |
| **server** | gRPC adapter、Gateway HTTP/WS、runtime logger、进程入口 |
| **storage** | SQLite 持久化、文档元数据、会话状态 |
| **semantic_cache** | L0 记忆适配器、Redis 语义缓存、上下文风险检测、缓存策略 |
| **memory** | L3 长期记忆压缩器、向量化记忆存储 |
| **llm** | 本地/云端 LLM 客户端、gRPC 推理适配、重试与降级 |
| **document** | 文档解析、分块、元数据存储、LLM chunk 缓存 |

### 情感层处理模式

主链路情感层以 BERT 情绪分类为 primary analyzer，并在 `FusedEmotionAnalyzer` 中融合多来源证据：

```text
用户/AI 文本
  -> gRPC BERT emotion analyzer
  -> keyword / vector / LLM evidence
  -> per-label fusion logits
  -> softmax emotion distribution
  -> confidence / margin gate
  -> V-A emotion state tracker
  -> prompt hint + generation params + memory metadata
```

融合头不再把异构证据直接归一化，而是先构建每个 label 的 logit：

```text
logit[label] =
    head_bias
  + bert_signal_weight * bert_weight * bert_prob[label] * label_reliability[label]
  + evidence_signal_weight * evidence_score[label]
  + margin_signal_weight * primary_margin_bonus
```

随后对 logits 做 softmax，得到主情绪分布。这样既保留 BERT 概率、Macro-F1 风格标签可靠性、关键词/向量/LLM 来源权重的可解释性，也保留多类情绪竞争关系。

门控策略基于融合后的 top confidence 和 top margin：

```text
top < accept_confidence || top1 - top2 < ambiguity_margin
```

低置信或类别接近时可触发 LLM fallback；fallback 结果作为 `llm` evidence 重新进入融合头，而不是直接覆盖 BERT。最终用户情绪和 AI 回复情绪共同更新 V-A 状态机，并序列化到会话状态与 L0 记忆元数据中。

### 主要构建目标

**服务器可执行文件**

| Target | 说明 |
|--------|------|
| `agent_gateway_server` | 生产级网关服务器：Persona Gateway + HTTP/WS + 静态文件托管 + 文档分析 |
| `multimodal_inference_server` | 多模态推理服务器：gRPC + BERT + VLM/llama.cpp（可选） |
| `emotion_inference_server` | BERT 情绪推理服务器：仅 CPU，gRPC |
| `persona_gateway_e2e_server` | E2E 测试网关服务器（手动测试用） |

**核心库**

| Library | 职责 |
|---------|------|
| `agent_core` | Result/Status、内存池、线程池、对象池、队列、RAII |
| `agent_net` | HTTP/WebSocket runtime、连接池、背压控制 |
| `agent_tls` | TLS context、SSL/TLS 封装 |
| `agent_http_client` | 出站 HTTP/HTTPS 客户端、重试策略 |
| `agent_llm` | LLM 客户端（OpenAI + 本地 gRPC） |
| `agent_models` | ONNX Runtime + llama.cpp 模型封装 |
| `agent_cache` | VLM 结果缓存 |
| `agent_vector` | 向量索引（Exact/Faiss）+ Embedding pipeline + HF Tokenizer FFI |
| `agent_semantic_cache` | 语义缓存管线 + Redis 连接池 + L0 记忆适配器 |
| `agent_storage` | SQLite 异步执行器 + 连接池 + 事务 |
| `agent_vector_storage` | 向量元数据存储 + 分区注册表 |
| `agent_memory` | L3 长期记忆压缩器 |
| `agent_document` | 文档分析 + OOXML 解析 + 分块 + LLM chunk 缓存 |
| `agent_service` | Persona 运行时 + 会话管理 + 课堂调度 + 网关服务 |
| `agent_config` | 配置系统 + section registry + CLI fallback |
| `server_runtime` | Logger + server common utilities |

**测试与工具**

| Target | 说明 |
|--------|------|
| `core_tests` / `net_tests` / `config_tests` | 核心模块单元测试 |
| `storage_tests` / `vector_storage_tests` | 存储层单元测试 |
| `semantic_cache_tests` / `document_tests` | 缓存与文档单元测试 |
| `memory_tests` / `vector_tests` / `service_tests` | 业务层单元测试 |
| `llm_tests` / `llm_integration_tests` | LLM 客户端测试 |
| `http_client_tests` / `tls_tests` | HTTP/TLS 客户端测试 |
| `l3_compression_e2e_test` | L3 记忆压缩 E2E 测试 |
| `document_analysis_e2e_test` | 文档分析 E2E 测试 |
| `bert_inference_client` / `bert_benchmark_client` | BERT 协议客户端工具 |

---

## 📚 文档索引

### 架构与设计

- [基础设施开发计划](docs/INFRASTRUCTURE_PLAN.md)
- [商业化架构方案](docs/COMMERCIAL_ARCHITECTURE.md)
- [对话热路径缓存与推理引擎路由](docs/CONVERSATION_CACHE_AND_INFERENCE_STRATEGY.md)
- [架构图](docs/ARCHITECTURE_DIAGRAM.md)
- [流式架构](docs/STREAMING_ARCHITECTURE.md)

### 实现细节

- [配置系统](docs/CONFIG_SYSTEM.md)
- [C++ 服务端安全工程准则](docs/SECURITY_ENGINEERING_STANDARD.md)
- [缓存与优化](docs/CACHE_AND_OPTIMIZATION.md)
- [Net API 设计笔记](docs/NET_API_NOTES.md)
- [Redis 兼容性](docs/REDIS_COMPATIBILITY.md)
- [向量化 Embedding 模型](docs/PHASE2_EMBEDDING_MODEL.md)

### 迁移与部署

- [部署指南](docs/DEPLOYMENT.md)
- [视觉模型迁移](docs/VISION_MIGRATION.md)
- [Redis++ 迁移](docs/REDIS_PLUSPLUS_MIGRATION_COMPLETE.md)
- [网关前端对齐](docs/GATEWAY_FRONTEND_SESSION_ALIGNMENT.md)

### 其他

- [性能报告](docs/PERFORMANCE_REPORT.md)
- [E2E 测试与情绪管道](docs/E2E_TEST_AND_EMOTION_PIPELINE.md)
- [团队实施计划](docs/TEAM_IMPLEMENTATION_PLAN.md)

---

## 🧪 测试覆盖

当前测试规模：**70+ CTest 用例**，覆盖率持续提升中。

### 测试覆盖领域

✅ **core** - 内存池、对象池、线程池、共享内存块、队列、UniqueHandle  
✅ **net** - HTTP/WebSocket runtime、连接池、背压、静态文件、请求接口  
✅ **http_client** - 出站 HTTP/HTTPS 客户端、URL 解析、重试策略  
✅ **tls** - TLS context、SSL 证书加载  
✅ **config** - 配置加载、CLI fallback、section 校验  
✅ **storage** - SQLite 异步执行器、连接池、事务、RAII  
✅ **vector_storage** - 向量元数据存储、分区注册表、指纹计算  
✅ **vector** - Tokenizer、Embedding、向量索引、Top-K 检索、索引管理器  
✅ **semantic_cache** - 缓存管线、L0 适配器、批量加载  
✅ **document** - OOXML 提取器、文档解析  
✅ **memory** - L3 长期记忆压缩器、向量化存储  
✅ **llm** - OpenAI 客户端、本地 gRPC 客户端、重试与降级  
✅ **service** - Persona 算法、会话管理、Persona 运行时、网关服务  

### 运行测试

```powershell
# 运行所有测试
ctest --test-dir build/tests -C Release --output-on-failure

# 运行特定测试
build\tests\Release\core_tests.exe
build\tests\Release\net_tests.exe
build\tests\Release\vector_tests.exe
build\tests\Release\semantic_cache_tests.exe
build\tests\Release\service_tests.exe

# 增量构建单个测试
cmake --build build/tests --target core_tests --config Release
cmake --build build/tests --target semantic_cache_tests --config Release
```

### E2E 测试工具

```powershell
# L3 记忆压缩 E2E
build\tests\Release\l3_compression_e2e_test.exe config\l3_test.json

# 文档分析 E2E
build\tests\Release\document_analysis_e2e_test.exe config\doc_test.json

# Persona Gateway E2E（手动测试）
build\tests\Release\persona_gateway_e2e_server.exe config\gateway_e2e.json
```

---

## ⚙️ 配置说明

服务通过 JSON 配置文件 + CLI 参数启动，支持多 section 分布式校验。

### 配置样例

```json
{
  "models": {
    "llm": "D:/models/qwen2-vl.gguf",
    "mmproj": "D:/models/mmproj.gguf",
    "bert": "D:/models/joint_model.onnx",
    "n_gpu_layers": -1
  },
  "grpc": {
    "host": "127.0.0.1",
    "port": "50051",
    "max_receive_message_mb": 100
  },
  "http": {
    "host": "0.0.0.0",
    "port": "8080",
    "num_threads": 4,
    "dist_root": "dist"
  },
  "auth": {
    "metadata_key": "x-agent-auth",
    "token_env": "AGENT_BACKEND_AUTH_TOKEN"
  },
  "vlm_cache": {
    "enabled": true,
    "max_entries": 512,
    "max_mb": 1024,
    "ttl_seconds": 3600
  },
  "semantic_cache": {
    "enabled": true,
    "redis_url": "tcp://127.0.0.1:6379",
    "sim_threshold": 0.93,
    "max_entries": 10000
  }
}
```

### CLI 覆盖参数

```powershell
multimodal_inference_server.exe `
  --config config/server.json `
  --llm "D:/models/custom.gguf" `
  --host 0.0.0.0 `
  --port 50051
```

完整配置说明见 [配置系统文档](docs/CONFIG_SYSTEM.md)。

---

## 🔗 依赖管理

### vcpkg 管理依赖

- **gRPC** + Protobuf - RPC 框架
- **OpenSSL** - TLS/SSL 加密
- **spdlog** - 高性能日志
- **GTest** - 单元测试框架
- **hiredis** + redis++ - Redis 客户端
- **libzip** - ZIP 文件解析（DOCX/PPTX）
- **pugixml** - XML 解析（OOXML）

### 预编译依赖

- **ONNX Runtime** 1.17.1 (CPU) / 1.20.1 (GPU) - BERT 推理
- **llama.cpp** with mtmd support - VLM 多模态推理
- **OpenCV** 4.10 - 图像处理
- **Boost** 1.85 (header-only) - Asio/Beast/Redis
- **Faiss** 1.14.1 CPU - 向量索引
- **SQLite** 3.53.1 - 本地持久化
- **Eigen** 5.0.1 - 线性代数（向量运算）
- **MKL** 2023.1 - BLAS 加速（Faiss 依赖）
- **HuggingFace Tokenizers** (Rust FFI) - 分词器

依赖按 `deps/` 和 `vcpkg_installed/` 两种方式集成。Linux 构建脚本自动下载和配置所有依赖。

---

## 🎯 与 EducationalAgentProject 的关系

**EducationalAgentProject** 是教育智能体主项目，负责 Persona、Prompt、OpenAI-compatible API、Python 侧业务编排和多 persona 调度。

**AgentBackendPredict** 负责 C++ 后端基础设施、推理服务、协议层 runtime、缓存和高性能服务端能力。

### 职责划分

| 层 | 负责方 | 内容 |
|----|--------|------|
| 业务逻辑 | Python | Persona、Prompt、记忆策略、业务编排 |
| 热路径 | C++ | 推理、缓存、向量检索、网关、连接池 |
| 主力 LLM | vLLM | 高 QPS 对话主路径 |
| 备用/VLM | llama.cpp | VLM、低显存、边缘设备、OOM 降级 |

---

## 🚧 路线图

### 当前阶段（v2.0）

- ✅ 基础设施分层架构
- ✅ HTTP/WebSocket runtime
- ✅ 向量化 Embedding pipeline
- ✅ 语义缓存与 Redis 集成
- ✅ 文档分析与元数据管理
- ✅ 网关层与课堂调度器
- 🔄 L3 长期记忆压缩（进行中）

### 下一阶段

- 🎯 语义缓存完整链路（预填充 + 在线更新）
- 🎯 RAG 知识库集成（教材、课标、经典题）
- 🎯 vLLM 推理引擎路由与降级
- 🎯 流式推理与 WebSocket 双工
- 🎯 性能优化（SIMD 向量运算、零拷贝传输）
- 🎯 多机部署与负载均衡

详见 [下一阶段 Runtime 路线图](docs/NEXT_RUNTIME_ROADMAP.md)。

---

## 🛠️ 开发指南

### 代码约定

1. C API 对象（HANDLE、SQL 连接、io_context）封装为 RAII
2. 错误码/异常统一包装为 `core::Status` 和 `core::Result<T>`
3. 业务代码错误需在 `core::Status` 基础上输出日志
4. 避免 `void*` 裸指针，优先使用 `std::optional<T>`
5. 内存分配使用 `core` 中的内存池/对象池
6. 所有模块需要单元测试和 E2E 测试
7. 每个业务模块提供扩展接口（`I` 前缀抽象类）
8. 避免平台单一 API（如 WIN32 API），使用条件编译分支
9. 所有文件读写、路径处理默认 UTF-8，不使用系统本地编码
10. 接口和源码文件能合并即合并，不超过 800 行上限

### 风格约定

- **注释**：少量准确的英文注释
- **命名**：遵循 C++ 标准库风格（snake_case for functions/variables，PascalCase for types）
- **模块边界**：core 层不依赖 gRPC/OpenCV/Faiss/Redis
- **测试覆盖**：失败路径、并发路径、资源释放路径

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

在贡献代码前，请确保：

1. 运行 `ctest` 确保所有测试通过
2. 新功能需要添加单元测试
3. 遵循项目代码约定和风格
4. 更新相关文档

---

## 📝 License

[MIT License](LICENSE)

---

## 📧 联系方式

- **项目维护**：Orange20000922
- **主项目**：[EducationalAgentProject](../EducationalAgentProject)
- **相关项目**：[Filerestore_CLI](https://github.com/Orange20000922/Filerestore_CLI)

---

<p align="center">
  <i>Built with ❤️ by Orange20000922</i>
</p>
