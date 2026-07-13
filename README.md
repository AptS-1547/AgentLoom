# AgentLoom

> 可组合的 Agent 服务端基础设施 —— HTTP/WebSocket Gateway、Persona Runtime、语义记忆、实时视觉输入与 BERT/VLM 推理，C++20 实现

中文 | [English](README_EN.md)

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20+-green.svg)](https://cmake.org/)
[![Tests](https://img.shields.io/badge/CTest-420_passing-brightgreen.svg)](#测试)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 项目定位

AgentLoom 是一个面向服务端智能体的 C++20 Runtime。它把 HTTP/WebSocket Gateway、Persona 与 Session Runtime、语义缓存、长期记忆、文档分析、实时媒体链路以及 BERT/VLM 推理组织成可组合的库目标和基础 Server。

项目源自教育智能体后端，但开源边界不绑定教育产品：通用 Runtime、协议和基础 Server 留在 AgentLoom；Persona、Prompt、领域评估、数据集与产品编排由下游项目持有。领域报告评估可通过 `IReportEvaluator` 注入，仓库不包含特定组织的评估器实现。

当前版本为 `0.1.0`。接口仍在活跃开发中，适合用于二次开发、系统集成和面向生产的工程验证，不保证ABI的绝对稳定。

## 核心特性

- **可组合架构**：核心能力以 CMake target 形式暴露，既可作为库集成到下游源码工程，也可以作为独立 Server 部署。
- **完整的对话热路径**：从 WebSocket 连接、会话管理、记忆召回、情绪感知到 LLM 生成的端到端 C++ 实现，避免跨语言边界开销。
- **实时多模态感知**：从浏览器摄像头 WebRTC 输入、GStreamer 解码、OpenCV 动态抽帧到 VLM 推理的完整链路。
- **推理成本优化**：语义缓存、Prompt KV Cache 与 VLM 结果缓存协同，降低重复推理开销；VRAM guard 支持低显存与 OOM 降级。
- **进程边界清晰**：BERT 与 VLM 推理作为独立 gRPC 进程，支持容器化与独立扩缩容；Gateway 通过 gRPC 或共享内存 IPC 对接。
- **生产工程质量**：420+ 单元测试与跨进程 E2E 测试，覆盖并发路径、错误恢复与资源释放；`core::Status`/`Result` 统一错误模型。

## 已实现能力

- **Gateway Runtime**：统一 HTTP/WebSocket 入口、JWT/cookie 认证、静态文件托管、请求过滤、背压和运行时维护任务。
- **Agent Runtime**：Persona、Session、Skill Session、多人格调度、主动发言状态机、trace 和情绪状态持久化。
- **模型服务**：ONNX Runtime BERT 情绪推理，以及基于 llama.cpp/mtmd 的流式与同步 VLM 推理。
- **情感融合**：BERT 主干结合关键词等证据，通过可配置 fusion head、置信度和 margin gate 更新 V-A 状态。
- **记忆与缓存**：Redis/SQLite L0 记忆、L3 压缩记忆、Exact/Faiss 向量检索、VLM 结果缓存，以及基于 llama.cpp sequence state 的 image-prefix Prompt KV Cache（memory/Redis 双后端）。
- **文档链路**：DOCX/PPTX OOXML 提取、受管文件存储、分块分析、元数据和 LLM/语义缓存。
- **实时多模态输入**：WebRTC signaling（offer/answer/ICE/resume）、GStreamer `webrtcbin` media pipeline、OpenCV 动态抽帧（MOG2/直方图/边缘变化/EMA/cooldown）、关键帧 JPEG/PNG 编码（NVIDIA/VAAPI/D3D11/QSV 硬件加速与软件回退）。
- **帧推理链路**：共享内存帧 IPC（MPMC sequence ring、RAII claim、epoch 重建）+ gRPC IPC 控制面（grant/revoke/probe、lease 协调、跨进程故障恢复）组成数据面/控制面分离的传输层；上层由有序准入、mmap 磁盘 spool 溢出回放、私有 backlog、VLM coordinator 和执行级封口聚合（running → sealing → replay → aggregate）构成受控的关键帧推理生命周期。
- **基础设施**：`core::Status`/`Result`、RAII 句柄封装、内存池、线程池、对象池、线程安全队列、keyed serial executor（按 key 串行、会话亲和保序）、task group（结构化并发）、TLS context 和并发 HTTP client。

## 运行时边界

```text
Browser / Downstream Application
              |
              v
 agent_gateway_server  <---->  OpenAI-compatible / local LLM
   HTTP · WebSocket              backend
   Persona · Session
   Memory · Document
   Media · Skill
              |
              +---- gRPC ----> emotion_inference_server
              |
              +---- gRPC ----> multimodal_inference_server
              |                   (VLM 推理 · 帧 coordinator)
              |
              +== shared memory (数据面) + gRPC (控制面) ==> 帧推理链路
```

模型推理默认以独立进程和 protobuf/gRPC 协议作为复用边界。关键帧传输采用共享内存数据面 + gRPC 控制面分离：大体积帧走共享内存零拷贝，grant/revoke/epoch 生命周期与故障恢复走 gRPC 控制信令。Gateway、Persona、Session、Memory、Media 与 IPC 也可以通过 CMake target 直接组合到下游源码工程。

### 基础 Server

| Target | 当前职责 |
| --- | --- |
| `agent_gateway_server` | HTTP/WebSocket Gateway，组合 Persona、认证、记忆、文档、Skill 与静态前端 |
| `emotion_inference_server` | ONNX BERT 情绪推理 gRPC Server |
| `multimodal_inference_server` | BERT + llama.cpp/mtmd VLM gRPC Server |
| `persona_gateway_e2e_server` | 用于手动集成验证的 Gateway E2E Server |

### 可复用 CMake 目标

下游项目使用 `add_subdirectory()` 时可以链接稳定别名：

```cmake
add_subdirectory(path/to/AgentLoom)

target_link_libraries(my_agent PRIVATE
    AgentLoom::core
    AgentLoom::runtime
    AgentLoom::service
    AgentLoom::gateway
)
```

当前公开别名包括 `core`、`net`、`tls`、`http_client`、`config`、`storage`、`vector_storage`、`vector`、`semantic_cache`、`memory`、`document`、`llm`、`models`、`cache`、`ipc`、`media_inference`、`media`、`runtime`、`gateway` 和 `service`。安装式 `find_package(AgentLoom)` 导出尚未提供。

## 构建

### 依赖

- Visual Studio 2026/v145（Windows），或 GCC 11+/Clang 14+（Linux）
- CMake 3.20+
- vcpkg manifest 依赖：gRPC、Protobuf、OpenSSL、spdlog、Redis clients、libzip、pugixml、nlohmann/json；测试另需 GTest
- 预编译/外部依赖：ONNX Runtime、llama.cpp（含 mtmd）、OpenCV、Boost、SQLite、Faiss、Eigen、MKL 和 HuggingFace Tokenizers C API
- **工具链说明**：截至 2026-07-11，使用最新 VS2026/v145 工具链构建启用 CUDA 的 llama.cpp 会在 CUDA 编译阶段因兼容性问题直接失败，表明当前 CUDA Toolkit 对 VS2026/v145 的支持仍不充分。因此，本仓库测试基线使用的 llama.cpp 依赖由 VS2022/v143 构建。理论上这可能引入 ABI 兼容风险，但现有单元测试、压力测试和集成测试均未复现相关问题。在官方支持完善前，建议对启用 CUDA 的依赖统一使用 VS2022/v143 构建。

仓库的 `deps/` 与 `vcpkg_installed/` 是本地依赖目录，不随源码分发。Linux 脚本可以准备对应依赖；Windows 需要按本机路径准备依赖，并确保 CMake generator、MSVC 工具集和 vcpkg ABI 一致。

### Windows

VS2026/v145 必须使用支持 `Visual Studio 18 2026` generator 的 CMake，例如 `C:\Program Files\CMake\bin\cmake.exe`：

```powershell
& "C:\Program Files\CMake\bin\cmake.exe" -B build/x64-Release `
  -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_CONFIGURATION_TYPES=Release `
  -DBERT_VCPKG_TRIPLET=x64-windows `
  -DBERT_USE_ONNXRUNTIME_GPU=OFF `
  -DLLAMA_CPP_ROOT="<path-to-llama.cpp>" `
  -DLLAMA_CPP_BUILD="<path-to-llama.cpp-build>"

& "C:\Program Files\CMake\bin\cmake.exe" --build build/x64-Release `
  --config Release --parallel
```

### Linux / WSL2

默认脚本构建 CPU Runtime、Gateway、Emotion Server 和测试；VLM Server 需要预先准备 CUDA llama.cpp：

```bash
linux/scripts/bootstrap_toolchain.sh
linux/scripts/prepare_deps.sh
linux/scripts/configure.sh
linux/scripts/build.sh
linux/scripts/test.sh

# 可选：启用 VLM inference target
linux/scripts/configure.sh --inference
linux/scripts/build.sh --inference
```

Linux 使用独立的 `build/linux-vcpkg-installed`，不会复用或写入 Windows 的仓库根 `vcpkg_installed/`。

## 配置与启动

配置系统使用 JSON section，并允许 CLI 覆盖。实际字段以 `src/config/sections/` 和公开样例为准：

| 进程 | 配置样例 |
| --- | --- |
| Gateway | [`config/agent_gateway.example.json`](config/agent_gateway.example.json) |
| Multimodal inference | [`config/server.example.json`](config/server.example.json) |
| Container emotion inference | [`config/emotion.container.example.json`](config/emotion.container.example.json) |

API Key 和认证 token 应通过环境变量或本地文件注入，不要写入受 Git 跟踪的 JSON。模型路径同理使用本地配置；`config/e2e_test.json` 和 `.clangd` 已忽略，仓库分别提供 example 文件。

```powershell
# Gateway：当前入口同时使用位置参数定位工作目录，并由 --config 加载统一配置
build\x64-Release\Release\agent_gateway_server.exe `
  config\agent_gateway.example.json `
  --config config\agent_gateway.example.json `
  --no-stdin-stop

# Multimodal gRPC inference
build\x64-Release\Release\multimodal_inference_server.exe `
  --config config\server.example.json

# Emotion gRPC inference；按需覆盖模型和端口
build\x64-Release\Release\emotion_inference_server.exe `
  --config config\emotion.container.example.json
```

公开样例包含占位模型路径，运行前必须改为本机文件。Gateway 样例默认要求 `AGENT_LLM_API_KEY`；认证、Redis、embedding、emotion analyzer、L0/L3 memory 和 document cache 都可以按部署环境配置。

## 模块结构

| 目录 | 职责 |
| --- | --- |
| `src/core` | `Status`/`Result`、RAII、内存/对象/线程池、并发队列、keyed serial executor 与 task group |
| `src/net` | HTTP/WebSocket Server、TLS、HTTP client、连接管理和背压 |
| `src/config` | JSON/CLI section registry、配置解析与跨 section 校验 |
| `src/models` | ONNX Runtime、llama.cpp/mtmd、runner pool 和模型生命周期 |
| `src/service` | Persona/Session Runtime、Gateway routes、调度与 inference service |
| `src/semantic_cache` | Redis 连接池、L0 adapter、缓存策略和 context risk detector |
| `src/storage` / `src/vector` | SQLite、向量元数据、embedding pipeline 与 Exact/Faiss index |
| `src/memory` / `src/document` | L3 压缩记忆、OOXML 文档分析与缓存 |
| `src/media` / `src/ipc` | WebRTC/GStreamer、帧编码抽样、共享内存数据面与 gRPC 控制面、有序准入、磁盘 spool 回放与 VLM coordination |
| `src/server` | Gateway 与 gRPC 进程入口、日志、状态映射和运行时统计 |

## 测试

当前 CMake 注册 **420 个 CTest**，覆盖 core、TLS/HTTP、LLM、配置、SQLite、向量检索、语义缓存、文档、记忆、Media/IPC、Persona/Gateway 和 gRPC 边界，并包含跨进程 E2E 与独立 benchmark target。

截至 2026-07-11，Windows VS2026/v145 Release 全量构建与 **420 项 CTest 均已通过**，并经过多轮重复验证，未观察到代码回归。

```powershell
& "C:\Program Files\CMake\bin\cmake.exe" -B build/x64-Release-Tests-v145 `
  -G "Visual Studio 18 2026" -A x64 `
  -DBERT_BUILD_TESTS=ON `
  -DBERT_VCPKG_TRIPLET=x64-windows `
  -DBERT_USE_ONNXRUNTIME_GPU=OFF `
  -DLLAMA_CPP_ROOT="<path-to-llama.cpp>" `
  -DLLAMA_CPP_BUILD="<path-to-llama.cpp-build>"

& "C:\Program Files\CMake\bin\cmake.exe" --build `
  build/x64-Release-Tests-v145 --config Release --parallel

ctest --test-dir build/x64-Release-Tests-v145 `
  -C Release --output-on-failure
```

## 扩展边界

| 层 | 推荐归属 | 内容 |
| --- | --- | --- |
| 领域业务 | 下游项目 | Persona、Prompt、领域评估、数据集与产品编排 |
| Runtime 热路径 | AgentLoom 库 | Session、缓存、向量检索、记忆、Media 与 IPC |
| 服务入口 | AgentLoom 基础 Server | HTTP/WebSocket/WebRTC、gRPC、认证、配置与生命周期 |
| 模型后端 | 独立进程/外部服务 | BERT、VLM、vLLM 或 OpenAI-compatible backend |

业务扩展优先通过 `I...` 接口注入。AgentLoom 提供基础训练/会话报告和 `IReportEvaluator` 扩展点，不包含特定学校、组织或商业项目的指标、权重与实现。

## 文档

完整文档目录见 [docs/README.md](docs/README.md)。建议从以下内容开始：

- [当前 Runtime 路线图](docs/CURRENT_RUNTIME_ROADMAP_2026_06.md)
- [配置系统](docs/CONFIG_SYSTEM.md)
- [部署指南](docs/DEPLOYMENT.md)
- [Frontend/Backend API 协议](docs/FRONTEND_BACKEND_API_PROTOCOL.md)
- [扩展 AgentLoom](docs/EXTENDING_AGENTLOOM.md)
- [安全工程规范](docs/SECURITY_ENGINEERING_STANDARD.md)

## 贡献与许可

提交更改前请运行相关测试、同步受影响文档，可以参考 [AGENTS.md](AGENTS.md) 的工程约定。安全问题请参阅 [SECURITY.md](SECURITY.md)，不要在公开 Issue 中披露凭据或未修复漏洞。

AgentLoom 使用 [MIT License](LICENSE)。
