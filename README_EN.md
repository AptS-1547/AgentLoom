# AgentLoom

> A composable C++20 agent runtime, gateway, and inference infrastructure project.

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20+-green.svg)](https://cmake.org/)
[![Tests](https://img.shields.io/badge/tests-70%2B%20passing-brightgreen.svg)](README.md#测试覆盖)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[中文](README.md) | English

## Overview

AgentLoom is a C++20 server-side infrastructure project for agent systems. It provides reusable runtime components for inference services, semantic cache, memory management, multimodal input, gateway orchestration, and service-level composition.

The project originally evolved from an educational agent backend, but its public boundary is now organized around reusable infrastructure and a basic server reference implementation. Domain-specific evaluation logic is intentionally kept outside the open-source core and can be injected by downstream projects through interfaces such as `IReportEvaluator`.

Inference services are designed around process boundaries. BERT and VLM inference can be deployed as standalone gRPC servers, while gateway, persona runtime, session, memory, media, and IPC modules can be reused as CMake targets by downstream systems.

## Highlights

- High-performance inference service integration with ONNX Runtime, llama.cpp/mtmd, BERT, and VLM backends.
- Multi-level memory with L0 session context and L3 compressed long-term memory.
- Semantic cache pipeline with local cache, Redis integration, and vector retrieval support.
- HTTP/WebSocket gateway runtime with connection management, backpressure, static file hosting, and API routing.
- Persona runtime, session management, classroom-style scheduling, and trace-aware service orchestration.
- Document analysis pipeline for OOXML extraction, chunking, metadata, and LLM cache integration.
- Shared infrastructure for `Status`/`Result`, RAII wrappers, memory pools, thread pools, object pools, and queues.
- Media, IPC, and multimodal runtime components for real-time perception and VLM coordination.

## Quick Start

### Requirements

- Compiler: Visual Studio 2026/v145 on Windows, or GCC 11+/Clang 14+ on Linux.
- CMake 3.20+.
- vcpkg for gRPC, Protobuf, OpenSSL, spdlog, GTest, Redis clients, and related dependencies.
- Prebuilt dependencies for ONNX Runtime, OpenCV, Boost, llama.cpp, Faiss, SQLite, Eigen, MKL, and HuggingFace Tokenizers where required.

Local model paths and machine-specific tooling files should not be committed. Copy `config/e2e_test.example.json` to `config/e2e_test.json` and fill in local paths when running E2E tests. Copy `.clangd.example` to `.clangd` if local language-server settings are needed.

### Windows Release Build

```powershell
& "C:\Program Files\CMake\bin\cmake.exe" -B build/x64-Release -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_CONFIGURATION_TYPES=Release `
  -DBERT_VCPKG_TRIPLET=x64-windows `
  -DBERT_USE_ONNXRUNTIME_GPU=OFF `
  -DLLAMA_CPP_ROOT="<path-to-llama.cpp>"

& "C:\Program Files\CMake\bin\cmake.exe" --build build/x64-Release `
  --target multimodal_inference_server --config Release --parallel
```

### Linux Release Build

```bash
linux/scripts/bootstrap_toolchain.sh
linux/scripts/prepare_deps.sh
linux/scripts/configure.sh
linux/scripts/build.sh
linux/scripts/test.sh
linux/scripts/package.sh
```

Use `linux/scripts/configure.sh --inference` and `linux/scripts/build.sh --inference` when building inference server targets that require CUDA-enabled llama.cpp.

## Runtime Targets

| Target | Description |
| ------ | ----------- |
| `agent_gateway_server` | Gateway server with Persona Gateway, HTTP/WebSocket APIs, static file hosting, and document analysis. |
| `multimodal_inference_server` | gRPC inference server for BERT and optional VLM/llama.cpp backends. |
| `emotion_inference_server` | Lightweight CPU-only BERT emotion inference server over gRPC. |
| `persona_gateway_e2e_server` | Manual E2E gateway server used by integration tests. |

Reusable CMake targets are exposed through aliases such as `AgentLoom::core`, `AgentLoom::service`, `AgentLoom::gateway`, `AgentLoom::media`, and `AgentLoom::ipc`. Installable `find_package(AgentLoom)` exports are planned after the public header boundary is stabilized.

## Architecture

```text
Application Layer
  Gateway · Persona Runtime · Classroom Scheduler · Session

Service Layer
  Inference · Document Analysis · Memory · Semantic Cache

Infrastructure Layer
  core · net · config · cache · vector · models · storage · media · ipc
```

| Module | Responsibility |
| ------ | -------------- |
| `core` | `Status`/`Result`, exceptions, memory pools, thread pools, object pools, RAII wrappers, queues. |
| `net` | HTTP/WebSocket runtime, connection pools, backpressure, request interfaces. |
| `config` | JSON configuration, CLI fallback, section registry, distributed validation. |
| `models` | ONNX Runtime and llama.cpp/mtmd wrappers, model lifecycle management. |
| `service` | Persona runtime, sessions, classroom scheduling, gateway service aggregation. |
| `semantic_cache` | Redis semantic cache, L0 memory adapter, context risk detection, cache policies. |
| `memory` | L3 long-term memory compression and vectorized memory storage. |
| `document` | OOXML parsing, chunking, metadata storage, LLM chunk cache. |
| `media` / `ipc` | Real-time media input, shared-memory IPC, frame transport, VLM coordination. |

## Tests

The project currently contains 70+ CTest cases across core infrastructure, networking, configuration, storage, vector retrieval, semantic cache, document analysis, memory, LLM clients, and service orchestration.

```powershell
& "C:\Program Files\CMake\bin\cmake.exe" -B build/x64-Release-Tests-v145 `
  -G "Visual Studio 18 2026" -A x64 `
  -DBERT_BUILD_TESTS=ON `
  -DBERT_VCPKG_TRIPLET=x64-windows `
  -DLLAMA_CPP_ROOT="<path-to-llama.cpp>"

& "C:\Program Files\CMake\bin\cmake.exe" --build build/x64-Release-Tests-v145 `
  --config Release --parallel

ctest --test-dir build/x64-Release-Tests-v145 -C Release --output-on-failure
```

## Documentation

The full documentation index is maintained in [docs/README.md](docs/README.md).

Recommended entry points:

- [Current runtime roadmap](docs/CURRENT_RUNTIME_ROADMAP_2026_06.md)
- [Configuration system](docs/CONFIG_SYSTEM.md)
- [Deployment guide](docs/DEPLOYMENT.md)
- [Frontend/backend API protocol](docs/FRONTEND_BACKEND_API_PROTOCOL.md)
- [Extending AgentLoom](docs/EXTENDING_AGENTLOOM.md)
- [Security engineering standard](docs/SECURITY_ENGINEERING_STANDARD.md)

## Extension Boundary

AgentLoom provides reusable core libraries and basic server infrastructure. Domain-specific product logic, prompts, evaluation metrics, datasets, and organization-specific policies should live in downstream projects.

Recommended ownership:

| Layer | Owner | Content |
| ----- | ----- | ------- |
| Domain business | Downstream projects | Persona definitions, prompts, evaluation logic, product orchestration. |
| Runtime hot path | AgentLoom libraries | Sessions, cache, vector retrieval, memory, media, IPC. |
| Service entry | AgentLoom basic servers | HTTP/WebSocket/WebRTC, gRPC, configuration, lifecycle. |
| Model service | Separate processes | BERT, VLM, vLLM, or OpenAI-compatible backends. |

## Contributing

Before submitting changes, run the relevant tests, keep local configuration out of Git, update affected documentation, and follow the project conventions in [AGENTS.md](AGENTS.md).

## License

[MIT License](LICENSE)
