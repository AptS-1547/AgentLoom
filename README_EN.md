# AgentLoom

> A composable C++20 agent runtime, gateway, and multimodal inference infrastructure project

[中文](README.md) | English

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20+-green.svg)](https://cmake.org/)
[![Tests](https://img.shields.io/badge/CTest-420_passing-brightgreen.svg)](#tests)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

AgentLoom is a C++20 server-side runtime for agent systems. It packages an HTTP/WebSocket gateway, persona and session runtime, semantic cache, long-term memory, document analysis, real-time media paths, and BERT/VLM inference as composable libraries and basic server implementations.

The project originated from an educational agent backend, but the open-source boundary is domain-neutral. AgentLoom owns reusable runtime code, protocols, and server foundations. Downstream projects own personas, prompts, domain evaluation, datasets, and product orchestration. Domain report evaluation is injected through `IReportEvaluator`; no organization-specific evaluator is included.

The current version is `0.1.0` and remains under active development. It is suitable for downstream development, system integration, and production-oriented engineering validation, but absolute ABI stability is not guaranteed.

## Implemented Capabilities

- HTTP/WebSocket gateway with JWT/cookie authentication, static files, request filtering, backpressure, and maintenance tasks.
- Persona, session, skill-session, multi-persona scheduling, proactive speech state, tracing, and persisted emotion state.
- ONNX Runtime BERT emotion inference and llama.cpp/mtmd streaming or synchronous VLM inference.
- Redis/SQLite L0 memory, compressed L3 memory, Exact/Faiss retrieval, VLM result cache, and prompt KV cache.
- DOCX/PPTX OOXML extraction, managed document storage, chunk analysis, metadata, and LLM/semantic caches.
- WebRTC signaling, GStreamer media pipelines, OpenCV sampling, shared-memory frame IPC, backlog, and VLM coordination.
- Shared `core::Status`/`Result`, RAII wrappers, memory/thread/object pools, concurrent queues, TLS, and HTTP clients.

## Runtime Boundaries

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
              |
              +---- shared memory IPC ----> local frame/VLM data plane
```

Inference is primarily reused through standalone processes and protobuf/gRPC contracts. Gateway, persona, session, memory, media, and IPC components can also be linked directly into downstream source builds.

| Server target | Current responsibility |
| --- | --- |
| `agent_gateway_server` | HTTP/WebSocket gateway combining persona, auth, memory, documents, skills, and static frontend hosting |
| `emotion_inference_server` | ONNX BERT emotion inference over gRPC |
| `multimodal_inference_server` | BERT plus llama.cpp/mtmd VLM inference over gRPC |
| `persona_gateway_e2e_server` | Manual gateway integration server |

## Reusable CMake Targets

Downstream projects using `add_subdirectory()` can link stable build-tree aliases:

```cmake
add_subdirectory(path/to/AgentLoom)

target_link_libraries(my_agent PRIVATE
    AgentLoom::core
    AgentLoom::runtime
    AgentLoom::service
    AgentLoom::gateway
)
```

Available aliases include `core`, `net`, `tls`, `http_client`, `config`, `storage`, `vector_storage`, `vector`, `semantic_cache`, `memory`, `document`, `llm`, `models`, `cache`, `ipc`, `media_inference`, `media`, `runtime`, `gateway`, and `service`. Installable `find_package(AgentLoom)` exports are not available yet.

## Build

Requirements:

- Visual Studio 2026/v145 on Windows, or GCC 11+/Clang 14+ on Linux
- CMake 3.20+
- vcpkg manifest dependencies: gRPC, Protobuf, OpenSSL, spdlog, Redis clients, libzip, pugixml, and nlohmann/json; GTest for tests
- Prebuilt or external ONNX Runtime, llama.cpp with mtmd, OpenCV, Boost, SQLite, Faiss, Eigen, MKL, and HuggingFace Tokenizers C API packages
- **Toolchain note:** As of 2026-07-11, building the CUDA-enabled llama.cpp dependency with the latest VS2026/v145 generator fails during CUDA compilation, indicating that the installed CUDA Toolkit does not yet provide sufficient VS2026/v145 compatibility. The llama.cpp dependency used by this repository's current test baseline was therefore built with VS2022/v143. This theoretically introduces an ABI compatibility risk, but no related failure has appeared across the existing unit, stress, or integration test runs. Until official support is available, using VS2022/v143 consistently for the CUDA-enabled dependency is recommended.

The local `deps/` and `vcpkg_installed/` directories are not distributed with the source. Linux scripts prepare the required packages. On Windows, keep the CMake generator, MSVC toolset, and vcpkg ABI aligned.

### Windows

Use a CMake version that supports the VS2026 generator, such as `C:\Program Files\CMake\bin\cmake.exe`:

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

The default path builds the CPU runtime, gateway, emotion server, and tests. The VLM server additionally requires a prepared CUDA llama.cpp build.

```bash
linux/scripts/bootstrap_toolchain.sh
linux/scripts/prepare_deps.sh
linux/scripts/configure.sh
linux/scripts/build.sh
linux/scripts/test.sh

# Optional VLM inference target
linux/scripts/configure.sh --inference
linux/scripts/build.sh --inference
```

Linux dependencies are installed under `build/linux-vcpkg-installed`, separate from the Windows root `vcpkg_installed/`.

## Configuration and Startup

Configuration uses JSON sections with CLI overrides. Treat `src/config/sections/` and the public examples as the source of truth:

| Process | Example |
| --- | --- |
| Gateway | [`config/agent_gateway.example.json`](config/agent_gateway.example.json) |
| Multimodal inference | [`config/server.example.json`](config/server.example.json) |
| Container emotion inference | [`config/emotion.container.example.json`](config/emotion.container.example.json) |

Inject API keys and authentication tokens through environment variables or local files. Do not commit them in JSON. Local E2E paths belong in ignored `config/e2e_test.json`; `.clangd.example` is provided for local language-server setup.

```powershell
# The current gateway entry uses the positional path for startup context and
# --config for the shared configuration loader.
build\x64-Release\Release\agent_gateway_server.exe `
  config\agent_gateway.example.json `
  --config config\agent_gateway.example.json `
  --no-stdin-stop

build\x64-Release\Release\multimodal_inference_server.exe `
  --config config\server.example.json

build\x64-Release\Release\emotion_inference_server.exe `
  --config config\emotion.container.example.json
```

Public examples contain placeholder model paths. The gateway example expects `AGENT_LLM_API_KEY` by default. Authentication, Redis, embedding, emotion analysis, L0/L3 memory, and document caches remain deployment-configurable.

## Source Layout

| Path | Responsibility |
| --- | --- |
| `src/core` | `Status`/`Result`, RAII, memory/object/thread pools, and concurrent queues |
| `src/net` | HTTP/WebSocket server, TLS, HTTP client, connection management, and backpressure |
| `src/config` | JSON/CLI section registry, parsing, and cross-section validation |
| `src/models` | ONNX Runtime, llama.cpp/mtmd, runner pools, and model lifecycle |
| `src/service` | Persona/session runtime, gateway routes, scheduling, and inference services |
| `src/semantic_cache` | Redis pools, L0 adapters, policies, and context risk detection |
| `src/storage` / `src/vector` | SQLite, vector metadata, embeddings, and Exact/Faiss indexes |
| `src/memory` / `src/document` | L3 compression, OOXML analysis, and document caches |
| `src/media` / `src/ipc` | WebRTC/GStreamer, frame processing, shared-memory IPC, and VLM coordination |
| `src/server` | Gateway/gRPC process entries, logging, status mapping, and runtime statistics |

## Tests

CMake currently registers **420 CTest cases** covering core infrastructure, TLS/HTTP, LLM clients, configuration, SQLite, vector retrieval, semantic cache, documents, memory, media/IPC, persona/gateway behavior, and gRPC boundaries. Cross-process E2E and standalone benchmark targets are also included.

As of 2026-07-11, the complete Windows VS2026/v145 Release build and **all 420 CTest cases pass** and have been verified across repeated runs, with no observed code regressions.

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

## Extension Boundary

| Layer | Recommended owner | Content |
| --- | --- | --- |
| Domain business | Downstream project | Personas, prompts, evaluation, datasets, and product orchestration |
| Runtime hot path | AgentLoom libraries | Sessions, caches, vector retrieval, memory, media, and IPC |
| Service entry | AgentLoom basic servers | HTTP/WebSocket/WebRTC, gRPC, auth, configuration, and lifecycle |
| Model backend | Separate process/service | BERT, VLM, vLLM, or OpenAI-compatible backends |

Business extensions should use `I...` interfaces. AgentLoom provides base session/training reports and the `IReportEvaluator` extension point, but no school-, organization-, or product-specific metrics and weights.

## Documentation

See [docs/README.md](docs/README.md) for the complete categorized index. Recommended entry points:

- [Current runtime roadmap](docs/CURRENT_RUNTIME_ROADMAP_2026_06.md)
- [Configuration system](docs/CONFIG_SYSTEM.md)
- [Deployment guide](docs/DEPLOYMENT.md)
- [Frontend/backend API protocol](docs/FRONTEND_BACKEND_API_PROTOCOL.md)
- [Extending AgentLoom](docs/EXTENDING_AGENTLOOM.md)
- [Security engineering standard](docs/SECURITY_ENGINEERING_STANDARD.md)

## Contributing and License

Run relevant tests, update affected documentation, and follow [AGENTS.md](AGENTS.md) before submitting changes. Report security issues according to [SECURITY.md](SECURITY.md); do not disclose credentials or unresolved vulnerabilities in public issues.

AgentLoom is licensed under the [MIT License](LICENSE).
