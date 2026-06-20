# Docker deployment

This directory provides a multi-container deployment layout for the Linux package
produced by `linux/scripts/package.sh`.

## Architecture

- `gateway`
  - runs `agent_gateway_server`
  - serves HTTP/WebSocket/static frontend
  - owns Persona runtime, document analysis, L0/L3 orchestration
- `emotion-inference`
  - runs `emotion_inference_server`
  - CPU-only BERT emotion inference over gRPC
- `multimodal-inference` (optional profile)
  - runs `multimodal_inference_server`
  - intended for VLM / llama.cpp / GPU-capable environments
- `redis`
  - shared cache backend

The image is shared across services. Different containers select their runtime
role through `APP_ROLE`.

## Prerequisites

Build the Linux package first inside Linux or WSL:

```bash
bash docker/build-linux-package.sh
```

This must create:

```text
build/linux-package/bin
build/linux-package/config
```

The wrapper uses the existing Linux build scripts and keeps the Linux vcpkg
install root at:

```text
build/linux-vcpkg-installed/x64-linux-release
```

Build with the optional multimodal inference target when a compatible
llama.cpp CUDA binary is prepared:

```bash
bash docker/build-linux-package.sh --inference
```

## Required config adjustments

Before starting containers, start from:

- `config/agent_gateway.container.example.json`
- `config/emotion.container.example.json`

The Gateway config already points Redis and gRPC dependencies at Docker
service DNS names. A reduced excerpt:

```json
{
  "persona_gateway": {
    "address": "0.0.0.0",
    "port": 8080
  },
  "gateway_auth": {
    "redis_host": "redis",
    "redis_port": "6379"
  },
  "l0_memory": {
    "redis_host": "redis",
    "redis_port": 6379
  },
  "document_llm_chunk_cache": {
    "redis_host": "redis",
    "redis_port": 6379
  },
  "document_semantic_cache": {
    "redis_host": "redis",
    "redis_port": 6379
  },
  "emotion_analyzer": {
    "target": "emotion-inference:50052"
  },
  "local_llm": {
    "target": "multimodal-inference:50051"
  }
}
```

For multimodal inference, provide a dedicated config derived from
`config/server.example.json` with container-visible model paths.

## Start

Gateway + Redis + BERT emotion service:

```bash
docker compose -f docker/compose.yml up --build
```

Include the multimodal service:

```bash
docker compose -f docker/compose.yml --profile vlm up --build
```

## Notes

- This layout intentionally separates Gateway and inference services into
  different containers for fault isolation and independent scaling.
- `emotion-inference` is the default lightweight inference backend.
- `multimodal-inference` is optional because many environments do not provide
  compatible NVIDIA GPUs.
