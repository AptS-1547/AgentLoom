# Infrastructure Development Plan

This document records the next-stage infrastructure plan for the C++ migration.

The goal is to replace heavy Python-side runtime dependencies with a lightweight C++ foundation while preserving the useful design assets from the existing research project.

## Direction

The project should move from dependency preparation into infrastructure interface development.

The first phase should not directly port Persona or memory business logic. The priority is to define stable low-level interfaces so that later modules share the same error model, logging model, threading model, storage model, and protocol boundaries.

## Core Layers

### 1. Core Infrastructure

Path: `src/core/`

Responsibilities:

- memory pool
- thread pool
- centralized error and exception types
- `Result<T>` or equivalent status-return model
- cancellation primitives
- logging adapter compatible with the existing spdlog setup

Rules:

- no dependency on gRPC
- no dependency on OpenCV
- no dependency on Faiss
- no dependency on SQLite
- no dependency on Redis
- should depend only on the C++ standard library and logging primitives

Expected files:

```text
src/core/result.h
src/core/error.h
src/core/exception.h
src/core/logger_adapter.h
src/core/thread_pool.h
src/core/memory_pool.h
src/core/cancellation.h
```

### 2. Network Layer

Path: `src/net/`

Responsibilities:

- HTTP server foundation
- HTTPS support
- WebSocket session handling
- request and response abstractions
- router and middleware interfaces
- timeout handling
- backpressure handling

Primary dependencies:

- Boost.Asio
- Boost.Beast
- OpenSSL

Expected files:

```text
src/net/http_request.h
src/net/http_response.h
src/net/router.h
src/net/http_server.h
src/net/websocket_session.h
src/net/tls_config.h
```

### 3. Streaming Layer

Path: `src/streaming/`

Responsibilities:

- basic media ingress interfaces
- frame packet representation
- stream session abstraction
- OpenCV frame integration
- encoded image payload handling
- raw byte payload handling
- backpressure policy for frame streams

Primary dependencies:

- OpenCV
- Boost.Asio when asynchronous stream handling is needed

Expected files:

```text
src/streaming/frame_packet.h
src/streaming/frame_source.h
src/streaming/stream_session.h
src/streaming/opencv_frame_source.h
```

### 4. Storage Layer

Path: `src/storage/`

Responsibilities:

- SQLite connection wrapper
- transaction management
- schema migration
- key-value storage
- append-only event log foundation
- memory record persistence

Primary dependency:

- SQLite

Rules:

- SQLite is the default local source of truth.
- Storage code should not depend on Faiss.
- Storage code should not depend on network protocols.
- Schema versioning should be available from the beginning.

Expected files:

```text
src/storage/sqlite_connection.h
src/storage/sqlite_transaction.h
src/storage/migration.h
src/storage/kv_store.h
```

### 5. Vector Layer

Path: `src/vector/`

Responsibilities:

- vector index abstraction
- Faiss-backed local vector index
- vector search result representation
- embedding record identifiers
- index snapshot and restore foundation

Primary dependency:

- Faiss

Rules:

- Public interfaces should not expose Faiss types.
- Faiss should be replaceable behind the interface.
- SQLite metadata should be referenced through stable ids, not direct storage coupling.

Expected files:

```text
src/vector/vector_index.h
src/vector/faiss_index.h
src/vector/embedding_record.h
```

### 6. Cache Layer

Path: `src/cache/`

Responsibilities:

- local memory cache
- optional Redis cache
- shared cache interface
- cache expiration policy
- cache miss and fallback handling

Primary dependencies:

- Boost.Redis
- Boost.Asio
- OpenSSL

Rules:

- Redis is optional.
- Redis should not be the source of truth.
- A local in-memory cache should exist as the default implementation.
- `boost/redis/src.hpp` should be included in one implementation translation unit only.

Expected files:

```text
src/cache/cache_store.h
src/cache/memory_cache_store.h
src/cache/redis_cache_store.h
```

### 7. Memory Infrastructure

Path: `src/memory/`

Responsibilities:

- memory record model
- memory repository
- memory vector index coordination
- SQLite and Faiss consistency management
- short-term memory persistence
- long-term memory retrieval foundation
- user profile storage foundation

Primary dependencies:

- SQLite
- Faiss
- core infrastructure

Rules:

- Memory code may compose storage and vector modules.
- Storage and vector modules should not depend on memory business logic.
- Redis may be used as a cache, not as persistent memory.
- The first implementation should focus on infrastructure, not advanced memory policy.

Expected files:

```text
src/memory/memory_record.h
src/memory/memory_repository.h
src/memory/memory_index.h
src/memory/memory_service_core.h
```

## Dependency Direction

The desired dependency direction is:

```text
core
  -> storage
  -> vector
  -> cache
  -> streaming
  -> net
  -> memory
```

The actual composition should follow these rules:

- `core` must stay independent.
- `storage` must not depend on Faiss.
- `vector` must not depend on SQLite.
- `memory` may compose `storage` and `vector`.
- `net` should call services through interfaces, not directly through SQLite or Faiss.
- Redis must remain optional.
- SQLite is the local persistent source of truth.
- Faiss is the local vector index.

## Target Runtime Model

The lightweight runtime model should be:

```text
SQLite     -> persistent local data
Faiss      -> local vector search
Redis      -> optional distributed cache
Boost.Asio -> asynchronous IO foundation
Boost.Beast -> HTTP and WebSocket protocol layer
OpenCV     -> image and stream processing foundation
ONNX Runtime -> BERT and related inference
llama.cpp  -> VLM and local LLM inference
```

Heavy dependencies such as mem0ai, Qdrant, LangChain, and LlamaIndex should not be part of the C++ hot path.

They may remain useful as migration references, compatibility services, or optional external integrations, but the core runtime should not depend on them.

## First Implementation Milestone

The first milestone should implement `src/core/` only.

Suggested scope:

- `ErrorCode`
- `Status`
- `Result<T>`
- `AppException`
- `LoggerAdapter`
- `ThreadPool`
- basic memory pool interface

Acceptance criteria:

- builds with Visual Studio 17 2022
- does not add new heavy dependencies
- uses the existing CMake dependency layout
- integrates cleanly with spdlog
- can be reused by SQLite, Faiss, HTTP, WebSocket, and Redis modules later

