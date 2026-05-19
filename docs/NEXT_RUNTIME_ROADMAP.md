# Next Runtime Roadmap

> Working roadmap for the next AgentBackendPredict runtime phase.  
> Status: planning note  
> Date: 2026-05-19

## 1. Direction

AgentBackendPredict remains a C++ primary runtime. The project should not be
rewritten in another language. Rust and Python are allowed where they have clear
ownership boundaries:

- C++ owns online hot-path runtime, native model/media integrations, resource
  governance, vector search, cache routing, and protocol infrastructure.
- Rust may be used for narrow native kernels exposed through a stable C ABI,
  especially HuggingFace tokenizer integration.
- Python owns offline data/model tooling, embedding artifact generation,
  evaluation, curated cache building, and fast-changing business orchestration.

The next phase should move from protocol/runtime foundations into storage,
embedding, memory, media, and inference infrastructure.

## 2. Guiding Principles

1. External library types must not leak into public module interfaces.
   - Hide `sqlite3*`, `faiss::Index*`, `GstElement*`, Redis connection handles,
     `Ort::Session`, and Rust tokenizer handles behind C++ adapters.
2. Runtime and tooling stay separate.
   - Offline embedding/data building is Python-first.
   - Online vector search and cache hit routing are C++-first.
3. Any buffer crossing async boundaries must be owned by a session/runtime-level
   pool or copied into one.
4. Every queue or streaming boundary needs backpressure or a bounded window.
5. Embedding/vector artifacts must be versioned with model, tokenizer, pooling,
   normalization, dimension, corpus, and policy fingerprints.
6. Global, user, and session memory scopes must be validated before cache or
   memory hits are reused.

## 3. Phase Plan

### Phase 1: Native Dependency Safety Layer

Goal: use existing `core` infrastructure and RAII patterns to wrap native
dependencies into safe, small C++ APIs.

Initial modules:

```text
src/storage/sqlite/
src/vector/faiss/
src/cache/redis/
src/media/gstreamer/
```

Recommended first implementation: SQLite.

SQLite first API sketch:

```text
SqliteConnection
SqliteStatement
SqliteTransaction
```

SQLite test coverage:

- open in-memory database
- execute schema
- prepare/bind/step/query
- commit transaction
- rollback on transaction destructor
- invalid SQL error mapping
- move semantics and no double close
- busy timeout

Why SQLite first:

- Small C API.
- Clear RAII value.
- Required for future memory metadata/payload store.
- Good template for Faiss, Redis, and GStreamer adapter style.

### Phase 2: Vector Similarity and Artifact Runtime

Goal: establish the C++ vector search foundation before connecting full
embedding model inference.

Core pieces:

```text
src/vector/vector_similarity.h/.cpp
src/vector/embedding_matrix.h/.cpp
src/vector/exact_vector_index.h/.cpp
```

Functions:

- `NormalizeInPlace`
- `DotDynamic`
- `DotFixed<384>`
- `DotFixed<512>`
- `DotFixed<768>`
- `TopKNormalizedDot`
- `ThresholdScan`

Artifact format:

```text
artifact/
  manifest.json
  vectors.f32
  entries.jsonl
  payload.jsonl
  optional semantic_cache.db
  optional faiss.index
```

Runtime should support loading Python-built artifacts and running search from a
given query vector before online embedding is integrated.

### Phase 3: Faiss Adapter

Goal: add Faiss as a replaceable vector index backend after the exact index
interface is stable.

Rules:

- Do not expose Faiss types in public headers.
- Use normalized vectors with inner product for cosine-like search.
- Keep exact scan backend for tests, fallback, and small hot buckets.
- Support save/load of index files.
- Keep row-id mapping external and versioned.

Public interface should look like:

```text
IVectorIndex
ExactVectorIndex
FaissFlatIpIndex
```

### Phase 4: Rust Tokenizer C ABI + ONNX Embedding Runtime

Goal: C++ hot path can perform online text vectorization:

```text
text -> tokenizer -> ONNX embedding -> pooling -> L2 normalize -> vector
```

Rust component:

```text
third_party/hf_tokenizers_capi/
  Cargo.toml
  Cargo.lock
  include/hf_tokenizers_capi.h
  src/lib.rs
```

C++ wrapper:

```text
src/embedding/
  tokenizer.h
  hf_tokenizer.h/.cpp
  text_embedder.h
  onnx_text_embedder.h/.cpp
  pooling.h
  normalization.h
```

Tokenizer C ABI requirements:

- Opaque tokenizer handle.
- `create_from_file`.
- `encode`.
- `encode_batch`.
- explicit free functions for all Rust-allocated memory.
- no panics/exceptions across FFI.
- output `input_ids`, `attention_mask`, and `token_type_ids` as `int64_t`.

Validation:

- C++ token ids exactly match Python HuggingFace tokenizer output.
- C++ ONNX embedding and Python embedding cosine similarity should be at least
  `0.999` for reference inputs.
- Benchmark tokenizer single/batch, ONNX single/batch, and full end-to-end
  embedding latency.

### Phase 5: Redis Importer and Cache Runtime

Goal: Redis is used as shared payload/cache layer, not as the primary vector
compute engine.

Recommended Redis key layout:

```text
semantic:<version>:manifest
semantic:<version>:payload:<entry_id>
semantic:<version>:meta:<entry_id>
semantic:<version>:bucket:<bucket_name>
semantic:<version>:vector:<entry_id>   optional
```

Hot-path pattern:

```text
local vector index / mmap matrix
  -> top-k row ids
  -> payload keys
  -> Redis/local payload fetch
  -> policy and scope verification
```

First Redis tool should consume artifact files rather than HuggingFace models:

```text
semantic_cache_importer
  --manifest artifact/manifest.json
  --entries artifact/entries.jsonl
  --payload artifact/payload.jsonl
  --redis redis://127.0.0.1:6379
```

### Phase 6: GStreamer Media Frame Runtime

Goal: move from GStreamer probe to decoded frame/audio ingestion.

Already completed:

- Windows MSVC GStreamer probe.
- `webrtcbin`, `appsink`, `appsrc`, `decodebin`, `videoconvert`,
  `audioconvert`, `audioresample`, `opusdec`, `vp8dec`, and `rtpbin` verified.

Next steps:

1. `videotestsrc -> videoconvert -> appsink` probe.
2. Define internal `VideoFrame` and `AudioChunk`.
3. Build `GstRuntime` wrapper with a GLib loop thread.
4. Add appsink callback that emits internal frames.
5. Later connect `webrtcbin` with signaling over the existing WebSocket runtime.

Media module should live under `src/media`, not `src/net`.

### Phase 7: Memory System Rewrite

Goal: use Faiss + SQLite to rebuild the original multi-level memory system.

Memory dimensions:

```text
MemoryScope:
  global
  user
  session
  persona
  course
  classroom

MemoryType:
  fact
  preference
  learning_state
  misconception
  recent_context
  knowledge_card
  dialogue_turn
```

Storage split:

- SQLite: metadata, payload, scope, version, lifecycle, citations.
- Faiss/exact index: vector search.
- Redis: hot shared payload, invalidation, multi-instance cache.

Query path:

```text
MemoryQuery
  -> scope validation
  -> online embedding
  -> vector search
  -> metadata filter
  -> recency/quality scoring
  -> payload load
  -> context pack
```

Critical rule: private user/session memory must never leak into global semantic
cache hits.

### Phase 8: Inference Runtime and Multi-GPU

Goal: evolve inference from model wrappers into schedulable, observable,
degradable runtime.

Scope:

- vLLM as external OpenAI-compatible main LLM path.
- llama.cpp as fallback/VLM/edge path.
- ONNX embedding runtime.
- VLM, ASR, and future TTS adapters.
- GPU device registry.
- VRAM guard.
- queue depth / overload routing.
- OOM fallback.
- request budget and max token/context degradation.
- multi-GPU routing.

Resource model should track:

```text
device_id
vram_total
vram_free
loaded_models
queue_depth
estimated_throughput
oom_count
health
```

### Phase 9: Full Business Logic

Goal: implement complete education-agent behavior on top of stable runtime
infrastructure.

Business logic should remain mostly outside the low-level runtime until hot paths
are stable:

- persona
- prompt orchestration
- classroom/session flow
- student learning model
- tutoring policy
- multi-agent/persona scheduling
- multimodal interaction loop
- assessment and feedback

Python can continue to own fast-changing orchestration, while stable hot paths
can be moved into C++ as needed.

## 4. Immediate Next Step

Start with SQLite RAII.

Recommended files:

```text
src/storage/sqlite/sqlite_error.h
src/storage/sqlite/sqlite_connection.h
src/storage/sqlite/sqlite_connection.cpp
src/storage/sqlite/sqlite_statement.h
src/storage/sqlite/sqlite_statement.cpp
src/storage/sqlite/sqlite_transaction.h
src/storage/sqlite/sqlite_transaction.cpp
tests/storage/sqlite_storage_test.cpp
```

CMake targets:

```text
agent_storage
storage_tests
```

This creates the storage adapter pattern that later Redis, Faiss, and GStreamer
wrappers should follow.

## 5. Discussion Anchors for Next Session

Open questions:

1. SQLite API shape: minimal statement/transaction API vs higher-level metadata
   store.
2. Whether `agent_storage` should depend only on `agent_core` at first.
3. Exact vector similarity implementation order and supported dimensions.
4. Rust tokenizer C ABI shape and build integration.
5. Redis library choice: Boost.Redis vs hiredis vs custom minimal RESP client.
6. GStreamer appsink probe design and `VideoFrame` ownership model.
7. Artifact manifest schema and Redis key layout.

Recommended next coding task:

```text
Implement SQLite RAII wrapper + storage tests.
```
