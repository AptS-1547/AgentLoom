# Next Runtime Roadmap

> Working roadmap for the next AgentBackendPredict runtime phase.  
> Status: active planning note  
> Date: 2026-05-20

## 1. Current Baseline

The protocol and storage foundations are now considered usable runtime
infrastructure for the next phase.

Already established:

- C++ remains the primary online runtime.
- HTTP/WebSocket is treated as a validated gateway baseline.
  - Public HTTPS/TLS termination should be handled by Nginx for the current
    commercial deployment path.
  - Gateway should remain HTTP/WS internally unless a later deployment target
    explicitly requires embedded TLS.
- SQLite RAII, statement, transaction, connection pool, timed acquire, close
  wakeup, and async executor are implemented and tested.
- The existing in-memory `VectorIndex` is useful as a small hot-bucket cache
  pattern, but it is not the final semantic retrieval backend.
- Faiss dependency discovery is already present in CMake, but no production
  Faiss wrapper is wired into `agent_vector` yet.

The next phase should move from runtime foundations into the text vectorization
and semantic retrieval chain:

```text
text
  -> Hugging Face tokenizer FFI
  -> ONNX text embedding model
  -> pooling / normalization
  -> Faiss vector index
  -> SQLite metadata filter
  -> semantic cache / RAG / memory routing
```

## 2. Runtime Ownership Model

AgentBackendPredict should keep the same language boundary policy:

- C++ owns online hot paths, resource governance, protocol infrastructure,
  storage access, vector search, model wrappers, cache routing, and degradation.
- Rust may be used for narrow native components exposed through a stable C ABI.
  The first concrete use is Hugging Face `tokenizers`.
- Python owns offline dataset processing, curated semantic cache construction,
  model export, evaluation, and fast-changing business orchestration.

The key rule is that external runtime types must not leak into public module
interfaces.

Do not expose these outside their adapters:

```text
sqlite3*
sqlite3_stmt*
faiss::Index*
Ort::Session
llama_model*
llama_context*
Rust tokenizer handles
GstElement*
Redis connection handles
```

## 3. Guiding Principles

1. Public C++ APIs should be stable, typed, and ownership-explicit.
2. Rust/C ABI details must be isolated behind a C++ RAII wrapper.
3. C++ must copy FFI output into owned `std::vector` buffers in the first
   implementation. Zero-copy FFI is not a first-version goal.
4. Batch APIs are required from the beginning for tokenizer, embedding, and
   vector search.
5. Every artifact must be fingerprinted by tokenizer, model, pooling,
   normalization, dimension, corpus, and policy version.
6. Faiss only owns vector search. Metadata, payload, scope, and cache policy
   belong outside Faiss.
7. Vector similarity alone is never sufficient for semantic cache reuse.
   Metadata and scope filters must run after top-k retrieval.
8. Private user/session memory must never leak into global semantic cache hits.
9. Every queue, pool, and async boundary needs bounded capacity, timeout, or
   explicit cancellation behavior.
10. Commercial functionality should be layered over stable runtime primitives,
    not mixed into low-level adapters.

## 4. Phase Plan

### Phase 1: Tokenizer FFI Boundary

Goal: create a stable bridge from C++ to Hugging Face `tokenizers` without
letting Rust or C ABI ownership rules leak into runtime code.

Recommended layout:

```text
third_party/hf_tokenizers_capi/
  Cargo.toml
  Cargo.lock
  include/hf_tokenizers_capi.h
  src/lib.rs

src/vector/
  tokenizer_types.h
  hf_tokenizer.h
  hf_tokenizer.cpp
  hf_tokenizer_internal.h
```

The C ABI should use opaque handles:

```c
typedef struct hf_tokenizer hf_tokenizer_t;
typedef struct hf_tokenized_batch hf_tokenized_batch_t;
```

Required C ABI operations:

```text
create_from_file
destroy tokenizer
encode one text
encode batch
read input_ids
read attention_mask
read token_type_ids
read batch size
read sequence length
destroy tokenized batch
destroy error string
```

Prefer byte-span input over null-terminated strings:

```c
const uint8_t* text
size_t text_len
```

This avoids hidden bugs around embedded NUL bytes and keeps UTF-8 handling
explicit.

C++ public API should look like this conceptually:

```cpp
struct TokenizerOptions {
    std::filesystem::path tokenizer_json;
    std::size_t max_length = 512;
    bool add_special_tokens = true;
    bool padding = true;
    bool truncation = true;
};

struct TokenizedBatch {
    std::vector<std::int64_t> input_ids;
    std::vector<std::int64_t> attention_mask;
    std::vector<std::int64_t> token_type_ids;
    std::size_t batch_size = 0;
    std::size_t sequence_length = 0;
};

class HfTokenizer {
public:
    core::Status Load(const TokenizerOptions& options);
    core::Result<TokenizedBatch> EncodeBatch(std::span<const std::string_view> texts) const;
    core::Result<TokenizedBatch> Encode(std::string_view text) const;
};
```

First-version threading rule:

- Use a conservative mutex around tokenizer calls unless Rust-side concurrency
  is explicitly validated.
- If tokenizer latency becomes a bottleneck, add a `TokenizerPool` later using
  cloned tokenizer instances.

Validation:

- C++ token IDs match Python Hugging Face output for fixed reference inputs.
- Batch padding/truncation shape is deterministic.
- Chinese text, empty text, long text, and invalid UTF-8 behavior are tested.
- Rust panic must not cross FFI.
- All Rust-allocated memory has explicit destroy functions.

### Phase 2: ONNX Text Embedding Runtime

Goal: add a dedicated embedding model wrapper rather than reusing the current
classification-oriented `OnnxBERTModel`.

Recommended layout:

```text
src/vector/
  text_embedding_model.h
  onnx_text_embedding_model.h
  onnx_text_embedding_model.cpp
  embedding_pipeline.h
  embedding_pipeline.cpp
  pooling.h
  normalization.h
```

Conceptual API:

```cpp
enum class PoolingStrategy {
    Cls,
    Mean,
    MeanSqrtLen,
    LastToken,
    ModelOutput
};

struct EmbeddingModelOptions {
    std::filesystem::path model_path;
    std::string execution_provider = "auto";
    std::size_t dimension = 768;
    PoolingStrategy pooling = PoolingStrategy::Mean;
    bool normalize = true;
};

struct EmbeddingBatch {
    std::vector<float> embeddings; // [batch, dim]
    std::size_t batch_size = 0;
    std::size_t dimension = 0;
};
```

Runtime path:

```text
TokenizedBatch
  -> ONNX Runtime session
  -> select output tensor
  -> pooling using attention_mask if needed
  -> L2 normalization
  -> EmbeddingBatch
```

Validation:

- Output dimension matches config.
- Normalized vectors have norm close to `1.0`.
- Reference C++ embeddings match Python-exported embeddings with cosine
  similarity at least `0.999`, when using the same tokenizer/model/pooling.
- Single and batch inference both work.

### Phase 3: Faiss RAII Adapter

Goal: add a replaceable Faiss backend behind a small C++ vector-index interface.

Recommended first backend:

```text
IndexFlatIP
```

Use normalized vectors with inner product to implement cosine-like search.

Recommended layout:

```text
src/vector/
  vector_index.h
  faiss_index.h
  faiss_index.cpp
  exact_vector_index.h
  exact_vector_index.cpp
```

Conceptual API:

```cpp
struct VectorSearchResult {
    std::int64_t id = 0;
    float score = 0.0f;
};

class IVectorIndex {
public:
    virtual ~IVectorIndex() = default;
    virtual core::Status Add(std::span<const float> vectors,
                             std::span<const std::int64_t> ids,
                             std::size_t count) = 0;
    virtual core::Result<std::vector<VectorSearchResult>> Search(
        std::span<const float> query,
        std::size_t top_k) const = 0;
};
```

Rules:

- Do not expose Faiss headers from public project headers unless the type is an
  internal implementation detail.
- Keep an exact scan backend for tests and small buckets.
- Store row-id to metadata mapping externally.
- Support save/load after the in-memory path is correct.
- Do not implement complex ANN variants until correctness and metadata filtering
  are stable.

Validation:

- Add/search self-retrieval test.
- Dimension mismatch returns `InvalidArgument`.
- Empty index returns empty results.
- Save/load preserves search behavior.
- Exact index and Faiss flat index agree on reference vectors.

### Phase 4: SQLite Vector Metadata Store

Goal: separate vector similarity from semantic safety policy.

Faiss returns candidate IDs. SQLite decides whether those candidates are legal
to reuse.

Initial tables:

```sql
vector_collections(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  embedding_model_fingerprint TEXT NOT NULL,
  tokenizer_fingerprint TEXT NOT NULL,
  pooling_strategy TEXT NOT NULL,
  dimension INTEGER NOT NULL,
  normalization TEXT NOT NULL,
  corpus_version TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL
);

vector_entries(
  id INTEGER PRIMARY KEY,
  collection_id INTEGER NOT NULL,
  tenant_id TEXT,
  scope TEXT NOT NULL,
  cache_key TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  subject TEXT,
  grade TEXT,
  topic TEXT,
  persona_scope TEXT,
  memory_scope TEXT,
  answer_type TEXT,
  quality_score REAL NOT NULL DEFAULT 0.0,
  expires_at_ms INTEGER,
  created_at_ms INTEGER NOT NULL
);
```

The exact schema can evolve, but the design rule should remain:

```text
Vector index:
  nearest-neighbor candidates only

SQLite metadata:
  scope, tenant, policy, corpus, quality, expiry, and cache safety

Payload store:
  actual response / RAG card / memory content
```

### Phase 5: Semantic Cache Pipeline

Goal: turn tokenizer + embedding + Faiss + metadata into a safe online semantic
cache lookup path.

Query path:

```text
Request text
  -> route / intent / context-risk precheck
  -> tokenizer
  -> embedding
  -> Faiss top-k
  -> SQLite metadata filter
  -> quality / expiry / policy check
  -> payload fetch
  -> response candidate
```

Hard cache safety rules:

- Do not use global semantic cache for strong context-dependent requests.
- Strong context markers include references like this/that/above/previous step,
  image-local references, and recent-turn dependencies.
- Tenant, subject, grade, topic, persona, memory scope, answer type, embedding
  model, tokenizer, corpus, and policy must be compatible.
- Private user/session memory can only hit user/session scoped entries.
- RAG authority answers must retain source/citation metadata.

Initial implementation should prefer correctness over hit rate.

### Phase 6: Artifact Importer

Goal: load offline-built semantic cache artifacts into the runtime store.

Offline Python can build:

```text
artifact/
  manifest.json
  vectors.f32
  entries.jsonl
  payload.jsonl
  optional faiss.index
```

C++ importer should validate:

- dimension
- vector count
- entry count
- model fingerprint
- tokenizer fingerprint
- corpus version
- policy version
- checksums

The importer should populate:

```text
Faiss index
SQLite metadata
payload store
```

Payload can start in SQLite or filesystem. Redis can be introduced later as a
shared hot payload layer.

### Phase 7: Redis Shared Cache Layer

Goal: use Redis as shared payload/cache coordination, not as the primary vector
compute engine.

Redis should be introduced after the local semantic cache path is correct.

Recommended layout:

```text
semantic:<version>:manifest
semantic:<version>:payload:<entry_id>
semantic:<version>:meta:<entry_id>
semantic:<version>:bucket:<bucket_name>
semantic:<version>:invalidate
```

Hot path:

```text
local Faiss / exact index
  -> top-k ids
  -> SQLite/local metadata filter
  -> Redis/local payload fetch
  -> response
```

Redis must not become the only source of truth.

### Phase 8: Memory and RAG Integration

Goal: reuse the vectorization chain for memory and RAG after semantic cache
lookup is stable.

Memory query path:

```text
MemoryQuery
  -> scope validation
  -> tokenizer / embedding
  -> vector search
  -> metadata filter
  -> recency / quality scoring
  -> payload load
  -> context pack
```

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

Critical rule:

```text
global semantic cache
  != user memory
  != session memory
  != RAG authority source
```

They can share vector infrastructure, but they must not share unsafe reuse
policy.

### Phase 9: Commercial Runtime Layers

Goal: build commercial behavior over the stable runtime primitives.

Priority order:

```text
storage repositories
  -> tenant / user / auth context
  -> session / conversation persistence
  -> usage accounting
  -> metrics / tracing
  -> semantic cache
  -> RAG / memory
  -> persona / orchestration
  -> billing / admin API
```

The gateway should remain a resource-governed orchestration layer. Fast-changing
education policy, prompt templates, and business workflow should stay
configuration-driven or service-isolated until they are stable enough to move
into C++.

## 5. Immediate Next Step

Start with the Hugging Face tokenizer FFI boundary.

Recommended next discussion/coding target:

```text
Design and implement the Rust C ABI + C++ RAII HfTokenizer wrapper.
```

First concrete deliverables:

```text
third_party/hf_tokenizers_capi/include/hf_tokenizers_capi.h
third_party/hf_tokenizers_capi/src/lib.rs
src/vector/tokenizer_types.h
src/vector/hf_tokenizer.h
src/vector/hf_tokenizer.cpp
tests/vector/tokenizer_test.cpp
```

First tests:

- Load tokenizer from `tokenizer.json`.
- Encode one UTF-8 Chinese sentence.
- Encode batch with deterministic padding/truncation.
- Compare token IDs with Python Hugging Face tokenizer fixture.
- Verify error path for missing tokenizer file.
- Verify repeated load/destroy does not crash or leak obvious ownership.

Do not start with Faiss. Faiss is lower risk and should be connected after the
tokenizer output shape and model fingerprint policy are stable.

## 6. Discussion Anchors for Next Session

Open questions:

1. Whether the Rust tokenizer library should be built as a static library or DLL
   on Windows for the first implementation.
2. Exact C ABI shape: null-terminated strings vs byte pointer + length. Current
   recommendation is byte pointer + length.
3. Tokenizer threading model: shared mutex first vs tokenizer pool from day one.
4. Where to store tokenizer/model fixtures for C++ tests.
5. Which embedding model should be the first supported ONNX text embedding
   model.
6. Pooling strategy for the first embedding model: model output vs mean pooling.
7. Whether vector metadata lives first in SQLite only or also writes a filesystem
   manifest.
8. Whether `agent_vector` should absorb tokenizer/embedding initially or split
   into `agent_tokenizer`, `agent_embedding`, and `agent_vector`.

Recommended next coding task:

```text
Tokenizer FFI minimal vertical slice:
  tokenizer.json -> Rust tokenizers -> C ABI -> C++ HfTokenizer -> GTest fixture
```
