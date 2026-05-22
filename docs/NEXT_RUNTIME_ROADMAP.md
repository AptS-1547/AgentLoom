# Next Runtime Roadmap

> Working roadmap for the next AgentBackendPredict runtime phase.  
> Status: active planning note  
> Date: 2026-05-22

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

### Phase 1: Tokenizer FFI Boundary — ✅ Completed (2026-05-20)

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

### Phase 2: ONNX Text Embedding Runtime — ✅ Completed (2026-05-20)

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

### Phase 3: Faiss RAII Adapter — ✅ Completed (2026-05-20)

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

### Phase 4: SQLite Vector Metadata Store + Index Manager — ✅ Completed (2026-05-22)

Goal: separate vector similarity from semantic safety policy, with SQLite as
the single source of truth and Faiss as a lazily-hydrated cache.

Resolved design decisions:

- **Fine-grained partitioning**: `memory_level` lives in the partition key
  alongside `(collection, tenant, user)`. Working memory and knowledge base
  are physically isolated, eliminating cross-level query pollution.
- **SQLite as single source of truth**: vectors live as BLOBs in
  `vector_entries` alongside metadata, payload, and lifecycle fields. Faiss
  indices are rebuilt from SQLite on first touch per partition, never
  persisted independently.
- **Lazy load + LRU**: `VectorIndexManager` hydrates partitions on first
  Search, caps resident partitions via LRU, and rebuilds on staleness
  (compares `partition.last_modified_at_ms` against the hydrated snapshot).
- **Wider scope than originally planned**: schema + repository + fingerprint
  policy + partition registry + index manager all land in Phase 4 to avoid
  Phase 5 having to revisit schema decisions.
- **Additional scope**: a layered C++ outbound HTTP/HTTPS stack
  (`agent_tls` → `agent_http_client`) and an OpenAI-compatible LLM client
  (`agent_llm`) are folded into Phase 4 so the new memory system can run
  maintenance LLM calls (state extraction, fact distillation, profile
  compaction) in-process without crossing back into Python.

Realized schema (see `src/storage/vector/schema.sql`):

```sql
vector_collections   -- fingerprint policy per logical group
vector_partitions    -- (collection_id, tenant_id, user_id, memory_level) unique
vector_entries       -- entry_id doubles as Faiss IDMap2 id;
                     -- vector BLOB + lifecycle + payload + extra_metadata JSON
```

Realized library layering:

```text
agent_vector_storage    (sqlite repository, partition registry, fingerprints)
agent_vector            (+ index manager, hydrating Faiss/Exact backends)
agent_tls               (reusable client-side TLS context)
agent_http_client       (Beast-based http+https with async deadline)
agent_llm               (pending — pure business logic, OpenAI-compatible)
```

Step status:

```text
Step 1  SQLite schema + EnsureSchema                  ✅
Step 2  IVectorRepository CRUD                        ✅
Step 3  VectorFingerprint policy                      ✅
Step 4  PartitionRegistry                             ✅
Step 5  VectorIndexManager (LRU + hydrate)            ✅
Step 6a TLS context abstraction (agent_tls)           ✅
Step 6b HTTP client (agent_http_client, http+https)   ✅
Step 7  LLM client (OpenAI-compatible)                ✅ (2026-05-22)
Step 8  Config sections + server entry integration    ✅ (2026-05-22)
Step 9  End-to-end integration test                   ✅ (2026-05-22)
```

Tests landed so far: 33 storage/index manager + 33 net layer + 31 LLM = 97 passing.

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

Phase 4 complete (2026-05-22). All 9 steps merged and tested: schema,
repository, fingerprint policy, partition registry, index manager, TLS
context, outbound HTTP client, OpenAI-compatible LLM client, config
integration, and end-to-end tests (mock + real DeepSeek API).

Phase 5 skeleton landed (2026-05-22): `agent_semantic_cache` library with
`ISemanticCache`, `IContextRiskDetector`, `IPolicyMatcher`, and
`SemanticCachePipeline` stub.  Implementation pending.

Next coding task: **Phase 5 — Semantic Cache Pipeline implementation**,
starting with `IContextRiskDetector` (keyword-based v1) and
`IPolicyMatcher` (fail-closed fingerprint + scope + metadata checks).
Once those are solid, wire them into `SemanticCachePipeline::Lookup`
with the existing tokenizer → embedding → index manager chain.

## 6. Discussion Anchors for Next Session

Open questions for Phase 4 Steps 7-9 and forward:

1. **LLM provider scope for v1** — confirmed OpenAI-compatible only
   (DeepSeek / vLLM / proxies all fit). Anthropic schema deferred until a
   real C++-side need appears.
2. **API key resolution** — env var primary (`api_key_env`), file fallback
   (`api_key_file`), no plaintext in config. Aligned with the existing
   `auth.token_env / auth.token_file` pattern.
3. **Maintenance prompts location** — hard-coded as `inline constexpr
   std::string_view` in `src/llm/llm_prompts.h`. Versioned with the binary;
   if Python wants to A/B iterate, they can still rev their own copy and
   sync once finalized.
4. **Streaming in v1** — out of scope. Maintenance prompts emit short JSON;
   SSE parsing is non-trivial and not needed until the conversation hot
   path lands.
5. **Memory lifecycle engine port** — Phase 4 only ensures the schema can
   hold every Python `MemoryLifecycleStore` field. The actual retention
   weight / bucket-by-weight pruning port to C++ remains a Phase 4.5 or
   Phase 5 question.
6. **Where the maintenance worker thread lives** — not in Phase 4.
   Belongs to the Phase 5 conversation hot path so it can hook into
   per-turn / per-token / close-session triggers.
7. **payload storage location** — currently inline in `vector_entries.payload`.
   Move to a separate `vector_payloads` table only if a future scan pattern
   genuinely wants vector-without-payload rows.

Phase 1-3 anchors (Rust build mode, FFI byte-pointer signatures, tokenizer
threading model, fixture layout, embedding model choice, pooling strategy,
filesystem manifest format) have all been answered by the merged
implementation and are no longer open.
