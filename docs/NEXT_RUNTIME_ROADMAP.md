# Next Runtime Roadmap

> Working roadmap for the next AgentBackendPredict runtime phase.  
> Status: active planning note  
> Date: 2026-05-24

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

### Phase 5: Semantic Cache Pipeline — ✅ Completed (2026-05-27)

Goal: turn tokenizer + embedding + Faiss + metadata into a safe online semantic
cache lookup path.

**Status**: L0 vector index infrastructure complete. L3 long-term memory compressor
refactored to use unified vector storage infrastructure with semantic search support.

#### Completed Components (2026-05-27)

**L0 Vector Index Manager** — ✅ (2026-05-24)
- Per-record write model: every `AddRecord` immediately persists to Redis
- Active batch tracking: SQLite-backed `active_batch` table stores current
  batch timestamp and count
- Batch promotion: when `active_count >= MAX_CACHE_RECORDS`, promote to
  `timestamp_index` and start fresh batch
- Intelligent search strategy:
  - `< 2 × MAX_CACHE_RECORDS`: full-scan (load active + all historical batches)
  - `≥ 2 × MAX_CACHE_RECORDS`: active + random historical batch (uniform sampling)
- Segmented neighbor extraction in `SearchWithContext`: active segment and
  historical segment boundaries respected to preserve temporal coherence
- Memory footprint: ~0 resident, ~3MB transient peak per request
- Concurrency model: per-request `VectorIndexManager` instances, no shared state
- DDoS resilience: bounded by request count × 3MB, released on completion

**Redis Connection Pool** — ✅ (2026-05-24)
- 4 `io_context` workers, round-robin dispatch
- Async exec with `std::promise` synchronization
- Timeout support via `future.wait_for`

**Binary Serialization** — ✅ (2026-05-24)
- `SerializeCacheRecord` / `DeserializeCacheRecord`
- Dimension validation, boundary checks
- ~1.5KB per 384-dim record

**SQLite Schema** — ✅ (2026-05-24)
- `active_batch(user_uuid, timestamp, count)` — per-user active batch metadata
- `cache_timestamp_index(user_uuid, timestamp)` — completed batch registry

**L3 Long-Term Memory Compressor** — ✅ (2026-05-27)
- Refactored from standalone SQLite table writes to unified vector infrastructure
- Architecture: `IVectorRepository` + `PartitionRegistry` + `VectorIndexManager` + `EmbeddingPipeline`
- Storage model: facts stored as `EntryRecord` with `memory_level="L3"`, `memory_type="fact"`
- Partition key: `(collection_id, user_id, memory_level="L3")`
- Metadata: `extra_metadata_json` stores `{"date": "YYYY-MM-DD", "source_record_count": N}`
- Operations:
  - `CompressDailyMemory`: Redis L0 fetch → LLM fact extraction → Embedding → Vector storage
  - `GetDailySummary`: ListEntries + date filter → reconstruct bullet-point summary
  - `GetUserSummaries`: ListEntries + group by date (DESC order) → return recent summaries
  - `SearchFacts`: Embed query → VectorIndexManager.Search → semantic retrieval
- Testing modes:
  - Production: requires `llm_client`, `embedding_pipeline`, `index_manager`
  - Testing: allows nullptr for LLM/embedding, uses zero-vectors as placeholder
- Tests: 4/4 unit tests pass (Testing mode), E2E test passes (Production mode with real LLM + Embedding)
- E2E validation: 3 L0 records → LLM compression (1438ms) → 3 facts stored → semantic search successful

**Critical Redis/Boost.redis Findings** — ✅ (2026-05-26)
- **Windows Redis port issue**: Native Windows Redis (port 6379) blocks on `LRANGE` 
  with large binary payloads (~1.6KB serialized `CacheRecord`). Bare connection test 
  passes, but E2E test with actual data hangs indefinitely.
- **WSL Redis workaround**: Switching to WSL Redis (port 5000) resolves the issue 
  completely. Same payload size, same Boost.redis code, no blocking.
- **Root cause hypothesis**: Windows Redis port has issues with binary data handling 
  or buffer management that WSL build does not exhibit.
- **Boost.redis KEYS command bug**: `KEYS` pattern matching aborts connection in 
  Boost.redis 1.85. Workaround: use `SCAN` with `generic_response` instead.
- **Health check interference**: `cfg.health_check_interval = std::chrono::seconds::zero()` 
  required to disable library auto-PING that can interfere with custom command sequences.
- **Production recommendation**: Deploy with Linux Redis (native or WSL), avoid 
  Windows Redis port for binary payload workloads.

**Design Rationale**:
The initial design assumed long-lived `VectorIndexManager` instances with
in-memory write buffers. After recognizing the per-request lifecycle pattern
(HTTP handler creates instance → search/add → destructor on response complete),
the architecture was refactored to:
1. Eliminate memory write buffers (data goes straight to Redis)
2. Use SQLite as lightweight coordination state (active batch metadata)
3. Adapt search strategy to total data volume (full-scan for small datasets,
   rotation for large)

This yields near-zero resident memory, immediate data durability, and natural
horizontal scalability (any process can hydrate from Redis + SQLite).

#### Pending Components

**Semantic Cache Pipeline** — 🔲
Query path:

```text
Request text
  -> route / intent / context-risk precheck
  -> tokenizer
  -> embedding
  -> VectorIndexManager.Search (L0)
  -> metadata filter
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

**IContextRiskDetector** — 🔲
Keyword-based v1: detect strong context markers (this/that/above/previous/etc.)
that disqualify global cache reuse.

**IPolicyMatcher** — 🔲
Fail-closed fingerprint + scope + metadata checks. Verify embedding model,
tokenizer version, corpus version, tenant/user/session scope compatibility.

**SemanticCachePipeline::Lookup** — 🔲
Wire precheck → tokenize → embed → L0 search → policy filter → payload load.

**SemanticCachePipeline::Store** — 🔲
Persist new cache entries after LLM generation.

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

Phase 5 complete (2026-05-27). L0 vector index infrastructure and L3 long-term 
memory compressor both operational with unified vector storage.

**Completed in Phase 5**:
- `agent_semantic_cache` library skeleton (`ISemanticCache`,
  `IContextRiskDetector`, `IPolicyMatcher`, `SemanticCachePipeline` stub)
- `RedisConnectionPool` with 4-slot async dispatch
- Binary `CacheRecord` serialization (~1.5KB per 384-dim record)
- `VectorIndexManager` with per-record Redis writes, intelligent search
  strategy (full-scan vs. rotation), and segmented neighbor extraction
- SQLite-backed `active_batch` and `cache_timestamp_index` tables
- AVX2 FMA-optimized 384-dim dot product (4-way unrolled)
- `LongTermMemoryCompressor` refactored to unified vector infrastructure
- Windows Redis port issue identified and documented (WSL Redis recommended)
- Crash dump infrastructure for Windows debugging (`crash_dump.h`)

**Next phase priorities** (Phase 6-9):
1. **Semantic Cache Pipeline completion** — wire `IContextRiskDetector`, 
   `IPolicyMatcher`, and full `Lookup`/`Store` operations
2. **Artifact Importer** — load offline-built semantic cache artifacts
3. **Redis Shared Cache Layer** — shared payload coordination
4. **Memory and RAG Integration** — reuse vectorization chain for memory/RAG
5. **Commercial Runtime Layers** — tenant/user/auth, session persistence, 
   usage accounting, metrics/tracing

### Phase 10: Media Processing Infrastructure (Planned)

Goal: create a reusable media processing library (`agent_media`) that encapsulates 
GStreamer + OpenCV multimodal pipeline, providing a clean C++ interface for frame 
reception, detection, and keyframe selection.

**Status**: Placeholder. Directory `src/media/` reserved for future implementation.

**Scope**: This is a **library module**, not business logic. It provides the toolkit 
for stream processing; how to use these tools (VLM trigger timing, event injection 
strategy, confidence thresholds) belongs in `service/` layer.

#### Planned Components

**Frame Reception & Decoding**:
- WebSocket frame receiver (reuses `agent_net` Beast infrastructure)
- GStreamer pipeline wrapper with hardware-accelerated decode (NVDEC/VAAPI)
- Fallback to OpenCV software decode for compatibility
- Bounded frame queue with backpressure (drop old frames when full)

**Frame Detection & Analysis**:
- MOG2 adaptive background model wrapper
- Three-signal fusion (foreground area + histogram distance + edge density)
- FFT frequency-domain analysis for adaptive sampling window
- Morphological operations (erosion/dilation) and connected component filtering

**Keyframe Selection**:
- Peak detection with exponential moving average (EMA) smoothing
- Laplacian variance-based sharpness scoring
- Cooldown interval to prevent duplicate submissions
- Batch accumulation for VLM inference optimization

**Proposed Directory Structure**:
```text
src/media/
  media_types.h              // Frame, KeyFrame, DetectionResult
  gstreamer_pipeline.h/cpp   // GStreamer wrapper (decode, format conversion)
  opencv_processor.h/cpp     // MOG2, morphology, edge detection
  frame_detector.h/cpp       // Three-signal fusion + FFT analysis
  keyframe_selector.h/cpp    // Peak detection + sharpness scoring
  frame_queue.h/cpp          // Bounded queue with backpressure
  media_processor.h/cpp      // High-level facade
```

**Proposed Public API** (C++ headers, no C ABI needed):
```cpp
namespace agent::media {

struct MediaProcessorOptions {
    std::size_t frame_queue_capacity = 10;
    std::size_t keyframe_batch_size = 4;
    float detection_threshold = 0.15f;
    std::chrono::milliseconds cooldown_interval{500};
    bool use_hardware_decode = true;
};

struct KeyFrameBatch {
    std::vector<cv::Mat> frames;
    std::vector<std::chrono::steady_clock::time_point> timestamps;
    std::vector<float> confidence_scores;
};

using KeyFrameCallback = std::function<void(KeyFrameBatch)>;

class MediaProcessor {
public:
    core::Status Initialize(const MediaProcessorOptions& options);
    
    // Submit raw frame (async, non-blocking)
    core::Status SubmitFrame(std::span<const uint8_t> frame_data,
                            std::chrono::steady_clock::time_point timestamp);
    
    // Register callback for keyframe batches
    void SetKeyFrameCallback(KeyFrameCallback callback);
    
    // Graceful shutdown
    void Stop();
};

} // namespace agent::media
```

**Integration with Three-Pool Architecture**:
- Frame reception runs in **Net Pool** (WebSocket protocol layer)
- Frame detection (MOG2/FFT/OpenCV) runs in **Compute Pool** (CPU-bound)
- VLM inference triggered by callback runs in **IO Pool** (gRPC async)

**Dependencies**:
- GStreamer 1.24.x (optional, for hardware decode)
- OpenCV 4.10.0 (required, for MOG2/morphology/edge detection)
- Boost.Lockfree (for bounded frame queue)
- `agent_net` (for WebSocket frame reception)
- `agent_core` (for Status/Result, thread pool integration)

**Design Principles**:
1. **Library, not service**: Provides frame processing primitives, not business logic
2. **No VLM coupling**: Keyframe callback is opaque; caller decides what to do with frames
3. **Hardware-agnostic**: Graceful fallback from GPU decode to CPU decode
4. **Backpressure-aware**: Bounded queues prevent memory explosion under load
5. **Testable**: Mock frame sources for unit tests, no real camera required

**Validation Criteria**:
- Frame queue respects capacity limit (drops old frames when full)
- MOG2 background model converges on static scenes
- FFT adaptive sampling adjusts window size based on scene dynamics
- Peak detection identifies keyframes with >90% precision on reference videos
- Hardware decode path works on NVIDIA/Intel platforms
- Software decode fallback works on CPU-only systems
- Keyframe batch callback fires with correct batch size

**Future Enhancements** (post-MVP):
- Multi-stream support (multiple cameras/screen shares per session)
- Recording/transcoding pipeline (save keyframes to disk)
- Real-time quality metrics (frame drop rate, detection latency)
- Adaptive quality scaling (reduce resolution under CPU pressure)

**References**:
- Detailed algorithm specifications: `docs/STREAMING_ARCHITECTURE.md`
- Integration with business logic: `dev_note/FUTURE_PLAN.md` (Section 11-12)

## 6. Discussion Anchors for Next Session

Resolved during Phase 5 (2026-05-24 to 2026-05-27):

1. **VectorIndexManager lifecycle** — per-request, not long-lived singleton.
   Each HTTP handler creates a fresh instance, hydrates state from SQLite +
   Redis, runs search/add operations, then destructs on response complete.
2. **Memory model** — near-zero resident, ~3MB transient peak per request.
   Data persists in Redis (per-record) and SQLite (batch metadata), not
   in-process.
3. **Batch rotation strategy** — uniform random sampling from completed
   batches, drain mode (one batch per Search). Active batch always loaded
   in full to keep fresh writes immediately searchable.
4. **Search strategy threshold** — `2 × MAX_CACHE_RECORDS`. Below: full-scan
   all batches. Above: active + 1 random historical batch.
5. **Neighbor extraction segmentation** — `SearchWithContext` respects
   active/historical segment boundaries to preserve temporal coherence in
   reconstructed LLM context.
6. **DDoS resilience** — bounded by per-user concurrent request count.
   Single-user 1000 QPS is unrealistic outside attack scenarios; rate
   limiting at HTTP gateway is the proper defense layer.
7. **Random vs. LRU batch rotation** — random chosen to match L0's
   "uniform approximation of full history" assumption when queries are
   uncorrelated with write recency (cross-session topic switches).
8. **L3 storage architecture** — unified vector infrastructure (`IVectorRepository` 
   + `PartitionRegistry` + `VectorIndexManager`) replaces standalone SQLite 
   table writes. Enables semantic search over compressed facts.
9. **Windows Redis compatibility** — Windows Redis port has binary payload 
   handling issues. Production deployments should use Linux Redis (native or WSL).
10. **Crash dump infrastructure** — `crash_dump.h` provides comprehensive 
    Windows minidump capture (vectored exception handler, SEH, signals, CRT 
    invalid parameter, std::terminate). Essential for debugging production crashes.

Open questions for Phase 6+ continuation:

1. **Context-risk detection scope** — keyword-based v1 only, or include
   simple syntactic heuristics (pronoun density, demonstrative count)?
2. **Policy matcher fingerprint format** — reuse `agent_vector_storage`
   `VectorFingerprint` schema directly, or introduce a slimmer cache-only
   variant?
3. **Payload storage location for cache hits** — inline in vector entries
   (current Phase 4 schema) or separate Redis-backed payload layer?
4. **Multi-user shared L0** — current design is per-user. If a future
   workload shows extreme write skew (one heavy user vs. many light
   users), consider a global LRU-managed L0 pool. Defer until metrics
   indicate need.
5. **Active batch race conditions** — multiple concurrent `AddRecord`
   calls crossing the `MAX_CACHE_RECORDS` threshold simultaneously may
   create duplicate `PromoteBatchToIndex` attempts. SQLite PRIMARY KEY
   handles the race, but slight count inflation is possible. Acceptable
   for now; revisit if metrics show problematic batch sizes.
6. **L3 compression scheduling** — currently manual invocation via 
   `CompressDailyMemory`. Consider adding cron-like scheduler for 
   automatic daily compression of L0 → L3 facts.
7. **L3 fact deduplication** — LLM may extract similar facts across 
   different days. Consider semantic deduplication before storage.

Phase 1-4 anchors have all been answered by merged implementations and
are no longer open.

## 7. Document Analysis Runtime Roadmap

> Status: active implementation track  
> Scope: C++ Gateway document analysis path, Office OOXML extraction, chunking,
> mindmap/diagnosis generation, LLM fallback, embedding clustering, result
> persistence, cache integration, and E2E quality evaluation.

### 7.1 Current Baseline

The document analysis path is now usable for real DOCX inputs and is wired into
the Gateway HTTP route:

```text
POST /api/document/analyze
  -> PersonaGatewayHttpAdapter
  -> DocumentAnalysisService
  -> compute pool: OOXML extraction + chunk/mindmap/diagnosis pipeline
  -> IO pool: optional SQLite result persistence
  -> response JSON + E2E quality report
```

Completed baseline:

- `agent_document` owns the C++ document analysis runtime.
- DOCX/PPTX OOXML extraction is implemented through `libzip` + `pugixml`.
- File/path/text/report handling uses UTF-8 explicitly.
- `DocumentAnalysisService` follows the existing Gateway three-pool model:
  - main/request pool handles HTTP routing and callback orchestration.
  - compute pool handles extraction, chunking, embedding, and local graph work.
  - IO pool handles SQLite result write-back.
- `/api/document/analyze` is connected through the Gateway HTTP adapter.
- Result schema includes blocks, chunks, mindmap, diagnosis, run nodes, warnings,
  token estimate, and schema version.
- SQLite result repository stores `document_analysis_results` asynchronously.
- Real 384-dim embedding path is supported through the existing ONNX embedding
  pipeline.
- Embedding similarity uses the existing L0 SIMD kernel
  `dot_product_unrolled<384>`.
- The local 128-dim hash embedding path has been removed.
- LLM chunk fallback is implemented for weak natural-language chunks that lack
  explicit title signals.
- Explicit Office headings, numbered headings, and title-like local structure
  are not overridden by LLM output.
- Redis-backed exact LLM chunk cache is implemented:
  - key: `prompt_version:model:text_hash`
  - role: deterministic repeat-analysis cache and same-text error fallback
  - TTL: default 7 days in E2E config
- E2E tool `document_analysis_e2e_test` runs the real Gateway HTTP route with
  real LLM, real embedding, Redis cache, and quality report output.
- E2E quality reports compare source text against generated summaries and
  indexed chunk text using ASCII term recall and UTF-8 non-ASCII unit recall.
- Embedding warmup is performed before E2E request timing so cold ONNX/CUDA
  first-inference cost is not counted as document request latency.
- Mindmap separator-only nodes such as `---`, `===`, `***`, and `___` are
  filtered.

Recent real-document E2E observations:

```text
大创文本降AI率版本.docx
  chunks=15
  blocks=20
  llm_chunks=7
  Redis exact cache hits=7/7 after warm cache
  request latency after embedding warmup: ~100-120 ms
  indexedTextAsciiTermRecall=1
  indexedTextUtf8NonAsciiUnitRecall=1

项目特色与创新.docx
  chunks=9
  blocks=13
  llm_chunks=4
  Redis exact cache hits=4/4 after warm cache
  request latency after embedding warmup: ~50-70 ms
  indexedTextAsciiTermRecall=1
  indexedTextUtf8NonAsciiUnitRecall=1
```

Important interpretation:

- The exact Redis LLM chunk cache is useful, but it only handles byte-identical
  natural-language chunks.
- Natural-language reuse across documents needs semantic cache integration,
  because exact text overlap probability is low.
- The current exact cache should remain as a low-risk fallback layer and
  deterministic E2E stabilizer.
- Document `diagnosis` remains aligned with the education intelligent-agent
  product domain even when ad-hoc test documents are project proposals.

### 7.2 Design Rules

Document analysis should continue to follow the existing runtime rules:

1. Keep C++ as the Gateway hot-path runtime.
2. Keep Python as the algorithm prototype and offline evaluation source.
3. Avoid platform-only APIs; do not rely on Win32 text conversion in runtime or
   CI tests.
4. Treat all file reads, test inputs, report outputs, and path/text display as
   UTF-8.
5. Avoid raw pointer ownership and raw pointer dependency flow. Use RAII,
   `std::shared_ptr`, `std::unique_ptr`, `std::optional`, and typed result
   objects.
6. Keep file/module granularity cohesive. Do not split document analysis into
   many tiny files unless a boundary is genuinely reusable.
7. Use existing Gateway pools and async callback strategy instead of adding a
   separate synchronization model.
8. Use existing L0/vector/semantic-cache infrastructure where possible.
9. LLM output supplements weak natural-language chunks; it must not replace
   original-text indexing or explicit Office structure.
10. Redis failures in optional cache layers must be non-fatal.

### 7.3 Planned Request and File Lifecycle

The final user-facing design should split upload from analysis:

```text
WebSocket upload
  -> Gateway receives file stream
  -> file stored in managed document cache directory
  -> metadata persisted in SQLite
  -> document_id generated from content hash / UUID

POST /api/document/analyze
  -> request carries document_id or explicit path in test/dev mode
  -> Gateway resolves metadata
  -> DocumentAnalysisService reads file
  -> analysis result persisted by document_id
```

Planned metadata model:

```text
document_id          stable hash / UUID
content_hash         content digest for deduplication
owner_user_uuid      optional user scope
session_id           optional session scope
file_name            original UTF-8 display name
file_type            docx / pptx / later xlsx/pdf/image
storage_path         internal resolved path
uploaded_at_ms
last_analyzed_at_ms
last_accessed_at_ms
analysis_status      uploaded / analyzing / completed / failed
analysis_trace_id
size_bytes
schema_version
```

Retention policy:

- Maintain recent uploaded files through a timestamp/LRU queue.
- Persist lifecycle metadata in SQLite.
- Evict physical files whose latest analysis/access timestamp is older than
  one week by default.
- Expose recent documents so the frontend can analyze recently uploaded files
  and re-open historical parse results while the file remains retained.
- Deletion must verify resolved paths stay inside the managed document cache
  directory before physical removal.

### 7.4 Office Format Expansion

DOCX is the current strongest path. PPTX should be the next real-file test
target because slide boundaries are natural scope boundaries.

PPTX next work:

- Run real PPTX files through `document_analysis_e2e_test`.
- Validate slide ordering against `ppt/slides/slideN.xml`.
- Validate title placeholder extraction.
- Validate body text box ordering within each slide.
- Preserve `slide` on blocks, chunks, slices, and mindmap nodes.
- Use slide equality as a strong parent-selection signal for chunk attachment.
- Treat bullet level and placeholder type as heading/title features.
- Ensure separator-only shapes and decorative text do not become nodes.
- Add quality report fields grouped by slide:
  - blocks per slide
  - chunks per slide
  - title coverage per slide
  - weak chunk count per slide

DOCX next work:

- Improve handling of tables:
  - preserve row/column text order
  - mark table-derived blocks in metadata
  - avoid merging unrelated table cells into one weak paragraph
- Improve handling of headers/footers:
  - default skip or mark as low-priority metadata
  - avoid repeated header/footer text polluting chunks
- Improve list/numbering extraction:
  - parse numbering definitions from `word/numbering.xml`
  - map abstract numbering to levels when present
  - preserve list scope in metadata
- Improve style heuristics:
  - stronger Chinese heading style detection
  - avoid treating short bold body fragments as top-level headings when they are
    table labels or inline emphasis

Future Office targets:

- XLSX: useful for structured education data, but should be separate from the
  DOCX/PPTX narrative parser. Sheet/table semantics are different enough to
  require a dedicated extraction policy.
- PDF: defer until Office path is stable. PDF needs layout/OCR handling and has
  a much higher false-structure risk.

### 7.5 Chunking and Semantic Structure Plan

Current chunking uses:

```text
Office explicit heading / inferred title
  -> deterministic local chunk

weak paragraph group
  -> embedding similarity clustering
  -> exact LLM chunk cache lookup
  -> LLM title/summary/slice fallback when cache misses
```

Planned improvements:

- Add stronger title/body grouping around numbered short titles and weak body
  paragraphs.
- Avoid fixed byte/character title splitting. Title extraction should stop at
  real body-feature boundaries, not arbitrary length thresholds.
- Keep title truncation only for display labels such as mindmap leaves.
- Continue preserving full indexed chunk text for retrieval and quality checks.
- Add chunk-level reason metadata:
  - `group_reason`
  - `title_source`
  - `fallback_reason`
  - `embedding_attached`
  - `explicit_title`
- Add per-group diagnostics for E2E:
  - source block ids
  - source block kinds
  - title candidate
  - whether LLM fallback was attempted
  - whether exact cache or semantic cache was used
- Introduce batch embedding for chunk grouping if per-document block count
  becomes large.
- Keep deterministic parser as the source of original-text indexing. LLM can
  produce labels and summaries, but should not remove source evidence.

### 7.6 Document Semantic Cache Integration

Exact LLM chunk cache should remain, but the main cross-document reuse path
should use the existing semantic cache infrastructure.

Recommended lookup order:

```text
explicit heading / numbered local structure
  -> deterministic local chunk

weak natural-language chunk
  -> exact Redis LLM chunk cache
  -> document semantic cache lookup
  -> real LLM fallback
  -> store exact cache
  -> store semantic cache
```

Exact cache role:

- Same document / same text repeat analysis.
- Deterministic E2E replay.
- Same-text LLM failure fallback.
- Low-risk cache layer with no semantic false-positive risk.

Semantic cache role:

- Similar chunk reuse across documents.
- OCR/layout/paragraph-break tolerant reuse.
- Natural-language title/summary/slice reuse when text differs but meaning is
  close.

Recommended semantic-cache namespace:

```text
collection: document_chunk_title
memory_level / scope: document_runtime_cache
payload: title, summary, slices, source evidence, model, prompt_version
embedding: normalized 384-dim text embedding
similarity: high threshold, initially 0.92-0.96
```

Semantic cache safety checks:

- `prompt_version` must be equal or explicitly compatible.
- embedding model, tokenizer, pooling, normalization, and dimension must match.
- document type should be compatible unless the cached item is format-agnostic.
- source text keyword/UTF-8 unit recall must exceed a minimum threshold.
- cached slice evidence should be present in or close to the current chunk text.
- confidence should be downgraded when evidence alignment is weak.
- semantic hits must not override explicit Office headings.

Implementation choices:

- Prefer adapting existing `ISemanticCache` / L0 storage over creating an
  independent document vector-cache implementation.
- Keep document-specific payload encoding in a thin adapter.
- Reuse Redis connection pool and vector infrastructure already used by Gateway
  E2E.
- Keep all cache failures non-fatal and visible in run-node metrics.

### 7.7 LLM Fallback and Prompt Cache

Current LLM fallback extracts strict JSON:

```json
{
  "title": "short title",
  "summary": "one sentence",
  "slices": [
    {
      "title": "slice title",
      "summary": "slice summary",
      "text": "source evidence",
      "kind": "concept|example|practice|objective|paragraph",
      "confidence": 0.8
    }
  ]
}
```

Planned improvements:

- Store and report explicit fallback reasons:
  - no explicit title feature
  - weak chunk after separator filtering
  - natural-language paragraph group lacks heading boundary
  - local parser confidence below threshold
- Add prompt version migration policy.
- Add model-version compatibility policy for cached payloads.
- Add stale-cache fallback mode:
  - only same text hash for exact cache
  - semantic cache only if evidence checks pass
  - metadata must mark `stale_fallback=true`
- Add bounded retry policy for malformed JSON.
- Keep LLM fallback less strict than deterministic parser. The parser should
  prefer original text and explicit structure; LLM should fill weak labels.

### 7.8 Quality Evaluation and E2E Coverage

Current E2E report already records:

- chunk count
- block count
- LLM chunk count
- local chunk count
- diagnosis score
- token count
- summary ASCII term recall
- summary UTF-8 non-ASCII unit recall
- indexed text ASCII term recall
- indexed text UTF-8 non-ASCII unit recall
- run nodes
- source sample
- summary sample
- indexed text sample
- chunks
- mindmap
- diagnosis

Planned E2E additions:

- Full block list in report, or a bounded block diagnostic section.
- Per-slide PPTX quality summary.
- Per-chunk fallback/caching explanation.
- Exact cache hit/miss summary.
- Future semantic cache hit/miss summary.
- LLM fallback failure reasons.
- Separator filtering count.
- Heading extraction accuracy notes:
  - explicit Office heading
  - inferred heading
  - numbered title
  - LLM title
- Latency percentiles across repeated runs:
  - cold model startup
  - warmed request
  - exact cache hit
  - semantic cache hit
  - real LLM fallback
- Golden-document corpus:
  - real DOCX teaching material
  - real project proposal DOCX
  - real PPTX classroom slides
  - PPTX with multiple text boxes per slide
  - DOCX with tables
  - DOCX with heavy numbering/list styles
  - Chinese filename/path cases
  - mixed Chinese/English technical terms

### 7.9 Runtime Warmup and Observability

The E2E tool has shown that ONNX/CUDA first inference can dominate the first
document request if the embedding model is not warmed.

Planned runtime behavior:

- Add optional Gateway startup warmup for document embedding provider.
- Warmup should run before readiness reports healthy for document analysis.
- Report warmup latency separately from request latency.
- Keep warmup failure configurable:
  - fail startup in strict production mode
  - degrade document semantic clustering in development mode
- Add document-analysis metrics:
  - extraction latency
  - chunk grouping latency
  - embedding request count / total ms
  - exact cache lookup count / hit count / ms
  - future semantic cache lookup count / hit count / ms
  - LLM direct count / ms
  - SQLite write-back ms
  - end-to-end latency

### 7.10 Open Implementation Checklist

Near-term:

- [ ] Test real PPTX files through `/api/document/analyze`.
- [ ] Add PPTX per-slide E2E report fields.
- [ ] Add full or bounded block diagnostics to quality reports.
- [ ] Add document file metadata repository for uploaded files.
- [ ] Add managed upload directory and safe physical eviction.
- [ ] Add WebSocket upload path or upload adapter compatible with existing
      Gateway WS design.
- [ ] Add document-id based analyze request path in addition to dev/test path.
- [ ] Add runtime embedding warmup to Gateway startup, not only E2E.

Cache track:

- [ ] Keep exact Redis LLM chunk cache as fallback/stability layer.
- [ ] Add document semantic cache adapter over existing L0 semantic cache.
- [ ] Add semantic-cache evidence alignment checks.
- [ ] Add semantic-cache metrics to `SemanticChunkQueue`.
- [ ] Store semantic cache entries after successful real LLM fallback.
- [ ] Define prompt/model compatibility policy for cache reuse.

Parser quality track:

- [ ] Improve DOCX table extraction.
- [ ] Improve DOCX numbering definition parsing.
- [ ] Improve repeated header/footer filtering.
- [ ] Improve inline bold/large-font false-heading suppression.
- [ ] Add separator filtering counters.
- [ ] Add fallback reason metadata per chunk.

Reliability track:

- [ ] Add corruption/unsupported-file tests for DOCX/PPTX.
- [ ] Add zip bomb / oversized XML tests.
- [ ] Add malformed XML recovery tests.
- [ ] Add Redis unavailable E2E for exact cache degradation.
- [ ] Add LLM unavailable E2E for local-only degradation.
- [ ] Add crash dump registration to standalone document E2E tools when running
      Windows debugging builds.

Deferred:

- [ ] XLSX extraction policy.
- [ ] PDF extraction/OCR policy.
- [ ] Visual OCR through existing LLM client protocol extension.
- [ ] Offline Python-generated document analysis evaluation corpus importer.
