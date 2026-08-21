# Gateway, Frontend, and Session Lifecycle Alignment

Version: 0.1  
Date: 2026-06-03  
Scope: Architecture notes preserved from the design discussion around the C++ gateway, frontend migration, deployment boundary, and session lifecycle.

## 1. Context

The project is moving from a Python-based prototype into a production-oriented C++ gateway and runtime. The original Python system was useful as a single-machine or very low-concurrency prototype, and it was sufficient for early competitions and demonstrations. However, it was not designed as a high-QPS, low-latency, observable, secure, and commercially deployable system.

The current C++ work is not a simple language port. It is a system-level upgrade intended to provide:

- Higher QPS and lower latency on hot paths.
- Explicit session lifecycle management.
- Traceable request execution across scheduling pools and modules.
- Clear separation between edge security, gateway protocol handling, business runtime, memory, cache, and inference services.
- A static frontend deployment model without Vite dev server or ADP-specific runtime dependencies.
- A codebase that can be migrated into the enterprise project after additional security hardening.

The old frontend under `<legacy-frontend-root>` is treated as a functional reference, not as a protocol contract. Its ADP/qbot/Vite-dev-server integration should not define the new production API.

## 2. Fixed Deployment Boundary

The intended deployment model is:

```text
Browser
  -> HTTPS / WSS
Nginx + TLS + Load Balancing + ModSecurity WAF
  -> HTTP / WS upstream
C++ Gateway container
  -> serves static frontend dist
  -> exposes /api/* REST endpoints
  -> exposes /ws/* WebSocket endpoints
  -> owns session lifecycle and runtime dispatch
```

This boundary should not be simplified into a temporary demo deployment. It is also not the old ADP/Vite/proxy model.

### 2.1 Nginx Responsibilities

Nginx remains the external edge layer. Its responsibilities are:

- TLS termination and HTTPS redirection.
- Load balancing across gateway containers.
- ModSecurity/WAF integration.
- Basic request boundary controls such as body size, connection limits, and rate limits.
- Reverse proxying to the internal C++ gateway.
- Forwarding client and request metadata through headers such as:
  - `X-Forwarded-For`
  - `X-Forwarded-Proto`
  - `X-Real-IP`
  - `X-Request-Id`
- Supporting WebSocket upgrades for `/ws/*`.

Nginx should not contain application-specific protocol adaptation for ADP, qbot, Vite dev server, or legacy frontend behavior.

### 2.2 C++ Gateway Responsibilities

The C++ gateway container is the application gateway and static SPA host. Its responsibilities are:

- Serve the compiled frontend `dist/`.
- Provide SPA fallback to `index.html`, while ensuring `/api/*` and `/ws/*` are never swallowed by static fallback.
- Expose first-party REST APIs under `/api/*`.
- Expose real-time WebSocket APIs under `/ws/*`.
- Create, own, close, persist, and clean up sessions.
- Inject and propagate Trace IDs.
- Perform application-level request parsing, schema validation, authorization checks, session ownership checks, resource limits, and business-level rejection.
- Dispatch work through the designed Net / Compute / IO pool architecture.
- Orchestrate Persona runtime, L0 memory, L3 facts, inference services, and future RAG/multimodal modules.

## 3. Frontend Migration Direction

The frontend must be rebuilt as a static SPA that can be compiled into `dist/` and hosted by the C++ gateway container. It must not depend on Vite dev server, Vite proxy, ADP qbot endpoints, or old Tencent workflow-specific request shapes in production.

The old frontend should be used to preserve functional boundaries:

- Special/single-persona training chat.
- Classroom simulation.
- Teacher broadcast and targeted student interaction.
- Proactive student speaking or polling.
- Training report generation.
- Emotion/state display.
- Teaching document analysis and export.
- E2E collection of latency, turn counts, trace IDs, and runtime metrics.

The legacy protocol layer should be removed or isolated:

- `bot_app_key`
- `visitor_biz_id`
- `workflow_status`
- `tcadp_user_id`
- ADP `custom_variables` as a generic business container
- qbot SSE response-shape parsing
- Vite production proxy dependencies
- COS/DescribeStorageCredential as the default document path
- OpenAI Chat Completions shape as the frontend-to-gateway business protocol

If minimal UI changes are needed during migration, a facade may temporarily keep old function names while internally calling the new first-party API. This facade must remain an adapter, not a new permanent protocol.

## 4. First-Party API Direction

The new frontend should call first-party business APIs. These API shapes should be stable, explicit, and independent of ADP/qbot/OpenAI-compatible historical protocols.

Recommended route groups:

```text
GET  /api/auth/me
POST /api/auth/register

POST /api/session/create
POST /api/session/close
GET  /api/session/{sessionId}
GET  /api/session/{sessionId}/emotion
GET  /api/session/{sessionId}/metrics

POST /api/chat/message

POST /api/classroom/message
POST /api/classroom/proactive
POST /api/classroom/poll

POST /api/report/training

POST /api/document/upload
POST /api/document/analyze
GET  /api/document/{documentId}/analysis
```

WebSocket routes:

```text
/ws/session
```

WebSocket authentication uses the same same-origin `agent_auth` Cookie or `Authorization: Bearer <jwt>` header as HTTP during the upgrade handshake. Frontend code should not put JWTs in the WebSocket query string.

`POST /api/auth/register` is the E2E/local SSO bootstrap endpoint. It is intentionally unauthenticated, creates a backend-local auth session row, signs an RS256 JWT, and returns `Set-Cookie: agent_auth=<jwt>; Path=/; HttpOnly; SameSite=Lax` by default. Production deployment should place this endpoint behind the intended registration policy, rate limit, and WAF boundary.

Long-term real-time interactions should converge on WebSocket for streaming replies, emotion updates, proactive events, and future multimodal events. HTTP chat endpoints may remain useful for initial migration, E2E, fallback, and debugging, but they should not permanently compete with WebSocket as a separate real-time protocol.

## 5. Response Envelope and Observability

REST responses should use a stable envelope:

```json
{
  "ok": true,
  "traceId": "trace_xxx",
  "sessionId": "sess_xxx",
  "latencyMs": 1234,
  "data": {}
}
```

Error responses:

```json
{
  "ok": false,
  "traceId": "trace_xxx",
  "sessionId": "optional-session-id",
  "error": {
    "code": "NOT_FOUND",
    "message": "session not found",
    "details": {}
  }
}
```

Frontend error-display contract:

- `ok=false` is the authoritative error signal.
- `error.code` is stable enough for coarse UI routing, such as unauthorized, missing session, unavailable backend, or validation failure.
- `error.message` is the user-visible fallback text for the frontend error panel.
- `error.details` is optional diagnostic metadata; frontend code must not require it.
- `traceId` should be visible or copyable in E2E/debug UI because backend logs and WAF logs use it for correlation.

Current gateway status-code mapping:

```text
400 INVALID_ARGUMENT
403 PERMISSION_DENIED
404 NOT_FOUND
409 ALREADY_EXISTS or FAILED_PRECONDITION
429 RESOURCE_EXHAUSTED
503 UNAVAILABLE
504 TIMEOUT
500 INTERNAL_ERROR or UNKNOWN
```

Temporarily exposed but not fully implemented features should return a non-fatal default response when the frontend can still proceed:

```json
{
  "ok": true,
  "traceId": "trace_xxx",
  "sessionId": "optional-session-id",
  "latencyMs": 0,
  "data": {
    "implemented": false,
    "status": "placeholder",
    "message": "This feature is not implemented yet.",
    "result": {}
  }
}
```

The frontend should render `implemented=false` as an available-but-basic feature result, not as a global error. Use an actual error envelope only for invalid input, authentication failure, missing state, unsafe uploads, backend unavailability, or any operation that would corrupt or misrepresent session state.

Each business response should carry enough data for E2E and production diagnostics:

- `traceId`
- `sessionId`
- `latencyMs`
- `turnIndex` where applicable
- `usage` where available
- stage timings where available, such as queue, memory, prompt, LLM, and persistence latency
- structured error code

Trace ID priority:

```text
X-Trace-Id > X-Request-Id > gateway-generated trace ID
```

The selected trace ID should be returned both in the response body and as `X-Trace-Id`.

## 6. Session Lifecycle Model

The session lifecycle is a core resource-management concept. A session is not only a chat identifier. It owns or references runtime state such as recent history, emotion state, memory/cache managers, WebSocket connection handles, proactive timers, and metrics.

The intended lifecycle is:

```text
Frontend enters a new conversation
  -> POST /api/session/create
  -> backend creates and initializes session resources
  -> business messages are processed
  -> frontend explicitly sends close request
     OR backend idle timer detects timeout
  -> backend closes session, persists data, destructs resources
```

Recommended state machine:

```text
CREATING
  -> ACTIVE
  -> CLOSING
  -> CLOSED

ACTIVE
  -> IDLE_EXPIRED
  -> CLOSING
  -> CLOSED

ACTIVE
  -> DISCONNECTED
  -> ACTIVE       // WebSocket reconnect succeeds within grace window
  -> CLOSING      // reconnect grace expires
  -> CLOSED
```

`CLOSING` is important. It prevents duplicate teardown, new business work entering a closing session, and concurrent persistence/destruction races.

### 6.1 Session Creation

Session creation should be explicit and preferably always use HTTP:

```text
POST /api/session/create
```

Creation should:

- Validate caller identity and requested session mode.
- Generate or validate `session_id`.
- Bind `tenant_id` and `user_uuid`.
- Initialize persona configuration.
- Initialize or load OU/emotion state.
- Initialize L0 manager and relevant memory/cache scope.
- Optionally prepare L3 facts or summaries.
- Initialize metrics.
- Return `sessionId`, `traceId`, and initial state.

Even if auth is not fully enabled in the competition build, the data model should preserve `tenant_id` and `user_uuid`, using default values if necessary.

### 6.2 Business Activity

Every valid business message should:

- Verify that the session exists and is `ACTIVE`.
- Verify session ownership and scope.
- Refresh `last_active`.
- Run through Trace-aware scheduling.
- Update recent raw history.
- Update emotion state.
- Write or schedule L0 cache records as appropriate.
- Update metrics.

Current memory semantics:

- L0 replaces the previous direct L1/L2 raw-history accumulation/compression path.
- The LLM API is stateless, so context must still be supplied through the message array.
- Messages are built from L0-screened historical context plus the current session's latest raw turns.
- Recent raw turns should include the latest 10 turns, or fewer if the current session has fewer turns.
- L3 facts follow the original logic.
- L4 is reserved for later implementation.

Prompt assembly should clearly distinguish:

```text
[Relevant Historical Context]
[L3 Long-Term Facts]
[Recent Session Turns]
[Current User Message]
```

This avoids confusing semantically retrieved history with the immediate conversational timeline.

### 6.3 Explicit Close

Frontend close should call:

```text
POST /api/session/close
```

Close should be idempotent:

- `ACTIVE` -> perform close.
- `CLOSING` -> return closing/closed status.
- `CLOSED` -> return closed status when a tombstone exists.

Suggested close reasons:

```text
client_close
idle_timeout
ws_disconnect_timeout
server_shutdown
auth_revoked
resource_exhausted
internal_error
```

Suggested close order:

```text
1. Mark CLOSING and reject new business requests.
2. Stop proactive timers.
3. Wait for, finish, or cancel in-flight session tasks according to policy.
4. Flush recent history and L0 active batch.
5. Store timestamp indexes.
6. Persist OU/emotion state.
7. Persist metrics summary.
8. Unbind WebSocket handle.
9. Release memory/cache/runtime resources.
10. Remove from active session map.
11. Keep a short-lived closed-session tombstone for idempotent close and E2E metrics.
```

L3 daily compression should generally be scheduled asynchronously and should not block session close.

### 6.4 Idle Timeout and WebSocket Disconnect

The backend timer remains necessary because browser close events are not reliable.

Recommended windows:

```text
ws_reconnect_grace: 5s - 30s
session_idle_timeout: 15min - 30min
```

The cleanup task should avoid closing sessions that still have active in-flight work unless a stronger shutdown policy is being applied.

For WebSocket:

- Session creation should remain HTTP-driven.
- WebSocket connection should bind to an existing session.
- Disconnect should mark the session as temporarily disconnected.
- Reconnect within the grace window should restore `ACTIVE`.
- Grace expiry should close the session if no other policy keeps it alive.

## 7. Threading and Scheduling Model

The high-level scheduling design remains:

```text
Net Pool
  -> HTTP/WS runtime, connection handling, protocol parsing, Trace injection

Compute Pool
  -> persona instance work, cache calculation, Faiss/search, OU state machine,
     prompt assembly, CPU-bound media frame processing

IO Pool
  -> Redis, SQLite, gRPC inference calls, external LLM calls, batch persistence,
     VLM/BERT requests, heavy storage interaction
```

This separation protects CPU-bound compute workers from high-latency IO. It also makes backpressure and fallback behavior easier to reason about.

SessionManager dispatch should preserve Trace ID and log submit/start/done/failure events for scheduled tasks.

Same-session business work should have serial semantics. This may be implemented through per-session locking, a session task queue, or a session strand. Long IO should not be performed while holding a session lock.

## 8. Load Balancing and Session Affinity

The architecture supports multiple C++ gateway instances behind Nginx, but active session state is process-affine unless and until a distributed session ownership model is implemented.

Early production or competition deployment should use sticky routing:

- Cookie-based affinity, or
- `session_id`-based consistent hash when possible.

If routing depends on `session_id`, frontend requests may include `session_id` in both query and JSON body:

```text
POST /api/chat/message?session_id=sess_xxx
```

The backend should verify that query and body session IDs match.

Longer-term enterprise scaling may require:

- Session owner registry.
- Session restore.
- Externalized state.
- WebSocket affinity.
- Distributed proactive timers or owner-only proactive scheduling.
- Graceful handoff.

## 9. Security Model

The security model is layered:

```text
Nginx + ModSecurity
  -> edge filtering, TLS, LB, coarse request controls

C++ Gateway
  -> schema validation, auth, session ownership, tenant/user scope,
     resource limits, memory/cache scope, business rejection, logging redaction
```

WAF does not replace application-level checks. AI systems also need protection against:

- Prompt injection.
- Cross-session memory leakage.
- Tenant/user scope violations.
- Oversized prompt or message payloads.
- Abusive WebSocket sessions.
- Unsafe document upload.
- Sensitive log leakage.
- Model output that violates business constraints.

JWT/SSO, tenant-aware memory access, encrypted sensitive user data, and key management remain important future hardening requirements.

## 10. Report and Document Analysis Boundaries

Training report generation should not be treated as ordinary chat with an `evaluation=true` flag in the new first-party architecture.

Recommended boundary:

```text
PersonaRuntime
  -> normal dialogue, emotion, prompt, memory

IReportEvaluator provider
  -> downstream-owned session data, indicators, domain scoring,
     optional LLM summary and structured evaluation
```

AgentLoom always provides base session metrics through the report route. Domain evaluation remains a separate provider path injected through `IReportEvaluator`; the core Gateway does not own organization-specific indicators, weights, datasets, Redis queries or evaluation configuration.

Document analysis should also be redesigned as a first-party flow, not a continuation of ADP/COS/qbot:

```text
POST /api/document/upload
POST /api/document/analyze
GET  /api/document/{documentId}/analysis
```

Open design questions for document analysis include:

- Whether documents are stored locally, in object storage, or as temporary session artifacts.
- Whether analysis is synchronous or asynchronous.
- Whether analyzed document content enters RAG, L0, L3, or session-only context.
- How document ownership and tenant scope are enforced.
- How large files are limited, parsed, and scanned.

## 11. Objective Architecture Assessment

The architecture is production-oriented and commercially meaningful. Its main strength is that it addresses the real limitations of the Python prototype:

- Low concurrency.
- Weak deployment model.
- Loose session state.
- ADP/Vite/proxy coupling.
- Limited observability.
- Weak failure isolation.
- Operational reliance instead of engineered resilience.

The C++ gateway path is justified because the system needs high-QPS protocol handling, low-latency cache/search paths, explicit resource ownership, and strong isolation between network, compute, IO, inference, and memory modules.

The main risks are:

- High complexity during migration.
- Session affinity issues under multi-instance load balancing.
- Possible confusion between L0-retrieved history and immediate conversational state.
- Duplicated HTTP and WebSocket chat paths if not converged.
- WAF being mistaken for complete application security.
- Report and document analysis pipelines becoming mixed with normal dialogue.
- SQLite or Redis bottlenecks under enterprise-level concurrency.
- Business logic becoming prematurely locked into C++ without versioned schemas and extension interfaces.

The recommended mitigation is to stabilize the core trunk first:

```text
static dist
  -> Nginx reverse proxy
  -> C++ Gateway
  -> session create
  -> chat/classroom
  -> L0 + recent turns + L3
  -> PersonaRuntime
  -> response
  -> Trace/metrics
  -> report
  -> session close/persist
```

Once this trunk is stable, WebSocket streaming, document analysis, RAG, L4, multimodal support, SSO, and enterprise security hardening can be added on a solid base.

## 12. Implementation Guidance

Future code should preserve these boundaries:

- Do not put business logic into HTTP handlers.
- Do not let legacy ADP frontend fields enter core service DTOs.
- Keep HTTP/WS protocol adapters separate from PersonaRuntime and memory algorithms.
- Keep session lifecycle centralized in SessionManager.
- Preserve Trace ID across Net, Compute, and IO scheduling.
- Keep response schemas versionable.
- Keep report schema versionable.
- Make close idempotent and observable.
- Keep frontend API clients first-party and static-dist compatible.
- Treat legacy frontend code as a functional migration reference only.

The guiding principle is:

```text
Old frontend functionality informs the product boundary.
New first-party APIs define the production boundary.
C++ service modules define the enterprise migration boundary.
```

## 13. Infrastructure Reuse and Library Packaging

The current infrastructure is considered a reusable asset, not a temporary competition-only implementation. The performance report in `docs/performance/PERFORMANCE_REPORT.md` shows that the core C++ infrastructure already has strong hot-path behavior:

- CPU embedding E2E latency is approximately 4 ms in single-item mode.
- CPU batch embedding reaches hundreds of QPS.
- CUDA batch embedding reaches several thousand QPS in the tested setup.
- SIMD dot product for 384-dimensional vectors reaches tens of millions of operations per second.
- Top-K search over small working sets is in the sub-millisecond range.
- Static site serving and HTTP Keep-Alive handling are sufficient for the planned gateway-hosted frontend model.
- WebSocket throughput is high enough to support future streaming and multimodal transport work.

This means the infrastructure should not be repeatedly redesigned. Future work should focus on stabilization, packaging, documentation, safety hardening, and business integration.

### 13.1 Reusable Modules

The following modules have clear reuse value beyond this project:

```text
core
  Status / Result
  ThreadPool
  BlockingQueue
  TraceContext
  resource and lifecycle primitives

config
  typed runtime options
  config parsing
  CLI / env / file integration

net
  HttpServer
  WebSocket handling
  static file serving
  request filtering
  access controller
  HTTP request/response types

vector
  tokenizer abstraction
  tokenizer pool
  embedding pipeline
  ONNX embedding backend
  exact vector index
  Faiss vector index
  SIMD similarity primitives

semantic_cache
  CacheRecord serialization
  L0 search modes
  Top-K retrieval
  Redis / SQLite backed cache lifecycle

memory
  L3 compressor interfaces
  vector repository integration
  memory-level abstractions

service
  SessionManager
  PersonaRuntime
  gateway-facing orchestration
```

These modules can support other C++ systems that need HTTP/WS gateways, static SPA hosting, vector search, embedding, semantic cache, trace-aware scheduling, or AI runtime orchestration.

### 13.2 Suggested Library Targets

The infrastructure should be packaged as layered CMake targets rather than one monolithic library:

```text
agent_core
agent_config
agent_net
agent_vector
agent_semantic_cache
agent_memory
agent_service
```

Recommended dependency direction:

```text
agent_core
  <- agent_config
  <- agent_net
  <- agent_vector
  <- agent_semantic_cache
  <- agent_memory
  <- agent_service
```

More specifically:

```text
agent_net depends on agent_core
agent_config depends on agent_core
agent_vector depends on agent_core
agent_semantic_cache depends on agent_core + agent_vector
agent_memory depends on agent_core + agent_vector
agent_service depends on the infrastructure targets it orchestrates
```

The dependency graph must remain acyclic. Lower-level infrastructure must not depend on Persona, frontend protocol details, competition-specific concepts, or enterprise business logic.

### 13.3 Public and Private Headers

Library packaging should distinguish public API headers from internal implementation headers.

Target public headers should expose stable abstractions such as:

```text
core::Status
core::Result<T>
core::ThreadPool
core::TraceContext
net::HttpServer
net::HttpServerOptions
vector::IVectorIndex
semantic_cache::CacheRecord
service::persona::ISessionManager
```

Internal headers should remain private to each target. Third-party implementation details should not leak into the public API unless they are intentionally part of an extension interface.

Examples:

```text
Public:
  include/agent/core/status.h
  include/agent/net/http_server.h
  include/agent/vector/vector_index.h

Private:
  src/net/internal/beast_session.h
  src/vector/internal/faiss_adapter_detail.h
  src/semantic_cache/internal/redis_batch_loader.h
```

The repository does not need to be physically reorganized immediately, but the CMake target model should gradually enforce this distinction through public/private include directories.

### 13.4 Optional Dependencies

Reusable infrastructure should not force every downstream project to link every heavy dependency.

Recommended feature flags:

```text
AGENT_WITH_FAISS
AGENT_WITH_ONNX
AGENT_WITH_CUDA
AGENT_WITH_REDIS
AGENT_WITH_SQLITE
AGENT_WITH_GRPC
AGENT_WITH_TLS
```

Expected behavior:

- `agent_core` should have minimal dependencies.
- `agent_net` should not require ONNX, Faiss, Redis, or SQLite.
- `agent_vector` may compile without Faiss if only exact index support is enabled.
- ONNX embedding should be optional.
- CUDA acceleration should be optional and selected by build configuration.
- Redis and SQLite integrations should be optional backends, not mandatory for all users of the cache abstractions.

This keeps the infrastructure reusable in smaller C++ projects.

### 13.5 API Stability Policy

Once the infrastructure is packaged as reusable libraries, public APIs should be treated as stable contracts.

Recommended policy:

- Public headers should change deliberately and infrequently.
- Public DTOs should be versioned when they represent serialized data or cross-service contracts.
- Internal implementation can continue to evolve behind stable interfaces.
- Business-specific DTOs must not become part of lower-level infrastructure targets.
- New features should prefer extension interfaces over modifying existing public APIs.

Examples:

```text
Stable:
  core::Result<T>
  net::IHttpRequestHandler
  vector::IVectorIndex
  service::persona::ISessionManager

Versioned:
  training_report.v1
  session_metrics.v1
  websocket_message.v1

Internal:
  exact scheduling policy
  concrete Redis command batching
  concrete ONNX session configuration
```

### 13.6 Error and Resource Model

All reusable infrastructure should continue to use the existing `core::Status` / `core::Result<T>` style. Third-party errors should be converted into project status codes with useful diagnostic messages.

Library code should avoid:

- Raw ownership transfer.
- Unclear pointer lifetime.
- Unstructured `void*` forwarding.
- Throwing exceptions across module boundaries unless explicitly documented.
- Returning `bool` without an error reason for meaningful failures.

RAII should remain the default for:

- HTTP/WS sessions.
- Thread pools.
- Redis and SQLite connections.
- ONNX sessions.
- Tokenizer handles.
- Faiss indexes.
- Static file resources.
- Timer and proactive session resources.

### 13.7 Benchmark and Regression Targets

The benchmark suite should remain part of the infrastructure asset. Performance claims should be protected against regression.

Important benchmark targets include:

```text
net_stability_bench
static_site_concurrency_bench
ws_download_bench
embedding_bench
simd_similarity_bench
semantic_cache_bench
```

Future business-pipeline benchmarks should be added separately:

```text
gateway_session_create_bench
gateway_chat_message_bench
gateway_classroom_message_bench
gateway_proactive_bench
gateway_report_generation_bench
gateway_session_close_bench
```

The current performance report proves that the infrastructure is strong. A later gateway business pipeline report should prove that the integrated product path is strong.

### 13.8 What Should Not Be Rewritten

The following infrastructure components should not be redesigned unless a concrete correctness, safety, or performance issue is found:

- `core::Status` / `core::Result`
- Thread pool and blocking queue foundations.
- Trace context propagation model.
- HTTP server core.
- Static file serving model.
- WebSocket transport core.
- Vector index abstractions.
- SIMD dot product implementation.
- CacheRecord binary serialization.
- Tokenizer and embedding abstractions.
- Config parser shape.

Future changes should be scoped to:

- API polish.
- Public/private boundary cleanup.
- Dependency optionalization.
- Documentation.
- Tests and benchmarks.
- Security hardening.
- Business integration.

This keeps the project focused on migrating and stabilizing the business pipeline instead of repeatedly revisiting infrastructure that is already performing well.

### 13.9 Enterprise Migration Value

The infrastructure can become a reusable foundation for the enterprise project if the following boundaries are preserved:

```text
Infrastructure libraries
  -> generic C++ runtime, net, vector, cache, memory primitives

Competition/business service layer
  -> PersonaRuntime, classroom, reports, document analysis

Enterprise service layer
  -> enterprise-specific auth, tenant policy, RAG, data governance, deployment policy
```

The core value is not only performance. It is also:

- Predictable ownership.
- Traceable scheduling.
- Clear failure isolation.
- Reusable gateway runtime.
- Typed configuration.
- Vector and semantic cache primitives.
- A path to package shared C++ libraries across multiple projects.

The infrastructure should therefore be treated as a long-term asset and a shared runtime foundation.
