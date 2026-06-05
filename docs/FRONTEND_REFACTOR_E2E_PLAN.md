# Frontend Refactor and E2E Plan

## 1. Purpose

This document defines the frontend refactor direction for the competition E2E phase.

The frontend should be rebuilt from the old `D:\front_end\Train` project as a functional reference, not as a protocol or architecture contract. The new frontend must exercise the C++ Gateway directly, collect latency data for the full maximum-latency path, and build to a static `dist` directory served by the C++ `HttpServer`.

The backend-side target path is:

```text
Browser
  -> Nginx TLS / ModSecurity WAF
  -> C++ Gateway HttpServer
       -> static dist
       -> /api/*
       -> /ws/session
       -> SessionManager
       -> PersonaRuntime
       -> memory context
       -> prompt build
       -> cloud LLM or local gRPC fallback
       -> response
```

For local E2E, Nginx/ModSecurity may be bypassed initially. The frontend must still use relative API paths so the same build works behind Nginx later.

## 2. Non-Goals

Do not keep these old frontend assumptions:

```text
Vite dev server as a runtime dependency
ADP protocol parsing
qbot protocol shim
old platform dispatcher
frontend-driven user UUID trust
hardcoded backend host in production code
```

Do not implement a marketing landing page. The first screen should be the usable chat/classroom/testing workspace.

Do not make the first E2E depend on semantic answer cache. The first E2E should measure the current maximum-latency path where `answerCache.hit=false`.

## 3. Frontend Architecture

Recommended source layout:

```text
src/
  api/
    gatewayClient.ts
    sessionClient.ts
    chatClient.ts
    reportClient.ts
    wsClient.ts
  auth/
    authState.ts
    cookieSession.ts
  state/
    sessionStore.ts
    chatStore.ts
    metricsStore.ts
  views/
    ChatWorkspace.tsx
    ClassroomWorkspace.tsx
    TrainingReportPanel.tsx
    MetricsPanel.tsx
  components/
    MessageList.tsx
    MessageComposer.tsx
    SessionToolbar.tsx
    LatencyTable.tsx
  utils/
    traceId.ts
    latency.ts
    schema.ts
```

The critical split is:

```text
UI components
  -> state stores
  -> typed gateway clients
  -> relative /api and /ws endpoints
```

No UI component should parse backend envelopes directly beyond calling typed API client functions.

## 4. Runtime Deployment

The frontend must build to static assets:

```text
npm run build
  -> dist/
```

The C++ gateway serves `dist`:

```text
GET /              -> dist/index.html
GET /assets/*      -> dist/assets/*
GET /any/spa/path  -> dist/index.html
POST /api/*        -> gateway API
WS /ws/session     -> gateway WebSocket
```

Frontend API paths must be relative:

```ts
fetch("/api/session/create", ...)
fetch("/api/chat/message", ...)
new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/session`)
```

This keeps the build portable across:

```text
local C++ server
container hosted gateway
Nginx reverse proxy
TLS deployment
```

## 5. Gateway API Contract

All HTTP API responses use this envelope:

```json
{
  "ok": true,
  "traceId": "...",
  "sessionId": "...",
  "latencyMs": 0,
  "data": {}
}
```

Error responses use:

```json
{
  "ok": false,
  "traceId": "...",
  "sessionId": "optional-session-id",
  "error": {
    "code": "PERMISSION_DENIED",
    "message": "...",
    "details": {}
  }
}
```

Error response rules for the frontend:

```text
ok=false is the authoritative signal for rendering an error state.
traceId must be shown or made copyable in developer/E2E diagnostics.
error.code should drive coarse UI behavior.
error.message is safe to display in the frontend error panel unless the endpoint explicitly documents otherwise.
error.details is optional and may be omitted; frontend code must tolerate its absence.
sessionId may be absent for auth, registration, document upload, and early request validation failures.
```

Current HTTP status mapping:

```text
400 -> INVALID_ARGUMENT
403 -> PERMISSION_DENIED
404 -> NOT_FOUND
409 -> ALREADY_EXISTS or FAILED_PRECONDITION
429 -> RESOURCE_EXHAUSTED
503 -> UNAVAILABLE
504 -> TIMEOUT
500 -> INTERNAL_ERROR or UNKNOWN
```

Known error code strings currently include:

```text
INVALID_ARGUMENT
NOT_FOUND
ALREADY_EXISTS
PERMISSION_DENIED
FAILED_PRECONDITION
RESOURCE_EXHAUSTED
UNAVAILABLE
TIMEOUT
INTERNAL_ERROR
UNKNOWN
```

The backend also returns `X-Trace-Id` in HTTP responses.

### 5.0 Default Response for Temporarily Unimplemented Features

Some competition E2E screens may exist before the full backend business algorithm is migrated. If the endpoint is intentionally exposed for frontend integration but the algorithm is not finished, prefer a successful default response over an error when the frontend can continue the flow.

Default successful placeholder shape:

```json
{
  "ok": true,
  "traceId": "...",
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

Frontend handling:

```text
Render this as an available-but-basic result, not as a fatal error.
Show data.message only in the feature result area, not in the global error panel.
Do not block the rest of the E2E flow when implemented=false and ok=true.
Preserve traceId for diagnostics.
```

Use an error only when the request cannot be accepted or the UI must stop the flow:

```json
{
  "ok": false,
  "traceId": "...",
  "error": {
    "code": "UNAVAILABLE",
    "message": "Document analysis pipeline is not available in this build.",
    "details": {
      "feature": "document_analysis",
      "implemented": false
    }
  }
}
```

Recommended boundary:

```text
Return ok=true + implemented=false for report/document/classroom helper data that can be displayed as a default result.
Return ok=false for auth failures, invalid input, missing session, unsafe upload, unavailable LLM/local inference, or any action that would make later state inconsistent.
```

### 5.1 Auth Identity

```http
POST /api/auth/register
Content-Type: application/json
X-Trace-Id: <trace-id>
```

Request:

```json
{
  "userUuid": "optional-client-or-test-user-uuid",
  "tenantId": "default",
  "subject": "optional-login-subject",
  "ttlSeconds": 28800
}
```

Response:

```json
{
  "ok": true,
  "traceId": "...",
  "data": {
    "authenticated": true,
    "userUuid": "e2e-user-001",
    "tenantId": "default",
    "subject": "e2e-user-001",
    "tokenId": "jwt-session-id",
    "issuedAt": 1234560000,
    "expiresAt": 1234567890,
    "token": "<jwt>"
  }
}
```

Headers:

```http
Set-Cookie: agent_auth=<jwt>; Path=/; Max-Age=<seconds>; HttpOnly; SameSite=Lax
```

Notes:

```text
This endpoint bypasses normal JWT authentication so a browser E2E flow can create its first login state.
The backend signs an RS256 JWT and stores tokenId -> userUuid/tenantId in SQLite.
The returned token is useful for Playwright and non-browser tests; browser code should rely on the Set-Cookie header.
```

```http
GET /api/auth/me
X-Trace-Id: <trace-id>
Cookie: agent_auth=<jwt>
```

Response:

```json
{
  "ok": true,
  "traceId": "...",
  "data": {
    "authenticated": true,
    "userUuid": "e2e-user-001",
    "tenantId": "default",
    "subject": "e2e-user-001"
  }
}
```

Frontend usage:

```text
Call this endpoint during app bootstrap.
If authenticated=false in dev mode, the UI may continue as anonymous local_user.
If SSO is required and the backend returns 401/403, show the unauthorized state.
Do not let UI state override userUuid or tenantId.
```

### 5.2 Create Session

```http
POST /api/session/create
Content-Type: application/json
X-Trace-Id: <trace-id>
```

Request:

```json
{
  "sessionId": "optional-client-session-id",
  "personaId": "lidazhi",
  "personality": {
    "name": "李大志",
    "description": "..."
  }
}
```

Notes:

```text
userUuid should come from auth/JWT on the backend when SSO is enabled.
body.userUuid may exist for local dev, but backend auth context should override it.
```

Additional session creation fields now supported by the gateway:

```json
{
  "classroomId": "optional-classroom-id",
  "contextIds": ["group_1"],
  "contextPatterns": ["private_*"],
  "proactiveLevel": "off",
  "defaultPersona": true
}
```

Notes:

```text
classroomId/contextIds/contextPatterns/defaultPersona register this session in the classroom scheduler.
proactiveLevel accepts "off", "low", or "medium".
The frontend should include classroomId when creating classroom simulation persona sessions.
```

Response:

```json
{
  "ok": true,
  "traceId": "...",
  "sessionId": "...",
  "latencyMs": 0,
  "data": {
    "sessionId": "...",
    "userUuid": "...",
    "personaId": "lidazhi",
    "status": "active",
    "metrics": {
      "turnCount": 0,
      "requestCount": 0,
      "failedRequestCount": 0,
      "lastLatencyMs": 0,
      "totalLatencyMs": 0,
      "avgLatencyMs": 0
    }
  }
}
```

### 5.3 Send Chat Message

```http
POST /api/chat/message
Content-Type: application/json
X-Trace-Id: <trace-id>
```

Request:

```json
{
  "sessionId": "...",
  "personaId": "lidazhi",
  "mode": "chat",
  "message": "你好",
  "model": "",
  "stream": false
}
```

Response:

```json
{
  "ok": true,
  "traceId": "...",
  "sessionId": "...",
  "latencyMs": 0,
  "data": {
    "personaId": "lidazhi",
    "turnIndex": 1,
    "reply": {
      "role": "assistant",
      "content": "..."
    },
    "userEmotion": {
      "primary": "neutral",
      "intensity": 0.0,
      "behavior": "unknown",
      "tone": "neutral"
    },
    "aiEmotion": {
      "primary": "neutral",
      "intensity": 0.0,
      "behavior": "unknown",
      "tone": "neutral"
    },
    "memory": {
      "l0Hit": false,
      "l3Hit": false
    },
    "answerCache": {
      "enabled": false,
      "hit": false,
      "bypassed": false,
      "source": "llm",
      "cacheKey": "",
      "similarityScore": 0.0
    },
    "pipelineLatency": {
      "memoryContextMs": 0,
      "answerCacheMs": 0,
      "promptBuildMs": 0,
      "llmTotalMs": 0,
      "totalMs": 0
    }
  }
}
```

The first E2E baseline should expect:

```text
answerCache.enabled=false
answerCache.hit=false
answerCache.source=llm
pipelineLatency.llmTotalMs >= 0
```

When semantic answer cache is later implemented, the frontend should not need schema changes.

### 5.4 Get Session

```http
GET /api/session/{sessionId}
GET /api/session/{sessionId}/metrics
GET /api/session/{sessionId}/emotion
```

These endpoints are used for polling session state and E2E assertions.

### 5.5 Training Report

```http
POST /api/report/training
```

Request:

```json
{
  "sessionId": "...",
  "includeRawTurns": true
}
```

Current response may contain a pending/default summary:

```json
{
  "ok": true,
  "data": {
    "generatedAt": "...",
    "totalTurns": 1,
    "summary": "Training report evaluation pipeline is pending; session metrics are available.",
    "metrics": {},
    "schemaVersion": "training_report.v1"
  }
}
```

This is acceptable for the first E2E. The frontend should render it as an available-but-basic report, not as an error.

### 5.6 Close Session

```http
POST /api/session/close
```

Request:

```json
{
  "sessionId": "...",
  "reason": "client_close"
}
```

After close, the frontend must prevent further sends for that session.

### 5.7 Classroom Message

```http
POST /api/classroom/message
Content-Type: application/json
X-Trace-Id: <trace-id>
```

Request:

```json
{
  "classroomId": "classroom-a",
  "sessionId": "",
  "targetPersonaId": "xiaozhi",
  "contextId": "group_2",
  "message": "please answer",
  "broadcast": false,
  "model": ""
}
```

Routing semantics:

```text
If sessionId is present, route directly to that session.
Else if targetPersonaId/personaId is present, route to that persona in classroomId.
Else if contextId is present, scheduler resolves exact contextIds, then contextPatterns.
Else scheduler uses default persona fallback.
```

Response:

```json
{
  "ok": true,
  "traceId": "...",
  "sessionId": "session-b",
  "latencyMs": 0,
  "data": {
    "classroomId": "classroom-a",
    "speakerPersonaId": "xiaozhi",
    "content": "...",
    "shouldSpeak": true,
    "turnIndex": 1,
    "userEmotion": {
      "primary": "neutral",
      "intensity": 0.0,
      "behavior": "unknown",
      "tone": "neutral"
    },
    "aiEmotion": {
      "primary": "neutral",
      "intensity": 0.0,
      "behavior": "unknown",
      "tone": "neutral"
    }
  }
}
```

### 5.8 Classroom Proactive

```http
POST /api/classroom/proactive
Content-Type: application/json
X-Trace-Id: <trace-id>
```

Request:

```json
{
  "classroomId": "classroom-a",
  "sessionId": "",
  "personaId": "xiaozhi",
  "contextId": "group_2",
  "model": ""
}
```

This endpoint forces a proactive generation request for the resolved persona. It is useful for manual UI controls and debugging. For normal classroom simulation polling, prefer `/api/classroom/poll`.

Response shape is the same as `/api/classroom/message`.

### 5.9 Classroom Poll

```http
POST /api/classroom/poll
Content-Type: application/json
X-Trace-Id: <trace-id>
```

Request:

```json
{
  "classroomId": "classroom-a",
  "personaId": "xiaozhi",
  "contextId": "group_2",
  "systemEvent": false,
  "systemEventContent": "",
  "model": ""
}
```

Semantics:

```text
off: never proactive.
low: only systemEvent can trigger proactive speech.
medium: idle trigger can produce proactive speech.
WAITING_RESPONSE state: poll returns shouldSpeak=false until timeout or user reply.
DORMANT state: poll returns shouldSpeak=false until user message resets state.
```

No-speech response:

```json
{
  "ok": true,
  "traceId": "...",
  "sessionId": "session-b",
  "latencyMs": 0,
  "data": {
    "classroomId": "classroom-a",
    "speakerPersonaId": "xiaozhi",
    "content": "",
    "shouldSpeak": false,
    "turnIndex": 0,
    "userEmotion": {
      "primary": "",
      "intensity": 0.0,
      "behavior": "",
      "tone": ""
    },
    "aiEmotion": {
      "primary": "",
      "intensity": 0.0,
      "behavior": "",
      "tone": ""
    }
  }
}
```

Speech response uses the same response shape as classroom message, with `shouldSpeak=true`.

## 6. WebSocket Contract

HTTP should be used for the first E2E. WS should be kept as a second path.

Endpoint:

```text
WS /ws/session
```

Auth:

```text
The WebSocket upgrade request uses the same Cookie/Bearer JWT auth as HTTP.
Browsers automatically send same-origin agent_auth cookies during WS upgrade.
If auth fails, the backend sends a type=error frame.
```

Send:

```json
{
  "type": "chat.message",
  "traceId": "...",
  "payload": {
    "sessionId": "...",
    "personaId": "lidazhi",
    "mode": "ws_chat",
    "message": "你好",
    "model": ""
  }
}
```

Receive:

```json
{
  "type": "chat.final",
  "payload": {
    "ok": true,
    "traceId": "...",
    "sessionId": "...",
    "latencyMs": 0,
    "data": {
      "personaId": "lidazhi",
      "turnIndex": 1,
      "reply": {
        "role": "assistant",
        "content": "..."
      },
      "memory": {
        "l0Hit": false,
        "l3Hit": false
      },
      "answerCache": {
        "enabled": false,
        "hit": false,
        "bypassed": false,
        "source": "llm",
        "cacheKey": "",
        "similarityScore": 0.0
      },
      "pipelineLatency": {
        "memoryContextMs": 0,
        "answerCacheMs": 0,
        "promptBuildMs": 0,
        "llmTotalMs": 0,
        "totalMs": 0
      }
    }
  }
}
```

Error:

```json
{
  "type": "error",
  "payload": {
    "ok": false,
    "traceId": "...",
    "error": {
      "code": "PERMISSION_DENIED",
      "message": "..."
    }
  }
}
```

Close:

```json
{
  "type": "session.close",
  "traceId": "...",
  "payload": {
    "sessionId": "...",
    "reason": "client_close"
  }
}
```

Current WS is final-response async, not token streaming. Do not build frontend UI that assumes token streaming yet.

## 7. Frontend Session State

Use an explicit state machine:

```text
idle
  -> creating
  -> active
  -> closing
  -> closed
  -> error
```

Allowed transitions:

```text
idle -> creating
creating -> active
creating -> error
active -> closing
closing -> closed
active -> error
error -> idle
```

Send message only when:

```text
state === "active"
sessionId is not empty
input message is not empty
```

## 8. Latency and Metrics Collection

The frontend should record both client-observed and backend-reported latency.

For each message:

```ts
const traceId = createTraceId()
const clientStart = performance.now()
const response = await sendChatMessage(...)
const clientEnd = performance.now()

const clientObservedMs = clientEnd - clientStart
const backendTotalMs = response.data.pipelineLatency.totalMs
const networkAndRenderMs = clientObservedMs - backendTotalMs
```

Store per-turn metrics:

```ts
type TurnMetric = {
  traceId: string
  turnIndex: number
  clientObservedMs: number
  backendTotalMs: number
  memoryContextMs: number
  answerCacheMs: number
  promptBuildMs: number
  llmTotalMs: number
  answerCacheHit: boolean
  source: string
}
```

The E2E report should include:

```text
first turn latency
average latency
max latency
backend total latency
LLM total latency
memory context latency
answer cache hit count
turn count
request count
failed request count
```

## 9. SSO for Competition E2E

Competition version SSO should be intentionally simple:

```text
Cookie + JWT
RSA signature
uuid claim mapping
SQLite persona config isolation by UUID
```

Recommended cookie:

```text
agent_auth=<jwt>
HttpOnly
SameSite=Lax
Secure when HTTPS is enabled
```

Recommended JWT claims:

```json
{
  "sub": "e2e-user-001",
  "uuid": "e2e-user-001",
  "tenant": "default",
  "jti": "e2e-session-001",
  "role": "student",
  "iss": "agent-e2e",
  "aud": "agent-gateway",
  "exp": 1234567890
}
```

Frontend E2E can inject this cookie before loading the page.

The frontend must not trust or manually edit `userUuid` for production-like E2E. The backend should derive it from JWT.

Development mode may allow anonymous fallback:

```text
uuid = local_user
tenant = default
role = student
```

Backend-local login state:

```text
When gateway_auth.session_database_path is configured, JWT jti/sid maps to SQLite table gateway_auth_sessions.
The backend may auto-provision this record for externally issued JWTs, require a pre-existing record in stricter mode, or create it through POST /api/auth/register.
SQLite can revoke a login session independently of JWT expiry.
```

Registration/signing config:

```json
{
  "gateway_auth": {
    "enabled": true,
    "allow_dev_identity": false,
    "require_auth_for_api": true,
    "cookie_name": "agent_auth",
    "public_key_file": "jwt_public.pem",
    "private_key_file": "jwt_private.pem",
    "issuer": "agent-e2e",
    "audience": "agent-gateway",
    "clock_skew_seconds": 60,
    "token_ttl_seconds": 28800,
    "cookie_http_only": true,
    "cookie_secure": false,
    "cookie_same_site": "Lax",
    "require_session_record": false,
    "auto_provision_session": true,
    "session_database_path": "gateway_auth.db"
  }
}
```

Current auth session table:

```sql
CREATE TABLE IF NOT EXISTS gateway_auth_sessions (
    token_id TEXT PRIMARY KEY,
    user_uuid TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    subject TEXT,
    issued_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0,
    revoked_reason TEXT,
    updated_at INTEGER NOT NULL
);
```

Frontend impact:

```text
The frontend only manages the browser cookie.
It should not persist or edit token_id/user_uuid mappings.
Logout should clear the cookie client-side; server-side revocation can be added as a future endpoint.
```

## 10. WAF Boundary

Competition version WAF should be:

```text
Nginx + ModSecurity fixed rules
C++ Gateway records WAF-related headers and trace IDs
```

Do not duplicate ModSecurity rules in C++.

Recommended C++ responsibilities:

```text
request body limit
basic suspicious header/path filter
trace id propagation
access/error logging
optional read of X-WAF-* headers
```

Recommended Nginx responsibilities:

```text
TLS
reverse proxy
ModSecurity CRS/fixed rule set
request blocking
WAF audit log
load balancing
```

This keeps security failures debuggable from Nginx/ModSecurity logs rather than C++ business code.

## 11. Persona Config Isolation

Old Python config files:

```text
D:\Users\21405\source\repos\EducationalAgentProject\agent\config\config_lidazhi.json
D:\Users\21405\source\repos\EducationalAgentProject\agent\config\config_lidazhi_no_emotion.json
D:\Users\21405\source\repos\EducationalAgentProject\agent\config\config_linnuan.json
D:\Users\21405\source\repos\EducationalAgentProject\agent\config\config_test_deepseek.json
D:\Users\21405\source\repos\EducationalAgentProject\agent\config\config_zhangyiming.json
D:\Users\21405\source\repos\EducationalAgentProject\agent\config\evaluation_indicators.json
```

Competition C++ approach:

```text
import Python JSON config
store raw JSON into SQLite
key by user_uuid + persona_id
parse only fields needed by C++ runtime
```

Minimal SQLite table:

```sql
CREATE TABLE IF NOT EXISTS persona_configs (
    user_uuid TEXT NOT NULL,
    persona_id TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (user_uuid, persona_id)
);
```

Optional evaluation table:

```sql
CREATE TABLE IF NOT EXISTS evaluation_indicator_sets (
    tenant_id TEXT NOT NULL,
    name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (tenant_id, name)
);
```

Frontend does not manage this config directly in the first E2E. The frontend only chooses:

```text
personaId
session mode
message
```

Backend maps:

```text
JWT uuid + personaId -> SQLite persona config -> CreateSessionRequest.personality
```

## 12. E2E Test Plan

### 12.1 Full HTTP Chat Path

Steps:

```text
1. Start C++ Gateway with static dist.
2. Open frontend page.
3. Inject JWT cookie or use dev anonymous mode.
3a. Call `GET /api/auth/me` and assert identity.
4. Create session.
5. Send first chat message.
6. Wait for assistant reply.
7. Assert reply visible in UI.
8. Assert answerCache.hit=false.
9. Assert pipelineLatency.totalMs is present.
10. Fetch /api/session/{sessionId}/metrics.
11. Assert turnCount >= 1.
12. Generate training report.
13. Close session.
14. Assert further sends are blocked in UI.
```

### 12.2 Optional Auth Path

```text
No cookie -> unauthorized UI state or 401.
Valid cookie -> session create succeeds.
Expired cookie -> unauthorized UI state or 401.
```

### 12.3 Optional WAF Path

Run only when Nginx/ModSecurity is part of the local E2E stack.

```text
normal chat request -> allowed
known blocked payload -> blocked by Nginx/ModSecurity
C++ Gateway should not see blocked request
```

### 12.4 Optional WS Path

```text
connect /ws/session
same-origin agent_auth cookie is sent during upgrade
send chat.message
receive chat.final
assert same UI rendering path as HTTP
```

### 12.5 Classroom Poll Path

```text
Create two sessions with the same classroomId.
Register contextIds or targetPersonaId for each persona.
POST /api/classroom/message with contextId and assert routed speakerPersonaId.
POST /api/classroom/poll and assert shouldSpeak=false for off/low without systemEvent.
POST /api/classroom/poll with systemEvent=true for low/medium and assert optional proactive reply.
```

## 13. Backend Test Responsibility

This repository should keep backend-focused tests:

```text
service/persona unit tests
gateway HTTP loopback tests
gateway server static dist tests
LLM fallback tests
auth/JWT tests once implemented
SQLite persona config repository tests once implemented
```

The frontend terminal can own:

```text
React/Vue/Svelte refactor
Playwright browser E2E
UI latency rendering
static dist build
```

## 14. Refactor Checklist

Frontend:

```text
[ ] Remove ADP parser dependency from main chat path.
[ ] Remove qbot/platform protocol shim from main chat path.
[ ] Replace dev-server-only backend paths with relative /api paths.
[ ] Add typed gateway client.
[ ] Add explicit session state machine.
[ ] Add latency metrics store.
[ ] Add answerCache display fields.
[ ] Add training report panel with pending/default support.
[ ] Add static build output.
[ ] Add Playwright full HTTP E2E.
```

Backend:

```text
[x] HttpServer can serve static dist and /api on one instance.
[x] Gateway HTTP chat is async.
[x] Gateway WS chat is async final-response.
[x] Answer cache schema is exposed in chat response.
[x] Add /api/auth/me.
[x] Add competition JWT cookie auth.
[x] Add SQLite auth session store.
[x] Add classroom poll endpoint.
[x] Add gateway E2E config sections.
[ ] Add /api/health.
[ ] Add SQLite persona config repository.
[ ] Add config import tool for Python persona JSON.
[ ] Add E2E server startup fixture or script.
```

## 15. First E2E Success Criteria

The first successful E2E run should produce:

```text
sessionId
traceId per request
assistant reply rendered
turnCount >= 1
requestCount >= 1
client observed latency
backend total latency
LLM total latency
answerCache.hit=false
training report panel rendered
session close success
```

This is the baseline maximum-latency chain. After semantic answer cache is implemented, later E2E runs should compare cache-hit latency against this baseline.
