# Frontend Backend API Protocol

本文档用于前端和当前 C++ Gateway 后端对齐接口。内容以 `agent_gateway_server` / `PersonaGatewayHttpAdapter` 当前实现为准，覆盖 HTTP API、WebSocket 消息、认证、主对话链路、文档链路和 Skill Session 控制接口。

更新时间：2026-07-11

## 1. 基础约定

### 1.1 Base URL

默认本地形式：

```text
http://<host>:<port>
```

端口由配置中的 `persona_gateway.port` / HTTP options 决定。WebSocket 默认路径：

```text
/ws/session
```

前端同源部署时建议直接使用相对路径：

```ts
fetch("/api/session/create", ...)
new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/session`)
```

### 1.2 JSON 和编码

- HTTP request / response 默认使用 `application/json; charset=utf-8`。
- WebSocket 文本帧为 JSON 字符串。
- 所有路径、文件名、消息内容、错误消息均按 UTF-8 处理。
- 字段命名对前端优先使用 camelCase。部分接口兼容 snake_case 输入，但前端新代码应使用 camelCase。

### 1.3 Trace ID

后端会为每个请求生成或透传 `traceId`。前端可通过以下方式传入：

- Header: `X-Trace-Id`
- Header: `X-Request-Id`
- JSON body: `traceId`

响应会带：

```http
X-Trace-Id: <trace-id>
```

JSON envelope 中也会包含：

```json
{
  "traceId": "trace-xxx"
}
```

前端日志、错误弹窗、问题反馈应保留 `traceId`。

### 1.4 成功响应 Envelope

大多数 HTTP 成功响应结构：

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "sessionId": "session-xxx",
  "latencyMs": 123,
  "data": {}
}
```

不是所有接口都有 `sessionId` 或 `latencyMs`。前端判断成功应以 `ok === true` 为准。

### 1.5 错误响应 Envelope

HTTP 和 WebSocket 错误 payload 均使用：

```json
{
  "ok": false,
  "traceId": "trace-xxx",
  "error": {
    "code": "InvalidArgument",
    "message": "sessionId is required"
  }
}
```

前端规则：

- `ok === false` 是权威错误信号。
- `error.code` 用于粗粒度 UI 分支。
- `error.message` 可用于开发态或用户友好化后的提示。

常见错误码包括：

```text
InvalidArgument
NotFound
FailedPrecondition
Unauthenticated
PermissionDenied
ResourceExhausted
Unavailable
InternalError
```

HTTP 状态码由后端根据错误码映射，前端不应只依赖 HTTP status。

## 2. 认证

后端支持 Cookie / Bearer JWT。浏览器端推荐使用 HttpOnly Cookie。

默认 Cookie 名称由配置决定，当前约定通常为：

```text
agent_auth
```

WebSocket upgrade 时浏览器会自动携带同源 Cookie。前端不要把 JWT 放进 WebSocket query string。

### 2.1 注册/登录

```http
POST /api/auth/signup
```

正式注册入口不要求已登录，但只接受用户凭据字段。`subject` 是 JWT 语义隔离字段，`tenantId` 是后续管理员/组策略域字段，前端注册时不要传入这两个字段。

Request:

```json
{
  "username": "student@example.test",
  "password": "correct-password",
  "ttlSeconds": 86400
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `username` | string | 是 | 登录名；`email` 可作为兼容别名 |
| `password` | string | 是 | 明文密码只出现在 HTTPS 请求体中，后端保存 PBKDF2 哈希、salt 和迭代次数 |
| `ttlSeconds` | number | 否 | token 有效期，缺省使用后端配置 |

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "data": {
    "authenticated": true,
    "userUuid": "user-001",
    "tenantId": "default",
    "subject": "user-001",
    "tokenId": "jti-xxx",
    "issuedAt": 1782230400,
    "expiresAt": 1782316800,
    "token": "eyJ..."
  }
}
```

同时响应头包含：

```http
Set-Cookie: agent_auth=<jwt>; Path=/; HttpOnly; SameSite=Lax; ...
```

Cookie 只保存 JWT/session token，不保存密码。前端同源请求应使用：

```ts
fetch("/api/auth/signup", {
  method: "POST",
  credentials: "include",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload)
})
```

登录入口：

```http
POST /api/auth/login
```

Request:

```json
{
  "username": "student@example.test",
  "password": "correct-password",
  "ttlSeconds": 86400
}
```

登录成功返回格式与 `/api/auth/signup` 相同，并重新签发 JWT 与 Cookie。密码校验使用后端注册记录，不从 Cookie 读取密码。

开发/测试兼容入口：

```http
POST /api/auth/register
```

此接口仅用于本地/E2E/比赛演示环境创建认证态，默认关闭，需要显式启用 `gateway_auth.enable_dev_registration` 或 CLI `--gateway-auth-enable-dev-registration`。该接口仍可接受 `userUuid`、`tenantId`、`subject` 以支持测试场景；生产入口不要使用它。

### 2.2 当前认证状态

```http
GET /api/auth/me
```

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "data": {
    "authenticated": true,
    "userUuid": "user-001",
    "tenantId": "default",
    "subject": "user-001"
  }
}
```

前端启动时应先调用此接口恢复登录态。

## 3. Session 接口

Session 是主对话链路的核心资源。前端必须先创建 session，再发送 chat/classroom/report/skill 请求。

### 3.1 Session Snapshot

多个接口会返回统一 session snapshot：

```json
{
  "sessionId": "session-001",
  "userUuid": "user-001",
  "personaId": "teacher",
  "status": "active",
  "closeReason": "",
  "recentTurnCount": 2,
  "emotion": {
    "valence": 0.12,
    "arousal": 0.34,
    "primary": "neutral",
    "sustainedLabel": "neutral"
  },
  "metrics": {
    "turnCount": 2,
    "requestCount": 2,
    "failedRequestCount": 0,
    "lastLatencyMs": 530,
    "totalLatencyMs": 1020,
    "avgLatencyMs": 510
  }
}
```

`status` 可取：

```text
creating
active
disconnected
closing
closed
```

### 3.2 创建 Session

```http
POST /api/session/create
```

Request:

```json
{
  "sessionId": "session-001",
  "classroomId": "classroom-001",
  "personaId": "teacher",
  "contextIds": ["ctx-001"],
  "contextPatterns": ["math"],
  "proactiveLevel": "off",
  "defaultPersona": false,
  "personality": {
    "name": "teacher",
    "description": "A concise teaching assistant",
    "traits": ["patient", "clear"],
    "openness": 0.5,
    "extraversion": 0.5,
    "humorTendency": 0.2,
    "empathyLevel": 0.7,
    "curiosityLevel": 0.5,
    "formality": 0.6
  },
  "emotionPrompts": {
    "emotionMap": {},
    "emotionReliability": {},
    "confidenceThresholds": {},
    "intensityLevels": {}
  }
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `sessionId` | string | 否 | 前端指定 ID；为空时后端可生成 |
| `classroomId` | string | 否 | 课堂/房间 ID |
| `personaId` | string | 否 | 人格/角色 ID |
| `contextIds` | string[] | 否 | 绑定上下文 ID |
| `contextPatterns` | string[] | 否 | 上下文匹配模式 |
| `proactiveLevel` | string | 否 | 主动发言等级，默认 `off` |
| `defaultPersona` | bool | 否 | 是否使用默认人格 |
| `personality` | object | 否 | 人格配置 |
| `emotionPrompts` | object | 否 | 情绪提示词配置 |

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "sessionId": "session-001",
  "latencyMs": 1,
  "data": {
    "sessionId": "session-001",
    "userUuid": "user-001",
    "personaId": "teacher",
    "status": "active",
    "closeReason": "",
    "recentTurnCount": 0,
    "emotion": {
      "valence": 0,
      "arousal": 0,
      "primary": "neutral",
      "sustainedLabel": "neutral"
    },
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

### 3.3 查询 Session

```http
GET /api/session/{sessionId}
```

Response 为 session envelope。

### 3.4 查询情绪状态

```http
GET /api/session/{sessionId}/emotion
```

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "sessionId": "session-001",
  "latencyMs": 1,
  "data": {
    "valence": 0.1,
    "arousal": 0.2,
    "primary": "neutral",
    "sustainedLabel": "neutral"
  }
}
```

### 3.5 查询指标

```http
GET /api/session/{sessionId}/metrics
```

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "sessionId": "session-001",
  "latencyMs": 1,
  "data": {
    "turnCount": 2,
    "requestCount": 2,
    "failedRequestCount": 0,
    "lastLatencyMs": 530,
    "totalLatencyMs": 1020,
    "avgLatencyMs": 510
  }
}
```

### 3.6 关闭 Session

```http
POST /api/session/close
```

Request:

```json
{
  "sessionId": "session-001",
  "reason": "client_close"
}
```

Response 为 session envelope，`data.status` 通常为 `closed`。

前端关闭成功后必须阻止该 session 继续发送消息。

## 4. 主对话接口

### 4.1 发送用户消息

```http
POST /api/chat/message
```

Request:

```json
{
  "sessionId": "session-001",
  "personaId": "teacher",
  "mode": "chat",
  "message": "请解释一下二次函数。",
  "model": "",
  "stream": false
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `sessionId` | string | 是 | 已创建的 session ID |
| `personaId` | string | 否 | 指定响应人格 |
| `mode` | string | 否 | 默认 `chat` |
| `message` | string | 是 | 用户输入 |
| `model` | string | 否 | 覆盖默认 LLM model |
| `stream` | bool | 否 | 当前 HTTP 路由仍返回最终结果，流式优先用 WS 后续扩展 |

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "sessionId": "session-001",
  "latencyMs": 1200,
  "data": {
    "personaId": "teacher",
    "turnIndex": 1,
    "reply": {
      "role": "assistant",
      "content": "二次函数通常写作 y = ax^2 + bx + c..."
    },
    "userEmotion": {
      "primary": "neutral",
      "intensity": 0.2,
      "behavior": "",
      "tone": ""
    },
    "aiEmotion": {
      "primary": "neutral",
      "intensity": 0.1,
      "behavior": "",
      "tone": ""
    },
    "memory": {
      "l0Hit": false,
      "l3Hit": false
    },
    "answerCache": {
      "enabled": true,
      "hit": false,
      "bypassed": false,
      "source": "",
      "cacheKey": "",
      "similarityScore": 0
    },
    "pipelineLatency": {
      "computeQueueWaitMs": 0,
      "computeStageMs": 1,
      "ioQueueWaitMs": 0,
      "ioStageMs": 1000,
      "memoryContextMs": 10,
      "answerCacheMs": 0,
      "promptBuildMs": 2,
      "llmTotalMs": 980,
      "callbackToResponseMs": 1,
      "totalMs": 1200
    }
  }
}
```

前端 UI 最少需要使用：

- `data.reply.content`
- `data.turnIndex`
- `data.userEmotion`
- `data.aiEmotion`
- `data.answerCache.hit`

## 5. Classroom 接口

课堂接口用于多 persona / classroom 场景。普通单人聊天前端可暂不接。

### 5.1 课堂消息

```http
POST /api/classroom/message
```

Request:

```json
{
  "sessionId": "session-001",
  "classroomId": "classroom-001",
  "targetPersonaId": "teacher",
  "contextId": "ctx-001",
  "message": "请继续讲解。",
  "broadcast": false,
  "model": ""
}
```

`targetPersonaId` 可用 `personaId` 兼容替代。

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "sessionId": "session-001",
  "latencyMs": 800,
  "data": {
    "classroomId": "classroom-001",
    "speakerPersonaId": "teacher",
    "content": "我们继续看这个例子。",
    "shouldSpeak": true,
    "turnIndex": 2,
    "userEmotion": {
      "primary": "neutral",
      "intensity": 0.2,
      "behavior": "",
      "tone": ""
    },
    "aiEmotion": {
      "primary": "neutral",
      "intensity": 0.1,
      "behavior": "",
      "tone": ""
    }
  }
}
```

### 5.2 主动发言

```http
POST /api/classroom/proactive
```

Request:

```json
{
  "sessionId": "session-001",
  "classroomId": "classroom-001",
  "personaId": "teacher",
  "contextId": "ctx-001",
  "model": ""
}
```

Response 同 `/api/classroom/message`。

### 5.3 课堂轮询

```http
POST /api/classroom/poll
```

Request:

```json
{
  "classroomId": "classroom-001",
  "personaId": "teacher",
  "contextId": "ctx-001",
  "systemEvent": false,
  "systemEventContent": "",
  "model": ""
}
```

Response 同 `/api/classroom/message`。当 `data.shouldSpeak === false` 时，前端不应追加一条可见 AI 消息。

## 6. 报告接口

### 6.1 训练报告

```http
POST /api/report/training
```

Request:

```json
{
  "sessionId": "session-001",
  "includeRawTurns": true
}
```

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "sessionId": "session-001",
  "latencyMs": 20,
  "data": {
    "generatedAt": "2026-06-24T12:00:00Z",
    "totalTurns": 8,
    "summary": "Session metrics report generated.",
    "evaluation": {},
    "metrics": {
      "turnCount": 8,
      "requestCount": 8,
      "failedRequestCount": 0,
      "lastLatencyMs": 600,
      "totalLatencyMs": 4800,
      "avgLatencyMs": 600
    },
    "schemaVersion": "training_report.v1"
  }
}
```

`evaluation` 是可选领域扩展结果。AgentLoom 基础 Server 未注入 `IReportEvaluator` 时返回空对象并保留 session metrics；下游项目可以注入自己的 evaluator。评估 provider 失败不会让基础报告请求失败，响应会保留 metrics，并通过 `summary`/`evaluation` 表示评估不可用。

## 7. 系统状态接口

### 7.1 Health

```http
GET /api/health
```

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "latencyMs": 1,
  "data": {
    "status": "ok",
    "sessionCount": 1,
    "pools": {
      "compute": {
        "workerCount": 8,
        "queuedTasks": 0,
        "activeWorkers": 0,
        "submittedTasks": 10,
        "completedTasks": 10,
        "failedTasks": 0,
        "rejectedTasks": 0
      },
      "io": {
        "workerCount": 24,
        "queuedTasks": 0,
        "activeWorkers": 0,
        "submittedTasks": 10,
        "completedTasks": 10,
        "failedTasks": 0,
        "rejectedTasks": 0
      }
    }
  }
}
```

### 7.2 System Stats

```http
GET /api/system/stats
```

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "latencyMs": 1,
  "data": {
    "sessionCount": 1,
    "pools": {
      "compute": {},
      "io": {}
    }
  }
}
```

## 8. Skill Session 接口

Skill Session 是后端统一管理外部能力生命周期的接口。当前前端最重要的是 `vision.observe`。

状态值：

```text
idle
starting
ready
running
waiting_input
closing
closed
failed
expired
```

### 8.1 Skill Session Snapshot

```json
{
  "skillId": "vision.observe",
  "sessionId": "session-001",
  "userUuid": "user-001",
  "personaId": "teacher",
  "traceId": "trace-xxx",
  "state": "running",
  "statusText": "vision event stream ready",
  "lastObservation": "画面中检测到明显移动",
  "lastError": "",
  "closeReason": "",
  "maxDurationMs": 120000,
  "recentObservations": [
    {
      "skillId": "vision.observe",
      "sessionId": "session-001",
      "traceId": "trace-xxx",
      "summary": "画面中检测到明显移动",
      "confidence": 0.76,
      "stale": false,
      "shouldInjectPrompt": true,
      "source": "vision_event",
      "metadata": {
        "event_id": "vision-event-1"
      }
    }
  ]
}
```

### 8.2 启动 Skill Session

```http
POST /api/skill/session/start
```

Request:

```json
{
  "skillId": "vision.observe",
  "sessionId": "session-001",
  "personaId": "teacher",
  "source": "frontend",
  "reason": "用户点击开启视觉观察",
  "arguments": {
    "mode": "observe"
  },
  "maxDurationMs": 120000
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `skillId` | string | 是 | 例如 `vision.observe` |
| `sessionId` | string | 是 | 绑定主对话 session |
| `personaId` | string | 否 | 角色 ID |
| `source` | string | 否 | 建议前端填 `frontend` |
| `reason` | string | 否 | 启动原因 |
| `arguments` | object | 否 | Skill 参数，会被序列化保存 |
| `maxDurationMs` | number | 否 | 本次 Skill 最大持续时间 |

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "data": {
    "skillId": "vision.observe",
    "sessionId": "session-001",
    "state": "starting",
    "statusText": "用户点击开启视觉观察",
    "recentObservations": []
  }
}
```

### 8.3 查询 Skill Session

```http
POST /api/skill/session/status
```

Request:

```json
{
  "skillId": "vision.observe",
  "sessionId": "session-001"
}
```

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "data": null
}
```

若存在则 `data` 为 Skill Session Snapshot；若不存在则为 `null`。

### 8.4 停止 Skill Session

```http
POST /api/skill/session/stop
```

Request:

```json
{
  "skillId": "vision.observe",
  "sessionId": "session-001",
  "source": "frontend",
  "reason": "用户关闭视觉观察",
  "summarize": true,
  "writeL3": false
}
```

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "data": {
    "skillId": "vision.observe",
    "sessionId": "session-001",
    "state": "closed",
    "closeReason": "用户关闭视觉观察"
  }
}
```

前端规则：

- 用户点击“开启视觉”时调用 start。
- UI 显示 starting/ready/running/failed/expired 状态。
- 用户点击“关闭视觉”或页面卸载时调用 stop。
- 如果状态为 failed/expired，应提示工具不可用，并停止发送视觉相关信令。

## 9. 文档接口

文档链路支持两种方式：

1. HTTP 直接传服务器可读路径。
2. WebSocket 上传文件，再注册为后端托管文件。

浏览器前端通常应使用 WebSocket 上传；HTTP path 方式更适合本地调试或服务端已有文件。

### 9.1 注册服务器本地文件

```http
POST /api/document/register
```

Request:

```json
{
  "path": "D:/tmp/demo.pptx",
  "fileName": "demo.pptx",
  "sessionId": "session-001"
}
```

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "documentId": "doc-xxx",
  "data": {
    "documentId": "doc-xxx",
    "contentHash": "sha256...",
    "ownerUserUuid": "user-001",
    "sessionId": "session-001",
    "fileName": "demo.pptx",
    "fileType": "pptx",
    "uploadedAtMs": 1782230400000,
    "lastAnalyzedAtMs": 0,
    "lastAccessedAtMs": 1782230400000,
    "analysisStatus": "registered",
    "analysisTraceId": "",
    "sizeBytes": 123456,
    "schemaVersion": "document.metadata.v1"
  }
}
```

### 9.2 解析文档

```http
POST /api/document/analyze
```

Request:

```json
{
  "documentId": "doc-xxx",
  "path": "",
  "fileName": "demo.pptx",
  "enableEmbeddingClustering": true,
  "chunkSimilarityThreshold": 0.86,
  "maxChunkSlices": 64,
  "enableLlmChunkFallback": false,
  "chunkLlmModel": "",
  "chunkLlmMaxTokens": 800
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `documentId` | string | 否 | 已注册/已上传文件的 ID |
| `path` | string | 否 | 服务器本地文件路径 |
| `fileName` | string | 否 | 文件名 |
| `enableEmbeddingClustering` | bool | 否 | 是否启用 embedding 聚类 |
| `chunkSimilarityThreshold` | number | 否 | chunk 相似阈值 |
| `maxChunkSlices` | number | 否 | 最大切片数 |
| `enableLlmChunkFallback` | bool | 否 | 是否允许 LLM chunk fallback |
| `chunkLlmModel` | string | 否 | fallback 使用模型 |
| `chunkLlmMaxTokens` | number | 否 | fallback 最大 token |

Response:

```json
{
  "ok": true,
  "traceId": "trace-xxx",
  "documentId": "doc-xxx",
  "latencyMs": 1200,
  "data": {
    "...": "document analysis result"
  },
  "pipelineLatency": {
    "computeQueueWaitMs": 0,
    "computeStageMs": 1200,
    "totalMs": 1200
  }
}
```

`data` 为文档解析算法输出，前端应按当前 UI 需要做防御式读取。

## 10. WebSocket 主协议

连接：

```text
ws://<host>:<port>/ws/session
wss://<host>:<port>/ws/session
```

所有普通 WS 文本消息使用：

```json
{
  "type": "chat.message",
  "traceId": "trace-xxx",
  "payload": {}
}
```

`traceId` 可省略，后端会生成。

错误帧：

```json
{
  "type": "error",
  "payload": {
    "ok": false,
    "traceId": "trace-xxx",
    "error": {
      "code": "InvalidArgument",
      "message": "..."
    }
  }
}
```

### 10.1 WS Chat

Client:

```json
{
  "type": "chat.message",
  "traceId": "trace-xxx",
  "payload": {
    "sessionId": "session-001",
    "personaId": "teacher",
    "mode": "ws_chat",
    "message": "你好",
    "model": ""
  }
}
```

Server:

```json
{
  "type": "chat.final",
  "payload": {
    "ok": true,
    "traceId": "trace-xxx",
    "sessionId": "session-001",
    "latencyMs": 1000,
    "data": {
      "reply": {
        "role": "assistant",
        "content": "你好，有什么我可以帮你？"
      }
    }
  }
}
```

### 10.2 WS Close Session

Client:

```json
{
  "type": "session.close",
  "payload": {
    "sessionId": "session-001",
    "reason": "client_close"
  }
}
```

Server:

```json
{
  "type": "session.closed",
  "payload": {
    "ok": true,
    "traceId": "trace-xxx",
    "sessionId": "session-001",
    "data": {
      "status": "closed"
    }
  }
}
```

### 10.3 WS Skill Session

Start:

```json
{
  "type": "skill.session.start",
  "payload": {
    "skillId": "vision.observe",
    "sessionId": "session-001",
    "personaId": "teacher",
    "source": "websocket",
    "reason": "用户点击开启视觉观察",
    "arguments": {
      "mode": "observe"
    },
    "maxDurationMs": 120000
  }
}
```

Server:

```json
{
  "type": "skill.session.started",
  "payload": {
    "ok": true,
    "traceId": "trace-xxx",
    "data": {
      "skillId": "vision.observe",
      "sessionId": "session-001",
      "state": "starting"
    }
  }
}
```

Status:

```json
{
  "type": "skill.session.status",
  "payload": {
    "skillId": "vision.observe",
    "sessionId": "session-001"
  }
}
```

Server:

```json
{
  "type": "skill.session.status",
  "payload": {
    "ok": true,
    "traceId": "trace-xxx",
    "data": null
  }
}
```

Stop:

```json
{
  "type": "skill.session.stop",
  "payload": {
    "skillId": "vision.observe",
    "sessionId": "session-001",
    "source": "websocket",
    "reason": "client_stop",
    "summarize": true,
    "writeL3": false
  }
}
```

Server:

```json
{
  "type": "skill.session.stopped",
  "payload": {
    "ok": true,
    "traceId": "trace-xxx",
    "data": {
      "skillId": "vision.observe",
      "sessionId": "session-001",
      "state": "closed"
    }
  }
}
```

### 10.4 WS 文档上传

#### 10.4.1 开始上传

Client:

```json
{
  "type": "document.upload.start",
  "payload": {
    "fileName": "demo.pptx",
    "totalBytes": 123456,
    "sessionId": "session-001",
    "mode": "base64"
  }
}
```

`mode` 可取：

```text
base64
binary
```

当前前端最简单稳定方式是 `base64` 分片。`binary` 模式用于同一 WebSocket 连接上后续发送二进制帧，需确保同一连接同时只有一个 binary upload。

Server:

```json
{
  "type": "document.upload.started",
  "payload": {
    "ok": true,
    "traceId": "trace-xxx",
    "uploadId": "upload-xxx",
    "mode": "base64",
    "receivedBytes": 0,
    "totalBytes": 123456
  }
}
```

#### 10.4.2 Base64 分片上传

Client:

```json
{
  "type": "document.upload.chunk",
  "payload": {
    "uploadId": "upload-xxx",
    "offset": 0,
    "data": "base64..."
  }
}
```

规则：

- `offset` 必须等于服务端已接收字节数。
- `data` 是原始 bytes 的 base64 编码。
- 前端应等待 `chunk_ack` 后再发送下一片，或者自行保证 offset 顺序。

Server:

```json
{
  "type": "document.upload.chunk_ack",
  "payload": {
    "ok": true,
    "traceId": "trace-xxx",
    "uploadId": "upload-xxx",
    "receivedBytes": 65536
  }
}
```

#### 10.4.3 Binary 分片上传

若 start 时 `mode` 或 `encoding` 为 `binary`，后续可发送 WebSocket binary frame。服务端会将该连接上的 binary frame 写入当前 active binary upload，并返回：

```json
{
  "type": "document.upload.chunk_ack",
  "payload": {
    "ok": true,
    "traceId": "trace-xxx",
    "uploadId": "upload-xxx",
    "receivedBytes": 65536,
    "totalBytes": 123456,
    "complete": false
  }
}
```

前端必须保证：

- 同一 WS 连接同一时间只开启一个 binary upload。
- 不发送超过 `totalBytes` 的数据。
- 收到错误后停止继续发送。

#### 10.4.4 完成上传

Client:

```json
{
  "type": "document.upload.finish",
  "payload": {
    "uploadId": "upload-xxx"
  }
}
```

Server:

```json
{
  "type": "document.upload.finished",
  "payload": {
    "ok": true,
    "traceId": "trace-xxx",
    "documentId": "doc-xxx",
    "data": {
      "documentId": "doc-xxx",
      "fileName": "demo.pptx",
      "analysisStatus": "registered"
    }
  }
}
```

完成后前端可调用 `/api/document/analyze`，传入 `documentId`。

#### 10.4.5 取消上传

Client:

```json
{
  "type": "document.upload.abort",
  "payload": {
    "uploadId": "upload-xxx"
  }
}
```

Server:

```json
{
  "type": "document.upload.aborted",
  "payload": {
    "ok": true,
    "traceId": "trace-xxx",
    "uploadId": "upload-xxx"
  }
}
```

## 11. 多模态 / WebRTC 边界

当前主 Gateway 已有 Skill Session 控制接口，可以让前端对齐：

```text
POST /api/skill/session/start
POST /api/skill/session/status
POST /api/skill/session/stop
WS skill.session.start/status/stop
```

`vision.observe` 的建议前端流程：

```text
1. 用户点击开启视觉。
2. 调用 skill.session.start(skillId="vision.observe")。
3. UI 显示 starting。
4. 建立 WebRTC signaling / media 链路。
5. 后端收到视觉事件后，Skill Session 进入 ready/running。
6. 前端定期 status 或等待后续事件推送。
7. 用户关闭视觉时调用 skill.session.stop。
```

注意：

- Skill Session API 是生命周期控制层，不直接承载视频帧。
- WebRTC media/signaling 第一版仍是独立多模态链路，具体 signaling message 以 WebRTC smoke server / media 文档为准。
- 主对话链路不会默认相信 VLM 结果；视觉 observation 会以低置信工具观察进入 Prompt。

相关设计文档：

```text
docs/SKILL_SESSION_PROTOCOL.md
docs/VISUAL_TOOL_SESSION_PROTOCOL.md
docs/VISION_SYSTEM_DESIGN.md
```

### 11.1 当前验证状态

多模态链路当前状态应理解为：

```text
本地单路 MVP 已验证，但不建议默认作为正式 Release 功能开启。
```

已经通过的本地 E2E 能力：

```text
1. 浏览器摄像头授权和 WebRTC signaling 成功。
2. 真实摄像头 30s 本地推流，GStreamer appsink 可收到连续 frame。
3. vision.observe Skill Session 可完成 start -> ready/running -> stop -> closed 生命周期。
```

仍建议在正式开放前继续验证：

```text
1. 刷新页面、关闭 tab、WebSocket close 后资源释放。
2. 重复 start/stop 不泄漏 session 或底层 pipeline。
3. stop 后继续到达的 frame 不应再写入 observation。
4. 断网、弱网、ICE 失败、前端异常退出后的 expired/cleanup。
5. 多路 session 同时推流时 sessionId / connectionId 不串线。
6. 30-60min 长稳压测，确认内存、句柄、GStreamer pipeline 不持续上涨。
```

前端实现建议：

- 可以开始接 `vision.observe` 的 UI、状态条、启动/关闭按钮和错误提示。
- 默认仍应放在 feature flag / dev-only / beta 开关之后。
- UI 不应承诺“助手一定看见画面”，应表达为“视觉观察工具已开启/正在分析/观察结果可能不稳定”。
- 若后端返回 skill session 不可用或 media 能力未编译，前端应隐藏视觉入口或显示不可用状态。

### 11.2 Linux Release 与 media 模块

`llama.cpp` / 本地推理依赖之所以可以较容易做成可选，是因为推理端通过 gRPC 与主进程隔离。media 模块不同：WebRTC、GStreamer、OpenCV、VisionEvent、SkillSession bridge 当前与主链路存在编译期耦合。如果 Linux Target 不具备 media 依赖，简单地关闭运行时开关并不足以保证编译通过。

因此 Linux Release 目前只有三类可行方案：

#### 方案 A：Linux 预处理脚本裁剪 media 源码

在 Linux 预处理 Python 脚本中，对 Linux 目标源码副本删除或替换 media 相关代码：

```text
remove / stub:
  src/media/*
  media tests/tools
  skill_vision_event_sink
  webrtc signaling smoke server
  CMake 中 media target / link
```

同时保留主链路通用接口：

```text
保留:
  SkillSessionManager
  skill.session.start/status/stop
  普通 chat/session/document/cache/memory/emotion/auth

禁用:
  vision.observe 的真实 WebRTC/media backend
```

优点：

- Linux Release 最轻。
- 不需要在比赛/交付环境解决 GStreamer/OpenCV 依赖。
- 主链路构建风险最低。

缺点：

- Linux Release 不支持视觉 Beta 功能。
- 预处理脚本会引入一套“源码副本裁剪规则”，需要维护，且容易和主分支代码演进发生偏差。
- 前端必须能识别该 Release 不具备 media capability。

#### 方案 B：Release 分支只保留完备功能

维护一个稳定 Release 分支，只包含已经充分测试的功能；开发主分支继续保留完整 Windows 本地测试功能和 Beta media 链路。media 经过足够 E2E、长稳和 Linux 依赖验证后再合入 Release 分支。

优点：

- 最符合工程发布习惯。
- Release 分支构建、测试、交付边界清晰。
- Beta 功能不会污染交付稳定性。

缺点：

- 需要维护分支同步。
- 需要明确哪些文档、配置、前端开关属于 Release，哪些属于开发分支。
- 如果比赛压缩包不是 Git 仓库，需要人工筛选 Release 分支内容。

#### 方案 C：准备 Linux GStreamer/OpenCV 依赖

直接为 Linux Target 准备 GStreamer、OpenCV、WebRTC 相关依赖，并让 media 模块在 Linux Release 中也参与编译。

优点：

- 功能最完整，Linux Release 与 Windows 开发功能一致。
- GStreamer/OpenCV 在 Linux 生态通常更成熟，长期看这是不可避免的正式方案。
- 后续公网、容器、多机器部署会更接近真实生产路径。

缺点：

- 容器镜像和依赖体积明显变大。
- CI/构建脚本复杂度上升。
- 需要额外验证 GStreamer plugin、OpenCV ABI、vcpkg/system package 混用、运行时动态库搜索路径。
- 当前 media 链路仍属 Beta，直接进入 Release 会放大交付风险。

### 11.3 当前建议

当前阶段建议采用：

```text
短期 Release:
  优先方案 A 或 B，Linux Release 去除或禁用真实 media backend。

开发 / Windows 本地测试:
  保留完整 media 模块，继续压测 WebRTC/GStreamer/OpenCV/VLM 链路。

中长期正式多模态:
  走方案 C，补齐 Linux GStreamer/OpenCV 依赖和容器化验证。
```

前端协议层不需要删除 Skill Session API。它是通用能力控制协议，不只服务于 media。但前端必须把 `vision.observe` 视为 capability-dependent 功能：只有后端明确启用并通过 health/config 暴露能力时才展示真实视觉入口。

## 12. 前端推荐调用顺序

### 12.1 普通聊天

```text
GET  /api/auth/me
若未登录:
  POST /api/auth/register
POST /api/session/create
POST /api/chat/message
GET  /api/session/{sessionId}/emotion
POST /api/session/close
```

### 12.2 WebSocket 聊天

```text
GET  /api/auth/me
POST /api/session/create
WS connect /ws/session
WS send chat.message
WS receive chat.final
WS send session.close
```

### 12.3 文档解析

```text
POST /api/session/create
WS document.upload.start
WS document.upload.chunk repeated
WS document.upload.finish -> documentId
POST /api/document/analyze(documentId)
```

### 12.4 视觉 Skill

```text
POST /api/session/create
POST /api/skill/session/start(skillId="vision.observe")
建立 WebRTC signaling/media
POST /api/skill/session/status
POST /api/chat/message
POST /api/skill/session/stop
```

## 13. 前端实现注意事项

- 所有 fetch 建议带 `credentials: "include"`。
- 所有 API 错误都先解析 JSON envelope，再显示 UI 状态。
- 用户关闭页面、切换 session、关闭摄像头时应调用 session/skill stop。
- 对 `data` 下的复杂字段做防御式读取，尤其是文档解析结果和未来视觉 observation。
- `sessionId` 是前端状态管理的主键，Skill session 以 `sessionId + skillId` 为主键。
- 不要把未确认的视觉 observation 当作长期事实展示。
- 文档上传大文件建议分片并等待 ack，避免浏览器内存峰值过高。
- 前端 UI 不应依赖 `latencyMs` 必然存在；它是诊断字段。

## 14. 当前接口清单

HTTP:

```text
GET  /api/health
GET  /api/system/stats

POST /api/auth/register
GET  /api/auth/me

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

POST /api/skill/session/start
POST /api/skill/session/status
POST /api/skill/session/stop

POST /api/document/register
POST /api/document/analyze
```

WebSocket text messages:

```text
chat.message
session.close

skill.session.start
skill.session.status
skill.session.stop

document.upload.start
document.upload.chunk
document.upload.finish
document.upload.abort
```

WebSocket server message types:

```text
chat.final
session.closed

skill.session.started
skill.session.status
skill.session.stopped

document.upload.started
document.upload.chunk_ack
document.upload.finished
document.upload.aborted

error
```
