# Net API Notes

This document records the intended shape of the lightweight `src/net` wrapper.
It is not a general-purpose web framework. The goal is to keep Boost.Beast /
Boost.Asio usage consistent across this repository while preserving access to
Beast request and response types when protocol details matter.

## HTTP Runtime

`HttpServer` owns the accept loop, connection accounting, request parsing,
optional request filtering, access control, route dispatch, static files, and
WebSocket upgrade handoff.

The HTTP request path is:

```text
accept
  -> Beast HTTP parse
  -> request_body_limit
  -> optional HttpRequestFilter
  -> optional AccessController
  -> WebSocket upgrade if the target has a WebSocket handler
  -> typed IHttpRequest handler
  -> generator-style HTTP handler
  -> static file handler
  -> 404
```

`SetHttpRequestHandler` and `SetHttpHandler` are alternatives. Prefer
`SetHttpRequestHandler` when the handler needs connection context, memory pool
access, delayed/asynchronous response completion, or explicit close behavior.
Prefer `SetHttpHandler` for simple request-response endpoints.

Typed request handlers must call `Respond()` once or explicitly `Close()` the
request. The server intentionally does not synthesize a fallback response after
a typed handler takes ownership.

## Request Filtering

`HttpRequestFilter` is an optional early guardrail enabled through
`HttpServerOptions::request_filter.enabled`.

It performs cheap checks on the parsed request:

- request target length
- per-field and total header size
- control characters in target/header values
- obvious suspicious patterns in target, headers, and body

The pattern rules are deliberately heuristic. They are meant to reject commodity
scanner traffic and clearly malformed requests before business handlers run.
They are not a complete WAF, SQL parser, or authorization layer. Product-level
policy should still live in the access controller or a fronting gateway.

## WebSocket Runtime

`HttpServer` only owns the upgrade decision. After a request is upgraded, the
connection is handled by the dedicated WebSocket session implementation.

`SetWebSocketHandler` receives `WebSocketMessage` values. A message can represent
one streamed fragment rather than a whole logical client message. Check
`final_fragment` to determine whether the client message has completed.

For typed WebSocket handling, `SetWebSocketStreamHandler` provides an
`IWebSocketStreamRequest` with the same connection and memory-pool access pattern
as typed HTTP requests.

Oversized input is recoverable. If a client message exceeds
`WebSocketOptions::max_message_bytes`, the handler receives a `WebSocketMessage`
where `ok()` is false and `status.code()` is `ResourceExhausted`. The session
drains the rest of that client message and keeps the connection open for the next
message. Internal errors, protocol errors, remote close, and explicit `Close()`
still close the connection.

Outbound WebSocket writes are serialized internally. `Send()` copies the payload
into session-owned memory before enqueueing the frame, so callers may use stack
or short-lived pools to build the input frame safely. This favors async lifetime
safety over zero-copy output. Large server-to-client media streaming should add
an owned-frame or session-allocation API instead of repeatedly copying large
payloads through `Send()`.

`Send()` can return `ResourceExhausted` when outbound queue limits are reached.

## Ownership

Payloads use `SharedBuffer` so large HTTP/WebSocket paths can avoid repeated
string copies. Handlers should treat received buffers as request-scoped data and
copy only when data must outlive the request object or session callback.

`ConnectionPool` tracks HTTP and WebSocket leases. HTTP leases are upgraded to
WebSocket leases during a successful upgrade, so per-protocol limits still apply
without double-counting the same socket.
