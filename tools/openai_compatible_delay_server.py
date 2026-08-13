from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class DelayState:
    def __init__(self, delay_ms: int) -> None:
        self._delay_ms = delay_ms
        self._lock = threading.Lock()
        self._inflight = 0
        self._peak_inflight = 0
        self._completed = 0

    def set_delay(self, delay_ms: int) -> None:
        with self._lock:
            self._delay_ms = max(0, delay_ms)

    def begin(self) -> int:
        with self._lock:
            self._inflight += 1
            self._peak_inflight = max(self._peak_inflight, self._inflight)
            return self._delay_ms

    def finish(self) -> None:
        with self._lock:
            self._inflight -= 1
            self._completed += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "delayMs": self._delay_ms,
                "inflight": self._inflight,
                "peakInflight": self._peak_inflight,
                "completed": self._completed,
            }

    def reset_metrics(self) -> None:
        with self._lock:
            self._peak_inflight = self._inflight
            self._completed = 0


class Handler(BaseHTTPRequestHandler):
    server: "DelayServer"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path in {"/health", "/metrics"}:
            self._send_json(200, {"ok": True, **self.server.state.snapshot()})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/control":
            body = self._json_body()
            if "delayMs" in body:
                self.server.state.set_delay(int(body["delayMs"]))
            if body.get("resetMetrics"):
                self.server.state.reset_metrics()
            self._send_json(200, {"ok": True, **self.server.state.snapshot()})
            return
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return

        request = self._json_body()
        delay_ms = self.server.state.begin()
        try:
            time.sleep(delay_ms / 1000)
            model = str(request.get("model") or "mock-chat")
            self._send_json(
                200,
                {
                    "id": f"mock-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"Mock response after {delay_ms} ms.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 16,
                        "completion_tokens": 8,
                        "total_tokens": 24,
                    },
                },
            )
        finally:
            self.server.state.finish()


class DelayServer(ThreadingHTTPServer):
    daemon_threads = True
    # 默认 socketserver backlog 只有 5，无法作为 100 并发 Gateway outbound 的稳定负载端。
    request_queue_size = 256

    def __init__(self, address: tuple[str, int], state: DelayState) -> None:
        super().__init__(address, Handler)
        self.state = state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--delay-ms", type=int, default=100)
    args = parser.parse_args()

    server = DelayServer((args.host, args.port), DelayState(args.delay_ms))
    print(f"[openai-mock] listening on http://{args.host}:{args.port}, delay={args.delay_ms} ms", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
