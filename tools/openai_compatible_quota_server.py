from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from typing import Any

from aiohttp import web


class QuotaState:
    def __init__(self, delay_ms: int, max_inflight: int) -> None:
        self.delay_ms = max(0, delay_ms)
        self.max_inflight = max(1, max_inflight)
        self._semaphore = asyncio.Semaphore(self.max_inflight)
        self.inflight = 0
        self.peak_inflight = 0
        self.queued = 0
        self.peak_queued = 0
        self.completed = 0
        self.rejected = 0

    async def acquire(self) -> None:
        self.queued += 1
        self.peak_queued = max(self.peak_queued, self.queued)
        await self._semaphore.acquire()
        self.queued -= 1
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)

    def release(self) -> None:
        self.inflight -= 1
        self.completed += 1
        self._semaphore.release()

    def snapshot(self) -> dict[str, int | bool]:
        return {
            "ok": True,
            "delayMs": self.delay_ms,
            "maxInflight": self.max_inflight,
            "inflight": self.inflight,
            "peakInflight": self.peak_inflight,
            "queued": self.queued,
            "peakQueued": self.peak_queued,
            "completed": self.completed,
            "rejected": self.rejected,
        }


async def health(request: web.Request) -> web.Response:
    state: QuotaState = request.app["state"]
    return web.json_response(state.snapshot())


async def complete(request: web.Request) -> web.Response:
    state: QuotaState = request.app["state"]
    body: dict[str, Any] = await request.json()
    await state.acquire()
    try:
        await asyncio.sleep(state.delay_ms / 1000)
        model = str(body.get("model") or "mock-enterprise-chat")
        return web.json_response(
            {
                "id": f"mock-enterprise-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Enterprise quota mock response after {state.delay_ms} ms.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 16,
                    "completion_tokens": 8,
                    "total_tokens": 24,
                },
            }
        )
    finally:
        state.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--delay-ms", type=int, default=2000)
    parser.add_argument("--max-inflight", type=int, default=500)
    args = parser.parse_args()

    app = web.Application(client_max_size=1024 * 1024)
    app["state"] = QuotaState(args.delay_ms, args.max_inflight)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", health)
    app.router.add_post("/v1/chat/completions", complete)
    web.run_app(
        app,
        host=args.host,
        port=args.port,
        backlog=1024,
        access_log=None,
        print=lambda message: print(f"[openai-quota-mock] {message}", flush=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
