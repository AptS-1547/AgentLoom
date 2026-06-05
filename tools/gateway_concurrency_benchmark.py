from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
import psutil


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:18080"
PERSONA_REGISTRY = ROOT / "data" / "persona_gateway_e2e" / "persona_registry.json"
REPORT_DIR = ROOT / "data" / "persona_gateway_e2e" / "reports"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "avg": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "avg": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def find_gateway_process() -> psutil.Process | None:
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            if (process.info.get("name") or "").lower() == "persona_gateway_e2e_server.exe":
                return process
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


async def sample_memory(stop: asyncio.Event, interval_s: float) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    process = find_gateway_process()
    if not process:
        return samples

    started = time.perf_counter()
    while not stop.is_set():
        try:
            mem = process.memory_info()
            samples.append(
                {
                    "tSec": time.perf_counter() - started,
                    "rssMb": mem.rss / 1024 / 1024,
                    "privateMb": getattr(mem, "private", 0) / 1024 / 1024,
                    "vmsMb": mem.vms / 1024 / 1024,
                    "numThreads": process.num_threads(),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        await asyncio.sleep(interval_s)
    return samples


def load_persona(persona_id: str) -> dict[str, Any]:
    with PERSONA_REGISTRY.open("r", encoding="utf-8") as file:
        registry = json.load(file)
    for persona in registry.get("personas", []):
        if persona.get("personaId") == persona_id:
            return persona
    raise RuntimeError(f"persona not found: {persona_id}")


async def post_json(client: httpx.AsyncClient, path: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = await client.post(f"{BASE_URL}{path}", json=payload, headers=headers)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        raise RuntimeError(f"POST {path} failed {response.status_code}: {response.text}")
    return response.json(), elapsed_ms


async def get_json(client: httpx.AsyncClient, path: str, headers: dict[str, str] | None = None) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = await client.get(f"{BASE_URL}{path}", headers=headers)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        raise RuntimeError(f"GET {path} failed {response.status_code}: {response.text}")
    return response.json(), elapsed_ms


async def run_virtual_user(
    client: httpx.AsyncClient,
    user_index: int,
    turns: int,
    persona: dict[str, Any],
    think_ms: int,
    start_delay_s: float = 0.0,
) -> list[dict[str, Any]]:
    if start_delay_s > 0:
        await asyncio.sleep(start_delay_s)

    now = int(time.time() * 1000)
    user_uuid = f"bench-user-{now}-{user_index}"
    session_id = f"bench-session-{now}-{user_index}"

    records: list[dict[str, Any]] = []

    def error_record(kind: str, error: Exception, turn: int | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": kind,
            "user": user_index,
            "ok": False,
            "errorType": type(error).__name__,
            "error": str(error),
        }
        if turn is not None:
            record["turn"] = turn
        return record

    try:
        register, register_ms = await post_json(
            client,
            "/api/auth/register",
            {
                "userUuid": user_uuid,
                "tenantId": "default",
                "subject": user_uuid,
                "ttlSeconds": 3600,
            },
        )
        token = register["data"]["token"]
        headers = {"Authorization": f"Bearer {token}"}
        records.append({"kind": "register", "user": user_index, "ok": True, "latencyMs": register_ms})

        _, me_ms = await get_json(client, "/api/auth/me", headers=headers)
        records.append({"kind": "authMe", "user": user_index, "ok": True, "latencyMs": me_ms})

        _, create_ms = await post_json(
            client,
            "/api/session/create",
            {
                "traceId": f"bench-session-{user_index}",
                "sessionId": session_id,
                "personaId": persona["personaId"],
                "personality": persona["personality"],
            },
            headers=headers,
        )
        records.append({"kind": "createSession", "user": user_index, "ok": True, "latencyMs": create_ms})
    except Exception as error:
        records.append(error_record("setup", error))
        return records

    prompts = [
        "李大志，老师刚讲完一元二次方程配方法，你现在听懂了吗？",
        "刚才你说后面没跟上，那你能说说卡在哪一步吗？",
        "那我们再看一次配方法，你觉得第一步应该先做什么？",
        "如果我把题目拆成更小的步骤，你愿意试着回答吗？",
    ]

    for turn in range(turns):
        message = prompts[turn % len(prompts)]
        try:
            body, chat_ms = await post_json(
                client,
                "/api/chat/message",
                {
                    "traceId": f"bench-chat-{user_index}-{turn}",
                    "sessionId": session_id,
                    "personaId": persona["personaId"],
                    "mode": "chat",
                    "message": message,
                    "stream": False,
                },
                headers=headers,
            )
        except Exception as error:
            records.append(error_record("chat", error, turn))
            continue
        data = body.get("data", {})
        pipeline = data.get("pipelineLatency", {})
        memory = data.get("memory", {})
        records.append(
            {
                "kind": "chat",
                "user": user_index,
                "ok": True,
                "turn": turn,
                "latencyMs": chat_ms,
                "backendTotalMs": pipeline.get("totalMs", body.get("latencyMs", 0)),
                "computeQueueWaitMs": pipeline.get("computeQueueWaitMs", 0),
                "computeStageMs": pipeline.get("computeStageMs", 0),
                "ioQueueWaitMs": pipeline.get("ioQueueWaitMs", 0),
                "ioStageMs": pipeline.get("ioStageMs", 0),
                "memoryContextMs": pipeline.get("memoryContextMs", 0),
                "promptBuildMs": pipeline.get("promptBuildMs", 0),
                "llmTotalMs": pipeline.get("llmTotalMs", 0),
                "answerCacheMs": pipeline.get("answerCacheMs", 0),
                "callbackToResponseMs": pipeline.get("callbackToResponseMs", 0),
                "l0Hit": memory.get("l0Hit", False),
                "l3Hit": memory.get("l3Hit", False),
                "replyPreview": str(data.get("reply", {}).get("content", ""))[:100],
            }
        )
        if think_ms > 0:
            await asyncio.sleep(think_ms / 1000)

    try:
        _, close_ms = await post_json(
            client,
            "/api/session/close",
            {
                "traceId": f"bench-close-{user_index}",
                "sessionId": session_id,
                "reason": "benchmark_complete",
            },
            headers=headers,
        )
        records.append({"kind": "closeSession", "user": user_index, "ok": True, "latencyMs": close_ms})
    except Exception as error:
        records.append(error_record("closeSession", error))
    return records


async def run(args: argparse.Namespace) -> dict[str, Any]:
    persona = load_persona(args.persona)
    stop_memory = asyncio.Event()
    memory_task = asyncio.create_task(sample_memory(stop_memory, args.memory_interval))
    started = time.perf_counter()

    limits = httpx.Limits(max_connections=max(args.concurrency * 4, 16), max_keepalive_connections=max(args.concurrency * 2, 8))
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = []
        for user in range(args.concurrency):
            if args.ramp_up_seconds > 0 and args.concurrency > 1:
                start_delay_s = args.ramp_up_seconds * user / (args.concurrency - 1)
            else:
                start_delay_s = 0.0
            tasks.append(run_virtual_user(client, user, args.turns, persona, args.think_ms, start_delay_s))
        nested = await asyncio.gather(*tasks)

    elapsed_s = time.perf_counter() - started
    stop_memory.set()
    memory_samples = await memory_task
    records = [record for user_records in nested for record in user_records]
    ok_records = [record for record in records if record.get("ok", True)]
    error_records = [record for record in records if not record.get("ok", True)]
    chat_records = [record for record in ok_records if record["kind"] == "chat"]

    chat_latency = [float(record["latencyMs"]) for record in chat_records]
    backend_latency = [float(record.get("backendTotalMs", 0)) for record in chat_records]
    memory_latency = [float(record.get("memoryContextMs", 0)) for record in chat_records]
    llm_latency = [float(record.get("llmTotalMs", 0)) for record in chat_records]
    compute_queue_latency = [float(record.get("computeQueueWaitMs", 0)) for record in chat_records]
    compute_stage_latency = [float(record.get("computeStageMs", 0)) for record in chat_records]
    io_queue_latency = [float(record.get("ioQueueWaitMs", 0)) for record in chat_records]
    io_stage_latency = [float(record.get("ioStageMs", 0)) for record in chat_records]
    callback_latency = [float(record.get("callbackToResponseMs", 0)) for record in chat_records]
    all_latency = [float(record["latencyMs"]) for record in ok_records if "latencyMs" in record]

    report = {
        "ok": not error_records,
        "baseUrl": BASE_URL,
        "personaId": args.persona,
        "concurrency": args.concurrency,
        "turnsPerUser": args.turns,
        "rampUpSeconds": args.ramp_up_seconds,
        "totalRequests": len(records),
        "successfulRequests": len(ok_records),
        "errorCount": len(error_records),
        "totalChatRequests": len(chat_records),
        "elapsedSec": elapsed_s,
        "throughputReqPerSec": len(records) / elapsed_s if elapsed_s > 0 else 0,
        "throughputChatPerSec": len(chat_records) / elapsed_s if elapsed_s > 0 else 0,
        "latencyMs": {
            "allRequests": summarize(all_latency),
            "chatEndToEnd": summarize(chat_latency),
            "backendTotal": summarize(backend_latency),
            "computeQueueWait": summarize(compute_queue_latency),
            "computeStage": summarize(compute_stage_latency),
            "ioQueueWait": summarize(io_queue_latency),
            "ioStage": summarize(io_stage_latency),
            "memoryContext": summarize(memory_latency),
            "llmTotal": summarize(llm_latency),
            "callbackToResponse": summarize(callback_latency),
        },
        "memory": {
            "samples": memory_samples,
            "rssMb": summarize([sample["rssMb"] for sample in memory_samples]),
            "privateMb": summarize([sample["privateMb"] for sample in memory_samples]),
            "vmsMb": summarize([sample["vmsMb"] for sample in memory_samples]),
            "peakRssMb": max((sample["rssMb"] for sample in memory_samples), default=0),
            "peakPrivateMb": max((sample["privateMb"] for sample in memory_samples), default=0),
        },
        "l0": {
            "hits": sum(1 for record in chat_records if record.get("l0Hit")),
            "misses": sum(1 for record in chat_records if not record.get("l0Hit")),
        },
        "errors": error_records,
        "records": records if args.include_records else [],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--persona", default="lidazhi")
    parser.add_argument("--think-ms", type=int, default=0)
    parser.add_argument("--ramp-up-seconds", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--memory-interval", type=float, default=0.5)
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = asyncio.run(run(args))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else REPORT_DIR / f"gateway_benchmark_{int(time.time())}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {k: v for k, v in report.items() if k not in {"records"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[benchmark] report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
