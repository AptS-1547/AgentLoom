"""Stress GenerateVLMSync and record inference-server memory usage.

Requires:
    pip install grpcio psutil

Example:
    python tools/vlm_grpc_memory_stress.py --pid 12345 --image frame.jpg --iterations 100
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import grpc
import psutil

_ROOT = Path(__file__).resolve().parents[1]
_E2E_VISUAL = _ROOT / "tools" / "e2e_visual"
sys.path.insert(0, str(_E2E_VISUAL))

import multimodal_inference_pb2 as pb2  # noqa: E402
import multimodal_inference_pb2_grpc as pb2_grpc  # noqa: E402


DEFAULT_PROMPT = (
    "请用中文简要描述这张图里最显著的可见内容：人物、动作、物体、场景。"
    "直接用自然语言描述，不用 JSON。控制在一两句话。"
)


def mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * (len(ordered) - 1)))
    return ordered[index]


def memory_sample(process: psutil.Process | None) -> dict[str, float]:
    if process is None:
        return {}
    try:
        with process.oneshot():
            info = process.memory_info()
            sample = {
                "rss_mb": mib(getattr(info, "rss", 0)),
                "vms_mb": mib(getattr(info, "vms", 0)),
            }
            if hasattr(info, "wset"):
                sample["working_set_mb"] = mib(info.wset)
            if hasattr(info, "private"):
                sample["private_mb"] = mib(info.private)
            return sample
    except psutil.NoSuchProcess:
        return {"process_exited": True}


def make_request(args: argparse.Namespace, image_data: bytes, index: int) -> pb2.VLMRequest:
    request = pb2.VLMRequest(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        context_size=args.context_size,
        session_id=args.session_id,
        request_id=f"{args.request_prefix}-{index}",
        task_type=args.task_type,
        allow_cache=not args.no_cache,
        force_refresh=args.force_refresh,
        allow_stale_cache=False,
        saliency_hint=args.saliency_hint,
    )
    if args.image_path_mode:
        request.image_path = str(args.image)
    else:
        request.image_data = image_data
    return request


def call_once(
    stub: pb2_grpc.MultimodalInferenceStub,
    args: argparse.Namespace,
    image_data: bytes,
    index: int,
) -> dict[str, Any]:
    request = make_request(args, image_data, index)
    started = time.perf_counter()
    try:
        response = stub.GenerateVLMSync(request, timeout=args.timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "index": index,
            "ok": not bool(response.error),
            "elapsed_ms": elapsed_ms,
            "error": response.error,
            "text_len": len(response.text or ""),
            "image_encode_ms": response.image_encode_ms,
            "prompt_eval_ms": response.prompt_eval_ms,
            "eval_ms": response.eval_ms,
            "prompt_tokens": response.prompt_tokens,
            "generated_tokens": response.generated_tokens,
            "cache_hit": response.cache_hit,
            "cache_stale": response.cache_stale,
            "result_source": response.result_source or "",
        }
    except grpc.RpcError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "index": index,
            "ok": False,
            "elapsed_ms": elapsed_ms,
            "error": f"{exc.code().name}: {exc.details()}",
            "text_len": 0,
            "image_encode_ms": 0.0,
            "prompt_eval_ms": 0.0,
            "eval_ms": 0.0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "cache_hit": False,
            "cache_stale": False,
            "result_source": "rpc_error",
        }


def write_row(writer: csv.DictWriter | None, row: dict[str, Any]) -> None:
    if writer is not None:
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--pid", type=int, help="multimodal_inference_server process id")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-path-mode", action="store_true", help="send image_path instead of image_data")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sleep-ms", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--session-id", default="vlm-memory-stress")
    parser.add_argument("--request-prefix", default="vlm-stress")
    parser.add_argument("--task-type", default="memory_stress")
    parser.add_argument("--saliency-hint", type=float, default=1.0)
    parser.add_argument("--no-cache", action="store_true", help="set allow_cache=false")
    parser.add_argument("--force-refresh", action="store_true", help="force fresh inference")
    parser.add_argument("--csv", type=Path, help="write per-request samples to CSV")
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")
    image_data = b"" if args.image_path_mode else args.image.read_bytes()
    process = psutil.Process(args.pid) if args.pid else None

    channel = grpc.insecure_channel(
        args.target,
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ],
    )
    stub = pb2_grpc.MultimodalInferenceStub(channel)
    grpc.channel_ready_future(channel).result(timeout=args.timeout)

    fieldnames = [
        "index", "ok", "elapsed_ms", "error", "text_len",
        "image_encode_ms", "prompt_eval_ms", "eval_ms",
        "prompt_tokens", "generated_tokens", "cache_hit", "cache_stale", "result_source",
        "rss_mb", "vms_mb", "working_set_mb", "private_mb",
    ]
    csv_file = args.csv.open("w", newline="", encoding="utf-8") if args.csv else None
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore") if csv_file else None
    if writer:
        writer.writeheader()

    try:
        print(json.dumps({"phase": "before", **memory_sample(process)}, ensure_ascii=False), flush=True)
        for i in range(args.warmup):
            row = call_once(stub, args, image_data, -args.warmup + i)
            print(json.dumps({"phase": "warmup", **row, **memory_sample(process)}, ensure_ascii=False), flush=True)

        results: list[dict[str, Any]] = []
        started = time.perf_counter()
        if args.concurrency <= 1:
            for i in range(args.iterations):
                row = call_once(stub, args, image_data, i)
                row.update(memory_sample(process))
                results.append(row)
                write_row(writer, row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if args.sleep_ms > 0:
                    time.sleep(args.sleep_ms / 1000.0)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [executor.submit(call_once, stub, args, image_data, i) for i in range(args.iterations)]
                for future in as_completed(futures):
                    row = future.result()
                    row.update(memory_sample(process))
                    results.append(row)
                    write_row(writer, row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)

        total_ms = (time.perf_counter() - started) * 1000.0
        ok = [r for r in results if r["ok"]]
        elapsed = [float(r["elapsed_ms"]) for r in ok]
        rss_values = [float(r["rss_mb"]) for r in results if "rss_mb" in r]
        private_values = [float(r["private_mb"]) for r in results if "private_mb" in r]
        summary = {
            "phase": "summary",
            "total": len(results),
            "ok": len(ok),
            "errors": len(results) - len(ok),
            "total_ms": total_ms,
            "qps": len(results) / max(total_ms / 1000.0, 1.0e-9),
            "latency_p50_ms": percentile(elapsed, 0.50),
            "latency_p90_ms": percentile(elapsed, 0.90),
            "latency_p99_ms": percentile(elapsed, 0.99),
            "latency_mean_ms": statistics.fmean(elapsed) if elapsed else 0.0,
            "rss_first_mb": rss_values[0] if rss_values else 0.0,
            "rss_last_mb": rss_values[-1] if rss_values else 0.0,
            "rss_max_mb": max(rss_values) if rss_values else 0.0,
            "private_first_mb": private_values[0] if private_values else 0.0,
            "private_last_mb": private_values[-1] if private_values else 0.0,
            "private_max_mb": max(private_values) if private_values else 0.0,
        }
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0 if summary["errors"] == 0 else 1
    finally:
        if csv_file:
            csv_file.close()
        channel.close()


if __name__ == "__main__":
    raise SystemExit(main())
