"""Run VLM inference over a directory of extracted video frames.

Example:
    python tools/vlm_frame_dir_quality_test.py --pid 12345 --frames-dir D:\\tmp\\frames --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
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
    "请用中文客观描述这一帧中可见的主要内容。"
    "只描述画面事实，不要猜测地点、身份或不可见信息。"
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


def list_frames(frames_dir: Path, limit: int) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    frames = sorted(p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if limit > 0:
        frames = frames[:limit]
    return frames


def call_frame(
    stub: pb2_grpc.MultimodalInferenceStub,
    args: argparse.Namespace,
    frame: Path,
    index: int,
) -> dict[str, Any]:
    request = pb2.VLMRequest(
        image_data=frame.read_bytes(),
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
    started = time.perf_counter()
    try:
        response = stub.GenerateVLMSync(request, timeout=args.timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "index": index,
            "frame": str(frame),
            "ok": not bool(response.error),
            "elapsed_ms": elapsed_ms,
            "error": response.error,
            "text": response.text,
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
            "frame": str(frame),
            "ok": False,
            "elapsed_ms": elapsed_ms,
            "error": f"{exc.code().name}: {exc.details()}",
            "text": "",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--pid", type=int, help="multimodal_inference_server process id")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--session-id", default="vlm-frame-dir-quality")
    parser.add_argument("--request-prefix", default="frame-quality")
    parser.add_argument("--task-type", default="frame_dir_quality")
    parser.add_argument("--saliency-hint", type=float, default=1.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--sleep-ms", type=float, default=0.0)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()

    if not args.frames_dir.exists():
        raise SystemExit(f"frames dir not found: {args.frames_dir}")
    frames = list_frames(args.frames_dir, args.limit)
    if not frames:
        raise SystemExit(f"no frames found in: {args.frames_dir}")

    process = psutil.Process(args.pid) if args.pid else None
    channel = grpc.insecure_channel(
        args.target,
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ],
    )
    grpc.channel_ready_future(channel).result(timeout=args.timeout)
    stub = pb2_grpc.MultimodalInferenceStub(channel)

    fieldnames = [
        "index", "frame", "ok", "elapsed_ms", "error", "text", "text_len",
        "image_encode_ms", "prompt_eval_ms", "eval_ms",
        "prompt_tokens", "generated_tokens", "cache_hit", "cache_stale",
        "result_source", "rss_mb", "vms_mb", "working_set_mb", "private_mb",
    ]
    csv_file = args.csv.open("w", newline="", encoding="utf-8") if args.csv else None
    jsonl_file = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore") if csv_file else None
    if writer:
        writer.writeheader()

    results: list[dict[str, Any]] = []
    try:
        print(json.dumps({"phase": "before", "frames": len(frames), **memory_sample(process)}, ensure_ascii=False), flush=True)
        started = time.perf_counter()
        for index, frame in enumerate(frames):
            row = call_frame(stub, args, frame, index)
            row.update(memory_sample(process))
            results.append(row)
            if writer:
                writer.writerow(row)
            if jsonl_file:
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

        total_ms = (time.perf_counter() - started) * 1000.0
        ok = [r for r in results if r["ok"]]
        elapsed = [float(r["elapsed_ms"]) for r in ok]
        prompt_eval = [float(r["prompt_eval_ms"]) for r in ok]
        rss = [float(r["rss_mb"]) for r in results if "rss_mb" in r]
        summary = {
            "phase": "summary",
            "total": len(results),
            "ok": len(ok),
            "errors": len(results) - len(ok),
            "total_ms": total_ms,
            "qps": len(results) / max(total_ms / 1000.0, 1.0e-9),
            "latency_p50_ms": percentile(elapsed, 0.50),
            "latency_p90_ms": percentile(elapsed, 0.90),
            "latency_mean_ms": statistics.fmean(elapsed) if elapsed else 0.0,
            "prompt_eval_mean_ms": statistics.fmean(prompt_eval) if prompt_eval else 0.0,
            "prompt_eval_p50_ms": percentile(prompt_eval, 0.50),
            "rss_first_mb": rss[0] if rss else 0.0,
            "rss_last_mb": rss[-1] if rss else 0.0,
            "rss_max_mb": max(rss) if rss else 0.0,
        }
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0 if summary["errors"] == 0 else 1
    finally:
        if csv_file:
            csv_file.close()
        if jsonl_file:
            jsonl_file.close()
        channel.close()


if __name__ == "__main__":
    raise SystemExit(main())
