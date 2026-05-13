"""
End-to-end visual perception test using local gRPC VLM server.

Uses MyNeuroLikeSystem's VisualPerceptionPipeline with a gRPC-backed
analyzer shim that replaces the cloud LLM.

Usage:
    python run_e2e_visual_test.py --source <video_path> [options]

Requires:
    - MyNeuroLikeSystem venv (grpcio, opencv-python)
    - multimodal_inference_server running on localhost:50051
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import io
import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── gRPC stubs (generated in this directory) ──
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import multimodal_inference_pb2 as pb2
import multimodal_inference_pb2_grpc as pb2_grpc
import grpc

# ── MyNeuroLikeSystem imports ──
_NEURO_ROOT = Path("D:/Users/21405/source/repos/MyNeuroLikeSystem/project_src")
sys.path.insert(0, str(_NEURO_ROOT))

from src.vision import (
    CV2_AVAILABLE,
    VisualPerceptionConfig,
    VisualPerceptionPipeline,
    derive_visual_emotion_signal,
    visual_event_to_agent_text,
)
from src.vision.visual_types import VisualAnalysis, VisualEvent


_RELAXED_PROMPT = (
    "请用中文简要描述这张图里最显著的可见内容：人物、动作、物体、场景。"
    "直接用自然语言描述，不用 JSON 结构，不要说'画面没变化''unchanged'之类的话，"
    "即使画面比较静止，也请描述你看到的东西。"
    "控制在一两句话，总长不超过 50 字。"
)


# ── Stats tracker ──
@dataclass
class TestStats:
    total_events: int = 0
    analyzed_events: int = 0
    grpc_errors: int = 0
    exact_cache_hits: int = 0
    vector_cache_hits: int = 0
    vector_cache_tentative: int = 0
    fresh_inferences: int = 0
    total_grpc_ms: float = 0.0
    result_sources: dict = field(default_factory=dict)
    fresh_latencies_ms: List[float] = field(default_factory=list)
    cache_latencies_ms: List[float] = field(default_factory=list)
    short_outputs: int = 0

    def record(self, source: str, elapsed_ms: float, tokens: int = 0):
        self.result_sources[source] = self.result_sources.get(source, 0) + 1
        self.total_grpc_ms += elapsed_ms
        if source == "exact_cache":
            self.exact_cache_hits += 1
            self.cache_latencies_ms.append(elapsed_ms)
        elif source == "vector_cache":
            self.vector_cache_hits += 1
            self.cache_latencies_ms.append(elapsed_ms)
        elif source == "vector_cache_tentative":
            self.vector_cache_tentative += 1
            self.cache_latencies_ms.append(elapsed_ms)
        elif source == "fresh":
            self.fresh_inferences += 1
            self.fresh_latencies_ms.append(elapsed_ms)
            if tokens < 8:
                self.short_outputs += 1

    @staticmethod
    def _pctl(samples: List[float], q: float) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        idx = min(len(ordered) - 1, int(q * len(ordered)))
        return ordered[idx]

    def summary(self) -> str:
        lines = [
            f"Events: total={self.total_events} analyzed={self.analyzed_events} errors={self.grpc_errors}",
            f"Cache:  exact={self.exact_cache_hits} vector={self.vector_cache_hits} "
            f"tentative={self.vector_cache_tentative} fresh={self.fresh_inferences} "
            f"short_fresh={self.short_outputs}",
            f"Fresh   p50={self._pctl(self.fresh_latencies_ms, 0.5):.0f}ms "
            f"p90={self._pctl(self.fresh_latencies_ms, 0.9):.0f}ms "
            f"p99={self._pctl(self.fresh_latencies_ms, 0.99):.0f}ms",
            f"Cache   p50={self._pctl(self.cache_latencies_ms, 0.5):.0f}ms "
            f"p90={self._pctl(self.cache_latencies_ms, 0.9):.0f}ms "
            f"p99={self._pctl(self.cache_latencies_ms, 0.99):.0f}ms",
            f"Sources: {self.result_sources}",
        ]
        return "\n".join(lines)


# ── gRPC VLM Analyzer (duck-compatible with VisualEventAnalyzer Protocol) ──
class GrpcVLMAnalyzer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        max_tokens: int = 120,
        temperature: float = 0.5,
        allow_cache: bool = True,
        stats: Optional[TestStats] = None,
    ):
        self.channel = grpc.insecure_channel(
            f"{host}:{port}",
            options=[
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ],
        )
        self.stub = pb2_grpc.MultimodalInferenceStub(self.channel)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.allow_cache = allow_cache
        self.stats = stats or TestStats()

    def analyze(self, event: VisualEvent) -> Optional[VisualAnalysis]:
        if not event.keyframes:
            return None

        # Pick representative keyframe (first one)
        kf = event.keyframes[0]
        image_bytes = base64.standard_b64decode(kf.base64_data)

        prompt = _RELAXED_PROMPT

        # Compute saliency hint from event peak score
        saliency_hint = event.peak_score

        request = pb2.VLMRequest(
            image_data=image_bytes,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            allow_cache=self.allow_cache,
            force_refresh=not self.allow_cache,
            saliency_hint=saliency_hint,
        )

        t0 = time.perf_counter()
        try:
            response = self.stub.GenerateVLMSync(request, timeout=60.0)
        except grpc.RpcError as e:
            self.stats.grpc_errors += 1
            print(f"  [gRPC ERROR] {e.code()}: {e.details()}")
            return None
        elapsed_ms = (time.perf_counter() - t0) * 1000

        source = response.result_source or "unknown"
        self.stats.record(source, elapsed_ms, response.generated_tokens)

        cache_tag = ""
        if response.cache_hit:
            cache_tag = f" [{source}]"

        text = response.text
        if not text:
            if response.error:
                print(f"  [gRPC] error: {response.error}")
                return None
            return None

        print(
            f"  [gRPC] {elapsed_ms:.0f}ms source={source}{cache_tag} "
            f"tokens={response.generated_tokens} text={text[:200]!r}",
            flush=True,
        )

        return VisualAnalysis.from_llm_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end visual perception test with local gRPC VLM."
    )
    parser.add_argument(
        "--source",
        default=str(_NEURO_ROOT / "data" / "visual_test_motion_trim_final.avi"),
        help="Video file path or camera index.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable VLM cache (force fresh inference every time).")
    parser.add_argument("--print-raw", action="store_true",
                        help="Print raw LLM analysis text.")
    parser.add_argument("--print-emotion", action="store_true",
                        help="Print derived emotion signals.")
    parser.add_argument(
        "--analysis-mode",
        choices=("none", "triggered", "per_event"),
        default="per_event",
        help="Vision analysis strategy. Default: per_event (analyze every event).",
    )
    parser.add_argument(
        "--calls-per-minute",
        type=int,
        default=120,
        help="Vision LLM rate limit. Default 120 (2/s) since local inference is fast.",
    )
    args = parser.parse_args()

    if not CV2_AVAILABLE:
        print("ERROR: OpenCV not available. Install opencv-python.")
        return 1

    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, str) and not Path(source).exists():
        print(f"ERROR: Video file not found: {source}")
        return 1

    stats = TestStats()
    analyzer = GrpcVLMAnalyzer(
        host=args.host,
        port=args.port,
        allow_cache=not args.no_cache,
        stats=stats,
    )

    config = VisualPerceptionConfig()
    config.vision_analysis_mode = args.analysis_mode
    config.vision_calls_per_minute = args.calls_per_minute
    config.vision_rate_limit_max_wait_seconds = 5.0

    def on_candidate(event: VisualEvent):
        stats.total_events += 1

    def on_promoted(event: VisualEvent):
        stats.analyzed_events += 1
        mode = str(event.metrics.get("mode", "trigger"))
        print(
            f"[{mode}] peak={event.peak_score:.3f} "
            f"text={visual_event_to_agent_text(event)}"
        )
        if event.analysis:
            if event.analysis.memory_candidate:
                print(f"  memory_candidate={event.analysis.memory_candidate}")
            if args.print_raw and event.analysis.raw_text:
                print(f"  raw_text={event.analysis.raw_text}")
        if args.print_emotion:
            signal = derive_visual_emotion_signal(event, config=config)
            print(f"  emotion={json.dumps(signal, ensure_ascii=False)}")

    def on_summary(summary):
        print(f"[summary] {summary.to_text()}")

    runner = VisualPerceptionPipeline(
        config=config,
        analyzer=analyzer,
        event_callback=on_candidate,
        promoted_event_callback=on_promoted,
        summary_callback=on_summary if config.summary_enabled else None,
    )

    print(f"Source: {source}")
    print(f"gRPC:   {args.host}:{args.port}")
    print(f"Cache:  {'disabled' if args.no_cache else 'enabled'}")
    print(f"Mode:   {args.analysis_mode}")
    print("=" * 60)

    try:
        runner.run(
            source,
            max_frames=args.max_frames,
            duration_seconds=args.duration_seconds,
        )
    except KeyboardInterrupt:
        runner.stop()
        print("\nInterrupted.")
    except Exception as exc:
        print(f"Pipeline error: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 60)
    print(
        f"Pipeline: candidates={len(runner.candidate_events)} "
        f"promoted={len(runner.promoted_events)} "
        f"summaries={len(runner.window_summaries)}"
    )
    print(stats.summary())

    # Compute vector cache hit rate
    total_cache = stats.exact_cache_hits + stats.vector_cache_hits + stats.vector_cache_tentative
    total_requests = total_cache + stats.fresh_inferences + stats.grpc_errors
    if total_requests > 0:
        print(
            f"\nCache hit rate: {total_cache}/{total_requests} "
            f"({100.0 * total_cache / total_requests:.1f}%)"
        )
        if stats.vector_cache_hits + stats.vector_cache_tentative > 0:
            print(
                f"  Vector cache contribution: "
                f"{stats.vector_cache_hits + stats.vector_cache_tentative} hits "
                f"({100.0 * (stats.vector_cache_hits + stats.vector_cache_tentative) / total_requests:.1f}%)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
