from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ReportSummary:
    path: str
    classification: str
    scenario: str
    concurrency: int
    planned_chats: int
    completed_chats: int
    error_count: int
    burst_seconds: float
    e2e_p95_ms: float
    backend_p95_ms: float
    queue_p95_ms: float
    llm_p95_ms: float
    memory_p95_ms: float
    callback_p95_ms: float
    peak_rss_mb: float


def number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def integer(value: Any, default: int = 0) -> int:
    return int(value) if isinstance(value, (int, float)) else default


def nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def classify(data: dict[str, Any]) -> str | None:
    if not isinstance(data.get("latencyMs"), dict) or not isinstance(data.get("concurrency"), (int, float)):
        return None
    planned = integer(data.get("concurrency")) * integer(data.get("turnsPerUser"), 1)
    completed = integer(data.get("totalChatRequests"))
    errors = integer(data.get("errorCount"))
    if completed == planned and errors == 0:
        return "complete"
    return "boundary" if completed > 0 else "diagnostic"


def display_path(path: Path, scan_root: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return f"{scan_root.name}/{path.relative_to(scan_root).as_posix()}"


def summarize(path: Path, root: Path) -> ReportSummary | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    classification = classify(data)
    if classification is None:
        return None
    concurrency = integer(data.get("concurrency"))
    planned = concurrency * integer(data.get("turnsPerUser"), 1)
    return ReportSummary(
        path=display_path(path, root),
        classification=classification,
        scenario=str(data.get("scenario") or path.stem),
        concurrency=concurrency,
        planned_chats=planned,
        completed_chats=integer(data.get("totalChatRequests")),
        error_count=integer(data.get("errorCount")),
        burst_seconds=number(nested(data, "chatMeasurementWindow", "elapsedSec")),
        e2e_p95_ms=number(nested(data, "latencyMs", "chatEndToEnd", "p95")),
        backend_p95_ms=number(nested(data, "latencyMs", "backendTotal", "p95")),
        queue_p95_ms=max(
            number(nested(data, "latencyMs", "computeQueueWait", "p95")),
            number(nested(data, "latencyMs", "ioQueueWait", "p95")),
        ),
        llm_p95_ms=number(nested(data, "latencyMs", "llmTotal", "p95")),
        memory_p95_ms=number(nested(data, "latencyMs", "memoryContext", "p95")),
        callback_p95_ms=number(nested(data, "latencyMs", "callbackToResponse", "p95")),
        peak_rss_mb=number(nested(data, "memory", "peakRssMb")),
    )


def report_files(roots: Iterable[Path]) -> Iterable[tuple[Path, Path]]:
    for root in roots:
        resolved = root.resolve()
        if resolved.is_file() and resolved.suffix.lower() == ".json":
            yield resolved, resolved.parent
        elif resolved.is_dir():
            for path in sorted(resolved.rglob("*.json")):
                yield path, resolved


def markdown(items: list[ReportSummary]) -> str:
    lines = [
        "| Class | Scenario | C | Chat | Errors | Burst s | E2E P95 | Backend P95 | Queue P95 | LLM P95 | Memory P95 | Callback P95 | RSS MiB | Path |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in items:
        lines.append(
            f"| {item.classification} | {item.scenario} | {item.concurrency} | "
            f"{item.completed_chats}/{item.planned_chats} | {item.error_count} | {item.burst_seconds:.3f} | "
            f"{item.e2e_p95_ms:.1f} | {item.backend_p95_ms:.1f} | {item.queue_p95_ms:.1f} | "
            f"{item.llm_p95_ms:.1f} | {item.memory_p95_ms:.1f} | {item.callback_p95_ms:.1f} | "
            f"{item.peak_rss_mb:.1f} | `{item.path}` |"
        )
    counts = {name: sum(item.classification == name for item in items) for name in ("complete", "boundary", "diagnostic")}
    lines.extend(["", f"Reports: {len(items)}; complete={counts['complete']}; boundary={counts['boundary']}; diagnostic={counts['diagnostic']}."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize AgentLoom Gateway E2E JSON reports.")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--classification", choices=("complete", "boundary", "diagnostic"))
    args = parser.parse_args()

    items: list[ReportSummary] = []
    seen: set[Path] = set()
    for path, root in report_files(args.roots):
        if path in seen:
            continue
        seen.add(path)
        item = summarize(path, root)
        if item and (not args.classification or item.classification == args.classification):
            items.append(item)
    items.sort(key=lambda item: (item.classification, item.concurrency, item.scenario, item.path))
    output = markdown(items)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
