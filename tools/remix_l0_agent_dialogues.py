#!/usr/bin/env python3
"""Remix generated Agent dialogue records into short clustered segments.

The generator may produce records in large topic blocks. L0's long-history
sampling assumption is closer to:

  short-term semantic clusters, long-term approximate random mixture

This script groups records by topic, then emits random-sized same-topic
segments in random topic order.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input must be a JSON array")
    records: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        user = str(item.get("user", "")).strip()
        assistant = str(item.get("assistant", "")).strip()
        if not user or not assistant:
            continue
        records.append(dict(item))
    return records


def remix(records: list[dict[str, Any]], min_segment: int, max_segment: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        topic = str(record.get("topic", "unknown")).strip() or "unknown"
        grouped[topic].append(record)

    queues: dict[str, deque[dict[str, Any]]] = {}
    for topic, items in grouped.items():
        rng.shuffle(items)
        queues[topic] = deque(items)

    topics = list(queues)
    out: list[dict[str, Any]] = []
    last_topic = ""

    while topics:
        available = [topic for topic in topics if topic != last_topic]
        if not available:
            available = topics
        topic = rng.choice(available)
        segment_size = rng.randint(min_segment, max_segment)
        segment: list[dict[str, Any]] = []
        q = queues[topic]
        while q and len(segment) < segment_size:
            segment.append(q.popleft())
        if not q:
            topics.remove(topic)
        for record in segment:
            remixed = dict(record)
            remixed["original_turn"] = remixed.get("turn")
            remixed["turn"] = len(out)
            remixed["remix_segment_topic"] = topic
            out.append(remixed)
        last_topic = topic

    return out


def topic_report(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record.get("topic", "unknown"))] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-segment", type=int, default=5)
    parser.add_argument("--max-segment", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    if args.min_segment <= 0 or args.max_segment < args.min_segment:
        raise ValueError("invalid segment size range")

    records = load_records(args.input)
    remixed = remix(records, args.min_segment, args.max_segment, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(remixed, f, ensure_ascii=False, indent=2)

    print(f"[remix] input={len(records)} output={len(remixed)}")
    print(f"[remix] segments={args.min_segment}-{args.max_segment} seed={args.seed}")
    print(f"[remix] topicCounts={json.dumps(topic_report(remixed), ensure_ascii=False)}")
    print(f"[remix] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
