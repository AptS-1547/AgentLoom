#!/usr/bin/env python3
"""Generate synthetic Agent conversation records for L0 retrieval tests.

The output schema is intentionally close to L0 cache records:

[
  {
    "topic": "machine_learning",
    "turn": 0,
    "user": "...",
    "assistant": "..."
  }
]

API key resolution follows the C++ E2E tools convention: environment first,
then api_key_file relative to the config file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_TOPICS = [
    "machine_learning",
    "math_study",
    "cpp_programming",
    "english_learning",
    "exam_anxiety",
    "time_management",
    "career_planning",
    "reading_comprehension",
    "debugging",
    "student_motivation",
]


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_api_key(config_path: Path | None, llm: dict[str, Any]) -> str:
    env_name = llm.get("api_key_env", "AGENT_LLM_API_KEY")
    if env_name:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    key_file = llm.get("api_key_file", "")
    if key_file:
        key_path = Path(key_file)
        if not key_path.is_absolute() and config_path is not None:
            key_path = config_path.parent / key_path
        if key_path.exists():
            return key_path.read_text(encoding="utf-8").strip()
    return ""


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_records(text: str) -> list[dict[str, str]]:
    text = strip_code_fence(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, list):
        raise ValueError("model response must be a JSON array")

    records: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        user = str(item.get("user", "")).strip()
        assistant = str(item.get("assistant", "")).strip()
        if len(user) < 4 or len(assistant) < 8:
            continue
        records.append({"user": user, "assistant": assistant})
    return records


def post_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def build_prompt(topic: str, batch_size: int, start_turn: int) -> list[dict[str, str]]:
    system = (
        "You generate realistic Chinese student-Agent dialogue records for retrieval testing. "
        "Return only a JSON array. Each item must contain string fields: user, assistant. "
        "The assistant is a patient educational companion, not a marketing bot. "
        "Make the records semantically coherent, diverse, and suitable for L0 memory retrieval. "
        "Avoid numbering outside JSON. Avoid markdown."
    )
    user = f"""
Generate {batch_size} independent but realistic dialogue turns for topic "{topic}".

Requirements:
- Chinese language.
- User messages should be natural student messages, 10-80 Chinese characters.
- Assistant replies should be useful and contextual, 30-140 Chinese characters.
- Include some paraphrases and recurring concepts so semantic retrieval can be tested.
- Do not mention this generation task.
- Start semantic variety around virtual turn index {start_turn}.

Return JSON array only:
[
  {{"user": "...", "assistant": "..."}}
]
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def fallback_records(total: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    templates = {
        "machine_learning": [
            ("我还是不理解过拟合和欠拟合的区别", "可以把欠拟合理解为模型太简单，训练集都学不好；过拟合是训练集记得太死，换到新题就容易出错。"),
            ("为什么神经网络要用激活函数", "激活函数给模型加入非线性能力，否则多层线性变换叠在一起仍然等价于一层线性模型。"),
            ("卷积里的参数共享到底有什么意义", "参数共享让同一个卷积核在不同位置识别相同模式，既减少参数量，也让模型更适合处理图像局部特征。"),
        ],
        "math_study": [
            ("我一看到导数应用题就不知道从哪下手", "先找题目里的变化率关系，再明确自变量和因变量。不要急着套公式，先把量之间的关系写出来。"),
            ("极限里的无穷小替换什么时候能用", "一般在乘除结构和局部等价时使用更安全，加减结构要特别小心，因为误差可能被放大。"),
            ("线性代数的特征值有什么直观意义", "特征值可以理解为某些特殊方向上的伸缩比例，矩阵作用后方向不变，只是长度按特征值缩放。"),
        ],
        "cpp_programming": [
            ("我总是搞混右值引用和移动语义", "右值引用是表达可被移动资源的类型工具，移动语义的核心是转移资源所有权，避免不必要的深拷贝。"),
            ("为什么析构函数里不能随便抛异常", "析构常发生在栈展开期间，如果再抛异常可能导致程序终止。资源释放失败最好记录状态或提供显式 close。"),
            ("智能指针是不是可以完全替代裸指针", "智能指针适合表达所有权，但非拥有关系仍可以用引用或裸指针。关键是让所有权边界清晰。"),
        ],
        "exam_anxiety": [
            ("我最近复习越多越觉得自己不会", "这通常是知识边界变清楚后的正常反应。先把不会的点列出来，再按高频和易提分程度排序处理。"),
            ("明天考试我现在很慌怎么办", "先停止无限刷题，做一轮错题回顾和公式梳理。今晚保证睡眠，比继续硬撑更能稳定发挥。"),
            ("我怕自己努力了还是考不好", "这种担心可以理解。我们把目标拆成可执行动作：今天只确认三类题型和两处薄弱点，不用一次解决全部压力。"),
        ],
    }
    topics = list(templates)
    records = []
    for i in range(total):
        topic = rng.choice(topics)
        user, assistant = rng.choice(templates[topic])
        if rng.random() < 0.35:
            user = user.replace("我", "我还是", 1) if user.startswith("我") else user
        records.append({"topic": topic, "turn": i, "user": user, "assistant": assistant})
    return records


def save_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("tools/persona_gateway_e2e_server.json"))
    parser.add_argument("--output", type=Path, default=Path("data/l0_long_context_benchmark/generated_agent_dialogues.json"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fallback-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    existing: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            raise ValueError("existing output is not a JSON array")

    if args.fallback_only:
        records = fallback_records(args.count, args.seed)
        save_records(args.output, records)
        print(f"[generate] wrote fallback records: {len(records)} -> {args.output}")
        return 0

    config = load_config(args.config)
    llm = config.get("llm", {})
    base_url = llm.get("base_url", "https://api.deepseek.com/v1")
    model = llm.get("model", "deepseek-chat")
    timeout_ms = int(llm.get("timeout_ms", 30000))
    api_key = resolve_api_key(args.config, llm)
    if not api_key:
        print("[generate] api key not found; use --fallback-only or configure llm api key", file=sys.stderr)
        return 1

    records = existing
    topics = DEFAULT_TOPICS
    batch_index = len(records) // max(args.batch_size, 1)
    while len(records) < args.count:
        topic = topics[batch_index % len(topics)]
        remaining = args.count - len(records)
        batch_size = min(args.batch_size, remaining)
        messages = build_prompt(topic, batch_size, len(records))
        try:
            content = post_chat_completion(
                base_url,
                api_key,
                model,
                messages,
                args.temperature,
                timeout_ms / 1000.0,
            )
            parsed = parse_records(content)
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"[generate] batch failed topic={topic} error={exc}", file=sys.stderr)
            save_records(args.output, records)
            return 1

        for item in parsed[:batch_size]:
            records.append({
                "topic": topic,
                "turn": len(records),
                "user": item["user"],
                "assistant": item["assistant"],
            })
        save_records(args.output, records)
        print(f"[generate] records={len(records)}/{args.count} topic={topic}")
        batch_index += 1
        time.sleep(0.2)

    save_records(args.output, records[: args.count])
    print(f"[generate] wrote records: {args.count} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
