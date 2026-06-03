#!/usr/bin/env python3
"""Run several OpenRouter models over the same task CSV.

This is the specialist-routing runner. It writes one JSONL record per example,
with one result object per named route, e.g.:

    cheap_general, math_specialist, science_specialist
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.graders import extract_answer_field, extract_confidence, grade_answer
from routing.prompts import (
    brief_reasoning_prompt,
    direct_prompt,
    full_reasoning_prompt,
    reasoning_prompt,
)


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class Route:
    name: str
    model: str
    prompt_mode: str


@dataclass(frozen=True)
class Generation:
    text: str
    elapsed_seconds: float
    usage: dict[str, Any]
    model_returned: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--route",
        action="append",
        required=True,
        help=(
            "Named model route. Format: name=model_slug or name=model_slug:prompt_mode. "
            "Prompt modes: direct, brief_reasoning, full_reasoning, reasoning."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.25)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--site-url", default="https://github.com/rli007/229")
    parser.add_argument("--app-title", default="CS229 Routing Project")
    return parser.parse_args()


def parse_route(raw: str) -> Route:
    if "=" not in raw:
        raise ValueError(f"Invalid --route {raw!r}; expected name=model")
    name, rest = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid --route {raw!r}; route name is empty")
    prompt_mode = "reasoning"
    valid_prompt_modes = {"direct", "brief_reasoning", "full_reasoning", "reasoning"}
    if ":" in rest:
        model, maybe_mode = rest.rsplit(":", 1)
        if maybe_mode in valid_prompt_modes:
            prompt_mode = maybe_mode
        else:
            model = rest
    else:
        model = rest
    model = model.strip()
    if not model:
        raise ValueError(f"Invalid --route {raw!r}; model slug is empty")
    return Route(name=name, model=model, prompt_mode=prompt_mode)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                completed.add(str(json.loads(line).get("example_id")))
            except json.JSONDecodeError:
                continue
    return completed


def prompt_for_mode(question: str, task_type: str, answer_type: str, mode: str) -> str:
    if mode == "direct":
        return direct_prompt(question, task_type, answer_type)
    if mode == "brief_reasoning":
        return brief_reasoning_prompt(question, task_type, answer_type)
    if mode == "full_reasoning":
        return full_reasoning_prompt(question, task_type, answer_type)
    return reasoning_prompt(question, task_type, answer_type)


def retry_delay_seconds(
    attempt: int,
    retry_after_header: str | None = None,
    response_body: str | None = None,
) -> float:
    candidates: list[float] = []
    if retry_after_header:
        try:
            candidates.append(float(retry_after_header))
        except ValueError:
            pass
    if response_body:
        try:
            body = json.loads(response_body)
            metadata = body.get("error", {}).get("metadata", {})
            retry_after = metadata.get("retry_after_seconds")
            if retry_after is not None:
                candidates.append(float(retry_after))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if candidates:
        return min(max(candidates), 60.0)
    return min(2**attempt, 30)


def chat_completion(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    site_url: str,
    app_title: str,
    max_retries: int,
) -> Generation:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": site_url,
        "X-Title": app_title,
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        retry_after_header = None
        response_body = None
        request = urllib.request.Request(
            OPENROUTER_CHAT_URL,
            data=data,
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0].get("message", {}).get("content", "")
            return Generation(
                text=content or "",
                elapsed_seconds=time.perf_counter() - started,
                usage=result.get("usage", {}) or {},
                model_returned=result.get("model"),
            )
        except urllib.error.HTTPError as exc:
            retry_after_header = exc.headers.get("Retry-After")
            response_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {response_body}")
            if exc.code not in {408, 429, 500, 502, 503}:
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(
            retry_delay_seconds(
                attempt,
                retry_after_header=retry_after_header,
                response_body=response_body,
            )
        )
    raise RuntimeError(f"OpenRouter request failed for {model}: {last_error}")


def run_route(
    route: Route,
    row: pd.Series,
    api_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = prompt_for_mode(
        question=str(row["prompt"]),
        task_type=str(row["task_type"]),
        answer_type=str(row["answer_type"]),
        mode=route.prompt_mode,
    )
    generation = chat_completion(
        api_key=api_key,
        model=route.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        site_url=args.site_url,
        app_title=args.app_title,
        max_retries=args.max_retries,
    )
    return {
        "output": generation.text,
        "parsed_answer": extract_answer_field(generation.text),
        "correct": grade_answer(generation.text, str(row["answer"]), str(row["answer_type"])),
        "cost": 1.0,
        "confidence": extract_confidence(generation.text),
        "samples": [extract_answer_field(generation.text)],
        "elapsed_seconds": generation.elapsed_seconds,
        "usage": generation.usage,
        "model_returned": generation.model_returned,
        "model_requested": route.model,
        "prompt_mode": route.prompt_mode,
    }


def run_example(
    row: pd.Series,
    row_number: int,
    total_rows: int,
    routes: list[Route],
    api_key: str,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    record: dict[str, Any] = {
        "example_id": str(row["example_id"]),
        "task_type": str(row["task_type"]),
        "answer_type": str(row["answer_type"]),
        "prompt": str(row["prompt"]),
        "answer": str(row["answer"]),
    }
    for route in routes:
        record[route.name] = run_route(route, row, api_key, args)
        if args.request_delay_seconds > 0:
            time.sleep(args.request_delay_seconds)
    prefix = f"[{row_number}/{total_rows}] {record['example_id']}"
    return prefix, record


def main() -> None:
    args = parse_args()
    routes = [parse_route(raw) for raw in args.route]
    route_names = [route.name for route in routes]
    if len(route_names) != len(set(route_names)):
        raise SystemExit("Route names must be unique.")

    load_env_file(PROJECT_ROOT / ".env")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set {args.api_key_env} in your shell or in "
            f"{PROJECT_ROOT / '.env'}."
        )

    tasks = pd.read_csv(args.tasks)
    if args.limit is not None:
        tasks = tasks.head(args.limit)

    completed = read_completed_ids(args.output) if args.resume else set()
    pending_rows = []
    for idx, row in tasks.iterrows():
        example_id = str(row["example_id"])
        row_number = idx + 1
        if example_id in completed:
            print(f"[{row_number}/{len(tasks)}] {example_id}: skipped")
            continue
        pending_rows.append((row_number, row))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as f:
        if not pending_rows:
            print("No new examples to run.")
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = [
                executor.submit(run_example, row, row_number, len(tasks), routes, api_key, args)
                for row_number, row in pending_rows
            ]
            for future in as_completed(futures):
                prefix, record = future.result()
                f.write(json.dumps(record) + "\n")
                f.flush()
                summary = " ".join(
                    f"{route.name}={record[route.name]['correct']}" for route in routes
                )
                print(f"{prefix}: {summary}")
    print(f"Wrote raw outputs to {args.output}")


if __name__ == "__main__":
    main()
