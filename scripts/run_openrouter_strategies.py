#!/usr/bin/env python3
"""Run OpenRouter models and write raw JSONL for routing experiments.

The default project setup is:
  - cheap_direct: Mistral Nemo
  - strong_reasoning: Qwen3 235B A22B

The script writes both direct and cascade route costs:
  - strong_reasoning: strong model cost only, for direct routing experiments.
  - escalate_strong: cheap feature cost + strong model cost, for cascade experiments.
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
from routing.prompts import brief_reasoning_prompt, direct_prompt, reasoning_prompt


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


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
        "--cheap-model",
        default="mistralai/mistral-nemo",
    )
    parser.add_argument(
        "--strong-model",
        default="qwen/qwen3-235b-a22b-2507",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cheap-samples", type=int, default=1)
    parser.add_argument(
        "--cheap-prompt-mode",
        choices=["direct", "brief_reasoning"],
        default="direct",
        help="Use `brief_reasoning` to expose cheap-model reasoning features for routing.",
    )
    parser.add_argument("--cheap-cost", type=float, default=1.0)
    parser.add_argument("--strong-cost", type=float, default=8.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.25)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of examples to process in parallel. Each example still runs cheap then strong sequentially.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the OpenRouter API key.",
    )
    parser.add_argument(
        "--site-url",
        default="https://github.com/rli007/229",
        help="Optional HTTP-Referer header for OpenRouter rankings/analytics.",
    )
    parser.add_argument("--app-title", default="CS229 Routing Project")
    return parser.parse_args()


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
        retry_after_header: str | None = None
        response_body: str | None = None
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
            choice = result["choices"][0]
            content = choice.get("message", {}).get("content", "")
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
        except urllib.error.URLError as exc:
            last_error = exc
        except TimeoutError as exc:
            last_error = exc
        sleep_seconds = retry_delay_seconds(
            attempt,
            error=last_error,
            retry_after_header=retry_after_header,
            response_body=response_body,
        )
        time.sleep(sleep_seconds)
    raise RuntimeError(f"OpenRouter request failed for {model}: {last_error}")


def retry_delay_seconds(
    attempt: int,
    error: Exception | None = None,
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
    if isinstance(error, TimeoutError):
        return min(2 ** (attempt + 1), 60)
    return min(2**attempt, 30)


def strategy_result(
    api_key: str,
    model: str,
    prompt: str,
    gold: str,
    answer_type: str,
    cost: float,
    args: argparse.Namespace,
    sample_count: int = 1,
) -> dict[str, Any]:
    generation = chat_completion(
        api_key=api_key,
        model=model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        site_url=args.site_url,
        app_title=args.app_title,
        max_retries=args.max_retries,
    )
    output = generation.text
    samples: list[str] = [extract_answer_field(output)]
    for _ in range(max(0, sample_count - 1)):
        sample_generation = chat_completion(
            api_key=api_key,
            model=model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=max(args.temperature, 0.7),
            site_url=args.site_url,
            app_title=args.app_title,
            max_retries=args.max_retries,
        )
        samples.append(extract_answer_field(sample_generation.text))
        time.sleep(args.request_delay_seconds)
    return {
        "output": output,
        "parsed_answer": extract_answer_field(output),
        "correct": grade_answer(output, gold, answer_type),
        "cost": cost * max(1, sample_count),
        "confidence": extract_confidence(output),
        "samples": samples,
        "elapsed_seconds": generation.elapsed_seconds,
        "usage": generation.usage,
        "model_returned": generation.model_returned,
    }


def make_cheap_prompt(
    question: str,
    task_type: str,
    answer_type: str,
    mode: str,
) -> str:
    if mode == "brief_reasoning":
        return brief_reasoning_prompt(question, task_type, answer_type)
    return direct_prompt(question, task_type, answer_type)


def run_example(
    row: pd.Series,
    row_number: int,
    total_rows: int,
    api_key: str,
    args: argparse.Namespace,
) -> tuple[int, str, dict[str, Any]]:
    example_id = str(row["example_id"])
    question = str(row["prompt"])
    task_type = str(row["task_type"])
    answer_type = str(row["answer_type"])
    gold = str(row["answer"])
    cheap_prompt = make_cheap_prompt(
        question,
        task_type,
        answer_type,
        args.cheap_prompt_mode,
    )
    strong_prompt = reasoning_prompt(question, task_type, answer_type)

    record: dict[str, Any] = {
        "example_id": example_id,
        "task_type": task_type,
        "answer_type": answer_type,
        "prompt": question,
        "answer": gold,
    }
    cheap_result = strategy_result(
        api_key=api_key,
        model=args.cheap_model,
        prompt=cheap_prompt,
        gold=gold,
        answer_type=answer_type,
        cost=args.cheap_cost,
        args=args,
        sample_count=args.cheap_samples,
    )
    if args.request_delay_seconds > 0:
        time.sleep(args.request_delay_seconds)
    strong_result = strategy_result(
        api_key=api_key,
        model=args.strong_model,
        prompt=strong_prompt,
        gold=gold,
        answer_type=answer_type,
        cost=args.strong_cost,
        args=args,
    )
    record["cheap_direct"] = cheap_result
    record["strong_reasoning"] = strong_result
    record["escalate_strong"] = {
        **strong_result,
        "cost": cheap_result["cost"] + args.strong_cost,
    }
    return row_number, f"[{row_number}/{total_rows}] {example_id}", record


def main() -> None:
    args = parse_args()
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
                executor.submit(
                    run_example,
                    row,
                    row_number,
                    len(tasks),
                    api_key,
                    args,
                )
                for row_number, row in pending_rows
            ]
            for future in as_completed(futures):
                row_number, prefix, record = future.result()
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(
                    f"{prefix}: "
                    f"cheap={record['cheap_direct']['correct']} "
                    f"strong={record['strong_reasoning']['correct']}"
                )

    print(f"Wrote raw outputs to {args.output}")


if __name__ == "__main__":
    main()
