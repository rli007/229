#!/usr/bin/env python3
"""Prepare StrategyQA examples for the routing pipeline.

Output schema matches scripts/run_openrouter_strategies.py:
    example_id,task_type,answer_type,prompt,answer
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional local StrategyQA JSON file. If omitted, load from Hugging Face.",
    )
    parser.add_argument("--dataset-name", default="ChilleD/StrategyQA")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=229)
    return parser.parse_args()


def load_strategyqa(path: Path | None, dataset_name: str, split: str) -> list[dict[str, Any]]:
    if path is not None:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Run `pip install -r requirements.txt` "
            "from your active environment."
        ) from exc
    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def normalize_answer(answer: Any) -> str:
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    text = str(answer).strip().lower()
    if text in {"true", "1", "yes"}:
        return "yes"
    if text in {"false", "0", "no"}:
        return "no"
    raise ValueError(f"Unsupported StrategyQA answer value: {answer!r}")


def main() -> None:
    args = parse_args()
    rows = load_strategyqa(args.source, args.dataset_name, args.split)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["example_id", "task_type", "answer_type", "prompt", "answer"],
        )
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "example_id": row.get("qid") or f"strategyqa_{idx:05d}",
                    "task_type": "strategyqa",
                    "answer_type": "yesno",
                    "prompt": f"{row['question']} Answer yes or no.",
                    "answer": normalize_answer(row["answer"]),
                }
            )
    print(f"Wrote {len(rows)} StrategyQA rows to {args.output}")


if __name__ == "__main__":
    main()
