#!/usr/bin/env python3
"""Build a router baseline CSV from per-strategy output records.

Input is JSONL with one example per line. Each record should contain fields like:
    {
      "example_id": "gsm8k_001",
      "task_type": "math",
      "prompt": "...",
      "cheap_direct": {"output": "...", "correct": 0, "cost": 1, "confidence": 0.4},
      "strong_reasoning": {"output": "...", "correct": 1, "cost": 8}
    }

Output is a flat CSV accepted by baselines/router_baselines.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.features import build_feature_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw JSONL input.")
    parser.add_argument("--output", type=Path, required=True, help="Router CSV output.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["cheap_direct", "medium_fewshot", "strong_reasoning"],
        help="Strategy names to flatten into <strategy>_correct and <strategy>_cost columns.",
    )
    parser.add_argument(
        "--cheap-strategy",
        default="cheap_direct",
        help="Strategy used to extract cheap-response cascade features.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def flatten_record(
    record: dict[str, Any],
    strategies: list[str],
    cheap_strategy: str,
) -> dict[str, Any]:
    row = build_feature_row(record, cheap_strategy=cheap_strategy)
    for strategy in strategies:
        result = record.get(strategy)
        if result is None:
            raise ValueError(
                f"Record {record.get('example_id')} is missing strategy {strategy!r}."
            )
        row[f"{strategy}_correct"] = int(result["correct"])
        row[f"{strategy}_cost"] = float(result["cost"])
    return row


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    rows = [
        flatten_record(record, args.strategies, args.cheap_strategy)
        for record in records
    ]
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    print("Columns:")
    print(", ".join(df.columns))


if __name__ == "__main__":
    main()
