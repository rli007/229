#!/usr/bin/env python3
"""Prepare a balanced mixed-task CSV for specialist routing experiments.

Output schema:
    example_id,task_type,answer_type,prompt,answer

The default mix is:
    - commonsense: StrategyQA yes/no
    - math: GSM8K numeric
    - science: SciQ multiple choice
    - humanities: MMLU history/philosophy/religion multiple choice
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task-limit", type=int, default=150)
    parser.add_argument("--seed", type=int, default=229)
    parser.add_argument("--split", default="train")
    parser.add_argument("--mmlu-split", default="test")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["commonsense", "math", "science", "humanities"],
    )
    parser.add_argument(
        "--humanities-subtasks",
        nargs="+",
        default=["high_school_world_history", "philosophy", "world_religions"],
    )
    parser.add_argument(
        "--social-science-subtasks",
        nargs="+",
        default=["high_school_government_and_politics", "sociology", "us_foreign_policy"],
    )
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def load_dataset_rows(dataset_name: str, *args: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Run `pip install -r requirements.txt` "
            "from your active environment."
        ) from exc
    dataset = load_dataset(dataset_name, *args, split=split)
    return [dict(row) for row in dataset]


def normalize_strategyqa_answer(answer: Any) -> str:
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    text = str(answer).strip().lower()
    if text in {"true", "1", "yes"}:
        return "yes"
    if text in {"false", "0", "no"}:
        return "no"
    raise ValueError(f"Unsupported StrategyQA answer value: {answer!r}")


def prepare_strategyqa(limit: int, split: str, rng: random.Random) -> list[dict[str, str]]:
    rows = load_dataset_rows("ChilleD/StrategyQA", split=split)
    rng.shuffle(rows)
    prepared = []
    for idx, row in enumerate(rows[:limit]):
        prepared.append(
            {
                "example_id": f"strategyqa_{row.get('qid') or idx}",
                "task_type": "commonsense",
                "answer_type": "yesno",
                "prompt": f"{row['question']} Answer yes or no.",
                "answer": normalize_strategyqa_answer(row["answer"]),
            }
        )
    return prepared


def extract_gsm8k_answer(answer: str) -> str:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer)
    if not numbers:
        raise ValueError(f"Could not extract GSM8K numeric answer: {answer[:120]!r}")
    return numbers[-1].replace(",", "")


def prepare_gsm8k(limit: int, split: str, rng: random.Random) -> list[dict[str, str]]:
    rows = load_dataset_rows("openai/gsm8k", "main", split=split)
    rng.shuffle(rows)
    prepared = []
    for idx, row in enumerate(rows[:limit]):
        prepared.append(
            {
                "example_id": f"gsm8k_{idx:05d}",
                "task_type": "math",
                "answer_type": "number",
                "prompt": row["question"],
                "answer": extract_gsm8k_answer(row["answer"]),
            }
        )
    return prepared


def prepare_sciq(limit: int, split: str, rng: random.Random) -> list[dict[str, str]]:
    rows = load_dataset_rows("allenai/sciq", split=split)
    rng.shuffle(rows)
    prepared = []
    letters = ["A", "B", "C", "D"]
    for idx, row in enumerate(rows[:limit]):
        options = [
            str(row["correct_answer"]),
            str(row["distractor1"]),
            str(row["distractor2"]),
            str(row["distractor3"]),
        ]
        rng.shuffle(options)
        gold_letter = letters[options.index(str(row["correct_answer"]))]
        option_text = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
        prepared.append(
            {
                "example_id": f"sciq_{idx:05d}",
                "task_type": "science",
                "answer_type": "choice",
                "prompt": f"{row['question']}\n{option_text}",
                "answer": gold_letter,
            }
        )
    return prepared


def prepare_mmlu_choice_task(
    task_type: str,
    subtasks: list[str],
    limit: int,
    split: str,
    rng: random.Random,
) -> list[dict[str, str]]:
    rows = []
    for subtask in subtasks:
        for row in load_dataset_rows("cais/mmlu", subtask, split=split):
            row = dict(row)
            row["_subtask"] = subtask
            rows.append(row)
    rng.shuffle(rows)
    prepared = []
    letters = ["A", "B", "C", "D"]
    for idx, row in enumerate(rows[:limit]):
        choices = list(row["choices"])
        answer = row["answer"]
        if isinstance(answer, str):
            gold_letter = answer.strip().upper()[:1]
        else:
            gold_letter = letters[int(answer)]
        option_text = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
        prepared.append(
            {
                "example_id": f"mmlu_{task_type}_{idx:05d}_{row['_subtask']}",
                "task_type": task_type,
                "answer_type": "choice",
                "prompt": f"{row['question']}\n{option_text}",
                "answer": gold_letter,
            }
        )
    return prepared


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    include = set(args.include)
    if "commonsense" in include:
        rows.extend(prepare_strategyqa(args.per_task_limit, args.split, rng))
    if "math" in include:
        rows.extend(prepare_gsm8k(args.per_task_limit, args.split, rng))
    if "science" in include:
        rows.extend(prepare_sciq(args.per_task_limit, args.split, rng))
    if "humanities" in include:
        rows.extend(
            prepare_mmlu_choice_task(
                "humanities",
                args.humanities_subtasks,
                args.per_task_limit,
                args.mmlu_split,
                rng,
            )
        )
    if "social_science" in include:
        rows.extend(
            prepare_mmlu_choice_task(
                "social_science",
                args.social_science_subtasks,
                args.per_task_limit,
                args.mmlu_split,
                rng,
            )
        )
    if args.shuffle:
        rng.shuffle(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["example_id", "task_type", "answer_type", "prompt", "answer"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} mixed-task rows to {args.output}")


if __name__ == "__main__":
    main()
