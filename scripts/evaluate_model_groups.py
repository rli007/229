#!/usr/bin/env python3
"""Evaluate accuracy routing within fixed model groups.

This script is for the "similar-cost models" question: if we hold the candidate
set roughly fixed in price, can learned routing improve accuracy over the best
single model?
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.router_baselines import (
    evaluate_for_cost_weight,
    results_table,
    split_data,
    validate_columns,
)


DEFAULT_FEATURES = [
    "task_type",
    "prompt_chars",
    "prompt_words",
    "question_start",
    "num_numbers",
    "has_math_symbols",
    "has_code_like_text",
    "question_mark_count",
    "capitalized_word_count",
    "contains_parenthesis",
    "contains_quote",
    "has_comparison_word",
    "has_negation_word",
    "has_quantity_word",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Router CSV.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Candidate models available in the CSV.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help=(
            "Named model group in the form name=model_a,model_b,... "
            "Can be repeated. If omitted, all pairs and triples are evaluated."
        ),
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Pre-routing features. Defaults to prompt/task metadata only.",
    )
    parser.add_argument(
        "--text-feature",
        default="prompt",
        help="Optional text column for TF-IDF routers. Use '' to disable.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[229, 230, 231, 232, 233],
        help="Random split seeds for repeated evaluation.",
    )
    parser.add_argument(
        "--skip-mlp",
        action="store_true",
        help="Skip MLP variants for faster sweeps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/model_group_accuracy_summary.csv"),
        help="Summary CSV output.",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=None,
        help="Optional per-seed result CSV output.",
    )
    return parser.parse_args()


def parse_groups(group_args: list[str], models: list[str]) -> dict[str, list[str]]:
    if group_args:
        groups: dict[str, list[str]] = {}
        for group_arg in group_args:
            if "=" not in group_arg:
                raise ValueError(f"Invalid group {group_arg!r}; expected name=m1,m2,...")
            name, raw_models = group_arg.split("=", 1)
            group_models = [model.strip() for model in raw_models.split(",") if model.strip()]
            if len(group_models) < 2:
                raise ValueError(f"Group {name!r} needs at least two models.")
            unknown = sorted(set(group_models) - set(models))
            if unknown:
                raise ValueError(
                    f"Group {name!r} references unknown models: {', '.join(unknown)}"
                )
            groups[name.strip()] = group_models
        return groups

    groups = {}
    for size in (2, 3):
        for combo in itertools.combinations(models, size):
            groups[f"{size}model_" + "_".join(combo)] = list(combo)
    return groups


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    grouped = details.groupby(["group", "models", "policy"], as_index=False)
    summary = grouped.agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        avg_cost_mean=("avg_cost", "mean"),
        utility_mean=("utility", "mean"),
        regret_mean=("regret", "mean"),
        route_acc_mean=("route_acc", "mean"),
    )
    summary["accuracy_std"] = summary["accuracy_std"].fillna(0.0)
    summary = summary.sort_values(
        ["group", "accuracy_mean", "regret_mean"],
        ascending=[True, False, True],
    )
    return summary


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    models = list(args.models)
    features = list(args.features)
    text_feature = args.text_feature or None
    validate_columns(df, models, features, text_feature=text_feature)

    groups = parse_groups(args.group, models)
    detail_tables = []
    for group_name, group_models in groups.items():
        validate_columns(df, group_models, features, text_feature=text_feature)
        for seed in args.seeds:
            train_df, test_df = split_data(
                df,
                split_column=None,
                heldout_task_types=None,
                test_size=args.test_size,
                random_state=seed,
            )
            results = evaluate_for_cost_weight(
                train_df=train_df,
                test_df=test_df,
                models=group_models,
                features=features,
                cost_weight=0.0,
                text_feature=text_feature,
                threshold_feature=None,
                thresholds=[],
                task_router_rules={},
                skip_mlp=args.skip_mlp,
                random_state=seed,
            )
            table = results_table(results)
            table.insert(0, "seed", seed)
            table.insert(0, "models", ",".join(group_models))
            table.insert(0, "group", group_name)
            detail_tables.append(table)

    details = pd.concat(detail_tables, ignore_index=True)
    summary = summarize(details)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    if args.details_output:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(args.details_output, index=False)

    print(f"Loaded {len(df)} rows")
    print(f"Evaluated {len(groups)} groups across {len(args.seeds)} random splits")
    print(f"Features: {', '.join(features)}")
    if text_feature:
        print(f"Text feature: {text_feature}")
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Saved summary to {args.output}")
    if args.details_output:
        print(f"Saved details to {args.details_output}")


if __name__ == "__main__":
    main()
