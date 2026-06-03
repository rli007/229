#!/usr/bin/env python3
"""Baseline experiments for instance-level model routing.

Expected CSV shape:
    one row per task/example, feature columns, and for each candidate model:
      - <model>_correct: 0/1 correctness on the example
      - <model>_cost: nonnegative inference cost, latency, or normalized price

Example:
    python baselines/router_baselines.py \
        --data data/example_router_data.csv \
        --models small medium large \
        --features prompt_len difficulty task_type \
        --cost-weight 0.05
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.router_models import (
    base_router_estimators,
    fit_router,
    text_router_estimators,
)


@dataclass(frozen=True)
class PolicyResult:
    name: str
    accuracy: float
    avg_cost: float
    utility: float
    regret: float
    cost_weight: float
    routing_accuracy: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Input CSV path.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Candidate model names. Requires <model>_correct and <model>_cost columns.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Feature columns available before routing.",
    )
    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.0,
        help="Utility is correctness - cost_weight * cost.",
    )
    parser.add_argument(
        "--cost-weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional list of cost weights to evaluate in one run.",
    )
    parser.add_argument(
        "--split-column",
        default=None,
        help="Optional column with train/val/test labels. If absent, a random split is used.",
    )
    parser.add_argument(
        "--heldout-task-types",
        nargs="+",
        default=None,
        help="Optional task_type values to hold out for evaluation.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=229)
    parser.add_argument(
        "--text-feature",
        default=None,
        help="Optional text column for TF-IDF router variants, usually `prompt`.",
    )
    parser.add_argument(
        "--threshold-feature",
        default=None,
        help=(
            "Optional numeric feature for a threshold cascade. "
            "If feature >= threshold, choose the first model; otherwise choose the last model."
        ),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.7, 0.9],
        help="Threshold values for --threshold-feature.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=None,
        help="Optional CSV path for saving the result table.",
    )
    return parser.parse_args()


def validate_columns(
    df: pd.DataFrame,
    models: Iterable[str],
    features: Iterable[str],
    text_feature: str | None = None,
) -> None:
    missing = []
    for col in features:
        if col not in df.columns:
            missing.append(col)
    if text_feature and text_feature not in df.columns:
        missing.append(text_feature)
    for model in models:
        for suffix in ("correct", "cost"):
            col = f"{model}_{suffix}"
            if col not in df.columns:
                missing.append(col)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def utility_matrix(df: pd.DataFrame, models: list[str], cost_weight: float) -> np.ndarray:
    correct = df[[f"{m}_correct" for m in models]].to_numpy(dtype=float)
    cost = df[[f"{m}_cost" for m in models]].to_numpy(dtype=float)
    return correct - cost_weight * cost


def oracle_labels(df: pd.DataFrame, models: list[str], cost_weight: float) -> np.ndarray:
    return utility_matrix(df, models, cost_weight).argmax(axis=1)


def chosen_values(
    df: pd.DataFrame,
    models: list[str],
    choices: np.ndarray,
    cost_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correct = df[[f"{m}_correct" for m in models]].to_numpy(dtype=float)
    cost = df[[f"{m}_cost" for m in models]].to_numpy(dtype=float)
    row_idx = np.arange(len(df))
    chosen_correct = correct[row_idx, choices]
    chosen_cost = cost[row_idx, choices]
    chosen_utility = chosen_correct - cost_weight * chosen_cost
    return chosen_correct, chosen_cost, chosen_utility


def evaluate_policy(
    name: str,
    df: pd.DataFrame,
    models: list[str],
    choices: np.ndarray,
    cost_weight: float,
    oracle: np.ndarray,
) -> PolicyResult:
    chosen_correct, chosen_cost, chosen_utility = chosen_values(
        df, models, choices, cost_weight
    )
    oracle_utility = utility_matrix(df, models, cost_weight).max(axis=1)
    routing_acc = accuracy_score(oracle, choices) if len(np.unique(oracle)) > 1 else None
    return PolicyResult(
        name=name,
        accuracy=float(chosen_correct.mean()),
        avg_cost=float(chosen_cost.mean()),
        utility=float(chosen_utility.mean()),
        regret=float((oracle_utility - chosen_utility).mean()),
        cost_weight=cost_weight,
        routing_accuracy=None if routing_acc is None else float(routing_acc),
    )


def split_data(
    df: pd.DataFrame,
    split_column: str | None,
    heldout_task_types: list[str] | None,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if heldout_task_types:
        if "task_type" not in df.columns:
            raise ValueError("--heldout-task-types requires a task_type column.")
        heldout = set(heldout_task_types)
        train = df[~df["task_type"].isin(heldout)].copy()
        test = df[df["task_type"].isin(heldout)].copy()
        if train.empty or test.empty:
            raise ValueError(
                "Held-out split produced empty train or eval data. "
                f"Requested heldout task types: {', '.join(heldout_task_types)}"
            )
        return train, test

    if split_column:
        if split_column not in df.columns:
            raise ValueError(f"Split column {split_column!r} not found.")
        train = df[df[split_column].isin(["train", "tr"])].copy()
        test = df[df[split_column].isin(["val", "valid", "validation", "test", "te"])].copy()
        if train.empty or test.empty:
            raise ValueError(
                f"{split_column!r} must contain train and val/test rows for this script."
            )
        return train, test

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train.copy(), test.copy()


def results_table(results: list[PolicyResult]) -> pd.DataFrame:
    table = pd.DataFrame([r.__dict__ for r in results])
    return table.rename(
        columns={
            "name": "policy",
            "avg_cost": "avg_cost",
            "routing_accuracy": "route_acc",
        }
    )


def print_results(results: list[PolicyResult]) -> None:
    table = results_table(results)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def evaluate_for_cost_weight(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models: list[str],
    features: list[str],
    cost_weight: float,
    text_feature: str | None,
    threshold_feature: str | None,
    thresholds: list[float],
    random_state: int,
) -> list[PolicyResult]:
    y_train = oracle_labels(train_df, models, cost_weight)
    y_test = oracle_labels(test_df, models, cost_weight)
    results: list[PolicyResult] = []

    for idx, model in enumerate(models):
        choices = np.full(len(test_df), idx, dtype=int)
        results.append(
            evaluate_policy(
                name=f"always_{model}",
                df=test_df,
                models=models,
                choices=choices,
                cost_weight=cost_weight,
                oracle=y_test,
            )
        )

    rng = np.random.default_rng(random_state)
    random_choices = rng.integers(0, len(models), size=len(test_df))
    results.append(
        evaluate_policy(
            name="random",
            df=test_df,
            models=models,
            choices=random_choices,
            cost_weight=cost_weight,
            oracle=y_test,
        )
    )

    results.append(
        evaluate_policy(
            name="oracle",
            df=test_df,
            models=models,
            choices=y_test,
            cost_weight=cost_weight,
            oracle=y_test,
        )
    )

    if threshold_feature:
        if threshold_feature not in test_df.columns:
            raise ValueError(f"Threshold feature {threshold_feature!r} not found.")
        feature_values = test_df[threshold_feature].to_numpy(dtype=float)
        for threshold in thresholds:
            choices = np.where(feature_values >= threshold, 0, len(models) - 1)
            results.append(
                evaluate_policy(
                    name=f"threshold_{threshold_feature}_{threshold:g}",
                    df=test_df,
                    models=models,
                    choices=choices,
                    cost_weight=cost_weight,
                    oracle=y_test,
                )
            )

    for name, estimator in base_router_estimators(random_state).items():
        if len(np.unique(y_train)) < 2:
            print(
                f"Skipping {name} at cost_weight={cost_weight:g}: "
                "training labels contain only one class."
            )
            continue
        router = fit_router(estimator, train_df, features, y_train)
        choices = router.predict(test_df[features])
        results.append(
            evaluate_policy(
                name=name,
                df=test_df,
                models=models,
                choices=choices,
                cost_weight=cost_weight,
                oracle=y_test,
            )
        )

    if text_feature:
        input_columns = features + [text_feature]
        for name, estimator in text_router_estimators(random_state).items():
            if len(np.unique(y_train)) < 2:
                print(
                    f"Skipping {name} at cost_weight={cost_weight:g}: "
                    "training labels contain only one class."
                )
                continue
            router = fit_router(
                estimator,
                train_df,
                features,
                y_train,
                text_feature=text_feature,
            )
            choices = router.predict(test_df[input_columns])
            results.append(
                evaluate_policy(
                    name=name,
                    df=test_df,
                    models=models,
                    choices=choices,
                    cost_weight=cost_weight,
                    oracle=y_test,
                )
            )
    return results


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    models = list(args.models)
    features = list(args.features)
    validate_columns(df, models, features, text_feature=args.text_feature)

    train_df, test_df = split_data(
        df,
        split_column=args.split_column,
        heldout_task_types=args.heldout_task_types,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    results: list[PolicyResult] = []
    cost_weights = args.cost_weights if args.cost_weights is not None else [args.cost_weight]
    for cost_weight in cost_weights:
        results.extend(
            evaluate_for_cost_weight(
                train_df=train_df,
                test_df=test_df,
                models=models,
                features=features,
                cost_weight=cost_weight,
                text_feature=args.text_feature,
                threshold_feature=args.threshold_feature,
                thresholds=args.thresholds,
                random_state=args.random_state,
            )
        )

    print(f"Loaded {len(df)} rows: {len(train_df)} train, {len(test_df)} eval")
    print(f"Models: {', '.join(models)}")
    print(f"Features: {', '.join(features)}")
    if args.text_feature:
        print(f"Text feature: {args.text_feature}")
    if args.heldout_task_types:
        print(f"Held-out task types: {', '.join(args.heldout_task_types)}")
    print(f"Cost weights: {', '.join(f'{weight:g}' for weight in cost_weights)}")
    print()
    print_results(results)
    if args.results_output:
        args.results_output.parent.mkdir(parents=True, exist_ok=True)
        results_table(results).to_csv(args.results_output, index=False)
        print()
        print(f"Saved results to {args.results_output}")


if __name__ == "__main__":
    main()
