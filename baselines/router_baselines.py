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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PolicyResult:
    name: str
    accuracy: float
    avg_cost: float
    utility: float
    regret: float
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
        "--split-column",
        default=None,
        help="Optional column with train/val/test labels. If absent, a random split is used.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=229)
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


def validate_columns(df: pd.DataFrame, models: Iterable[str], features: Iterable[str]) -> None:
    missing = []
    for col in features:
        if col not in df.columns:
            missing.append(col)
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
        routing_accuracy=None if routing_acc is None else float(routing_acc),
    )


def feature_pipeline(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    categorical = [
        col
        for col in features
        if pd.api.types.is_object_dtype(df[col])
        or isinstance(df[col].dtype, pd.CategoricalDtype)
    ]
    numeric = [col for col in features if col not in categorical]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )


def split_data(
    df: pd.DataFrame,
    split_column: str | None,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def fit_router(
    model_name: str,
    estimator,
    train_df: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("features", feature_pipeline(train_df, features)),
            ("clf", estimator),
        ]
    )
    pipe.fit(train_df[features], labels)
    return pipe


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


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    models = list(args.models)
    features = list(args.features)
    validate_columns(df, models, features)

    train_df, test_df = split_data(
        df,
        split_column=args.split_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    y_train = oracle_labels(train_df, models, args.cost_weight)
    y_test = oracle_labels(test_df, models, args.cost_weight)
    results: list[PolicyResult] = []

    for idx, model in enumerate(models):
        choices = np.full(len(test_df), idx, dtype=int)
        results.append(
            evaluate_policy(
                name=f"always_{model}",
                df=test_df,
                models=models,
                choices=choices,
                cost_weight=args.cost_weight,
                oracle=y_test,
            )
        )

    rng = np.random.default_rng(args.random_state)
    random_choices = rng.integers(0, len(models), size=len(test_df))
    results.append(
        evaluate_policy(
            name="random",
            df=test_df,
            models=models,
            choices=random_choices,
            cost_weight=args.cost_weight,
            oracle=y_test,
        )
    )

    results.append(
        evaluate_policy(
            name="oracle",
            df=test_df,
            models=models,
            choices=y_test,
            cost_weight=args.cost_weight,
            oracle=y_test,
        )
    )

    if args.threshold_feature:
        if args.threshold_feature not in test_df.columns:
            raise ValueError(f"Threshold feature {args.threshold_feature!r} not found.")
        feature_values = test_df[args.threshold_feature].to_numpy(dtype=float)
        for threshold in args.thresholds:
            choices = np.where(feature_values >= threshold, 0, len(models) - 1)
            results.append(
                evaluate_policy(
                    name=f"threshold_{args.threshold_feature}_{threshold:g}",
                    df=test_df,
                    models=models,
                    choices=choices,
                    cost_weight=args.cost_weight,
                    oracle=y_test,
                )
            )

    supervised_models = {
        "logistic_router": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest_router": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=args.random_state,
            class_weight="balanced_subsample",
        ),
    }
    for name, estimator in supervised_models.items():
        router = fit_router(name, estimator, train_df, features, y_train)
        choices = router.predict(test_df[features])
        results.append(
            evaluate_policy(
                name=name,
                df=test_df,
                models=models,
                choices=choices,
                cost_weight=args.cost_weight,
                oracle=y_test,
            )
        )

    print(f"Loaded {len(df)} rows: {len(train_df)} train, {len(test_df)} eval")
    print(f"Models: {', '.join(models)}")
    print(f"Features: {', '.join(features)}")
    print(f"Cost weight: {args.cost_weight}")
    print()
    print_results(results)
    if args.results_output:
        args.results_output.parent.mkdir(parents=True, exist_ok=True)
        results_table(results).to_csv(args.results_output, index=False)
        print()
        print(f"Saved results to {args.results_output}")


if __name__ == "__main__":
    main()
