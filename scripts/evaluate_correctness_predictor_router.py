#!/usr/bin/env python3
"""Evaluate routers that predict each model's correctness separately.

Instead of learning one multiclass "best model" label, this trains one binary
classifier per candidate model to estimate P(model is correct | features), then
routes to the model with the highest predicted score. This is useful when the
oracle route labels are highly imbalanced but several models have complementary
errors.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.router_models import feature_pipeline


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
    "cheap_answer_chars",
    "cheap_answer_words",
    "cheap_parsed_answer",
    "cheap_contains_uncertainty",
    "cheap_self_confidence",
    "cheap_sample_agreement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Router CSV.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Optional named group: name=model_a,model_b,... Can be repeated.",
    )
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument(
        "--text-feature",
        default="prompt",
        help="Optional text feature for TF-IDF variants. Use '' to disable.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seeds", nargs="+", type=int, default=[229, 230, 231, 232, 233])
    parser.add_argument("--cost-weight", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/correctness_predictor_router_summary.csv"),
    )
    parser.add_argument("--details-output", type=Path, default=None)
    return parser.parse_args()


def parse_groups(group_args: list[str], models: list[str]) -> dict[str, list[str]]:
    if group_args:
        groups = {}
        for group_arg in group_args:
            if "=" not in group_arg:
                raise ValueError(f"Invalid group {group_arg!r}; expected name=m1,m2,...")
            name, raw_models = group_arg.split("=", 1)
            group_models = [model.strip() for model in raw_models.split(",") if model.strip()]
            if len(group_models) < 2:
                raise ValueError(f"Group {name!r} needs at least two models.")
            unknown = sorted(set(group_models) - set(models))
            if unknown:
                raise ValueError(f"Unknown models in {name!r}: {', '.join(unknown)}")
            groups[name.strip()] = group_models
        return groups

    groups = {}
    for size in (2, 3):
        for combo in itertools.combinations(models, size):
            groups[f"{size}model_" + "_".join(combo)] = list(combo)
    return groups


def validate_columns(
    df: pd.DataFrame,
    models: list[str],
    features: list[str],
    text_feature: str | None,
) -> None:
    missing = [col for col in features if col not in df.columns]
    if text_feature and text_feature not in df.columns:
        missing.append(text_feature)
    for model in models:
        for suffix in ("correct", "cost"):
            col = f"{model}_{suffix}"
            if col not in df.columns:
                missing.append(col)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def estimators(random_state: int) -> dict[str, object]:
    return {
        "correctness_logistic": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
        "correctness_random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
        "correctness_extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
        ),
    }


def utility_matrix(df: pd.DataFrame, models: list[str], cost_weight: float) -> np.ndarray:
    correct = df[[f"{model}_correct" for model in models]].to_numpy(dtype=float)
    cost = df[[f"{model}_cost" for model in models]].to_numpy(dtype=float)
    return correct - cost_weight * cost


def evaluate_choices(
    name: str,
    df: pd.DataFrame,
    models: list[str],
    choices: np.ndarray,
    cost_weight: float,
    oracle: np.ndarray,
) -> dict[str, float | str]:
    correct = df[[f"{model}_correct" for model in models]].to_numpy(dtype=float)
    cost = df[[f"{model}_cost" for model in models]].to_numpy(dtype=float)
    idx = np.arange(len(df))
    chosen_correct = correct[idx, choices]
    chosen_cost = cost[idx, choices]
    utility = chosen_correct - cost_weight * chosen_cost
    oracle_utility = utility_matrix(df, models, cost_weight).max(axis=1)
    return {
        "policy": name,
        "accuracy": float(chosen_correct.mean()),
        "avg_cost": float(chosen_cost.mean()),
        "utility": float(utility.mean()),
        "regret": float((oracle_utility - utility).mean()),
        "route_acc": float(accuracy_score(oracle, choices)),
    }


def fit_predictor_router(
    estimator,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models: list[str],
    features: list[str],
    text_feature: str | None,
    cost_weight: float,
) -> np.ndarray:
    input_cols = features + ([text_feature] if text_feature else [])
    expected_cost = train_df[[f"{model}_cost" for model in models]].mean().to_numpy()
    scores = []
    for model in models:
        y = train_df[f"{model}_correct"].to_numpy(dtype=int)
        if len(np.unique(y)) == 1:
            prob = np.repeat(float(y[0]), len(test_df))
        else:
            pipe = Pipeline(
                steps=[
                    ("features", feature_pipeline(train_df, features, text_feature)),
                    ("clf", estimator),
                ]
            )
            pipe.fit(train_df[input_cols], y)
            prob = pipe.predict_proba(test_df[input_cols])[:, 1]
        scores.append(prob)
    score_matrix = np.column_stack(scores)
    score_matrix = score_matrix - cost_weight * expected_cost.reshape(1, -1)
    return score_matrix.argmax(axis=1)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    summary = (
        details.groupby(["group", "models", "policy"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            avg_cost_mean=("avg_cost", "mean"),
            utility_mean=("utility", "mean"),
            regret_mean=("regret", "mean"),
            route_acc_mean=("route_acc", "mean"),
        )
        .sort_values(["group", "accuracy_mean", "regret_mean"], ascending=[True, False, True])
    )
    summary["accuracy_std"] = summary["accuracy_std"].fillna(0.0)
    return summary


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    models = list(args.models)
    features = list(args.features)
    text_feature = args.text_feature or None
    validate_columns(df, models, features, text_feature)
    groups = parse_groups(args.group, models)

    rows = []
    for group_name, group_models in groups.items():
        validate_columns(df, group_models, features, text_feature)
        model_names = ",".join(group_models)
        for seed in args.seeds:
            train_df, test_df = train_test_split(
                df,
                test_size=args.test_size,
                random_state=seed,
                stratify=df["task_type"] if "task_type" in df.columns else None,
            )
            oracle = utility_matrix(test_df, group_models, args.cost_weight).argmax(axis=1)
            for model_idx, model in enumerate(group_models):
                choices = np.full(len(test_df), model_idx)
                row = evaluate_choices(
                    f"always_{model}",
                    test_df,
                    group_models,
                    choices,
                    args.cost_weight,
                    oracle,
                )
                row.update(group=group_name, models=model_names, seed=seed)
                rows.append(row)
            oracle_row = evaluate_choices(
                "oracle",
                test_df,
                group_models,
                oracle,
                args.cost_weight,
                oracle,
            )
            oracle_row.update(group=group_name, models=model_names, seed=seed)
            rows.append(oracle_row)
            rng = np.random.default_rng(seed)
            random_choices = rng.integers(0, len(group_models), size=len(test_df))
            random_row = evaluate_choices(
                "random",
                test_df,
                group_models,
                random_choices,
                args.cost_weight,
                oracle,
            )
            random_row.update(group=group_name, models=model_names, seed=seed)
            rows.append(random_row)

            for name, estimator in estimators(seed).items():
                choices = fit_predictor_router(
                    estimator=estimator,
                    train_df=train_df,
                    test_df=test_df,
                    models=group_models,
                    features=features,
                    text_feature=None,
                    cost_weight=args.cost_weight,
                )
                row = evaluate_choices(
                    name,
                    test_df,
                    group_models,
                    choices,
                    args.cost_weight,
                    oracle,
                )
                row.update(group=group_name, models=model_names, seed=seed)
                rows.append(row)

            if text_feature:
                choices = fit_predictor_router(
                    estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
                    train_df=train_df,
                    test_df=test_df,
                    models=group_models,
                    features=features,
                    text_feature=text_feature,
                    cost_weight=args.cost_weight,
                )
                row = evaluate_choices(
                    "correctness_tfidf_logistic",
                    test_df,
                    group_models,
                    choices,
                    args.cost_weight,
                    oracle,
                )
                row.update(group=group_name, models=model_names, seed=seed)
                rows.append(row)

    details = pd.DataFrame(rows)
    summary = summarize(details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    if args.details_output:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(args.details_output, index=False)
    print(f"Loaded {len(df)} rows")
    print(f"Evaluated {len(groups)} groups across {len(args.seeds)} random splits")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved summary to {args.output}")
    if args.details_output:
        print(f"Saved details to {args.details_output}")


if __name__ == "__main__":
    main()
