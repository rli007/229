#!/usr/bin/env python3
"""CS229 class analysis: re-runs routing experiments on the existing data.

Loads `data/openrouter_mixed_10models_787_router_dataset.csv` and produces
result tables that are designed for a CS229 writeup:

  results/class_pair_sweep_summary.csv
      30-seed mean +/- std accuracy / utility / regret for the top
      complementary pairs across multiple routers.

  results/class_feature_ablation_summary.csv
      30-seed feature ablation on the most promising pair.

  results/class_no_dominator_summary.csv
      30-seed accuracy routing with `gemini_flash_lite` excluded so the
      multiclass label is not collapsed to a single dominator.

  results/class_heldout_task_summary.csv
      Train router on four task types, evaluate on the fifth.

  results/class_cost_pareto.csv
      Cost-aware Pareto frontier: best fixed, oracle, and best learned
      router accuracy/cost across a range of cost weights with
      normalized [0, 1] per-call costs.

  results/class_per_task_breakdown.csv
      Per-task accuracy for the best learned router vs the best fixed
      model on the most promising pair, averaged over 30 seeds.

The script is self-contained: it does not call OpenRouter, it only
trains and evaluates classical ML routers on the cached labels.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.router_baselines import evaluate_for_cost_weight
from routing.router_models import (
    base_router_estimators,
    feature_pipeline,
    text_router_estimators,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


DATA_PATH = PROJECT_ROOT / "data" / "openrouter_mixed_10models_787_router_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
SEEDS = list(range(229, 259))


METADATA_FEATURES = [
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

CHEAP_PROBE_FEATURES = [
    "cheap_answer_chars",
    "cheap_answer_words",
    "cheap_parsed_answer",
    "cheap_contains_uncertainty",
    "cheap_self_confidence",
    "cheap_sample_agreement",
]

ALL_FEATURES = METADATA_FEATURES + CHEAP_PROBE_FEATURES


VIABLE_MODELS = [
    "gemini_flash_lite",
    "gemini_flash",
    "qwen_30b",
    "llama_70b",
    "deepseek_v3",
    "granite_8b",
]

NO_DOMINATOR_MODELS = [
    "gemini_flash",
    "qwen_30b",
    "llama_70b",
    "deepseek_v3",
    "granite_8b",
]


# Pairs ranked by oracle gap from earlier analysis. The first four are the
# main candidates for "can a router beat the best fixed model?" The last is
# the one already reported in the project notes; included as a baseline.
TOP_PAIRS = [
    ("gemini_flash_granite", ["gemini_flash", "granite_8b"]),
    ("gemini_flash_llama", ["gemini_flash", "llama_70b"]),
    ("llama_granite", ["llama_70b", "granite_8b"]),
    ("qwen_llama", ["qwen_30b", "llama_70b"]),
    ("llama_deepseek", ["llama_70b", "deepseek_v3"]),
    ("gemini_lite_granite", ["gemini_flash_lite", "granite_8b"]),
]

ABLATION_PAIR = ("llama_granite", ["llama_70b", "granite_8b"])

NO_DOMINATOR_GROUPS = [
    ("triple_gemini_qwen_llama", ["gemini_flash", "qwen_30b", "llama_70b"]),
    ("triple_qwen_llama_granite", ["qwen_30b", "llama_70b", "granite_8b"]),
    ("triple_qwen_llama_deepseek", ["qwen_30b", "llama_70b", "deepseek_v3"]),
    ("triple_gemini_llama_granite", ["gemini_flash", "llama_70b", "granite_8b"]),
]

HELDOUT_PAIR = ("llama_granite", ["llama_70b", "granite_8b"])


@dataclass
class ResultRow:
    bucket: str
    group: str
    models: str
    feature_set: str
    policy: str
    accuracy_mean: float
    accuracy_std: float
    utility_mean: float
    regret_mean: float
    avg_cost_mean: float
    route_acc_mean: float
    n_seeds: int


def normalize_costs(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Return a copy of df with `<model>_cost_norm` columns in [0, 1].

    Each model's actual OpenRouter cost (already in `<model>_cost`) is
    divided by the maximum observed mean cost across the candidate set.
    This lets us sweep a cost weight in roughly [0, 1] and recover a
    sensible accuracy/cost tradeoff.
    """
    df = df.copy()
    cost_cols = [f"{m}_cost" for m in models]
    max_cost = max(df[col].max() for col in cost_cols)
    if max_cost <= 0:
        max_cost = 1.0
    for m in models:
        df[f"{m}_cost_norm"] = df[f"{m}_cost"] / max_cost
    return df


def use_normalized_cost(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Return a view where `<model>_cost` is replaced by `<model>_cost_norm`."""
    df = df.copy()
    for m in models:
        df[f"{m}_cost"] = df[f"{m}_cost_norm"]
    return df


def aggregate_seed_results(
    bucket: str,
    group: str,
    models: list[str],
    feature_set: str,
    seed_rows: list[pd.DataFrame],
) -> list[ResultRow]:
    """Roll up per-seed result tables into one row per policy."""
    if not seed_rows:
        return []
    df = pd.concat(seed_rows, ignore_index=True)
    out: list[ResultRow] = []
    for policy, sub in df.groupby("policy", sort=False):
        out.append(
            ResultRow(
                bucket=bucket,
                group=group,
                models=",".join(models),
                feature_set=feature_set,
                policy=policy,
                accuracy_mean=float(sub["accuracy"].mean()),
                accuracy_std=float(sub["accuracy"].std(ddof=1)) if len(sub) > 1 else 0.0,
                utility_mean=float(sub["utility"].mean()),
                regret_mean=float(sub["regret"].mean()),
                avg_cost_mean=float(sub["avg_cost"].mean()),
                route_acc_mean=float(sub["route_acc"].mean()) if "route_acc" in sub.columns else float("nan"),
                n_seeds=int(len(sub)),
            )
        )
    return out


def to_dataframe(rows: list[ResultRow]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows])


def run_pair_sweep(
    df: pd.DataFrame,
    pairs: list[tuple[str, list[str]]],
    features: list[str],
    text_feature: str | None,
    feature_set_name: str,
    bucket: str,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for group_name, models in pairs:
        seed_tables: list[pd.DataFrame] = []
        for seed in SEEDS:
            train_df, test_df = train_test_split(
                df,
                test_size=0.25,
                random_state=seed,
                stratify=df["task_type"] if "task_type" in df.columns else None,
            )
            results = evaluate_for_cost_weight(
                train_df=train_df.copy(),
                test_df=test_df.copy(),
                models=models,
                features=features,
                cost_weight=0.0,
                text_feature=text_feature,
                threshold_feature=None,
                thresholds=[],
                task_router_rules={},
                skip_mlp=False,
                random_state=seed,
            )
            seed_tables.append(pd.DataFrame([r.__dict__ for r in results]).rename(
                columns={"name": "policy", "routing_accuracy": "route_acc"}
            ))
        rows.extend(
            aggregate_seed_results(bucket, group_name, models, feature_set_name, seed_tables)
        )
    return rows


def run_heldout_task(
    df: pd.DataFrame,
    pair: tuple[str, list[str]],
    features: list[str],
    text_feature: str | None,
) -> list[ResultRow]:
    group_name, models = pair
    rows: list[ResultRow] = []
    task_types = sorted(df["task_type"].unique())
    for held in task_types:
        train_df = df[df["task_type"] != held].copy()
        test_df = df[df["task_type"] == held].copy()
        if train_df.empty or test_df.empty:
            continue
        seed_tables: list[pd.DataFrame] = []
        for seed in SEEDS[:5]:
            results = evaluate_for_cost_weight(
                train_df=train_df,
                test_df=test_df,
                models=models,
                features=features,
                cost_weight=0.0,
                text_feature=text_feature,
                threshold_feature=None,
                thresholds=[],
                task_router_rules={},
                skip_mlp=False,
                random_state=seed,
            )
            seed_tables.append(pd.DataFrame([r.__dict__ for r in results]).rename(
                columns={"name": "policy", "routing_accuracy": "route_acc"}
            ))
        rows.extend(
            aggregate_seed_results(
                bucket="heldout_task",
                group=f"{group_name}_holdout_{held}",
                models=models,
                feature_set="metadata+cheap_probe",
                seed_rows=seed_tables,
            )
        )
    return rows


def run_cost_pareto(
    df: pd.DataFrame,
    models: list[str],
    features: list[str],
    text_feature: str | None,
    cost_weights: list[float],
) -> pd.DataFrame:
    df_norm = normalize_costs(df, models)
    df_view = use_normalized_cost(df_norm, models)
    rows: list[dict[str, object]] = []
    for cost_weight in cost_weights:
        seed_tables: list[pd.DataFrame] = []
        for seed in SEEDS[:10]:
            train_df, test_df = train_test_split(
                df_view,
                test_size=0.25,
                random_state=seed,
                stratify=df_view["task_type"],
            )
            results = evaluate_for_cost_weight(
                train_df=train_df.copy(),
                test_df=test_df.copy(),
                models=models,
                features=features,
                cost_weight=cost_weight,
                text_feature=text_feature,
                threshold_feature=None,
                thresholds=[],
                task_router_rules={},
                skip_mlp=False,
                random_state=seed,
            )
            seed_tables.append(pd.DataFrame([r.__dict__ for r in results]).rename(
                columns={"name": "policy", "routing_accuracy": "route_acc"}
            ))
        merged = pd.concat(seed_tables, ignore_index=True)
        for policy, sub in merged.groupby("policy", sort=False):
            rows.append({
                "cost_weight": cost_weight,
                "policy": policy,
                "accuracy_mean": float(sub["accuracy"].mean()),
                "accuracy_std": float(sub["accuracy"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "avg_cost_mean": float(sub["avg_cost"].mean()),
                "utility_mean": float(sub["utility"].mean()),
                "regret_mean": float(sub["regret"].mean()),
                "route_acc_mean": float(sub["route_acc"].mean()),
                "n_seeds": len(sub),
            })
    return pd.DataFrame(rows)


def per_task_breakdown(
    df: pd.DataFrame,
    pair: tuple[str, list[str]],
    features: list[str],
    text_feature: str | None,
) -> pd.DataFrame:
    """Per-task-type accuracy of the best fixed model and best learned router.

    Best learned router is selected by 30-seed mean accuracy.
    """
    group_name, models = pair
    rows: list[dict[str, object]] = []
    estimators = base_router_estimators(229)
    estimator_name = "linear_svm_router"
    estimator = estimators[estimator_name]
    per_task_records: list[dict[str, object]] = []
    for seed in SEEDS:
        train_df, test_df = train_test_split(
            df,
            test_size=0.25,
            random_state=seed,
            stratify=df["task_type"],
        )
        oracle_label = (
            test_df[[f"{m}_correct" for m in models]].to_numpy().argmax(axis=1)
        )
        best_fixed_idx = int(
            np.argmax(
                [train_df[f"{m}_correct"].mean() for m in models]
            )
        )
        best_fixed_model = models[best_fixed_idx]
        y_train = (
            train_df[[f"{m}_correct" for m in models]]
            .to_numpy()
            .argmax(axis=1)
        )
        pipe = Pipeline(
            steps=[
                ("features", feature_pipeline(train_df, features, text_feature=None)),
                ("clf", estimator),
            ]
        )
        pipe.fit(train_df[features], y_train)
        choices = pipe.predict(test_df[features])
        idx = np.arange(len(test_df))
        correct_matrix = test_df[[f"{m}_correct" for m in models]].to_numpy()
        routed_correct = correct_matrix[idx, choices]
        fixed_correct = test_df[f"{best_fixed_model}_correct"].to_numpy()
        oracle_correct = correct_matrix[idx, oracle_label]
        for task in sorted(test_df["task_type"].unique()):
            mask = (test_df["task_type"] == task).to_numpy()
            if not mask.any():
                continue
            per_task_records.append({
                "seed": seed,
                "task_type": task,
                "best_fixed_model": best_fixed_model,
                "fixed_acc": float(fixed_correct[mask].mean()),
                "router_acc": float(routed_correct[mask].mean()),
                "oracle_acc": float(oracle_correct[mask].mean()),
                "n": int(mask.sum()),
                "router_chooses_fixed_model_share": float(
                    (choices[mask] == best_fixed_idx).mean()
                ),
            })
    detail = pd.DataFrame(per_task_records)
    summary = (
        detail.groupby(["task_type"], as_index=False)
        .agg(
            best_fixed_model=("best_fixed_model", "first"),
            fixed_acc_mean=("fixed_acc", "mean"),
            router_acc_mean=("router_acc", "mean"),
            oracle_acc_mean=("oracle_acc", "mean"),
            router_acc_std=("router_acc", "std"),
            router_chooses_fixed_share=("router_chooses_fixed_model_share", "mean"),
            n_seeds=("seed", "nunique"),
            avg_n_per_split=("n", "mean"),
        )
    )
    summary["gain_pp"] = (summary["router_acc_mean"] - summary["fixed_acc_mean"]) * 100
    summary.insert(0, "group", group_name)
    summary.insert(1, "models", ",".join(models))
    summary.insert(2, "router", estimator_name)
    return summary


def main() -> None:
    print(f"Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  rows={len(df)}  task_type counts:")
    print(df["task_type"].value_counts().to_string())
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_pair_rows: list[ResultRow] = []
    all_ablation_rows: list[ResultRow] = []

    print("=== 1. 30-seed pair sweep on top complementary pairs ===")
    pair_rows = run_pair_sweep(
        df=df,
        pairs=TOP_PAIRS,
        features=ALL_FEATURES,
        text_feature="prompt",
        feature_set_name="metadata+cheap_probe+tfidf",
        bucket="pair_sweep",
    )
    all_pair_rows.extend(pair_rows)
    pair_df = to_dataframe(pair_rows)
    pair_df.to_csv(RESULTS_DIR / "class_pair_sweep_summary.csv", index=False)
    print(f"  Saved {len(pair_df)} rows to results/class_pair_sweep_summary.csv")
    print()

    print("=== 2. Feature ablation on llama_70b + granite_8b ===")
    feature_sets = [
        ("metadata", METADATA_FEATURES, None),
        ("metadata+cheap_probe", ALL_FEATURES, None),
        ("metadata+cheap_probe+tfidf", ALL_FEATURES, "prompt"),
    ]
    for name, feats, text_feat in feature_sets:
        rows = run_pair_sweep(
            df=df,
            pairs=[ABLATION_PAIR],
            features=feats,
            text_feature=text_feat,
            feature_set_name=name,
            bucket="feature_ablation",
        )
        all_ablation_rows.extend(rows)
    abl_df = to_dataframe(all_ablation_rows)
    abl_df.to_csv(RESULTS_DIR / "class_feature_ablation_summary.csv", index=False)
    print(f"  Saved {len(abl_df)} rows to results/class_feature_ablation_summary.csv")
    print()

    print("=== 3. No-dominator track (excluding gemini_flash_lite) ===")
    no_dom_rows = run_pair_sweep(
        df=df,
        pairs=NO_DOMINATOR_GROUPS,
        features=ALL_FEATURES,
        text_feature="prompt",
        feature_set_name="metadata+cheap_probe+tfidf",
        bucket="no_dominator",
    )
    nd_df = to_dataframe(no_dom_rows)
    nd_df.to_csv(RESULTS_DIR / "class_no_dominator_summary.csv", index=False)
    print(f"  Saved {len(nd_df)} rows to results/class_no_dominator_summary.csv")
    print()

    print("=== 4. Held-out-task generalization ===")
    holdout_rows = run_heldout_task(
        df=df,
        pair=HELDOUT_PAIR,
        features=ALL_FEATURES,
        text_feature="prompt",
    )
    ho_df = to_dataframe(holdout_rows)
    ho_df.to_csv(RESULTS_DIR / "class_heldout_task_summary.csv", index=False)
    print(f"  Saved {len(ho_df)} rows to results/class_heldout_task_summary.csv")
    print()

    print("=== 5. Cost-aware Pareto frontier (6 viable models, normalized costs) ===")
    cost_weights = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    pareto_df = run_cost_pareto(
        df=df,
        models=VIABLE_MODELS,
        features=ALL_FEATURES,
        text_feature="prompt",
        cost_weights=cost_weights,
    )
    pareto_df.to_csv(RESULTS_DIR / "class_cost_pareto.csv", index=False)
    print(f"  Saved {len(pareto_df)} rows to results/class_cost_pareto.csv")
    print()

    print("=== 6. Per-task breakdown for the best learned router (linear_svm) ===")
    per_task = per_task_breakdown(
        df=df,
        pair=ABLATION_PAIR,
        features=ALL_FEATURES,
        text_feature=None,
    )
    per_task.to_csv(RESULTS_DIR / "class_per_task_breakdown.csv", index=False)
    print(per_task.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    print("=== Headline: best learned router vs best fixed per pair ===")
    best_table = build_headline(pair_df)
    best_table.to_csv(RESULTS_DIR / "class_pair_headline.csv", index=False)
    print(best_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def build_headline(pair_df: pd.DataFrame) -> pd.DataFrame:
    """For each pair, pick the best fixed model and the best learned router."""
    rows = []
    for (group, models), sub in pair_df.groupby(["group", "models"]):
        fixed = sub[sub["policy"].str.startswith("always_")]
        learned = sub[~sub["policy"].isin(["random", "oracle"]) & ~sub["policy"].str.startswith("always_")]
        oracle = sub[sub["policy"] == "oracle"].iloc[0]
        best_fixed = fixed.loc[fixed["accuracy_mean"].idxmax()]
        best_learned = learned.loc[learned["accuracy_mean"].idxmax()]
        gain_pp = (best_learned["accuracy_mean"] - best_fixed["accuracy_mean"]) * 100
        rows.append({
            "group": group,
            "models": models,
            "best_fixed": best_fixed["policy"],
            "fixed_acc_mean": best_fixed["accuracy_mean"],
            "fixed_acc_std": best_fixed["accuracy_std"],
            "best_learned": best_learned["policy"],
            "learned_acc_mean": best_learned["accuracy_mean"],
            "learned_acc_std": best_learned["accuracy_std"],
            "gain_pp": gain_pp,
            "oracle_acc_mean": oracle["accuracy_mean"],
            "headroom_pp": (oracle["accuracy_mean"] - best_fixed["accuracy_mean"]) * 100,
        })
    return pd.DataFrame(rows).sort_values("gain_pp", ascending=False)


if __name__ == "__main__":
    main()
