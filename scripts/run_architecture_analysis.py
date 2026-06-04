#!/usr/bin/env python3
"""Architecture deep-dive for the report.

Runs four things end-to-end on the gemini_flash + llama_70b pair (the headline
pair from the main sweep):
  - Gini-based feature importance from ExtraTrees / RandomForest, averaged
    over 30 stratified seeds so the rankings are actually stable.
  - Per-task routing decision distribution (which model gets picked, by task
    type), again over 30 seeds.
  - Train/test accuracy gap per router family, which is the bias-variance
    diagnostic the report leans on.
  - A 30-seed sweep over the 6 "viable" models (i.e. excluding the broken
    four: mistral_nemo, minimax_m3, ring_1t, step_flash) to confirm the
    headline gain isn't an artifact of the bigger 10-model set.

Each of the four steps writes its own CSV to results/.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.router_baselines import evaluate_for_cost_weight
from routing.router_models import base_router_estimators, feature_pipeline

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


DATA_PATH = PROJECT_ROOT / "data" / "openrouter_mixed_10models_787_router_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
# 30 seeds is honestly overkill for the smaller pairs, but it's cheap and the
# std on the headline pair is borderline (~0.3pp), so I'd rather be safe.
SEEDS = list(range(229, 259))  # cs229

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

STRONG_PAIR = ("gemini_flash_llama", ["gemini_flash", "llama_70b"])

VIABLE_MODELS = [
    "gemini_flash_lite",
    "gemini_flash",
    "qwen_30b",
    "llama_70b",
    "deepseek_v3",
    "granite_8b",
]


def feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    _, models = STRONG_PAIR
    rows: list[dict] = []
    for seed in SEEDS:
        train_df, _ = train_test_split(
            df, test_size=0.25, random_state=seed, stratify=df["task_type"]
        )
        y_train = train_df[[f"{m}_correct" for m in models]].to_numpy().argmax(axis=1)

        for est_name, estimator in [
            (
                "extra_trees_router",
                ExtraTreesClassifier(
                    n_estimators=1000,
                    min_samples_leaf=2,
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
            (
                "random_forest_router",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    random_state=seed,
                    class_weight="balanced_subsample",
                ),
            ),
        ]:
            pipe = Pipeline(
                steps=[
                    ("features", feature_pipeline(train_df, ALL_FEATURES, text_feature=None)),
                    ("clf", estimator),
                ]
            )
            pipe.fit(train_df[ALL_FEATURES], y_train)
            transformer = pipe.named_steps["features"]
            try:
                feature_names = transformer.get_feature_names_out(ALL_FEATURES)
            except Exception:
                feature_names = np.array([f"f{i}" for i in range(len(pipe.named_steps["clf"].feature_importances_))])
            importances = pipe.named_steps["clf"].feature_importances_
            for fname, imp in zip(feature_names, importances):
                rows.append(
                    {
                        "seed": seed,
                        "router": est_name,
                        "feature": fname,
                        "importance": float(imp),
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["router", "feature"], as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["router", "importance_mean"], ascending=[True, False])
    )
    return summary


def router_decision_per_task(df: pd.DataFrame) -> pd.DataFrame:
    _, models = STRONG_PAIR
    rows: list[dict] = []
    for seed in SEEDS:
        train_df, test_df = train_test_split(
            df, test_size=0.25, random_state=seed, stratify=df["task_type"]
        )
        y_train = train_df[[f"{m}_correct" for m in models]].to_numpy().argmax(axis=1)
        estimator = ExtraTreesClassifier(
            n_estimators=1000,
            min_samples_leaf=2,
            random_state=seed,
            class_weight="balanced",
        )
        pipe = Pipeline(
            steps=[
                ("features", feature_pipeline(train_df, ALL_FEATURES, text_feature=None)),
                ("clf", estimator),
            ]
        )
        pipe.fit(train_df[ALL_FEATURES], y_train)
        choices = pipe.predict(test_df[ALL_FEATURES])
        correct_matrix = test_df[[f"{m}_correct" for m in models]].to_numpy()
        idx = np.arange(len(test_df))
        routed_correct = correct_matrix[idx, choices]
        for task in sorted(test_df["task_type"].unique()):
            mask = (test_df["task_type"] == task).to_numpy()
            if not mask.any():
                continue
            for model_idx, model in enumerate(models):
                rows.append(
                    {
                        "seed": seed,
                        "task_type": task,
                        "model_picked": model,
                        "share_picked": float((choices[mask] == model_idx).mean()),
                        "fixed_acc": float(test_df.loc[mask, f"{model}_correct"].mean()),
                        "n": int(mask.sum()),
                    }
                )
            rows.append(
                {
                    "seed": seed,
                    "task_type": task,
                    "model_picked": "router_total_acc",
                    "share_picked": float(routed_correct[mask].mean()),
                    "fixed_acc": float("nan"),
                    "n": int(mask.sum()),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["task_type", "model_picked"], as_index=False)
        .agg(
            share_or_acc_mean=("share_picked", "mean"),
            share_or_acc_std=("share_picked", "std"),
            fixed_acc_mean=("fixed_acc", "mean"),
            n=("n", "mean"),
        )
        .sort_values(["task_type", "model_picked"])
    )
    return summary


def bias_variance_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    _, models = STRONG_PAIR
    rows: list[dict] = []
    estimators = base_router_estimators(229)
    for seed in SEEDS[:10]:
        train_df, test_df = train_test_split(
            df, test_size=0.25, random_state=seed, stratify=df["task_type"]
        )
        train_correct_matrix = train_df[[f"{m}_correct" for m in models]].to_numpy()
        test_correct_matrix = test_df[[f"{m}_correct" for m in models]].to_numpy()
        y_train = train_correct_matrix.argmax(axis=1)

        for router_name, estimator in estimators.items():
            try:
                pipe = Pipeline(
                    steps=[
                        ("features", feature_pipeline(train_df, ALL_FEATURES, text_feature=None)),
                        ("clf", estimator),
                    ]
                )
                pipe.fit(train_df[ALL_FEATURES], y_train)
                train_choices = pipe.predict(train_df[ALL_FEATURES])
                test_choices = pipe.predict(test_df[ALL_FEATURES])
                train_acc = float(
                    train_correct_matrix[np.arange(len(train_df)), train_choices].mean()
                )
                test_acc = float(
                    test_correct_matrix[np.arange(len(test_df)), test_choices].mean()
                )
                rows.append(
                    {
                        "seed": seed,
                        "router": router_name,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "gap": train_acc - test_acc,
                    }
                )
            except Exception:
                continue
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("router", as_index=False)
        .agg(
            train_acc_mean=("train_acc", "mean"),
            test_acc_mean=("test_acc", "mean"),
            gap_mean=("gap", "mean"),
            gap_std=("gap", "std"),
        )
        .sort_values("test_acc_mean", ascending=False)
    )
    return summary


def viable_only_pair_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """30-seed sweep over all 15 pairs of the 6 viable models.

    MLP is skipped here because its runtime dominates and the bias-variance
    diagnostic already shows it's overfitting on this dataset size.
    """
    pairs = []
    for i in range(len(VIABLE_MODELS)):
        for j in range(i + 1, len(VIABLE_MODELS)):
            a, b = VIABLE_MODELS[i], VIABLE_MODELS[j]
            pairs.append((f"{a}_{b}", [a, b]))

    all_summaries: list[pd.DataFrame] = []
    for group_name, models in pairs:
        per_seed = []
        for seed in SEEDS:
            train_df, test_df = train_test_split(
                df, test_size=0.25, random_state=seed, stratify=df["task_type"]
            )
            results = evaluate_for_cost_weight(
                train_df=train_df.copy(),
                test_df=test_df.copy(),
                models=models,
                features=ALL_FEATURES,
                cost_weight=0.0,
                text_feature=None,
                threshold_feature=None,
                thresholds=[],
                task_router_rules={},
                skip_mlp=True,
                random_state=seed,
            )
            per_seed.append(
                pd.DataFrame([r.__dict__ for r in results]).rename(
                    columns={"name": "policy", "routing_accuracy": "route_acc"}
                )
            )
        merged = pd.concat(per_seed, ignore_index=True)
        summary = (
            merged.groupby("policy", as_index=False, sort=False)
            .agg(
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                avg_cost_mean=("avg_cost", "mean"),
            )
        )
        summary["accuracy_std"] = summary["accuracy_std"].fillna(0.0)
        summary.insert(0, "group", group_name)
        summary.insert(1, "models", ",".join(models))
        all_summaries.append(summary)
    return pd.concat(all_summaries, ignore_index=True)


def main() -> None:
    print(f"Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- feature importance on gemini_flash + llama_70b ---")
    fi = feature_importance(df)
    fi.to_csv(RESULTS_DIR / "class_feature_importance.csv", index=False)
    print(fi.head(30).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- router decision per task_type ---")
    rdt = router_decision_per_task(df)
    rdt.to_csv(RESULTS_DIR / "class_router_decision_per_task.csv", index=False)
    print(rdt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- bias-variance diagnostic (train vs test acc) ---")
    bv = bias_variance_diagnostic(df)
    bv.to_csv(RESULTS_DIR / "class_bias_variance_diagnostic.csv", index=False)
    print(bv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- 15 pairs of 6 viable models, 30 seeds, no MLP ---")
    vo = viable_only_pair_sweep(df)
    vo.to_csv(RESULTS_DIR / "class_viable_only_summary.csv", index=False)

    # collapse each pair down to a single (best fixed, best learned, oracle)
    # row for the headline-table view
    headline_rows = []
    for (group, models), sub in vo.groupby(["group", "models"]):
        fixed = sub[sub["policy"].str.startswith("always_")]
        learned = sub[
            ~sub["policy"].isin(["random", "oracle"])
            & ~sub["policy"].str.startswith("always_")
        ]
        oracle_row = sub[sub["policy"] == "oracle"].iloc[0]
        bf = fixed.loc[fixed["accuracy_mean"].idxmax()]
        bl = learned.loc[learned["accuracy_mean"].idxmax()]
        gain_pp = (bl["accuracy_mean"] - bf["accuracy_mean"]) * 100
        headline_rows.append(
            {
                "group": group,
                "models": models,
                "best_fixed": bf["policy"],
                "fixed_acc": bf["accuracy_mean"],
                "best_learned": bl["policy"],
                "learned_acc": bl["accuracy_mean"],
                "gain_pp": gain_pp,
                "oracle": oracle_row["accuracy_mean"],
            }
        )
    headline = pd.DataFrame(headline_rows).sort_values("gain_pp", ascending=False)
    headline.to_csv(RESULTS_DIR / "class_viable_only_headline.csv", index=False)
    print("\nViable-only headline, sorted by gain:")
    print(headline.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
