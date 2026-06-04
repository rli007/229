#!/usr/bin/env python3
"""Decidable-subset routing skill analysis.

Aggregate accuracy understates and overstates router skill in opposite ways:
- Examples where every candidate is correct give the router a free win.
- Examples where every candidate is wrong give the router a guaranteed loss.

The honest measure of routing skill is accuracy on the decidable subset:
examples where the candidates disagree, so a real routing decision is being
made. We compute three numbers for each (group, router) across 30 seeds:

  decidable_router_acc   - router's accuracy on the disagreement subset
  decidable_best_fixed   - the best fixed model's accuracy on the disagreement subset
                           (= "win share" of the strongest single model in disagreements)
  decidable_lift_pp      - decidable_router_acc - decidable_best_fixed (in pp)

`decidable_lift_pp > 0` means the router can identify which model is right
on disagreements better than just always picking the strongest single model.
This is the metric that captures predictive skill independent of how often
disagreements happen.

Output: results/class_decidable_routing_summary.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


GROUPS: list[tuple[str, list[str]]] = [
    # Strong pairs.
    ("gemini_flash_llama", ["gemini_flash", "llama_70b"]),
    ("gemini_flash_granite", ["gemini_flash", "granite_8b"]),
    ("llama_granite", ["llama_70b", "granite_8b"]),
    ("llama_deepseek", ["llama_70b", "deepseek_v3"]),
    ("qwen_llama", ["qwen_30b", "llama_70b"]),
    # No-dominator triples.
    ("triple_gemini_llama_granite", ["gemini_flash", "llama_70b", "granite_8b"]),
    ("triple_qwen_llama_granite", ["qwen_30b", "llama_70b", "granite_8b"]),
    # Weak pairs (hypothesis-failed setups).
    ("mistral_minimax", ["mistral_nemo", "minimax_m3"]),
    ("mistral_qwen", ["mistral_nemo", "qwen_30b"]),
    ("gemini_flash_mistral", ["gemini_flash", "mistral_nemo"]),
]


def fit_predict_one(
    estimator,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    text_feature: str | None,
    y_train: np.ndarray,
) -> np.ndarray:
    input_cols = features + ([text_feature] if text_feature else [])
    pipe = Pipeline(
        steps=[
            ("features", feature_pipeline(train_df, features, text_feature=text_feature)),
            ("clf", estimator),
        ]
    )
    pipe.fit(train_df[input_cols], y_train)
    return pipe.predict(test_df[input_cols])


def evaluate_group(
    df: pd.DataFrame,
    group_name: str,
    models: list[str],
    features: list[str],
    text_feature: str | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    estimator_factories = {
        **base_router_estimators(229),
        **(text_router_estimators(229) if text_feature else {}),
    }
    correct_cols = [f"{m}_correct" for m in models]

    # We evaluate every router family alongside fixed-baseline reference rows.
    for seed in SEEDS:
        train_df, test_df = train_test_split(
            df,
            test_size=0.25,
            random_state=seed,
            stratify=df["task_type"],
        )
        train_correct = train_df[correct_cols].to_numpy()
        test_correct = test_df[correct_cols].to_numpy()
        n_test = len(test_df)
        idx_test = np.arange(n_test)

        # Decidable subset: at least one model correct AND not all models correct.
        any_correct_test = test_correct.sum(axis=1) >= 1
        all_correct_test = test_correct.sum(axis=1) == len(models)
        decidable_mask = any_correct_test & ~all_correct_test
        n_decidable = int(decidable_mask.sum())

        # Oracle on test set, with cost_weight=0: just argmax of correctness.
        # Tiebreaks to first model among the all-correct examples; doesn't matter
        # for decidable-subset metrics because we only inspect decidable rows.
        oracle = test_correct.argmax(axis=1)
        oracle_acc_full = test_correct[idx_test, oracle].mean()

        # Always-X policies, computed once.
        always_acc = test_correct.mean(axis=0)
        # On decidable subset: per-model "win share among disagreements"
        if n_decidable > 0:
            decidable_correct = test_correct[decidable_mask]
            always_decidable_acc = decidable_correct.mean(axis=0)
            best_fixed_idx = int(np.argmax(always_acc))
            best_fixed_decidable_acc = float(always_decidable_acc[best_fixed_idx])
        else:
            best_fixed_idx = int(np.argmax(always_acc))
            best_fixed_decidable_acc = float("nan")

        # Train labels for multiclass routing: oracle on train.
        train_any_correct = train_correct.sum(axis=1) >= 1
        if not train_any_correct.any():
            continue
        y_train = train_correct.argmax(axis=1)

        # Reference rows.
        rows.append(
            {
                "group": group_name,
                "models": ",".join(models),
                "seed": seed,
                "policy": f"always_{models[best_fixed_idx]}",
                "n_test": n_test,
                "n_decidable": n_decidable,
                "decidable_share": float(decidable_mask.mean()),
                "router_acc_full": float(always_acc[best_fixed_idx]),
                "router_acc_decidable": best_fixed_decidable_acc,
                "decidable_lift_pp": 0.0,
                "router_disagrees_with_best_fixed_share": 0.0,
                "router_correct_when_disagrees_share": float("nan"),
            }
        )
        rows.append(
            {
                "group": group_name,
                "models": ",".join(models),
                "seed": seed,
                "policy": "oracle",
                "n_test": n_test,
                "n_decidable": n_decidable,
                "decidable_share": float(decidable_mask.mean()),
                "router_acc_full": float(oracle_acc_full),
                "router_acc_decidable": 1.0 if n_decidable > 0 else float("nan"),
                "decidable_lift_pp": (
                    100.0 * (1.0 - best_fixed_decidable_acc) if n_decidable > 0 else float("nan")
                ),
                "router_disagrees_with_best_fixed_share": float(
                    (oracle != best_fixed_idx).mean()
                ),
                "router_correct_when_disagrees_share": float("nan"),
            }
        )

        # Each learned router.
        for router_name, estimator in estimator_factories.items():
            tf = text_feature if router_name.startswith("tfidf_") else None
            try:
                preds = fit_predict_one(
                    estimator=estimator,
                    train_df=train_df,
                    test_df=test_df,
                    features=features,
                    text_feature=tf,
                    y_train=y_train,
                )
            except Exception as exc:  # noqa: BLE001
                # Some estimators may fail on degenerate seeds. Skip gracefully.
                continue
            chosen_correct = test_correct[idx_test, preds]
            full_acc = float(chosen_correct.mean())
            if n_decidable > 0:
                router_dec_acc = float(chosen_correct[decidable_mask].mean())
            else:
                router_dec_acc = float("nan")
            disagreement_with_fixed = preds != best_fixed_idx
            disagree_correct_share = (
                float(chosen_correct[disagreement_with_fixed].mean())
                if disagreement_with_fixed.any()
                else float("nan")
            )
            rows.append(
                {
                    "group": group_name,
                    "models": ",".join(models),
                    "seed": seed,
                    "policy": router_name,
                    "n_test": n_test,
                    "n_decidable": n_decidable,
                    "decidable_share": float(decidable_mask.mean()),
                    "router_acc_full": full_acc,
                    "router_acc_decidable": router_dec_acc,
                    "decidable_lift_pp": (
                        100.0 * (router_dec_acc - best_fixed_decidable_acc)
                        if n_decidable > 0
                        else float("nan")
                    ),
                    "router_disagrees_with_best_fixed_share": float(
                        disagreement_with_fixed.mean()
                    ),
                    "router_correct_when_disagrees_share": disagree_correct_share,
                }
            )
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail.groupby(["group", "models", "policy"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_decidable_mean=("n_decidable", "mean"),
            decidable_share_mean=("decidable_share", "mean"),
            acc_full_mean=("router_acc_full", "mean"),
            acc_full_std=("router_acc_full", "std"),
            acc_decidable_mean=("router_acc_decidable", "mean"),
            acc_decidable_std=("router_acc_decidable", "std"),
            decidable_lift_pp_mean=("decidable_lift_pp", "mean"),
            decidable_lift_pp_std=("decidable_lift_pp", "std"),
            router_disagrees_share_mean=(
                "router_disagrees_with_best_fixed_share",
                "mean",
            ),
            router_correct_when_disagrees_mean=(
                "router_correct_when_disagrees_share",
                "mean",
            ),
        )
    )
    summary["acc_full_std"] = summary["acc_full_std"].fillna(0.0)
    summary["acc_decidable_std"] = summary["acc_decidable_std"].fillna(0.0)
    summary["decidable_lift_pp_std"] = summary["decidable_lift_pp_std"].fillna(0.0)
    return summary


def main() -> None:
    print(f"Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for group_name, models in GROUPS:
        print(f"\n=== {group_name} = {models} ===")
        rows = evaluate_group(
            df=df,
            group_name=group_name,
            models=models,
            features=ALL_FEATURES,
            text_feature="prompt",
        )
        all_rows.extend(rows)

    detail = pd.DataFrame(all_rows)
    summary = summarize(detail)
    summary.to_csv(RESULTS_DIR / "class_decidable_routing_summary.csv", index=False)
    detail.to_csv(RESULTS_DIR / "class_decidable_routing_details.csv", index=False)
    print(f"\nSaved summary to {RESULTS_DIR / 'class_decidable_routing_summary.csv'}")

    # Headline: best learned router per group (by full accuracy) + its decidable lift.
    print("\n=== Headline ===")
    print(f"{'group':35s} {'best_learned':30s} {'full_acc':>10s} {'decidable_acc':>14s} {'best_fixed_dec':>16s} {'lift_pp':>10s} {'disagree%':>10s} {'correct|disagree':>17s}")
    for (group, models), sub in summary.groupby(["group", "models"]):
        learned = sub[
            ~sub["policy"].str.startswith("always_")
            & ~sub["policy"].isin(["random", "oracle"])
        ]
        always = sub[sub["policy"].str.startswith("always_")].iloc[0]
        oracle = sub[sub["policy"] == "oracle"].iloc[0]
        if learned.empty:
            continue
        bl = learned.loc[learned["acc_full_mean"].idxmax()]
        print(
            f"{group:35s} {bl['policy']:30s} {bl['acc_full_mean']:>10.4f} {bl['acc_decidable_mean']:>14.4f} {always['acc_decidable_mean']:>16.4f} {bl['decidable_lift_pp_mean']:>+10.2f} {bl['router_disagrees_share_mean']:>10.3f} {bl['router_correct_when_disagrees_mean']:>17.3f}"
        )


if __name__ == "__main__":
    main()
