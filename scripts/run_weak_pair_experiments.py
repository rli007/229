#!/usr/bin/env python3
"""Routing experiments on weak/cheap models.

Hypothesis from the README and the existing strong-model experiments: when
one model is dominant, routing has nowhere to go. With weaker models that
disagree often, the oracle headroom is much larger and a learned router has
more room to win.

This script reuses the cached 787-example dataset and the existing router
families to evaluate the most promising weak-model groups across 30 seeds.

Outputs:

  results/class_weak_pair_summary.csv
      Mean +/- std accuracy / utility / regret across 30 seeds for each
      group and router.

  results/class_weak_pair_headline.csv
      Best fixed model and best learned router per group.

  results/class_weak_disagreement_subset.csv
      Per-group analysis on the subset where the candidate models disagree:
      what fraction of the test set is "decidable", and how does the
      router accuracy compare to fixed policies on that subset only.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.router_baselines import evaluate_for_cost_weight

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


# Selected based on oracle-gap analysis. Each tuple is (group_name, models).
WEAK_GROUPS: list[tuple[str, list[str]]] = [
    # Pure weak pairs.
    ("mistral_minimax", ["mistral_nemo", "minimax_m3"]),  # 13.2 pp gap (largest)
    ("mistral_ring", ["mistral_nemo", "ring_1t"]),         # 8.3 pp gap
    ("ring_minimax", ["ring_1t", "minimax_m3"]),           # 5.3 pp gap
    # Weak + cheap-strong pairs.
    ("mistral_granite", ["mistral_nemo", "granite_8b"]),   # 6.6 pp gap
    ("gemini_flash_mistral", ["gemini_flash", "mistral_nemo"]),  # 5.8 pp gap
    ("mistral_qwen", ["mistral_nemo", "qwen_30b"]),        # 3.7 pp gap
    # Weak triples.
    ("triple_weak", ["mistral_nemo", "minimax_m3", "ring_1t"]),
    # Weak diversity + strong safety net.
    ("triple_weak_safety", ["mistral_nemo", "minimax_m3", "granite_8b"]),
    ("triple_weak_strong", ["mistral_nemo", "minimax_m3", "qwen_30b"]),
]


def run_group_30seed(
    df: pd.DataFrame,
    group_name: str,
    models: list[str],
    features: list[str],
    text_feature: str | None,
) -> pd.DataFrame:
    """Run the standard router sweep across 30 stratified seeds for one group."""
    rows: list[pd.DataFrame] = []
    for seed in SEEDS:
        train_df, test_df = train_test_split(
            df,
            test_size=0.25,
            random_state=seed,
            stratify=df["task_type"],
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
        rows.append(
            pd.DataFrame([r.__dict__ for r in results]).rename(
                columns={"name": "policy", "routing_accuracy": "route_acc"}
            )
        )
    detail = pd.concat(rows, ignore_index=True)
    summary = (
        detail.groupby("policy", as_index=False, sort=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            utility_mean=("utility", "mean"),
            regret_mean=("regret", "mean"),
            avg_cost_mean=("avg_cost", "mean"),
            route_acc_mean=("route_acc", "mean"),
        )
    )
    summary["accuracy_std"] = summary["accuracy_std"].fillna(0.0)
    summary.insert(0, "group", group_name)
    summary.insert(1, "models", ",".join(models))
    summary.insert(2, "n_seeds", len(SEEDS))
    return summary


def disagreement_subset_analysis(
    df: pd.DataFrame,
    group_name: str,
    models: list[str],
) -> dict[str, float]:
    """Analyze the routing problem restricted to examples where models disagree.

    "Decidable" examples are those where at least one candidate is correct
    AND not all candidates agree on correctness. For these examples there is
    a non-trivial routing decision; for everyone-correct or everyone-wrong
    examples no router can help.
    """
    correct = df[[f"{m}_correct" for m in models]].to_numpy()
    n = len(df)
    all_correct = (correct.sum(axis=1) == len(models)).mean()
    all_wrong = (correct.sum(axis=1) == 0).mean()
    any_correct = (correct.sum(axis=1) >= 1).mean()
    decidable = ((correct.sum(axis=1) >= 1) & (correct.sum(axis=1) < len(models))).mean()
    best_indiv = correct.mean(axis=0).max()
    oracle = (correct.sum(axis=1) >= 1).astype(int).mean()
    gap = oracle - best_indiv
    return {
        "group": group_name,
        "models": ",".join(models),
        "n_examples": n,
        "share_all_correct": float(all_correct),
        "share_all_wrong": float(all_wrong),
        "share_any_correct": float(any_correct),
        "share_decidable": float(decidable),
        "best_fixed_acc": float(best_indiv),
        "oracle_acc": float(oracle),
        "oracle_gap_pp": float(gap * 100),
    }


def build_headline(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, models), sub in summary_df.groupby(["group", "models"], sort=False):
        fixed = sub[sub["policy"].str.startswith("always_")]
        learned = sub[
            ~sub["policy"].isin(["random", "oracle"])
            & ~sub["policy"].str.startswith("always_")
        ]
        oracle = sub[sub["policy"] == "oracle"].iloc[0]
        bf = fixed.loc[fixed["accuracy_mean"].idxmax()]
        bl = learned.loc[learned["accuracy_mean"].idxmax()]
        gain_pp = (bl["accuracy_mean"] - bf["accuracy_mean"]) * 100
        rows.append(
            {
                "group": group,
                "models": models,
                "best_fixed": bf["policy"],
                "fixed_acc_mean": bf["accuracy_mean"],
                "fixed_acc_std": bf["accuracy_std"],
                "best_learned": bl["policy"],
                "learned_acc_mean": bl["accuracy_mean"],
                "learned_acc_std": bl["accuracy_std"],
                "gain_pp": gain_pp,
                "oracle_acc_mean": oracle["accuracy_mean"],
                "headroom_pp": (oracle["accuracy_mean"] - bf["accuracy_mean"]) * 100,
                "fraction_of_oracle_captured": (
                    gain_pp / max(oracle["accuracy_mean"] * 100 - bf["accuracy_mean"] * 100, 1e-9)
                )
                if oracle["accuracy_mean"] > bf["accuracy_mean"]
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("gain_pp", ascending=False)


def main() -> None:
    print(f"Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Disagreement / decidability summary ===")
    decidable_rows = [
        disagreement_subset_analysis(df, name, models) for name, models in WEAK_GROUPS
    ]
    decidable_df = pd.DataFrame(decidable_rows)
    decidable_df.to_csv(RESULTS_DIR / "class_weak_disagreement_subset.csv", index=False)
    print(
        decidable_df.to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )
    print(f"  Saved to {RESULTS_DIR / 'class_weak_disagreement_subset.csv'}")

    all_summaries: list[pd.DataFrame] = []
    for group_name, models in WEAK_GROUPS:
        print(f"\n=== Pair sweep · {group_name} = {models} ===")
        summary = run_group_30seed(
            df=df,
            group_name=group_name,
            models=models,
            features=ALL_FEATURES,
            text_feature="prompt",
        )
        all_summaries.append(summary)
        # Headline for this group, in-place.
        fixed = summary[summary["policy"].str.startswith("always_")]
        learned = summary[
            ~summary["policy"].isin(["random", "oracle"])
            & ~summary["policy"].str.startswith("always_")
        ]
        oracle = summary[summary["policy"] == "oracle"].iloc[0]
        bf = fixed.loc[fixed["accuracy_mean"].idxmax()]
        bl = learned.loc[learned["accuracy_mean"].idxmax()]
        gain = (bl["accuracy_mean"] - bf["accuracy_mean"]) * 100
        print(
            f"  best_fixed={bf['policy']:30s} {bf['accuracy_mean']:.4f} +/- {bf['accuracy_std']:.4f}"
        )
        print(
            f"  best_learned={bl['policy']:30s} {bl['accuracy_mean']:.4f} +/- {bl['accuracy_std']:.4f}  gain={gain:+.2f}pp  oracle={oracle['accuracy_mean']:.4f}"
        )

    big = pd.concat(all_summaries, ignore_index=True)
    big.to_csv(RESULTS_DIR / "class_weak_pair_summary.csv", index=False)
    print(f"\nSaved {len(big)} rows to {RESULTS_DIR / 'class_weak_pair_summary.csv'}")

    headline = build_headline(big)
    headline.to_csv(RESULTS_DIR / "class_weak_pair_headline.csv", index=False)
    print("\n=== Final headline (sorted by gain) ===")
    print(headline.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to {RESULTS_DIR / 'class_weak_pair_headline.csv'}")


if __name__ == "__main__":
    main()
