#!/usr/bin/env python3
"""Hyperparameter ablation on the headline pair.

For each of the 7 base router families, sweep a small grid of hyperparameter
settings on gemini_flash + llama_70b across 5 stratified seeds and report
mean test accuracy. The point is not to find a new best configuration; it is
to show that the ranking of router families and the magnitude of the headline
gain are stable across reasonable alternative settings.

The grid for each family is intentionally narrow (3-9 configs) so the whole
sweep runs in under a few minutes. The reference configuration (the one used
in the main report) is marked with a "*" in the printed output.

Output: results/class_hyperparam_sweep.csv
"""

from __future__ import annotations

import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.router_models import feature_pipeline

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


DATA_PATH = PROJECT_ROOT / "data" / "openrouter_mixed_10models_787_router_dataset.csv"
OUT_CSV = PROJECT_ROOT / "results" / "class_hyperparam_sweep.csv"
SEEDS = [229, 230, 231, 232, 233]
PAIR = ["gemini_flash", "llama_70b"]

FEATURES = [
    "task_type", "prompt_chars", "prompt_words", "question_start", "num_numbers",
    "has_math_symbols", "has_code_like_text", "question_mark_count",
    "capitalized_word_count", "contains_parenthesis", "contains_quote",
    "has_comparison_word", "has_negation_word", "has_quantity_word",
    "cheap_answer_chars", "cheap_answer_words", "cheap_parsed_answer",
    "cheap_contains_uncertainty", "cheap_self_confidence", "cheap_sample_agreement",
]


def fit_eval(estimator, train_df, test_df) -> tuple[float, float]:
    """Return (train_acc, test_acc) where 'accuracy' here means routed
    correctness, i.e. the candidate the router picks happens to be the one
    that got the example right.
    """
    y_train = train_df[[f"{m}_correct" for m in PAIR]].to_numpy().argmax(axis=1)
    pipe = Pipeline(
        steps=[
            ("features", feature_pipeline(train_df, FEATURES, text_feature=None)),
            ("clf", estimator),
        ]
    )
    pipe.fit(train_df[FEATURES], y_train)

    train_pick = pipe.predict(train_df[FEATURES])
    test_pick = pipe.predict(test_df[FEATURES])
    train_correct = train_df[[f"{m}_correct" for m in PAIR]].to_numpy()
    test_correct = test_df[[f"{m}_correct" for m in PAIR]].to_numpy()
    train_acc = float(train_correct[np.arange(len(train_df)), train_pick].mean())
    test_acc = float(test_correct[np.arange(len(test_df)), test_pick].mean())
    return train_acc, test_acc


def configs() -> list[tuple[str, str, dict, dict, bool]]:
    """Yield (family, label, kwargs, ctor_extra, is_reference)."""
    out = []

    # decision tree: depth x min_samples_leaf
    for depth, msl in product([3, 6, 10, None], [2, 5, 10]):
        out.append((
            "decision_tree", f"depth={depth}, msl={msl}",
            {"max_depth": depth, "min_samples_leaf": msl, "class_weight": "balanced"},
            {"ctor": DecisionTreeClassifier},
            depth == 6 and msl == 5,
        ))

    # random forest: n_estimators x min_samples_leaf
    for n, msl in product([100, 300, 500], [2, 3, 5]):
        out.append((
            "random_forest", f"n={n}, msl={msl}",
            {"n_estimators": n, "min_samples_leaf": msl, "class_weight": "balanced_subsample"},
            {"ctor": RandomForestClassifier},
            n == 300 and msl == 3,
        ))

    # extra trees: n_estimators x min_samples_leaf
    for n, msl in product([100, 500, 1000], [1, 2, 5]):
        out.append((
            "extra_trees", f"n={n}, msl={msl}",
            {"n_estimators": n, "min_samples_leaf": msl, "class_weight": "balanced"},
            {"ctor": ExtraTreesClassifier},
            n == 500 and msl == 2,
        ))

    # kNN: k in a range
    for k in [5, 10, 15, 25, 50]:
        out.append((
            "knn", f"k={k}",
            {"n_neighbors": k, "weights": "distance"},
            {"ctor": KNeighborsClassifier},
            k == 15,
        ))

    # MLP: hidden layers x alpha
    for hidden, alpha in product([(32,), (64, 32), (128, 64)], [1e-4, 1e-3, 1e-2]):
        out.append((
            "mlp", f"hidden={hidden}, alpha={alpha}",
            {"hidden_layer_sizes": hidden, "activation": "relu", "alpha": alpha,
             "max_iter": 1000},
            {"ctor": MLPClassifier},
            hidden == (64, 32) and alpha == 1e-3,
        ))

    # logistic regression: C
    for C in [0.1, 1.0, 10.0]:
        out.append((
            "logistic", f"C={C}",
            {"C": C, "max_iter": 2000, "class_weight": "balanced"},
            {"ctor": LogisticRegression},
            C == 1.0,
        ))

    # linear SVM: C
    for C in [0.1, 1.0, 10.0]:
        out.append((
            "linear_svm", f"C={C}",
            {"C": C, "max_iter": 5000, "class_weight": "balanced"},
            {"ctor": LinearSVC},
            C == 1.0,
        ))

    return out


def main() -> None:
    print(f"Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    rows = []
    for family, label, kwargs, extra, is_ref in configs():
        ctor = extra["ctor"]
        train_accs, test_accs = [], []
        for seed in SEEDS:
            train_df, test_df = train_test_split(
                df, test_size=0.25, random_state=seed, stratify=df["task_type"]
            )
            ctor_kwargs = dict(kwargs)
            if "random_state" in ctor.__init__.__code__.co_varnames:
                ctor_kwargs["random_state"] = seed
            est = ctor(**ctor_kwargs)
            tr, te = fit_eval(est, train_df, test_df)
            train_accs.append(tr)
            test_accs.append(te)
        rows.append({
            "family": family,
            "config": label,
            "is_reference": is_ref,
            "train_acc_mean": float(np.mean(train_accs)),
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
            "gap_pp": (float(np.mean(train_accs)) - float(np.mean(test_accs))) * 100,
        })
        marker = "*" if is_ref else " "
        print(f"  {marker} {family:14s} {label:32s} "
              f"test={float(np.mean(test_accs)):.4f} "
              f"+/- {float(np.std(test_accs)):.4f} "
              f"gap={(float(np.mean(train_accs)) - float(np.mean(test_accs))) * 100:.1f}pp")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    print("\n--- best per family vs reference ---")
    for family, sub in out.groupby("family", sort=False):
        ref = sub[sub["is_reference"]]
        best = sub.loc[sub["test_acc_mean"].idxmax()]
        if len(ref):
            ref_row = ref.iloc[0]
            delta = (best["test_acc_mean"] - ref_row["test_acc_mean"]) * 100
            print(
                f"  {family:14s} "
                f"ref={ref_row['config']:30s} test={ref_row['test_acc_mean']:.4f}  |  "
                f"best={best['config']:30s} test={best['test_acc_mean']:.4f}  "
                f"delta={delta:+.2f}pp"
            )


if __name__ == "__main__":
    main()
