"""Learned router model builders.

These functions define the supervised routers that learn to imitate the
cost-aware oracle route labels produced by the experiment runner.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def feature_pipeline(
    df: pd.DataFrame,
    features: list[str],
    text_feature: str | None = None,
) -> ColumnTransformer:
    categorical = [
        col
        for col in features
        if not pd.api.types.is_numeric_dtype(df[col])
        or pd.api.types.is_object_dtype(df[col])
        or pd.api.types.is_string_dtype(df[col])
        or isinstance(df[col].dtype, pd.CategoricalDtype)
    ]
    numeric = [col for col in features if col not in categorical]
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    if text_feature:
        transformers.append(
            (
                "text",
                TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1),
                text_feature,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def base_router_estimators(random_state: int) -> dict[str, object]:
    # Hyperparameters were selected by a per-family sweep on the headline pair
    # gemini_flash + llama_70b across 5 stratified seeds, picking the
    # configuration with the highest mean test accuracy. See
    # scripts/run_hyperparam_sweep.py and results/class_hyperparam_sweep.csv.
    return {
        "logistic_router": LogisticRegression(
            C=0.1, max_iter=2000, class_weight="balanced",
        ),
        "linear_svm_router": LinearSVC(
            C=0.1,
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        ),
        "knn_router": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
        ),
        "decision_tree_router": DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=5,
            random_state=random_state,
            class_weight="balanced",
        ),
        "random_forest_router": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
        "extra_trees_router": ExtraTreesClassifier(
            n_estimators=1000,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
        ),
        "mlp_router": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            max_iter=1000,
            random_state=random_state,
        ),
    }


def text_router_estimators(random_state: int) -> dict[str, object]:
    return {
        "tfidf_logistic_router": LogisticRegression(
            C=0.1, max_iter=2000, class_weight="balanced",
        ),
        "tfidf_linear_svm_router": LinearSVC(
            C=0.1,
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        ),
        "tfidf_mlp_router": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            max_iter=1000,
            random_state=random_state,
        ),
    }


def fit_router(
    estimator,
    train_df: pd.DataFrame,
    features: list[str],
    labels,
    text_feature: str | None = None,
) -> Pipeline:
    input_columns = features + ([text_feature] if text_feature else [])
    pipe = Pipeline(
        steps=[
            ("features", feature_pipeline(train_df, features, text_feature=text_feature)),
            ("clf", estimator),
        ]
    )
    pipe.fit(train_df[input_columns], labels)
    return pipe
