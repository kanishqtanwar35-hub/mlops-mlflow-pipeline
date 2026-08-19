"""One tracked training run.

Every run logs: the params, the metrics (headline and per-class), the data
fingerprint, the git commit, and the model itself registered as a new version.
Nothing about the run lives only in someone's terminal history.
"""

import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.mlops.logger import logger

REGISTRY = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "LogisticRegression": LogisticRegression,
}


def ensure_data(url: str, local_path: Path, column_names) -> Path:
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if not local_path.exists():
        logger.info("downloading %s", url)
        urllib.request.urlretrieve(url, local_path)
        if column_names:
            df = pd.read_csv(local_path, header=None)
            if len(df.columns) == len(column_names):
                df.columns = list(column_names)
                df.to_csv(local_path, index=False)
    return local_path


def build_pipeline(model_name: str, model_params: Dict, feature_names) -> Pipeline:
    if model_name not in REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. Available: {sorted(REGISTRY)}"
        )
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(feature_names)),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", REGISTRY[model_name](**model_params)),
    ])


def split(df: pd.DataFrame, target: str, test_size: float, seed: int) -> Tuple:
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def evaluate(pipeline: Pipeline, X_test, y_test, class_labels) -> Dict:
    preds = pipeline.predict(X_test)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, preds, average="macro", zero_division=0)),
    }

    if len(set(y_test)) == 2 and hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))

    # Per-class F1 is logged as separate metrics so the promotion gate can see
    # a single class collapsing. An average alone hides exactly that.
    per_class = f1_score(y_test, preds, average=None, zero_division=0)
    for label, value in zip(class_labels, per_class):
        metrics[f"f1_class_{label}"] = float(value)

    return metrics
