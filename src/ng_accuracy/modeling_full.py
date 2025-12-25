"""Model training for the full pipeline (simplified)."""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

logger = logging.getLogger(__name__)


PRIMARY_FEATURES = [
    "ng_dist_tss",
    "ng_dist_gap_1_2",
    "dist_to_nearest_all_tss",
    "dist_to_nearest_pc_tss",
    "cs_max_pip",
    "cs_neglog10p",
]


METRIC_FUNCS = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "logloss": log_loss,
    "brier": brier_score_loss,
}


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def train_models(features: pd.DataFrame, output_dir: pathlib.Path, seed: int = 1) -> Dict[str, Dict[str, float]]:
    y = features["y"].astype(int)
    X = features.drop(columns=["y"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}

    # Baseline prevalence
    prevalence = float(y.mean()) if len(y) else 0.0
    results["baseline_prevalence"] = {"pr_auc": prevalence, "balanced_accuracy": prevalence}

    # Distance baseline
    baseline_feats = features[[c for c in ["ng_dist_tss", "ng_dist_gap_1_2", "ng_gene_density_within_250000bp"] if c in features.columns]].copy()
    preproc = build_preprocessor(baseline_feats)
    clf = Pipeline([
        ("preprocess", preproc),
        ("model", LogisticRegression(max_iter=500, random_state=seed)),
    ])
    clf.fit(baseline_feats, y)
    proba = clf.predict_proba(baseline_feats)[:, 1]
    results["baseline_distance_logreg"] = {
        "roc_auc": roc_auc_score(y, proba) if len(y.unique()) > 1 else 0.0,
        "pr_auc": average_precision_score(y, proba),
    }
    joblib.dump(clf, output_dir / "baseline_distance_logreg.joblib")

    # Full logistic regression
    preproc_full = build_preprocessor(X)
    logreg = Pipeline([
        ("preprocess", preproc_full),
        ("model", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    logreg.fit(X, y)
    logreg_proba = logreg.predict_proba(X)[:, 1]
    results["logreg_full"] = {
        "roc_auc": roc_auc_score(y, logreg_proba) if len(y.unique()) > 1 else 0.0,
        "pr_auc": average_precision_score(y, logreg_proba),
    }
    joblib.dump(logreg, output_dir / "logreg_full.joblib")

    # Gradient boosting as quick baseline
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[num_cols]
    hgbt = HistGradientBoostingClassifier(random_state=seed)
    hgbt.fit(X_num, y)
    hgbt_proba = hgbt.predict_proba(X_num)[:, 1]
    results["hgbt_full"] = {
        "roc_auc": roc_auc_score(y, hgbt_proba) if len(y.unique()) > 1 else 0.0,
        "pr_auc": average_precision_score(y, hgbt_proba),
    }
    joblib.dump(hgbt, output_dir / "hgbt_full.joblib")

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


__all__ = ["train_models"]
