"""Training and evaluation utilities for the PILOT pipeline."""

from __future__ import annotations

import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = [
    "ng_dist_tss",
    "ng_dist_gap_1_2",
    "ng_gene_count_250kb",
    "cs_neglog10p",
    "cs_size",
    "cs_max_pip",
    "cs_entropy",
]


def train_test_split_group(df: pd.DataFrame, test_size: float, seed: int, group_by_study: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if group_by_study:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        groups = df["studyId"]
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        return df.iloc[train_idx], df.iloc[test_idx]
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["y"])
    return train_df, test_df


def _prep_features(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    return df[cols].to_numpy()


def prevalence_baseline(train_y: np.ndarray) -> float:
    return float(np.mean(train_y)) if len(train_y) else 0.5


def distance_only_features(df: pd.DataFrame) -> np.ndarray:
    arr = df[["ng_dist_tss", "ng_dist_gap_1_2", "ng_gene_count_250kb"]].copy()
    arr = arr.apply(np.log1p)
    return arr.to_numpy()


def logistic_regression_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def hgbt_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_depth=6, random_state=0)),
        ]
    )


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    y_pred = (clipped >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, clipped) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": _pr_auc(y_true, clipped),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, clipped, labels=[0, 1]),
        "brier": brier_score_loss(y_true, clipped),
    }
    return metrics


def _pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "pilot_roc.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "pilot_pr.png")
    plt.close()

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed fraction")
    plt.title("Calibration")
    plt.tight_layout()
    plt.savefig(out_dir / "pilot_calibration.png")
    plt.close()


def build_report(metrics: Dict[str, Dict[str, float]]) -> str:
    lines = ["# PILOT metrics", "", "Model | ROC-AUC | PR-AUC | Accuracy | Balanced Acc | Log Loss | Brier", "--- | --- | --- | --- | --- | --- | ---"]
    for name, vals in metrics.items():
        lines.append(
            f"{name} | {vals.get('roc_auc', float('nan')):.3f} | {vals.get('pr_auc', float('nan')):.3f} | "
            f"{vals.get('accuracy', float('nan')):.3f} | {vals.get('balanced_accuracy', float('nan')):.3f} | "
            f"{vals.get('log_loss', float('nan')):.3f} | {vals.get('brier', float('nan')):.3f}"
        )
    return "\n".join(lines)


def feature_matrix(df: pd.DataFrame, columns: List[str] | None = None) -> np.ndarray:
    cols = columns or NUMERIC_FEATURES
    return df[cols].to_numpy()


__all__ = [
    "NUMERIC_FEATURES",
    "train_test_split_group",
    "distance_only_features",
    "logistic_regression_model",
    "hgbt_model",
    "evaluate_predictions",
    "plot_curves",
    "build_report",
    "feature_matrix",
    "prevalence_baseline",
]
