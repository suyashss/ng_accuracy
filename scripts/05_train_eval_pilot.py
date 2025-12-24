#!/usr/bin/env python
"""Train baseline models for the PILOT pipeline."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ng_accuracy.config import add_config_arg, load_config
from ng_accuracy.logging_utils import get_logger
from ng_accuracy.modeling_pilot import (
    NUMERIC_FEATURES,
    build_report,
    distance_only_features,
    evaluate_predictions,
    feature_matrix,
    hgbt_model,
    logistic_regression_model,
    plot_curves,
    prevalence_baseline,
    train_test_split_group,
)


def constant_predictions(n: int, p: float) -> np.ndarray:
    return np.array([p] * n)


def heuristic_predictions(df: pd.DataFrame) -> np.ndarray:
    return ((df["ng_dist_tss"] < 10000) & (df["ng_gene_count_250kb"] <= 5)).astype(int).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("train_eval")

    features_path = pathlib.Path(cfg["paths"]["processed_dir"]) / "pilot_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file {features_path}; run 04_build_pilot_features.py")

    df = pd.read_parquet(features_path)

    train_df, test_df = train_test_split_group(df, cfg["split"]["test_size"], cfg["split"]["seed"], cfg["split"].get("group_by_study", True))

    metrics = {}

    # Baseline A: prevalence
    prev = prevalence_baseline(train_df["y"].to_numpy())
    y_prob = constant_predictions(len(test_df), prev)
    metrics["baseline_prevalence"] = evaluate_predictions(test_df["y"].to_numpy(), y_prob)

    # Baseline B: distance-only logistic
    dist_features_train = distance_only_features(train_df)
    dist_features_test = distance_only_features(test_df)
    dist_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    dist_model.fit(dist_features_train, train_df["y"].to_numpy())
    y_prob = dist_model.predict_proba(dist_features_test)[:, 1]
    metrics["baseline_distance_logreg"] = evaluate_predictions(test_df["y"].to_numpy(), y_prob)

    # Baseline C: heuristic
    y_prob = heuristic_predictions(test_df).astype(float)
    metrics["baseline_heuristic"] = evaluate_predictions(test_df["y"].to_numpy(), y_prob)

    # Model 1: logistic regression on all numeric features
    X_train = feature_matrix(train_df, NUMERIC_FEATURES)
    X_test = feature_matrix(test_df, NUMERIC_FEATURES)
    log_model = logistic_regression_model()
    log_model.fit(X_train, train_df["y"].to_numpy())
    log_prob = log_model.predict_proba(X_test)[:, 1]
    metrics["logreg_all"] = evaluate_predictions(test_df["y"].to_numpy(), log_prob)

    # Model 2: gradient boosting
    hgbt = hgbt_model()
    hgbt.fit(X_train, train_df["y"].to_numpy())
    y_prob_hgbt = hgbt.predict_proba(X_test)[:, 1]
    metrics["hgbt_all"] = evaluate_predictions(test_df["y"].to_numpy(), y_prob_hgbt)

    reports_dir = pathlib.Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for name, vals in metrics.items():
        for metric, value in vals.items():
            metrics_rows.append({"model": name, "metric": metric, "value": value})
    pd.DataFrame(metrics_rows).to_csv(reports_dir / "pilot_metrics.csv", index=False)

    # Use logistic regression curve plots
    plot_curves(test_df["y"].to_numpy(), log_prob, reports_dir)

    report_text = [
        "# PILOT REPORT",
        "",
        f"Samples: {len(df)}",
        f"Train: {len(train_df)} | Test: {len(test_df)}",
        f"Label prevalence (overall): {df['y'].mean():.3f}",
        "",
        build_report(metrics),
    ]
    (reports_dir / "PILOT_REPORT.md").write_text("\n".join(report_text), encoding="utf-8")
    logger.info("Saved metrics and report to %s", reports_dir)


if __name__ == "__main__":
    main()
