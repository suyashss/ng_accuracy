#!/usr/bin/env python
"""Evaluate trained models for the full pipeline."""

from __future__ import annotations

import argparse
import json
import pathlib

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, log_loss, roc_auc_score

from src.ng_accuracy.config import load_config
from src.ng_accuracy.logging_utils import configure_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging()
    features = pd.read_parquet(pathlib.Path(cfg["paths"]["processed_dir"]) / "full_features.parquet")
    models_dir = pathlib.Path(cfg["paths"]["processed_dir"]) / "models" / "full"
    report_dir = pathlib.Path(cfg["paths"]["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    y_true = features["y"].astype(int)
    metrics = {}
    for model_path in models_dir.glob("*.joblib"):
        model = joblib.load(model_path)
        proba = model.predict_proba(features.drop(columns=["y"]))[:, 1]
        metrics[model_path.stem] = {
            "roc_auc": roc_auc_score(y_true, proba) if len(y_true.unique()) > 1 else 0.0,
            "pr_auc": average_precision_score(y_true, proba),
            "accuracy": accuracy_score(y_true, proba > 0.5),
            "balanced_accuracy": balanced_accuracy_score(y_true, proba > 0.5),
            "logloss": log_loss(y_true, proba),
        }
    (report_dir / "full_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
