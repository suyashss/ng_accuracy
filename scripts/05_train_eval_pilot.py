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

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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
    logistic_regression_model,
    prevalence_baseline,
    train_test_split_group,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("xgboost is required for nested CV training. Please install it.") from exc


def constant_predictions(n: int, p: float) -> np.ndarray:
    return np.array([p] * n)


def heuristic_predictions(df: pd.DataFrame) -> np.ndarray:
    return ((df["ng_dist_tss"] < 10000) & (df["ng_gene_count_250kb"] <= 5)).astype(int).to_numpy()


def build_variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{chrom}_{pos}_{ref.upper()}_{alt.upper()}"


def filter_to_gwas_high_medium(df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    gold_path = pathlib.Path(cfg["gold"]["out_path"])
    if not gold_path.exists():
        raise FileNotFoundError(f"Missing gold file {gold_path}; run 00_download_gold.py")

    gold = pd.read_csv(gold_path, sep="\t")
    is_gwas = gold["association_info.gwas_catalog_id"].notna()
    confidence = gold["gold_standard_info.evidence.confidence"].fillna("")
    conf_ok = confidence.isin(["High", "High|High", "Medium"])
    gold = gold[is_gwas & conf_ok].copy()

    gold["chrom"] = gold["sentinel_variant.locus_GRCh38.chromosome"].astype(str).str.replace("chr", "", regex=False)
    gold["pos"] = gold["sentinel_variant.locus_GRCh38.position"].astype(int)
    gold["variantId"] = gold.apply(
        lambda r: build_variant_id(r["chrom"], int(r["pos"]), r["sentinel_variant.alleles.reference"], r["sentinel_variant.alleles.alternative"]),
        axis=1,
    )
    gold = gold[["association_info.otg_id", "variantId"]].drop_duplicates()
    gold = gold.rename(columns={"association_info.otg_id": "studyId"})

    merged = df.merge(gold, on=["studyId", "variantId"], how="inner")
    logger.info("Filtered to GWAS high/medium confidence loci: %d -> %d", len(df), len(merged))
    return merged


def assign_chrom_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    chrom_col = "chromosome" if "chromosome" in df.columns else "chromosome_cs"
    if chrom_col not in df.columns:
        raise ValueError("No chromosome column found for fold assignment.")
    chroms = df[chrom_col].astype(str)

    pos_counts = df.groupby(chroms)["y"].sum().to_dict()
    rng = np.random.default_rng(seed)
    chrom_list = list(pos_counts.items())
    rng.shuffle(chrom_list)
    chrom_list.sort(key=lambda x: x[1], reverse=True)

    fold_pos = [0.0] * n_folds
    chrom_to_fold = {}
    for chrom, pos in chrom_list:
        idx = int(np.argmin(fold_pos))
        chrom_to_fold[chrom] = idx
        fold_pos[idx] += pos

    return chroms.map(chrom_to_fold)


def xgb_param_space() -> dict:
    return {
        "n_estimators": np.arange(200, 701, 100),
        "max_depth": np.arange(2, 5),
        "learning_rate": np.linspace(0.05, 0.2, 16),
        "subsample": np.linspace(0.7, 1.0, 4),
        "colsample_bytree": np.linspace(0.7, 1.0, 4),
        "min_child_weight": np.arange(5, 11),
        "gamma": np.linspace(0.0, 1.0, 6),
        "reg_alpha": np.linspace(0.0, 0.5, 6),
        "reg_lambda": np.linspace(0.5, 2.0, 7),
    }


def nested_cv(df: pd.DataFrame, cfg: dict, logger) -> tuple[pd.DataFrame, dict]:
    seed = int(cfg["split"]["seed"])
    n_outer = 5
    n_inner = 5

    fold_ids = assign_chrom_folds(df, n_outer, seed)
    metrics_rows = []
    best_params = []

    for fold in range(n_outer):
        train_df = df[fold_ids != fold].reset_index(drop=True)
        test_df = df[fold_ids == fold].reset_index(drop=True)
        logger.info("Outer fold %d: train=%d test=%d", fold + 1, len(train_df), len(test_df))

        # Baseline A: prevalence
        prev = prevalence_baseline(train_df["y"].to_numpy())
        y_prob = constant_predictions(len(test_df), prev)
        metrics = {"baseline_prevalence": evaluate_predictions(test_df["y"].to_numpy(), y_prob)}

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

        # Model 2: XGBoost with nested CV
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
        search = RandomizedSearchCV(
            xgb,
            param_distributions=xgb_param_space(),
            n_iter=200,
            scoring="balanced_accuracy",
            cv=inner_cv,
            random_state=seed,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, train_df["y"].to_numpy())
        best_params.append(search.best_params_)

        calibrator = CalibratedClassifierCV(search.best_estimator_, cv=3, method="sigmoid")
        calibrator.fit(X_train, train_df["y"].to_numpy())
        y_prob_xgb = calibrator.predict_proba(X_test)[:, 1]
        metrics["xgb_nested"] = evaluate_predictions(test_df["y"].to_numpy(), y_prob_xgb)
        metrics["xgb_nested"]["balanced_accuracy"] = balanced_accuracy_score(test_df["y"].to_numpy(), (y_prob_xgb >= 0.5).astype(int))

        for name, vals in metrics.items():
            for metric, value in vals.items():
                metrics_rows.append({"fold": fold + 1, "model": name, "metric": metric, "value": value})

    metrics_df = pd.DataFrame(metrics_rows)

    summary = {}
    for (model, metric), sub in metrics_df.groupby(["model", "metric"]):
        summary.setdefault(model, {})[metric] = (float(sub["value"].mean()), float(sub["value"].std()))

    return metrics_df, {"summary": summary, "best_params": best_params}


def build_nested_report(summary: dict) -> str:
    lines = [
        "# PILOT metrics (Nested CV)",
        "",
        "Model | ROC-AUC | PR-AUC | Accuracy | Balanced Acc | Log Loss | Brier",
        "--- | --- | --- | --- | --- | --- | ---",
    ]
    for name, vals in summary.items():
        def _fmt(metric):
            mean, std = vals.get(metric, (float("nan"), float("nan")))
            return f"{mean:.3f} ± {std:.3f}"

        lines.append(
            f"{name} | {_fmt('roc_auc')} | {_fmt('pr_auc')} | {_fmt('accuracy')} | "
            f"{_fmt('balanced_accuracy')} | {_fmt('log_loss')} | {_fmt('brier')}"
        )
    return "\n".join(lines)


def single_split_metrics(df: pd.DataFrame, cfg: dict) -> dict:
    train_df, test_df = train_test_split_group(
        df,
        cfg["split"]["test_size"],
        cfg["split"]["seed"],
        cfg["split"].get("group_by_study", True),
    )

    metrics = {}

    prev = prevalence_baseline(train_df["y"].to_numpy())
    y_prob = constant_predictions(len(test_df), prev)
    metrics["baseline_prevalence"] = evaluate_predictions(test_df["y"].to_numpy(), y_prob)

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

    y_prob = heuristic_predictions(test_df).astype(float)
    metrics["baseline_heuristic"] = evaluate_predictions(test_df["y"].to_numpy(), y_prob)

    X_train = feature_matrix(train_df, NUMERIC_FEATURES)
    X_test = feature_matrix(test_df, NUMERIC_FEATURES)
    log_model = logistic_regression_model()
    log_model.fit(X_train, train_df["y"].to_numpy())
    log_prob = log_model.predict_proba(X_test)[:, 1]
    metrics["logreg_all"] = evaluate_predictions(test_df["y"].to_numpy(), log_prob)

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(cfg["split"]["seed"]),
        n_jobs=-1,
    )
    calibrator = CalibratedClassifierCV(xgb, cv=3, method="sigmoid")
    calibrator.fit(X_train, train_df["y"].to_numpy())
    xgb_prob = calibrator.predict_proba(X_test)[:, 1]
    metrics["xgb_single"] = evaluate_predictions(test_df["y"].to_numpy(), xgb_prob)

    return metrics


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
    df = filter_to_gwas_high_medium(df, cfg, logger)

    single_metrics = single_split_metrics(df, cfg)
    metrics_df, meta = nested_cv(df, cfg, logger)

    reports_dir = pathlib.Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(reports_dir / "pilot_metrics.csv", index=False)

    report_text = [
        "# PILOT REPORT",
        "",
        f"Samples: {len(df)}",
        f"Label prevalence (overall): {df['y'].mean():.3f}",
        "",
        "# Single-split metrics",
        "",
        build_report(single_metrics),
        "",
        build_nested_report(meta["summary"]),
    ]
    (reports_dir / "PILOT_REPORT.md").write_text("\n".join(report_text), encoding="utf-8")
    logger.info("Saved nested CV metrics and report to %s", reports_dir)


if __name__ == "__main__":
    main()
