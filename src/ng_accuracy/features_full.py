"""Feature orchestration for the full pipeline."""

from __future__ import annotations

import json
import logging
import pathlib

import pandas as pd

from .feature_blocks.credible_set import compute_credible_set_features
from .feature_blocks.proximity import build_proximity_features
from .feature_blocks.variant_block import compute_variant_features
from .feature_blocks.coloc_block import compute_coloc_features
from .feature_blocks.e2g_block import compute_e2g_features
from .feature_blocks.l2g_block import compute_l2g_features
from .normalize import normalize_gene_id
from .target_index import TargetGeneIndex

logger = logging.getLogger(__name__)


def assemble_features(
    mapped_loci: pd.DataFrame,
    target_index: TargetGeneIndex,
    cs_df: pd.DataFrame,
    study_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    coloc_df: pd.DataFrame,
    e2g_df: pd.DataFrame,
    l2g_df: pd.DataFrame,
    definition: str,
    windows,
    output_path: pathlib.Path,
    reports_dir: pathlib.Path,
) -> pd.DataFrame:
    mapped_loci = mapped_loci.copy()
    mapped_loci["goldGeneId_base"] = mapped_loci["goldGeneId"].apply(normalize_gene_id)
    nearest = build_proximity_features(target_index, mapped_loci, definition, windows)
    cs_feat = compute_credible_set_features(cs_df)
    merged = mapped_loci.merge(nearest, on="studyLocusId", how="left")
    merged = merged.merge(cs_feat, on="studyLocusId", how="left")
    variant_feat = compute_variant_features(variant_df, nearest) if not variant_df.empty else pd.DataFrame()
    if not variant_feat.empty:
        merged = merged.merge(variant_feat, on="studyLocusId", how="left")
    coloc_feat = compute_coloc_features(coloc_df, nearest, study_df)
    if not coloc_feat.empty:
        merged = merged.merge(coloc_feat, on="studyLocusId", how="left")
    e2g_feat = compute_e2g_features(e2g_df, nearest) if not e2g_df.empty else pd.DataFrame()
    if not e2g_feat.empty:
        merged = merged.merge(e2g_feat, on="studyLocusId", how="left")
    l2g_feat = compute_l2g_features(l2g_df, nearest) if not l2g_df.empty else pd.DataFrame()
    if not l2g_feat.empty:
        merged = merged.merge(l2g_feat, on="studyLocusId", how="left")
    merged["nearestGeneId_base"] = merged["nearestGeneId"].apply(normalize_gene_id)
    merged["y"] = (merged["nearestGeneId_base"] == merged["goldGeneId_base"]).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    merged.to_csv(output_path.with_suffix(".csv"), index=False)
    reports_dir.mkdir(parents=True, exist_ok=True)
    missingness = merged.isna().mean().sort_values(ascending=False)
    missingness.to_csv(reports_dir / "full_feature_missingness.csv")

    coloc_cols = [
        "coloc_max_h4_nearest_gene",
        "coloc_nearest_vs_best_h4_ratio",
        "coloc_nearest_vs_best_clpp_ratio",
    ]
    coloc_dist_summary = {}
    for col in coloc_cols:
        if col in merged.columns:
            series = merged[col]
            coloc_dist_summary[col] = {
                "min": float(series.min(skipna=True)) if len(series) else None,
                "median": float(series.median(skipna=True)) if len(series) else None,
                "max": float(series.max(skipna=True)) if len(series) else None,
                "frac_zero": float((series == 0).mean()) if len(series) else None,
            }

    summary = {"num_rows": len(merged), "prevalence": float(merged["y"].mean()) if len(merged) else None}
    (reports_dir / "full_feature_summary.json").write_text(json.dumps(summary, indent=2))
    if coloc_dist_summary:
        (reports_dir / "coloc_feature_distribution.json").write_text(json.dumps(coloc_dist_summary, indent=2))

    _write_coloc_diagnostics(merged, reports_dir)
    return merged


def _write_coloc_diagnostics(df: pd.DataFrame, reports_dir: pathlib.Path) -> None:
    if "coloc_status_no_pairs" not in df.columns:
        return

    status_cols = [
        "coloc_status_no_pairs",
        "coloc_status_pairs_no_mapped_gene",
        "coloc_status_mapped_gene_no_nearest",
        "coloc_status_nearest_match",
    ]
    lines = ["# Colocalisation diagnostics", ""]
    total = len(df)
    lines.append("## Status counts")
    for col in status_cols:
        count = int(df[col].sum())
        pct = (count / total * 100) if total else 0
        lines.append(f"- {col}: {count} ({pct:.2f}%)")

    lines.append("")
    if "coloc_has_any_pairs" in df.columns:
        count_pairs = int(df["coloc_has_any_pairs"].sum())
        count_mapped = int(df.get("coloc_has_any_mapped_qtl_gene", pd.Series()).sum())
        count_nearest = int(df.get("coloc_nearest_gene_in_coloc", pd.Series()).sum())
        lines.append("## Pair and mapping coverage")
        lines.append(f"- loci with any coloc pairs: {count_pairs}")
        lines.append(f"- loci with mapped QTL genes: {count_mapped}")
        lines.append(f"- loci where nearest gene appears in coloc: {count_nearest}")

    report_path = reports_dir / "coloc_diagnostics_full.md"
    report_path.write_text("\n".join(lines))


__all__ = ["assemble_features"]
