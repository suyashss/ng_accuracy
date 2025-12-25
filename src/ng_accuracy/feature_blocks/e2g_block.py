"""Enhancer-to-gene aggregation placeholder."""

from __future__ import annotations

import pandas as pd

from ..normalize import normalize_gene_id


def compute_e2g_features(e2g_df: pd.DataFrame, nearest: pd.DataFrame) -> pd.DataFrame:
    if e2g_df.empty or nearest.empty:
        return pd.DataFrame()
    e2g_df = e2g_df.copy()
    if "targetGeneId" in e2g_df.columns:
        e2g_df["targetGeneId_base"] = e2g_df["targetGeneId"].apply(normalize_gene_id)
    nearest_lookup = nearest.set_index("studyLocusId")["nearestGeneId_base"].to_dict()
    agg_rows = []
    for study_locus_id, group in e2g_df.groupby("studyLocusId"):
        ng = nearest_lookup.get(study_locus_id)
        if ng is None:
            continue
        nearest_rows = group[group.get("targetGeneId_base") == ng] if "targetGeneId_base" in group.columns else group.iloc[0:0]
        agg_rows.append(
            {
                "studyLocusId": study_locus_id,
                "e2g_nearest_num_intervals": float(len(nearest_rows)) if not nearest_rows.empty else 0.0,
                "e2g_best_gene_max_score": float(group.get("score", pd.Series([pd.NA])).max()) if "score" in group.columns else None,
                "e2g_nearest_max_score": float(nearest_rows.get("score", pd.Series([pd.NA])).max()) if not nearest_rows.empty and "score" in nearest_rows.columns else None,
            }
        )
    return pd.DataFrame(agg_rows)


__all__ = ["compute_e2g_features"]
