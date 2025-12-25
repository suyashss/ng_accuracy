"""L2G baseline features."""

from __future__ import annotations

import pandas as pd

from ..normalize import normalize_gene_id


def compute_l2g_features(l2g_df: pd.DataFrame, nearest: pd.DataFrame) -> pd.DataFrame:
    if l2g_df.empty or nearest.empty:
        return pd.DataFrame()
    l2g_df = l2g_df.copy()
    l2g_df["geneId_base"] = l2g_df.get("geneId").apply(normalize_gene_id)
    nearest_lookup = nearest.set_index("studyLocusId")["nearestGeneId_base"].to_dict()
    rows = []
    for study_locus_id, group in l2g_df.groupby("studyLocusId"):
        ng = nearest_lookup.get(study_locus_id)
        if ng is None:
            continue
        group_sorted = group.sort_values("score", ascending=False)
        nearest_row = group_sorted[group_sorted["geneId_base"] == ng].head(1)
        best_score = group_sorted["score"].max() if "score" in group_sorted.columns else None
        nearest_score = nearest_row["score"].iloc[0] if not nearest_row.empty and "score" in nearest_row.columns else None
        rank = int(nearest_row.index[0]) if not nearest_row.empty else None
        rows.append(
            {
                "studyLocusId": study_locus_id,
                "l2g_score_nearest": float(nearest_score) if nearest_score is not None else None,
                "l2g_rank_nearest": rank,
                "l2g_gap_best_minus_nearest": float(best_score - nearest_score) if best_score is not None and nearest_score is not None else None,
            }
        )
    return pd.DataFrame(rows)


__all__ = ["compute_l2g_features"]
