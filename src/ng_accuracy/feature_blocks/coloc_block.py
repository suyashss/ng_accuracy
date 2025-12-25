"""Colocalisation feature aggregation (lightweight placeholder)."""

from __future__ import annotations

import pandas as pd

from ..normalize import normalize_gene_id


def compute_coloc_features(coloc_df: pd.DataFrame, nearest: pd.DataFrame) -> pd.DataFrame:
    if coloc_df.empty or nearest.empty:
        return pd.DataFrame()
    nearest_lookup = nearest.set_index("studyLocusId")["nearestGeneId_base"].to_dict()
    coloc_df = coloc_df.copy()
    if "qtlGeneId" in coloc_df.columns:
        coloc_df["qtlGeneId_base"] = coloc_df["qtlGeneId"].apply(normalize_gene_id)
    agg_rows = []
    for study_locus_id, group in coloc_df.groupby("leftStudyLocusId"):
        ng = nearest_lookup.get(study_locus_id)
        if ng is None:
            continue
        nearest_rows = group[group.get("qtlGeneId_base") == ng] if "qtlGeneId_base" in group.columns else group.iloc[0:0]
        agg_rows.append(
            {
                "studyLocusId": study_locus_id,
                "coloc_num_pairs": float(len(group)),
                "coloc_max_h4_any": float(group.get("h4", pd.Series([pd.NA])).max()) if "h4" in group.columns else None,
                "coloc_max_h4_nearest_gene": float(nearest_rows.get("h4", pd.Series([pd.NA])).max()) if not nearest_rows.empty and "h4" in nearest_rows.columns else None,
            }
        )
    return pd.DataFrame(agg_rows)


__all__ = ["compute_coloc_features"]
