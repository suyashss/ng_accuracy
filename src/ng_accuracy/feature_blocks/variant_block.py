"""Variant-level evidence for causal gene."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..normalize import normalize_gene_id


def compute_variant_features(variant_df: pd.DataFrame, nearest: pd.DataFrame) -> pd.DataFrame:
    if variant_df.empty or nearest.empty:
        return pd.DataFrame()
    nearest_lookup = nearest.set_index("studyLocusId")["nearestGeneId_base"].to_dict()
    variant_df = variant_df.copy()
    variant_df["targetId_base"] = variant_df.get("targetId").apply(normalize_gene_id) if "targetId" in variant_df.columns else np.nan
    agg_rows = []
    for study_locus_id, group in variant_df.groupby("studyLocusId"):
        ng = nearest_lookup.get(study_locus_id)
        if ng is None:
            continue
        hits = group[group["targetId_base"] == ng]
        row = {
            "studyLocusId": study_locus_id,
            "var_any_cs_variant_hits_nearest_gene": bool(not hits.empty),
            "var_cs_num_variants": float(len(group)),
        }
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


__all__ = ["compute_variant_features"]
