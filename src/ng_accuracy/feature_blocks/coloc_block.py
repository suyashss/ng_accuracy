"""Colocalisation feature aggregation with explicit status flags."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..normalize import normalize_gene_id


EPS = 1e-6


def _build_qtl_gene_lookup(study_df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Build lookup from study locus or study ID to gene ID (base-normalised).

    The Open Targets colocalisation table sometimes omits ``qtlGeneId`` but includes
    ``rightStudyLocusId``. In those cases we fall back to mapping via the study
    metadata, accepting either ``studyLocusId`` or ``studyId`` keys.
    """

    lookup: Dict[str, Optional[str]] = {}
    if study_df is None or study_df.empty:
        return lookup

    if {"studyLocusId", "geneId"}.issubset(study_df.columns):
        lookup.update(
            study_df.set_index("studyLocusId")["geneId"]
            .apply(lambda g: normalize_gene_id(g) if pd.notna(g) else None)
            .to_dict()
        )

    if {"studyId", "geneId"}.issubset(study_df.columns):
        lookup.update(
            study_df.set_index("studyId")["geneId"]
            .apply(lambda g: normalize_gene_id(g) if pd.notna(g) else None)
            .to_dict()
        )

    return lookup


def _prepare_coloc_frame(coloc_df: pd.DataFrame, study_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns and provide best-effort QTL gene mapping."""

    coloc_df = coloc_df.copy()
    lookup = _build_qtl_gene_lookup(study_df)

    if "leftStudyLocusId" not in coloc_df.columns:
        coloc_df["leftStudyLocusId"] = None

    if "qtlGeneId" in coloc_df.columns:
        coloc_df["qtlGeneId_base"] = coloc_df["qtlGeneId"].apply(
            lambda g: normalize_gene_id(g) if pd.notna(g) else None
        )
    elif "rightGeneId" in coloc_df.columns:
        coloc_df["qtlGeneId_base"] = coloc_df["rightGeneId"].apply(
            lambda g: normalize_gene_id(g) if pd.notna(g) else None
        )
    else:
        if "rightStudyLocusId" in coloc_df.columns:
            coloc_df["qtlGeneId_base"] = coloc_df["rightStudyLocusId"].map(lookup)
        elif "rightStudyId" in coloc_df.columns:
            coloc_df["qtlGeneId_base"] = coloc_df["rightStudyId"].map(lookup)
        else:
            coloc_df["qtlGeneId_base"] = np.nan

    # ensure numeric columns exist and are float
    coloc_df["h4"] = pd.to_numeric(coloc_df.get("h4", 0.0), errors="coerce").fillna(0.0).astype(float)
    coloc_df["clpp"] = pd.to_numeric(coloc_df.get("clpp", 0.0), errors="coerce").fillna(0.0).astype(float)

    # normalise study type to lower-case for consistent matching
    if "rightStudyType" in coloc_df.columns:
        coloc_df["rightStudyType"] = coloc_df["rightStudyType"].astype(str).str.lower()
    else:
        coloc_df["rightStudyType"] = ""

    return coloc_df


def _type_max(group: pd.DataFrame, study_type: str, field: str) -> float:
    type_rows = group[group["rightStudyType"] == study_type]
    return float(type_rows[field].max()) if not type_rows.empty else 0.0


def compute_coloc_features(coloc_df: pd.DataFrame, nearest: pd.DataFrame, study_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Aggregate coloc evidence for every locus with explicit missingness semantics."""

    if nearest.empty:
        return pd.DataFrame()

    coloc_df = _prepare_coloc_frame(coloc_df if coloc_df is not None else pd.DataFrame(), study_df if study_df is not None else pd.DataFrame())
    nearest_lookup = nearest.set_index("studyLocusId")["nearestGeneId_base"].to_dict()

    agg_rows = []
    for study_locus_id, nearest_gene in nearest_lookup.items():
        group = coloc_df[coloc_df.get("leftStudyLocusId") == study_locus_id]
        num_pairs = int(len(group))
        has_any_pairs = int(num_pairs > 0)

        # generic maxima
        max_h4_any = float(group["h4"].max()) if has_any_pairs else 0.0
        max_clpp_any = float(group["clpp"].max()) if has_any_pairs else 0.0

        # type-specific
        type_maxima = {
            f"coloc_max_h4_{t}": _type_max(group, t, "h4") for t in ("eqtl", "pqtl", "sqtl")
        }
        type_maxima.update({f"coloc_max_clpp_{t}": _type_max(group, t, "clpp") for t in ("eqtl", "pqtl", "sqtl")})
        type_indicators = {f"coloc_has_{t}": int((group["rightStudyType"] == t).any()) for t in ("eqtl", "pqtl", "sqtl")}

        mapped = group[group["qtlGeneId_base"].notna()]
        num_pairs_mapped_gene = int(len(mapped))
        has_any_mapped = int(num_pairs_mapped_gene > 0)

        best_gene_h4 = 0.0
        best_gene_clpp = 0.0
        if has_any_mapped:
            grouped_gene = mapped.groupby("qtlGeneId_base")
            best_gene_h4 = float(grouped_gene["h4"].max().max())
            best_gene_clpp = float(grouped_gene["clpp"].max().max())

        nearest_rows = mapped[mapped["qtlGeneId_base"] == nearest_gene] if has_any_mapped else mapped.iloc[0:0]
        coloc_nearest_gene_in_coloc = int(not nearest_rows.empty)
        coloc_num_pairs_nearest_gene = int(len(nearest_rows)) if coloc_nearest_gene_in_coloc else 0
        coloc_max_h4_nearest_gene = float(nearest_rows["h4"].max()) if coloc_nearest_gene_in_coloc else 0.0
        coloc_max_clpp_nearest_gene = float(nearest_rows["clpp"].max()) if coloc_nearest_gene_in_coloc else 0.0

        coloc_nearest_vs_best_h4_ratio = float(coloc_max_h4_nearest_gene / (best_gene_h4 + EPS))
        coloc_nearest_vs_best_clpp_ratio = float(coloc_max_clpp_nearest_gene / (best_gene_clpp + EPS))

        # status flags (mutually exclusive)
        coloc_status_no_pairs = int(num_pairs == 0)
        coloc_status_pairs_no_mapped_gene = int((num_pairs > 0) and (has_any_mapped == 0))
        coloc_status_mapped_gene_no_nearest = int((has_any_mapped == 1) and (coloc_nearest_gene_in_coloc == 0))
        coloc_status_nearest_match = int(coloc_nearest_gene_in_coloc == 1)

        agg_rows.append(
            {
                "studyLocusId": study_locus_id,
                "coloc_num_pairs": float(num_pairs),
                "coloc_has_any_pairs": has_any_pairs,
                "coloc_max_h4_any": max_h4_any,
                "coloc_max_clpp_any": max_clpp_any,
                **type_maxima,
                **type_indicators,
                "coloc_num_pairs_mapped_gene": float(num_pairs_mapped_gene),
                "coloc_has_any_mapped_qtl_gene": has_any_mapped,
                "coloc_best_gene_h4": best_gene_h4,
                "coloc_best_gene_clpp": best_gene_clpp,
                "coloc_best_gene_h4_is_zero": int(best_gene_h4 == 0.0),
                "coloc_best_gene_clpp_is_zero": int(best_gene_clpp == 0.0),
                "coloc_nearest_gene_in_coloc": coloc_nearest_gene_in_coloc,
                "coloc_num_pairs_nearest_gene": float(coloc_num_pairs_nearest_gene),
                "coloc_max_h4_nearest_gene": coloc_max_h4_nearest_gene,
                "coloc_max_clpp_nearest_gene": coloc_max_clpp_nearest_gene,
                "coloc_nearest_vs_best_h4_ratio": coloc_nearest_vs_best_h4_ratio,
                "coloc_nearest_vs_best_clpp_ratio": coloc_nearest_vs_best_clpp_ratio,
                "coloc_status_no_pairs": coloc_status_no_pairs,
                "coloc_status_pairs_no_mapped_gene": coloc_status_pairs_no_mapped_gene,
                "coloc_status_mapped_gene_no_nearest": coloc_status_mapped_gene_no_nearest,
                "coloc_status_nearest_match": coloc_status_nearest_match,
            }
        )

    return pd.DataFrame(agg_rows)


__all__ = ["compute_coloc_features"]
