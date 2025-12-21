"""Feature engineering for the PILOT pipeline."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import duckdb
import numpy as np
import pandas as pd

from .gene_gtf import build_gene_index


def _nearest_gene_for_row(pos: int, chrom: str, gene_index: Dict[str, pd.DataFrame], window: int) -> Tuple[str | None, float | None, float | None, float | None, float | None]:
    genes = gene_index.get(chrom)
    if genes is None or genes.empty:
        return None, None, None, None, None
    tss = genes["gene_tss"].to_numpy()
    idx = np.searchsorted(tss, pos)
    candidates: List[int] = []
    if idx < len(tss):
        candidates.append(idx)
    if idx - 1 >= 0:
        candidates.append(idx - 1)
    dists = [(abs(int(tss[i]) - pos), i) for i in candidates]
    if not dists:
        return None, None, None, None, None
    dists.sort()
    nearest_dist, nearest_idx = dists[0]
    nearest_gene = genes.iloc[nearest_idx]["gene_id"]
    second_dist = dists[1][0] if len(dists) > 1 else None
    dist_gap = float(second_dist - nearest_dist) if second_dist is not None else None
    left = np.searchsorted(tss, pos - window, side="left")
    right = np.searchsorted(tss, pos + window, side="right")
    gene_count = float(right - left)
    return (
        nearest_gene,
        float(nearest_dist),
        float(second_dist) if second_dist is not None else None,
        dist_gap,
        gene_count,
    )


def annotate_nearest_genes(loci: pd.DataFrame, genes_df: pd.DataFrame, window: int) -> pd.DataFrame:
    gene_index = build_gene_index(genes_df)
    results = []
    for _, row in loci.iterrows():
        nearest_gene, nearest_dist, second_dist, dist_gap, gene_count = _nearest_gene_for_row(
            int(row["position"]), str(row["chromosome"]), gene_index, window
        )
        results.append(
            {
                "studyLocusId": row["studyLocusId"],
                "nearestGeneId": nearest_gene,
                "ng_dist_tss": nearest_dist,
                "ng_dist_gap_1_2": dist_gap,
                "ng_gene_count_250kb": gene_count,
                "ng_second_dist_tss": second_dist,
            }
        )
    return pd.DataFrame(results)


def compute_credible_set_features(parquet_glob: str, study_locus_ids: Iterable[str]) -> pd.DataFrame:
    ids = list(study_locus_ids)
    if not ids:
        return pd.DataFrame()
    con = duckdb.connect()
    placeholders = ", ".join(["?"] * len(ids))
    base_query = f"""
    SELECT studyLocusId, studyId, variantId, chromosome, position, locus,
           pValue, pValueMantissa, pValueExponent
    FROM read_parquet('{parquet_glob}')
    WHERE studyLocusId IN ({placeholders})
    """
    base_df = con.execute(base_query, ids).df()

    cs_size = None
    if "locus" in base_df.columns:
        try:
            size_df = con.execute(
                "SELECT studyLocusId, max(list_length(locus)) AS cs_size FROM base_df GROUP BY studyLocusId"
            ).df()
            cs_size = size_df
        except Exception:
            cs_size = None

    cs_pip = None
    if "locus" in base_df.columns:
        try:
            pip_query = """
            SELECT studyLocusId, max(entry.posteriorProbability) AS cs_max_pip
            FROM base_df
            CROSS JOIN UNNEST(locus) AS entry
            GROUP BY studyLocusId
            """
            cs_pip = con.execute(pip_query).df()
        except Exception:
            cs_pip = None

    def _compute_neglog(row: pd.Series) -> float | None:
        mantissa = row.get("pValueMantissa")
        exponent = row.get("pValueExponent")
        pval = row.get("pValue")
        val = None
        if not pd.isna(mantissa) and not pd.isna(exponent):
            val = float(mantissa) * (10 ** float(exponent))
        elif not pd.isna(pval):
            val = float(pval)
        if val is None or val <= 0:
            return None
        return -math.log10(val)

    base_df["cs_neglog10p"] = base_df.apply(_compute_neglog, axis=1)
    pval_df = base_df[["studyLocusId", "cs_neglog10p"]].dropna()
    agg = pval_df.groupby("studyLocusId", as_index=False)["cs_neglog10p"].max()

    merged = base_df.drop_duplicates(subset=["studyLocusId"]).copy()
    merged = merged[["studyLocusId", "studyId", "variantId", "chromosome", "position"]]
    merged = merged.merge(agg, on="studyLocusId", how="left")
    if cs_size is not None:
        merged = merged.merge(cs_size, on="studyLocusId", how="left")
    else:
        merged["cs_size"] = np.nan
    if cs_pip is not None:
        merged = merged.merge(cs_pip, on="studyLocusId", how="left")
    else:
        merged["cs_max_pip"] = np.nan
    con.close()
    return merged


def finalize_features(mapped: pd.DataFrame, nearest_df: pd.DataFrame, cs_df: pd.DataFrame) -> pd.DataFrame:
    df = mapped.merge(nearest_df, on="studyLocusId", how="left")
    df = df.merge(cs_df, on="studyLocusId", how="left", suffixes=("", "_cs"))
    df["y"] = (df["nearestGeneId"] == df["goldGeneId"]).astype(int)
    return df


__all__ = [
    "annotate_nearest_genes",
    "compute_credible_set_features",
    "finalize_features",
]
