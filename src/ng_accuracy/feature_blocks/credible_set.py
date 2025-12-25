"""Credible set feature computation with robust locus parsing."""

from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np
import pandas as pd

from ..normalize import normalize_variant_id


def _locus_entries(val) -> list:
    if isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    if isinstance(val, np.ndarray):
        return list(val)
    return []


def _safe_get(entry, key):
    if isinstance(entry, dict):
        return entry.get(key)
    return None


def parse_locus_features(locus_col) -> pd.DataFrame:
    df = pd.DataFrame({"locus": locus_col})
    df["cs_size"] = df["locus"].apply(lambda v: float(len(_locus_entries(v))) if _locus_entries(v) else np.nan)

    def _max_pip(val):
        entries = _locus_entries(val)
        pips = [_safe_get(e, "posteriorProbability") for e in entries if isinstance(e, dict)]
        pips = [p for p in pips if p is not None]
        return float(max(pips)) if pips else np.nan

    def _entropy(val):
        entries = _locus_entries(val)
        pips = [_safe_get(e, "posteriorProbability") for e in entries if isinstance(e, dict)]
        pips = [p for p in pips if p is not None and p >= 0]
        if not pips:
            return np.nan
        total = float(sum(pips))
        if total <= 0:
            return np.nan
        probs = [p / total for p in pips]
        return float(-sum(p * math.log(p) for p in probs if p > 0))

    def _top2_gap(val):
        entries = _locus_entries(val)
        pips = sorted([
            _safe_get(e, "posteriorProbability") for e in entries if isinstance(e, dict) and _safe_get(e, "posteriorProbability") is not None
        ], reverse=True)
        if len(pips) < 2:
            return np.nan
        return float(pips[0] - pips[1])

    df["cs_max_pip"] = df["locus"].apply(_max_pip)
    df["cs_entropy"] = df["locus"].apply(_entropy)
    df["cs_top2_pip_gap"] = df["locus"].apply(_top2_gap)
    return df


def compute_credible_set_features(cs_df: pd.DataFrame) -> pd.DataFrame:
    if cs_df.empty:
        return pd.DataFrame()
    parsed = parse_locus_features(cs_df["locus"]) if "locus" in cs_df.columns else pd.DataFrame()
    cs_df = cs_df.copy()
    cs_df["variantId"] = cs_df.get("variantId").apply(normalize_variant_id)

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

    cs_df["cs_neglog10p"] = cs_df.apply(_compute_neglog, axis=1)
    agg_cols = ["cs_neglog10p"]
    if not parsed.empty:
        cs_df = pd.concat([cs_df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
        agg_cols += ["cs_size", "cs_max_pip", "cs_entropy", "cs_top2_pip_gap"]
    grouped = cs_df.groupby("studyLocusId", as_index=False)[agg_cols].max()
    meta_cols = [c for c in ["cs_finemappingMethod", "finemappingMethod", "cs_confidence", "confidence", "qualityControls"] if c in cs_df.columns]
    result = grouped
    if meta_cols:
        meta = cs_df[["studyLocusId", *meta_cols]].drop_duplicates("studyLocusId")
        result = result.merge(meta, on="studyLocusId", how="left")
    return result


__all__ = ["compute_credible_set_features"]
