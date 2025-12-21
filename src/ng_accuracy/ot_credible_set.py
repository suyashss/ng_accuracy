"""Utilities for Open Targets credible set files."""

from __future__ import annotations

import pathlib
from typing import Iterable, List

import duckdb
import pandas as pd


def get_credible_set_parquet_path(base_dir: str | pathlib.Path) -> str:
    """Return glob path for credible set parquet files."""
    base = pathlib.Path(base_dir)
    return str(base / "*.parquet")


def load_filtered_study_loci(parquet_glob: str, study_locus_ids: Iterable[str]) -> pd.DataFrame:
    ids = list(study_locus_ids)
    if not ids:
        return pd.DataFrame()
    con = duckdb.connect()
    query = f"""
    SELECT * FROM read_parquet('{parquet_glob}')
    WHERE studyLocusId IN ({', '.join(['?'] * len(ids))})
    """
    df = con.execute(query, ids).df()
    con.close()
    return df


def compute_pvalue(mantissa: float | None, exponent: float | None, pval: float | None) -> float | None:
    if mantissa is not None and exponent is not None:
        return float(mantissa) * (10 ** float(exponent))
    if pval is not None:
        return float(pval)
    return None


__all__ = ["get_credible_set_parquet_path", "load_filtered_study_loci", "compute_pvalue"]
