"""Lightweight helpers to read Open Targets parquet datasets with DuckDB."""

from __future__ import annotations

import logging
import pathlib
from typing import Iterable, List, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def describe_parquet(parquet_glob: str) -> List[str]:
    con = duckdb.connect()
    cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_glob}')").df()["column_name"].tolist()
    con.close()
    return cols


def read_filtered(parquet_glob: str, column: str, values: Iterable[str]) -> pd.DataFrame:
    ids = list(values)
    if not ids:
        return pd.DataFrame()
    con = duckdb.connect()
    placeholders = ", ".join(["?"] * len(ids))
    query = f"SELECT * FROM read_parquet('{parquet_glob}') WHERE {column} IN ({placeholders})"
    df = con.execute(query, ids).df()
    con.close()
    return df


def scan_to_df(parquet_glob: str, limit: Optional[int] = None) -> pd.DataFrame:
    con = duckdb.connect()
    query = f"SELECT * FROM read_parquet('{parquet_glob}')"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    df = con.execute(query).df()
    con.close()
    return df


def write_subset(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


__all__ = ["describe_parquet", "read_filtered", "scan_to_df", "write_subset"]
