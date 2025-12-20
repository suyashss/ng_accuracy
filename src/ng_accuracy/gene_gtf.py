"""Utilities for parsing GENCODE GTF gene annotations."""

from __future__ import annotations

import gzip
import pathlib
from typing import Dict, Iterable, List

import pandas as pd

from .io_utils import ensure_dir


REQUIRED_GENE_COLS = [
    "gene_id",
    "gene_name",
    "gene_type",
    "chrom",
    "start",
    "end",
    "strand",
    "gene_tss",
]


def _parse_attributes(attr_field: str) -> Dict[str, str]:
    parts = [p.strip() for p in attr_field.strip().split(";") if p.strip()]
    attrs: Dict[str, str] = {}
    for part in parts:
        if " " not in part:
            continue
        key, value = part.split(" ", 1)
        attrs[key] = value.strip('"')
    return attrs


def parse_gtf(gtf_gz_path: str | pathlib.Path) -> pd.DataFrame:
    gtf_path = pathlib.Path(gtf_gz_path)
    rows: List[Dict[str, object]] = []
    with gzip.open(gtf_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = fields
            if feature != "gene":
                continue
            attr_map = _parse_attributes(attrs)
            gene_id = attr_map.get("gene_id")
            if gene_id is None:
                continue
            gene_name = attr_map.get("gene_name", "")
            gene_type = attr_map.get("gene_type", "")
            start_i = int(start)
            end_i = int(end)
            gene_tss = start_i if strand == "+" else end_i
            rows.append(
                {
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "gene_type": gene_type,
                    "chrom": chrom.replace("chr", ""),
                    "start": start_i,
                    "end": end_i,
                    "strand": strand,
                    "gene_tss": gene_tss,
                }
            )
    df = pd.DataFrame(rows, columns=REQUIRED_GENE_COLS)
    return df


def write_gene_tss_parquet(df: pd.DataFrame, out_path: str | pathlib.Path) -> None:
    path = pathlib.Path(out_path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def load_gene_tss(path: str | pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_gene_index(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return dict of chrom -> df sorted by gene_tss for nearest search."""
    index: Dict[str, pd.DataFrame] = {}
    for chrom, sub in df.groupby("chrom"):
        index[chrom] = sub.sort_values("gene_tss").reset_index(drop=True)
    return index


__all__ = [
    "parse_gtf",
    "write_gene_tss_parquet",
    "load_gene_tss",
    "build_gene_index",
    "REQUIRED_GENE_COLS",
]
