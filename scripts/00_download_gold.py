#!/usr/bin/env python
"""Download and normalize Open Targets Genetics gold standards."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

import pandas as pd

from ng_accuracy.config import add_config_arg, load_config
from ng_accuracy.io_utils import download_file, ensure_dir, write_json
from ng_accuracy.logging_utils import get_logger

REQUIRED_COLS = [
    "association_info.otg_id",
    "gold_standard_info.gene_id",
    "sentinel_variant.locus_GRCh38.chromosome",
    "sentinel_variant.locus_GRCh38.position",
    "sentinel_variant.alleles.reference",
    "sentinel_variant.alleles.alternative",
]


def normalize_gold(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "goldRowId": range(len(df)),
            "studyId": df["association_info.otg_id"],
            "goldGeneId": df["gold_standard_info.gene_id"],
            "chrom": df["sentinel_variant.locus_GRCh38.chromosome"].astype(str).str.replace("chr", "", regex=False),
            "pos": df["sentinel_variant.locus_GRCh38.position"].astype(int),
            "ref": df["sentinel_variant.alleles.reference"],
            "alt": df["sentinel_variant.alleles.alternative"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--include-medium", action="store_true", help="Include Medium confidence loci")
    parser.add_argument("--force", action="store_true", help="Redownload even if file exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("download_gold")

    gold_url = cfg["gold"]["url"]
    gold_path = pathlib.Path(cfg["gold"]["out_path"])
    ensure_dir(gold_path.parent)

    download_file(gold_url, gold_path, force=args.force)

    df = pd.read_csv(gold_path, sep="\t")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    confidence = df.get("gold_standard_info.evidence.confidence", pd.Series([], dtype=str)).fillna("")
    keep = confidence.isin(["High", "High|High"])
    if args.include_medium:
        keep = keep | confidence.eq("Medium")
    filtered = df[keep].copy()
    logger.info("Filtered gold rows: %d -> %d", len(df), len(filtered))

    norm = normalize_gold(filtered)
    interim_path = pathlib.Path(cfg["paths"]["interim_dir"]) / "gold_normalized.parquet"
    ensure_dir(interim_path.parent)
    norm.to_parquet(interim_path, index=False)

    summary = {"rows": int(len(norm)), "unique_studies": int(norm["studyId"].nunique())}
    write_json(pathlib.Path("reports") / "gold_summary.json", summary)
    logger.info("Wrote normalized gold to %s", interim_path)


if __name__ == "__main__":
    main()
