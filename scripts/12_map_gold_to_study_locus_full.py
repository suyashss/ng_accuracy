#!/usr/bin/env python
"""Map gold standards to Open Targets study loci for the full pipeline."""

from __future__ import annotations

import argparse
import pathlib

import duckdb
import pandas as pd

from src.ng_accuracy.config import load_config
from src.ng_accuracy.logging_utils import configure_logging
from src.ng_accuracy.normalize import normalize_gene_id, normalize_variant_id


def build_variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{chrom}_{pos}_{ref.upper()}_{alt.upper()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging()
    gold_path = pathlib.Path(cfg["paths"]["interim_dir"]) / "gold_normalized.parquet"
    if not gold_path.exists():
        raise FileNotFoundError("gold_normalized.parquet missing; run 00_download_gold.py")
    gold_df = pd.read_parquet(gold_path)
    gold_df["variantId"] = gold_df.apply(lambda r: build_variant_id(r["chrom"], int(r["pos"]), r["ref"], r["alt"]), axis=1)
    gold_df["variantId"] = gold_df["variantId"].apply(normalize_variant_id)
    gold_df["goldGeneId_base"] = gold_df["goldGeneId"].apply(normalize_gene_id)

    release = cfg["release"]
    parquet_glob = pathlib.Path(cfg["paths"]["raw_dir"]) / "opentargets" / release / "credible_set" / "*.parquet"
    con = duckdb.connect()
    con.register("gold", gold_df)
    query = f"""
    SELECT g.goldRowId, g.studyId, cs.studyLocusId, cs.variantId, cs.chromosome AS chrom, cs.position AS pos,
           cs.locusStart, cs.locusEnd, g.goldGeneId, g.goldGeneId_base
    FROM gold g
    INNER JOIN read_parquet('{parquet_glob}') cs
    ON g.studyId = cs.studyId AND g.variantId = cs.variantId
    """
    mapped = con.execute(query).df()
    con.close()
    mapped["variantId"] = mapped["variantId"].apply(normalize_variant_id)
    mapped = mapped.sort_values(["studyId", "variantId", "cs_neglog10p"], ascending=[True, True, False]) if "cs_neglog10p" in mapped.columns else mapped
    mapped = mapped.drop_duplicates(subset=["studyId", "variantId"], keep="first")

    out_path = pathlib.Path(cfg["paths"]["interim_dir"]) / "mapped_loci_full.parquet"
    if out_path.exists() and not args.force:
        print(f"Existing mapped loci at {out_path}; skipping")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mapped.to_parquet(out_path, index=False)
    report_lines = [
        "# Mapping summary (full)",
        "",
        f"Gold rows: {len(gold_df)}",
        f"Mapped rows: {len(mapped)}",
        f"Unmapped: {len(gold_df) - len(mapped)}",
    ]
    report_path = pathlib.Path(cfg["paths"]["reports_dir"]) / "mapping_summary_full.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
