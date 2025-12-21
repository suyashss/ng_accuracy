#!/usr/bin/env python
"""Map gold loci to Open Targets credible set entries using DuckDB."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import duckdb
import pandas as pd

from ng_accuracy.config import add_config_arg, load_config, resolve_release_path
from ng_accuracy.io_utils import ensure_dir
from ng_accuracy.logging_utils import get_logger



def build_variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{chrom}_{pos}_{ref.upper()}_{alt.upper()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("map_gold")

    gold_path = pathlib.Path(cfg["paths"]["interim_dir"]) / "gold_normalized.parquet"
    if not gold_path.exists():
        raise FileNotFoundError(f"Missing gold normalized file {gold_path}")
    gold_df = pd.read_parquet(gold_path)
    gold_df["variantId"] = gold_df.apply(
        lambda r: build_variant_id(r["chrom"], int(r["pos"]), r["ref"], r["alt"]), axis=1
    )

    release = cfg["release"]
    cred_dir = resolve_release_path(cfg["opentargets"]["credible_set_local_dir"], release)
    parquet_glob = str(pathlib.Path(cred_dir) / "*.parquet")

    con = duckdb.connect()
    con.register("gold", gold_df)
    query = f"""
    SELECT g.goldRowId, g.studyId, cs.studyLocusId, cs.variantId, cs.chromosome, cs.position, g.goldGeneId
    FROM gold g
    INNER JOIN read_parquet('{parquet_glob}') cs
    ON g.studyId = cs.studyId AND g.variantId = cs.variantId
    """
    mapped = con.execute(query).df()
    con.close()

    out_path = pathlib.Path(cfg["paths"]["interim_dir"]) / "mapped_loci.parquet"
    ensure_dir(out_path.parent)
    mapped.to_parquet(out_path, index=False)

    mapped_count = len(mapped)
    total = len(gold_df)
    unmapped = total - mapped_count
    summary_lines = [
        "# Mapping summary",
        "",
        f"Total gold rows: {total}",
        f"Mapped to credible_set: {mapped_count}",
        f"Unmapped: {unmapped}",
        "",
        "Mapping performed via studyId + variantId join using DuckDB.",
    ]
    report_path = pathlib.Path("reports") / "mapping_summary.md"
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Mapped %d/%d loci", mapped_count, total)


if __name__ == "__main__":
    main()
