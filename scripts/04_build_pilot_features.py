#!/usr/bin/env python
"""Build pilot features combining gold, credible set, and gene annotations."""

from __future__ import annotations

import argparse
import pathlib
import pandas as pd
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_accuracy.config import add_config_arg, load_config, resolve_release_path
from ng_accuracy.features_pilot import annotate_nearest_genes, compute_credible_set_features, finalize_features
from ng_accuracy.gene_gtf import parse_gtf, write_gene_tss_parquet, load_gene_tss
from ng_accuracy.io_utils import ensure_dir
from ng_accuracy.logging_utils import get_logger


def ensure_gene_tss(cfg: dict, logger) -> pathlib.Path:
    genes_parquet = pathlib.Path(cfg["paths"]["interim_dir"]) / "genes_tss.parquet"
    if genes_parquet.exists():
        return genes_parquet
    gtf_path = pathlib.Path(cfg["gene_gtf"]["local_path"])
    if not gtf_path.exists():
        raise FileNotFoundError(f"Missing GTF file {gtf_path}; run 02_download_gene_gtf.py")
    logger.info("Parsing GTF to gene TSS parquet... this may take a moment")
    genes_df = parse_gtf(gtf_path)
    write_gene_tss_parquet(genes_df, genes_parquet)
    return genes_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("build_features")

    mapped_path = pathlib.Path(cfg["paths"]["interim_dir"]) / "mapped_loci.parquet"
    if not mapped_path.exists():
        raise FileNotFoundError(f"Missing mapped loci file {mapped_path}; run 03_map_gold_to_credible_set.py")
    mapped = pd.read_parquet(mapped_path)

    genes_parquet = ensure_gene_tss(cfg, logger)
    genes_df = load_gene_tss(genes_parquet)

    release = cfg["release"]
    cred_dir = resolve_release_path(cfg["opentargets"]["credible_set_local_dir"], release)
    parquet_glob = str(pathlib.Path(cred_dir) / "*.parquet")

    window_bp = int(cfg["features"]["gene_window_bp"])
    nearest_df = annotate_nearest_genes(mapped, genes_df, window_bp)

    cs_df = compute_credible_set_features(parquet_glob, mapped["studyLocusId"].unique())

    features_df = finalize_features(mapped, nearest_df, cs_df)

    processed_dir = pathlib.Path(cfg["paths"]["processed_dir"])
    ensure_dir(processed_dir)
    out_parquet = processed_dir / "pilot_features.parquet"
    out_csv = processed_dir / "pilot_features.csv"
    features_df.to_parquet(out_parquet, index=False)
    features_df.to_csv(out_csv, index=False)

    missingness = features_df.isna().mean().reset_index()
    missingness.columns = ["column", "missing_fraction"]
    reports_dir = pathlib.Path("reports")
    ensure_dir(reports_dir)
    missingness.to_csv(reports_dir / "feature_missingness.csv", index=False)

    logger.info("Wrote features to %s", out_parquet)


if __name__ == "__main__":
    main()
