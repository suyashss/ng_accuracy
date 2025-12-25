#!/usr/bin/env python
"""Build full feature table."""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from src.ng_accuracy.config import load_config
from src.ng_accuracy.features_full import assemble_features
from src.ng_accuracy.logging_utils import configure_logging
from src.ng_accuracy.target_index import TargetGeneIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging()
    interim = pathlib.Path(cfg["paths"]["interim_dir"])
    processed = pathlib.Path(cfg["paths"]["processed_dir"])
    mapped = pd.read_parquet(interim / "mapped_loci_full.parquet")
    subsets = interim / "subsets_full"
    cs_df = pd.read_parquet(subsets / "credible_set_subset.parquet") if (subsets / "credible_set_subset.parquet").exists() else pd.DataFrame()
    study_df = pd.read_parquet(subsets / "study_subset.parquet") if (subsets / "study_subset.parquet").exists() else pd.DataFrame()
    variant_df = pd.DataFrame()
    coloc_df = pd.DataFrame()
    e2g_df = pd.DataFrame()
    l2g_df = pd.DataFrame()
    target_index = TargetGeneIndex.from_target_parquet(str(pathlib.Path(cfg["paths"]["interim_dir"]) / "target_genes.parquet"))
    output_path = processed / "full_features.parquet"
    if output_path.exists() and not args.force:
        print(f"Features already exist at {output_path}")
        return
    assemble_features(
        mapped,
        target_index,
        cs_df,
        study_df,
        variant_df,
        coloc_df,
        e2g_df,
        l2g_df,
        cfg["nearest_gene"]["definition"],
        cfg["nearest_gene"]["windows_bp"],
        output_path,
        pathlib.Path(cfg["paths"]["reports_dir"]),
    )
    print(f"Wrote features to {output_path}")


if __name__ == "__main__":
    main()
