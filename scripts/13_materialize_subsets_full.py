#!/usr/bin/env python
"""Materialize subset parquets for mapped loci."""

from __future__ import annotations

import argparse
import json
import pathlib

import duckdb
import pandas as pd

from src.ng_accuracy.config import load_config
from src.ng_accuracy.logging_utils import configure_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging()

    interim_dir = pathlib.Path(cfg["paths"]["interim_dir"])
    mapped_path = interim_dir / "mapped_loci_full.parquet"
    if not mapped_path.exists():
        raise FileNotFoundError("Run mapping step first")
    mapped = pd.read_parquet(mapped_path)
    study_locus_ids = mapped["studyLocusId"].unique().tolist()
    release = cfg["release"]
    raw_dir = pathlib.Path(cfg["paths"]["raw_dir"]) / "opentargets" / release

    subsets_dir = interim_dir / "subsets_full"
    subsets_dir.mkdir(parents=True, exist_ok=True)

    # credible set subset
    cs_glob = raw_dir / "credible_set" / "*.parquet"
    con = duckdb.connect()
    cs_subset = con.execute(
        f"SELECT * FROM read_parquet('{cs_glob}') WHERE studyLocusId IN ({', '.join(['?'] * len(study_locus_ids))})",
        study_locus_ids,
    ).df()
    con.close()
    cs_subset.to_parquet(subsets_dir / "credible_set_subset.parquet", index=False)

    # study subset
    study_ids = mapped["studyId"].unique().tolist()
    study_glob = raw_dir / "study" / "*.parquet"
    con = duckdb.connect()
    study_subset = con.execute(
        f"SELECT * FROM read_parquet('{study_glob}') WHERE studyId IN ({', '.join(['?'] * len(study_ids))})",
        study_ids,
    ).df()
    con.close()
    study_subset.to_parquet(subsets_dir / "study_subset.parquet", index=False)

    # optional datasets: write empty if missing
    manifest = {
        "credible_set_subset": len(cs_subset),
        "study_subset": len(study_subset),
    }
    manifest_path = pathlib.Path(cfg["paths"]["reports_dir"]) / "subset_manifest_full.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
