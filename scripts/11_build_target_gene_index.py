#!/usr/bin/env python
"""Build target gene index from Open Targets target parquet."""

from __future__ import annotations

import argparse
import pathlib

from src.ng_accuracy.config import load_config
from src.ng_accuracy.logging_utils import configure_logging
from src.ng_accuracy.target_index import build_gene_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging()
    raw_dir = pathlib.Path(config["paths"]["raw_dir"])
    release = config["release"]
    parquet_glob = raw_dir / "opentargets" / release / "target" / "*.parquet"
    output = pathlib.Path(config["paths"]["interim_dir"]) / "target_genes.parquet"
    if output.exists() and not args.force:
        print(f"Target gene index already exists at {output}")
        return
    build_gene_table(str(parquet_glob), output)
    print(f"Wrote gene index to {output}")


if __name__ == "__main__":
    main()
