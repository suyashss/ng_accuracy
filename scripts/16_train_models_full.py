#!/usr/bin/env python
"""Train models for the full pipeline."""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from src.ng_accuracy.config import load_config
from src.ng_accuracy.logging_utils import configure_logging
from src.ng_accuracy.modeling_full import train_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging()
    features = pd.read_parquet(pathlib.Path(cfg["paths"]["processed_dir"]) / "full_features.parquet")
    output_dir = pathlib.Path(cfg["paths"]["processed_dir"]) / "models" / "full"
    if output_dir.exists() and not args.force:
        print(f"Models already exist in {output_dir}")
        return
    metrics = train_models(features, output_dir, seed=cfg["splits"].get("seed", 1))
    print(metrics)


if __name__ == "__main__":
    main()
