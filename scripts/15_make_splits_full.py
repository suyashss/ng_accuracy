#!/usr/bin/env python
"""Create train/test splits for the full pipeline."""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold

from src.ng_accuracy.config import load_config
from src.ng_accuracy.logging_utils import configure_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging()
    features = pd.read_parquet(pathlib.Path(cfg["paths"]["processed_dir"]) / "full_features.parquet")
    strategy = cfg["splits"].get("strategy", "chrom_nested_cv")
    rows = []
    if strategy == "group_by_study":
        splitter = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=cfg["splits"].get("seed", 1))
        for i, (train_idx, test_idx) in enumerate(splitter.split(features, groups=features["studyId"])):
            for idx in train_idx:
                rows.append({"studyLocusId": features.iloc[idx]["studyLocusId"], "split_id": i, "fold": "train", "is_train": True, "is_test": False})
            for idx in test_idx:
                rows.append({"studyLocusId": features.iloc[idx]["studyLocusId"], "split_id": i, "fold": "test", "is_train": False, "is_test": True})
    else:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg["splits"].get("seed", 1))
        for i, (train_idx, test_idx) in enumerate(skf.split(features, features["y"])):
            for idx in train_idx:
                rows.append({"studyLocusId": features.iloc[idx]["studyLocusId"], "split_id": i, "fold": "train", "is_train": True, "is_test": False})
            for idx in test_idx:
                rows.append({"studyLocusId": features.iloc[idx]["studyLocusId"], "split_id": i, "fold": "test", "is_train": False, "is_test": True})
    splits_df = pd.DataFrame(rows)
    out_path = pathlib.Path(cfg["paths"]["processed_dir"]) / "full_splits.parquet"
    if out_path.exists() and not args.force:
        print(f"Splits already exist at {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_parquet(out_path, index=False)
    print(f"Wrote splits to {out_path}")


if __name__ == "__main__":
    main()
