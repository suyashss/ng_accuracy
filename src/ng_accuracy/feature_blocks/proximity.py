"""Proximity and nearest-gene feature block."""

from __future__ import annotations

import pandas as pd

from ..target_index import TargetGeneIndex, compute_nearest_features


def build_proximity_features(index: TargetGeneIndex, loci: pd.DataFrame, definition: str, windows) -> pd.DataFrame:
    return compute_nearest_features(index, loci, definition, windows)


__all__ = ["build_proximity_features"]
