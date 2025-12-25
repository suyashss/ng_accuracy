"""Target gene indexing and nearest-gene queries."""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

from .normalize import normalize_chrom, normalize_gene_id

logger = logging.getLogger(__name__)


@dataclass
class TargetGeneIndex:
    genes: pd.DataFrame
    by_chrom_tss: Dict[str, np.ndarray]
    by_chrom_body: Dict[str, np.ndarray]

    @classmethod
    def from_target_parquet(cls, parquet_glob: str) -> "TargetGeneIndex":
        con = duckdb.connect()
        df = con.execute(
            f"SELECT id AS geneId, approvedSymbol, biotype, chr AS chrom, start, `end`, strand, tss FROM read_parquet('{parquet_glob}')"
        ).df()
        con.close()
        df["geneId_base"] = df["geneId"].apply(normalize_gene_id)
        df["chrom"] = df["chrom"].apply(normalize_chrom)
        by_chrom_tss: Dict[str, np.ndarray] = {}
        by_chrom_body: Dict[str, np.ndarray] = {}
        for chrom, sub in df.groupby("chrom"):
            by_chrom_tss[chrom] = sub.sort_values("tss")["tss"].to_numpy()
            by_chrom_body[chrom] = sub.sort_values("start")["start"].to_numpy()
        return cls(df, by_chrom_tss, by_chrom_body)

    def nearest_by_tss(self, chrom: str, pos: int, protein_coding_only: bool = False) -> Tuple[Optional[pd.Series], Optional[float], Optional[float]]:
        chrom = normalize_chrom(chrom) or chrom
        genes = self.genes
        if protein_coding_only:
            genes = genes[genes["biotype"] == "protein_coding"]
        sub = genes[genes["chrom"] == chrom]
        if sub.empty:
            return None, None, None
        tss = sub["tss"].to_numpy()
        idx = np.searchsorted(tss, pos)
        candidates: List[int] = []
        if idx < len(tss):
            candidates.append(idx)
        if idx - 1 >= 0:
            candidates.append(idx - 1)
        if not candidates:
            return None, None, None
        dists = [(abs(int(tss[i]) - pos), i) for i in candidates]
        dists.sort()
        nearest_dist, nearest_idx = dists[0]
        second_dist = dists[1][0] if len(dists) > 1 else None
        return sub.iloc[nearest_idx], float(nearest_dist), float(second_dist) if second_dist is not None else None

    def nearest_by_gene_body(self, chrom: str, pos: int, protein_coding_only: bool = False) -> Tuple[Optional[pd.Series], Optional[float]]:
        chrom = normalize_chrom(chrom) or chrom
        genes = self.genes
        if protein_coding_only:
            genes = genes[genes["biotype"] == "protein_coding"]
        sub = genes[genes["chrom"] == chrom]
        if sub.empty:
            return None, None
        starts = sub["start"].to_numpy()
        idx = np.searchsorted(starts, pos)
        candidates: List[int] = []
        if idx < len(starts):
            candidates.append(idx)
        if idx - 1 >= 0:
            candidates.append(idx - 1)
        best = None
        best_dist = None
        for cand in candidates:
            row = sub.iloc[cand]
            start, end = int(row["start"]), int(row["end"])
            if start <= pos <= end:
                dist = 0
            elif pos < start:
                dist = start - pos
            else:
                dist = pos - end
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = row
        return best, float(best_dist) if best_dist is not None else None


def build_gene_table(parquet_glob: str, output_path: pathlib.Path) -> TargetGeneIndex:
    index = TargetGeneIndex.from_target_parquet(parquet_glob)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.genes.to_parquet(output_path, index=False)
    return index


def compute_nearest_features(index: TargetGeneIndex, loci: pd.DataFrame, definition: str, windows: Iterable[int]) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in loci.iterrows():
        chrom = row.get("chrom") or row.get("chromosome")
        pos = int(row.get("pos") or row.get("position"))
        protein_only = definition.endswith("protein_coding")
        use_gene_body = definition.startswith("gene_body")
        if use_gene_body:
            nearest, dist_primary = index.nearest_by_gene_body(chrom, pos, protein_only)
            _, dist_secondary = index.nearest_by_tss(chrom, pos, protein_only)
        else:
            nearest, dist_primary, dist_secondary = index.nearest_by_tss(chrom, pos, protein_only)
        nearest_gene_id = nearest["geneId"] if nearest is not None else None
        nearest_gene_base = normalize_gene_id(nearest_gene_id)
        feature_row = {
            "studyLocusId": row["studyLocusId"],
            "nearestGeneId": nearest_gene_id,
            "nearestGeneId_base": nearest_gene_base,
            "ng_def": definition,
            "ng_dist_tss": dist_primary,
            "ng_dist_2nd_tss": dist_secondary,
            "ng_dist_gap_1_2": (dist_secondary - dist_primary) if dist_primary is not None and dist_secondary is not None else None,
            "ng_is_protein_coding": bool(nearest["biotype"] == "protein_coding") if nearest is not None else None,
        }
        # alternative distances
        alt_all_tss = index.nearest_by_tss(chrom, pos, protein_coding_only=False)[1]
        alt_pc_tss = index.nearest_by_tss(chrom, pos, protein_coding_only=True)[1]
        alt_all_body = index.nearest_by_gene_body(chrom, pos, protein_coding_only=False)[1]
        alt_pc_body = index.nearest_by_gene_body(chrom, pos, protein_coding_only=True)[1]
        feature_row.update(
            {
                "dist_to_nearest_all_tss": alt_all_tss,
                "dist_to_nearest_pc_tss": alt_pc_tss,
                "dist_to_nearest_all_gene_body": alt_all_body,
                "dist_to_nearest_pc_gene_body": alt_pc_body,
            }
        )
        for window in windows:
            # rough density using tss sorted arrays
            chrom_tss = index.by_chrom_tss.get(normalize_chrom(chrom) or chrom, np.array([]))
            left = np.searchsorted(chrom_tss, pos - window, side="left")
            right = np.searchsorted(chrom_tss, pos + window, side="right")
            feature_row[f"ng_gene_density_within_{window}bp"] = float(right - left)
        rows.append(feature_row)
    return pd.DataFrame(rows)


__all__ = ["TargetGeneIndex", "build_gene_table", "compute_nearest_features"]
