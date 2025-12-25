"""Normalization helpers shared across pipelines."""

from __future__ import annotations

import re
from typing import Optional


def normalize_gene_id(gene_id: Optional[str]) -> Optional[str]:
    """Strip Ensembl version suffixes and whitespace.

    The pilot pipeline normalized nearest gene IDs when creating labels. The full
    pipeline reuses this utility everywhere gene identifiers are compared.
    """

    if gene_id is None:
        return None
    gene_id = str(gene_id).strip()
    if not gene_id:
        return None
    return re.sub(r"\.[0-9]+$", "", gene_id)


def normalize_chrom(chrom: Optional[str]) -> Optional[str]:
    if chrom is None:
        return None
    c = str(chrom).strip()
    if not c:
        return None
    if c.lower().startswith("chr"):
        c = c[3:]
    return c


def normalize_variant_id(variant_id: Optional[str]) -> Optional[str]:
    """Normalize a variant identifier of the form chr_pos_ref_alt."""

    if variant_id is None:
        return None
    vid = str(variant_id).strip()
    if not vid:
        return None
    parts = vid.split("_")
    if len(parts) == 4:
        chrom = normalize_chrom(parts[0])
        return f"{chrom}_{parts[1]}_{parts[2]}_{parts[3]}"
    return vid


__all__ = ["normalize_gene_id", "normalize_chrom", "normalize_variant_id"]
