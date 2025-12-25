"""Robust Open Targets dataset downloader.

This module mirrors the pilot downloader but supports multiple datasets. It
tries wget first and falls back to Python requests with HTML index parsing.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import pathlib
import re
import subprocess
from typing import Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)


INDEX_PATTERN = re.compile(r"href=\"([^\"]+\.parquet)\"")


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _wget(url: str, dest: pathlib.Path) -> bool:
    try:
        subprocess.run(["wget", "-q", "-O", str(dest), url], check=True)
        return True
    except Exception:
        return False


def _download_python(url: str, dest: pathlib.Path) -> bool:
    with requests.get(url, stream=True, timeout=30) as r:
        if r.status_code != 200:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1_048_576):
                if chunk:
                    fh.write(chunk)
    return True


def _list_parquet_urls(index_url: str) -> List[str]:
    resp = requests.get(index_url, timeout=30)
    resp.raise_for_status()
    matches = INDEX_PATTERN.findall(resp.text)
    return [index_url.rstrip("/") + "/" + m for m in matches]


def download_dataset(base_url: str, release: str, dataset: str, raw_dir: pathlib.Path, force: bool = False) -> Dict[str, int]:
    dest_dir = raw_dir / "opentargets" / release / dataset
    _ensure_dir(dest_dir)
    index_url = base_url.format(release=release) + f"/{dataset}/"
    try:
        urls = _list_parquet_urls(index_url)
    except Exception:
        urls = []
    if not urls:
        # fallback to wildcard assuming predictable names
        urls = [index_url + "part-0.parquet"]
    stats = {"downloaded": 0, "skipped": 0}
    for url in urls:
        filename = url.split("/")[-1]
        dest = dest_dir / filename
        if dest.exists() and not force:
            stats["skipped"] += 1
            continue
        if not _wget(url, dest):
            logger.info("wget failed for %s, falling back to python", url)
            _download_python(url, dest)
        stats["downloaded"] += 1
    return stats


def download_many(base_url: str, release: str, datasets: Iterable[str], raw_dir: pathlib.Path, max_workers: int = 4, force: bool = False) -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_dataset, base_url, release, dataset, raw_dir, force): dataset
            for dataset in datasets
        }
        for fut in concurrent.futures.as_completed(futures):
            dataset = futures[fut]
            try:
                stats = fut.result()
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("download failed for %s: %s", dataset, exc)
                stats = {"downloaded": 0, "skipped": 0, "error": 1}
            results[dataset] = stats
    return results


def write_manifest(manifest_path: pathlib.Path, base_dir: pathlib.Path, release: str, results: Dict[str, Dict[str, int]]) -> None:
    manifest: Dict[str, Dict[str, int | str]] = {}
    for dataset, stats in results.items():
        dest_dir = base_dir / "opentargets" / release / dataset
        files = sorted(dest_dir.glob("*.parquet"))
        total_bytes = sum(f.stat().st_size for f in files)
        manifest[dataset] = {
            "num_files": len(files),
            "total_bytes": total_bytes,
            "first_file": files[0].name if files else None,
            "last_file": files[-1].name if files else None,
        }
        manifest[dataset].update(stats)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


__all__ = ["download_many", "write_manifest", "download_dataset"]
