"""I/O helper utilities."""

from __future__ import annotations

import json
import pathlib
import shutil
import urllib.request
from typing import Any, Dict

from tqdm import tqdm


class TqdmUpTo(tqdm):
    """Progress bar for urlretrieve."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def ensure_dir(path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, dest: str | pathlib.Path, force: bool = False) -> pathlib.Path:
    dest_path = pathlib.Path(dest)
    ensure_dir(dest_path.parent)
    if dest_path.exists() and not force:
        return dest_path

    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)
    return dest_path


def write_json(path: str | pathlib.Path, payload: Dict[str, Any]) -> None:
    path = pathlib.Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def copytree(src: pathlib.Path, dst: pathlib.Path, overwrite: bool = False) -> None:
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    if not dst.exists():
        shutil.copytree(src, dst)


__all__ = ["download_file", "ensure_dir", "write_json", "copytree"]
