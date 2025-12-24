#!/usr/bin/env python
"""Download Open Targets credible set parquet directory for a release."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

import re
import subprocess
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from ng_accuracy.config import add_config_arg, load_config, resolve_release_path
from ng_accuracy.io_utils import download_file, ensure_dir
from ng_accuracy.logging_utils import get_logger


class IndexLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.hrefs.append(value)


def list_parquet_urls(base_url: str) -> list[str]:
    with urllib.request.urlopen(base_url) as response:
        body = response.read().decode("utf-8", errors="replace")

    parser = IndexLinkParser()
    parser.feed(body)

    urls = [urljoin(base_url, href) for href in parser.hrefs if href.endswith(".parquet")]
    if not urls:
        matches = re.findall(r"[A-Za-z0-9._-]+\\.parquet", body)
        urls = [urljoin(base_url, name) for name in matches]
    return sorted(set(urls))


def download_with_wget(base_url: str, local_dir: pathlib.Path, logger) -> bool:
    path_parts = [part for part in urlparse(base_url).path.strip("/").split("/") if part]
    cut_dirs = len(path_parts)
    wget_cmd = [
        "wget",
        "--recursive",
        "--level=1",
        "--no-parent",
        "--accept",
        "*.parquet",
        "--reject",
        "index.html*",
        f"--cut-dirs={cut_dirs}",
        "--no-host-directories",
        "--directory-prefix",
        str(local_dir),
        base_url,
    ]

    logger.info("Running wget download... this may take time")
    try:
        subprocess.run(wget_cmd, check=True)
    except subprocess.CalledProcessError:
        logger.warning("wget failed; falling back to Python downloader")
        return False

    parquet_files = list(local_dir.rglob("*.parquet"))
    if not parquet_files:
        logger.warning("wget completed but no parquet files were downloaded; falling back to Python downloader")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--force", action="store_true", help="Redownload even if marker exists")
    parser.add_argument(
        "--wget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use wget for recursive download (default: enabled)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("download_credible_set")

    release = cfg["release"]
    base_url = resolve_release_path(cfg["opentargets"]["credible_set_base_url"], release)
    local_dir = pathlib.Path(resolve_release_path(cfg["opentargets"]["credible_set_local_dir"], release))
    marker = local_dir.parent / "CREDIBLE_SET_DOWNLOADED.txt"

    if marker.exists() and not args.force:
        logger.info("Credible set already downloaded: %s", marker)
        return

    ensure_dir(local_dir)

    downloaded = False
    if args.wget:
        downloaded = download_with_wget(base_url, local_dir, logger)

    if not downloaded:
        logger.info("Fetching index and downloading parquet files")
        parquet_urls = list_parquet_urls(base_url)
        if not parquet_urls:
            raise RuntimeError(f"No parquet files found at {base_url}")
        for url in parquet_urls:
            filename = pathlib.Path(urlparse(url).path).name
            download_file(url, local_dir / filename, force=args.force)

    marker.write_text("Downloaded from %s\n" % base_url, encoding="utf-8")
    logger.info("Wrote marker %s", marker)


if __name__ == "__main__":
    main()
