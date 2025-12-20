#!/usr/bin/env python
"""Download Open Targets credible set parquet directory for a release."""

from __future__ import annotations

import argparse
import pathlib
import subprocess

from ng_accuracy.config import add_config_arg, load_config, resolve_release_path
from ng_accuracy.io_utils import ensure_dir
from ng_accuracy.logging_utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--force", action="store_true", help="Redownload even if marker exists")
    parser.add_argument("--wget", action="store_true", help="Use wget for recursive download (default)")
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

    wget_cmd = [
        "wget",
        "--recursive",
        "--no-parent",
        "--accept",
        "*.parquet",
        "--reject",
        "index.html*",
        "--cut-dirs=6",
        "--no-host-directories",
        "--directory-prefix",
        str(local_dir),
        base_url,
    ]

    logger.info("Running wget download... this may take time")
    subprocess.run(wget_cmd, check=True)

    marker.write_text("Downloaded from %s\n" % base_url, encoding="utf-8")
    logger.info("Wrote marker %s", marker)


if __name__ == "__main__":
    main()
