#!/usr/bin/env python
"""Download the configured GENCODE GTF file."""

from __future__ import annotations

import argparse
import pathlib

from ng_accuracy.config import add_config_arg, load_config
from ng_accuracy.io_utils import download_file, ensure_dir
from ng_accuracy.logging_utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--force", action="store_true", help="Redownload even if exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("download_gtf")

    url = cfg["gene_gtf"]["url"]
    out_path = pathlib.Path(cfg["gene_gtf"]["local_path"])
    ensure_dir(out_path.parent)

    download_file(url, out_path, force=args.force)
    logger.info("Downloaded gene annotation to %s", out_path)


if __name__ == "__main__":
    main()
