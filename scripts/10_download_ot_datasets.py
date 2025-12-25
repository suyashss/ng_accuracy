#!/usr/bin/env python
"""Download Open Targets datasets for the full pipeline."""

from __future__ import annotations

import argparse
import json
import pathlib

from src.ng_accuracy.config import load_config
from src.ng_accuracy.logging_utils import configure_logging
from src.ng_accuracy.ot_download import download_many, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging()
    raw_dir = pathlib.Path(config["paths"]["raw_dir"])
    datasets = config["opentargets"]["datasets"]
    base_url = config["opentargets"]["base_url"]
    release = config["release"]
    results = download_many(base_url, release, datasets, raw_dir, max_workers=config["opentargets"]["download"].get("max_workers", 4), force=args.force)
    manifest_path = pathlib.Path(config["paths"]["reports_dir"]) / "download_manifest.json"
    write_manifest(manifest_path, raw_dir, release, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
