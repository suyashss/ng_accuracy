#!/usr/bin/env python
"""Assemble a lightweight full pipeline report."""

from __future__ import annotations

import argparse
import json
import pathlib

from src.ng_accuracy.config import load_config
from src.ng_accuracy.logging_utils import configure_logging
from src.ng_accuracy.reports_full import write_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging()
    metrics_path = pathlib.Path(cfg["paths"]["reports_dir"]) / "full_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    report_path = pathlib.Path(cfg["paths"]["reports_dir"]) / "FULL_REPORT.md"
    mapping_report = pathlib.Path(cfg["paths"]["reports_dir"]) / "mapping_summary_full.md"
    write_summary(report_path, cfg, mapping_report, metrics)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
