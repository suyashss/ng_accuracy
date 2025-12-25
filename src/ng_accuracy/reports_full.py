"""Reporting utilities for the full pipeline."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from typing import Dict

import pandas as pd


def write_summary(report_path: pathlib.Path, config: dict, mapping_report: pathlib.Path | None, metrics: dict | None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# FULL_REPORT", ""]
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append("## Configuration")
    lines.append("```")
    lines.append(json.dumps(config, indent=2))
    lines.append("```")
    lines.append("")
    if mapping_report and mapping_report.exists():
        lines.append("## Mapping report")
        lines.append(mapping_report.read_text())
    if metrics:
        lines.append("## Metrics")
        lines.append(json.dumps(metrics, indent=2))
    report_path.write_text("\n".join(lines))


__all__ = ["write_summary"]
