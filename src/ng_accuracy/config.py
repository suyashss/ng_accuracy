"""Configuration helpers for the PILOT pipeline."""

from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = pathlib.Path("configs/pilot.yaml")


def load_config(path: str | pathlib.Path | None = None) -> Dict[str, Any]:
    """Load YAML configuration as a dict."""
    cfg_path = pathlib.Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Add a --config argument with default."""
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config (default configs/pilot.yaml)",
    )


def resolve_release_path(template: str, release: str) -> str:
    """Fill release placeholder in URLs/paths."""
    return template.format(release=release)


__all__ = ["load_config", "add_config_arg", "resolve_release_path", "DEFAULT_CONFIG_PATH"]
