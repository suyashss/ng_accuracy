"""Logging utilities."""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str, level: int = logging.INFO, verbose: bool = False) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s" if verbose else "%(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level if not verbose else logging.DEBUG)
    logger.propagate = False
    return logger


def set_verbosity(logger: logging.Logger, verbose: Optional[bool]) -> None:
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


__all__ = ["get_logger", "set_verbosity"]
