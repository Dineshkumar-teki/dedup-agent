"""Structured logging setup for CLI and Streamlit."""

from __future__ import annotations

import logging
import sys

from common.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    root = logging.getLogger()
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())
