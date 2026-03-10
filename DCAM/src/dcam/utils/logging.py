"""Logging helpers."""

from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler



def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a rich logger with a stable formatter."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler: Optional[logging.Handler] = RichHandler(rich_tracebacks=True, show_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
