#!/usr/bin/env python3
"""Backward-compatible wrapper for the new VaDE training entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vade.train import main


if __name__ == "__main__":
    main()
