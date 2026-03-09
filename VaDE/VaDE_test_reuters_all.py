#!/usr/bin/env python3
"""Backward-compatible wrapper for Reuters-all VaDE evaluation."""

from __future__ import annotations

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vade.infer import main


if __name__ == "__main__":
    sys.argv.insert(1, "reuters_all")
    main()
