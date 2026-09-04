#!/usr/bin/env python
"""Run the lid-driven cavity ground-truth solve from a YAML config."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._dispatch import run_module

if __name__ == "__main__":
    rc = run_module("cavity_transfer_truth_ladder")
    raise SystemExit(0 if rc is None else rc)
