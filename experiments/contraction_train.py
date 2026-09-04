#!/usr/bin/env python
"""Train a memory closure on contraction flow from a YAML config.

The published settings for each condition live in the configs; for example
``fenep_single_rate_u05.yaml`` holds the single-rate FENE-P run.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._dispatch import run_module

if __name__ == "__main__":
    rc = run_module("visco_opt_tbnn_contraction_run")
    raise SystemExit(0 if rc is None else rc)
