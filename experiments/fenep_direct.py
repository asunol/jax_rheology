#!/usr/bin/env python
"""Calibrate the four FENE-P parameters directly against contraction data, from a YAML config.

A single L-BFGS-B fit with no neural closure.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._dispatch import run_module

if __name__ == "__main__":
    rc = run_module("visco_opt_fenep_direct_contraction_run")
    raise SystemExit(0 if rc is None else rc)
