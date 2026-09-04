#!/usr/bin/env python
"""Train against PIV-like observations from a YAML config.

The config sets the interrogation-window size and the noise level used for
Supplementary Figs. S4-S6.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._dispatch import run_module

if __name__ == "__main__":
    rc = run_module("run_tbnn_debug_constriction_cluster_new_piv")
    raise SystemExit(0 if rc is None else rc)
