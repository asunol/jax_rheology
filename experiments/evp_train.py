#!/usr/bin/env python
"""Train an elastoviscoplastic channel closure from a YAML config.

The config selects the drive set and horizon, and the network seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._dispatch import run_module

if __name__ == "__main__":
    rc = run_module("visco_opt_tbnn_evp_run")
    raise SystemExit(0 if rc is None else rc)
