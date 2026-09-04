#!/usr/bin/env python
"""Evaluate trained elastoviscoplastic closures from a YAML config.

The config selects either the training-protocol arms or the seed ensemble.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._dispatch import apply_precision, dispatch, peek_config

_WRAPPER = {
    "arms": "evp_fix_eval",
    "seeds": "evp_fix_seed_eval",
}


def main():
    data = peek_config()
    apply_precision(data)
    dispatch(_WRAPPER, "wrapper", data.get("wrapper", "arms"))


if __name__ == "__main__":
    main()
