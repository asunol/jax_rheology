#!/usr/bin/env python
"""Re-run a trained closure forward from a YAML config, for the contraction or the channel."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._dispatch import apply_precision, dispatch, peek_config

_FAMILY = {
    "giesekus": "regen_contraction",
    "fene": "regen_contraction",
    "evp": "regen_evp",
}


def main():
    data = peek_config()
    apply_precision(data)
    family = data.get("family")
    if family is None and ("--help" in sys.argv or "-h" in sys.argv):
        print("usage: experiments/regen.py --config experiments/configs/regen_giesekus.yaml")
        print("family (yaml): giesekus | fene | evp")
        family = "giesekus"
    dispatch(_FAMILY, "family", family)


if __name__ == "__main__":
    main()
