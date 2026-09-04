#!/usr/bin/env python
"""Run elastoviscoplastic forward diagnostics, or a yield sweep, from a YAML config."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._dispatch import apply_precision, dispatch, peek_config

_WRAPPER = {
    "diag": "evp_forward_diag",
    "sweep": "fwd_yield_sweep",
}


def main():
    data = peek_config()
    if "--help" in sys.argv or "-h" in sys.argv:
        print("usage: experiments/evp_forward.py --config experiments/configs/evp_forward_diag.yaml")
        print("wrapper (yaml): diag | sweep")
        if not data:
            return
    apply_precision(data)
    dispatch(_WRAPPER, "wrapper", data.get("wrapper", "diag"))


if __name__ == "__main__":
    main()
