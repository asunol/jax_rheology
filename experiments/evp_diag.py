#!/usr/bin/env python
"""Run elastoviscoplastic post-training diagnostics from a YAML config."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._dispatch import apply_precision, dispatch, peek_config

_WRAPPER = {
    "baseline": "evp_baseline_diag",
    "phase0": "evp_baseline_diag",  # older configs still pass this wrapper name
    "flowcurve": "evp_learned_flowcurve",
    "precheck": "ablation_targets_precheck",
}


def main():
    data = peek_config()
    if "--help" in sys.argv or "-h" in sys.argv:
        print("usage: experiments/evp_diag.py --config experiments/configs/evp_baseline_diag.yaml")
        print("wrapper (yaml): baseline | flowcurve | precheck")
        if not data:
            return
    apply_precision(data)
    dispatch(_WRAPPER, "wrapper", data.get("wrapper", "baseline"))


if __name__ == "__main__":
    main()
