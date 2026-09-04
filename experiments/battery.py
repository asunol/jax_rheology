#!/usr/bin/env python
"""Bayesian-information-criterion model-selection battery (prepare / fit / merge-target / list).

Runs in the rheometry environment, which provides diffrax; see
``environment_diff_rheo.yml``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os

from experiments._dispatch import apply_precision, inject_positionals, peek_config, run_module

# The battery must see a CPU platform before JAX loads. Double precision is
# pinned from the config in apply_precision, and must be true for this campaign.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_USAGE = """usage: battery.py --config experiments/configs/battery_list.yaml

Model-selection battery over the rheometry model library. The mode (prepare,
fit, merge-target, or list) and every other option come from the config file;
the prepare / fit / merge configs sit beside battery_list.yaml.

Requires the rheometry environment (diffrax): see environment_diff_rheo.yml.
Full argument help is available once that environment is active."""


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        # Print usage without importing diff_rheo, so --help works in either
        # environment rather than failing on a missing dependency.
        print(_USAGE)
        return 0
    data = peek_config()
    apply_precision(data)
    if data.get("mode"):
        inject_positionals(data["mode"])
    try:
        rc = run_module("tbnn_bic_final_battery")
    except ImportError as exc:
        raise SystemExit(
            f"battery.py needs the rheometry environment (missing: {exc.name}).\n"
            "Create it from environment_diff_rheo.yml and run this entrypoint "
            "with that interpreter."
        )
    return 0 if rc is None else rc


if __name__ == "__main__":
    raise SystemExit(main())
