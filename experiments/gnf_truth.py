#!/usr/bin/env python
"""Generalized-Newtonian ground-truth forward solve: constriction, obstacle, or porous."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._dispatch import apply_precision, dispatch, inject_positionals, peek_config

_GEOMETRY = {
    "constriction": "channel_constriction_flow",
    "obstacle": "channel_obstacle_flow",
    "porous": "porous_media_flow",
}


def main():
    data = peek_config()
    apply_precision(data)
    if data.get("model") is not None:
        params = data.get("params") or []
        inject_positionals(data["model"], *[str(p) for p in params])
    dispatch(_GEOMETRY, "geometry", data.get("geometry", "constriction"))


if __name__ == "__main__":
    main()
