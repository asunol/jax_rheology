"""Duplicated Flax TBNN source vs the library copy.

Compares named model-definition blocks after whitespace normalization.
No JAX. Login-safe.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

DST = Path(__file__).resolve().parents[1]

# Regions listed in verify/expected.json fork_parity.expected.regions
DEFAULT_REGIONS = (
    "_logit",
    "_softplus_inv",
    "_default_mu_centers",
    "BoundedSlopeViscosity",
    "build_tbnn_bounded_model",
)

FORK = DST / "tbnn_model_selection" / "model_selection_tbnn.py"
LIBRARY = DST / "jax_rheology" / "models" / "tbnn_instantaneous.py"


def _defs(path: Path) -> dict[str, str]:
    src = path.read_text()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.get_source_segment(src, node) or ""
    return out


def normalize(text: str) -> str:
    """Whitespace-normalized source (comments kept; layout collapsed)."""
    return re.sub(r"\s+", " ", text).strip()


def compare_regions(regions=DEFAULT_REGIONS) -> dict:
    fork_defs = _defs(FORK)
    lib_defs = _defs(LIBRARY)
    rows = []
    all_ok = True
    for name in regions:
        fa = fork_defs.get(name)
        lb = lib_defs.get(name)
        if fa is None or lb is None:
            rows.append({
                "name": name,
                "identical": False,
                "reason": f"missing fork={fa is not None} library={lb is not None}",
            })
            all_ok = False
            continue
        ok = normalize(fa) == normalize(lb)
        rows.append({
            "name": name,
            "identical": ok,
            "fork_chars": len(fa),
            "library_chars": len(lb),
        })
        if not ok:
            all_ok = False
    return {
        "identical": all_ok,
        "n_regions": len(regions),
        "n_match": sum(1 for r in rows if r["identical"]),
        "regions": rows,
        "fork": str(FORK.relative_to(DST)),
        "library": str(LIBRARY.relative_to(DST)),
    }


if __name__ == "__main__":
    import json
    rec = compare_regions()
    print("ORACLE_JSON:" + json.dumps({
        "identical": rec["identical"],
        "n_match": rec["n_match"],
        "n_regions": rec["n_regions"],
    }))
    raise SystemExit(0 if rec["identical"] else 2)
