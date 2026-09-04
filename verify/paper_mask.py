"""Rebuild the constriction_focused (paper) native-grid loss mask.

Exec the shipped mask-construction block; do not reimplement the geometry
math. constriction_focused feeds solver-native (T, nx, ny) = (T, 256, 128)
with no transpose -- that is the frozen paper path.
"""
from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import numpy as np

DST = Path(__file__).resolve().parents[1]
EXPECTED_PATH = DST / "verify" / "expected.json"

MARKER = "# Create particle mask for constriction geometry"
END = "mask = jnp.broadcast_to(fluid_mask_2d[None, :, :], (T, H, W)).astype(err_all.dtype)"
DOMAIN = ((0.0, 8.0), (0.0, 4.0))
T, NX, NY = 2, 256, 128


def extract_mask_block(src: str) -> str:
    idx = src.find(MARKER)
    if idx < 0:
        raise RuntimeError("mask block marker missing")
    line_start = src.rfind("\n", 0, idx) + 1
    rest = src[line_start:]
    end_idx = rest.find(END)
    if end_idx < 0:
        raise RuntimeError("mask block end missing")
    return textwrap.dedent(rest[: end_idx + len(END)])


def fluid_mask_from_block(block: str, err_all, domain=DOMAIN):
    import jax.numpy as jnp

    ns = {
        "jnp": jnp,
        "err_all": err_all,
        "flow_cond": {"grid": type("G", (), {"domain": domain})()},
    }
    exec(compile(block, "mask_block", "exec"), ns, ns)
    return np.asarray(ns["fluid_mask_2d"])


def constriction_focused_err(dtype="float32"):
    import jax.numpy as jnp

    return jnp.ones((T, NX, NY), dtype=dtype)


def mask_from_trainer_source(src: str):
    """Paper / constriction_focused path: no transpose, (T, 256, 128)."""
    block = extract_mask_block(src)
    return fluid_mask_from_block(block, constriction_focused_err())


def repo_constriction_focused_mask():
    src = (DST / "jax_rheology" / "training" / "instantaneous.py").read_text()
    return mask_from_trainer_source(src)


def mask_sha256(fluid_2d) -> str:
    arr = np.ascontiguousarray(np.asarray(fluid_2d).astype(np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def recorded_sha256() -> str:
    """The published mask hash, read from the recorded check values."""
    import json

    doc = json.loads(EXPECTED_PATH.read_text())
    for check in doc["checks"]:
        if check["id"] == "paper_mask":
            return check["expected"]["sha256"]
    raise KeyError("paper_mask has no recorded sha256 in verify/expected.json")
