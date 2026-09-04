"""Repository root, vendored trees, and optional run archives.

``REPO_ROOT`` is derived from this file. ``diff_rheo`` resolves to the
vendored copy unless ``TBNN_DIFF_RHEO`` points elsewhere.

``FROZEN_MEM`` and ``FROZEN_INST`` name archives of the original runs. They
are not part of the release and have no default: set ``TBNN_FROZEN_MEM`` or
``TBNN_FROZEN_INST`` if you have them. Code that needs one should call
``require_frozen_mem`` / ``require_frozen_inst``, which raise a message
naming the variable rather than failing on a missing path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
def _archive(var: str) -> Path | None:
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else None


FROZEN_MEM = _archive("TBNN_FROZEN_MEM")
FROZEN_INST = _archive("TBNN_FROZEN_INST")


def _require(value: Path | None, var: str, what: str) -> Path:
    if value is None:
        raise RuntimeError(
            f"this step reads {what}, which is not part of the release. "
            f"Set {var} to a local copy, or use the published data bundle "
            f"(DATA_ROOT) where the step supports it."
        )
    return value


def require_frozen_mem() -> Path:
    return _require(FROZEN_MEM, "TBNN_FROZEN_MEM",
                    "the archive of the original viscoelastic runs")


def require_frozen_inst() -> Path:
    return _require(FROZEN_INST, "TBNN_FROZEN_INST",
                    "the archive of the original instantaneous runs")
DIFF_RHEO = Path(os.environ.get(
    "TBNN_DIFF_RHEO",
    str(REPO_ROOT / "diff_rheo"),
))


def insert_diff_rheo() -> Path:
    """Put ``DIFF_RHEO/src`` at ``sys.path[0]`` so the vendored package wins.

    The ``diff_rheo`` conda env still has an editable install of the paper-era
    checkout; battery scripts must call this *before* ``import diff_rheo``.
    ``TBNN_DIFF_RHEO`` still overrides the tree.
    """
    src = str((DIFF_RHEO / "src").resolve())
    while src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)
    return DIFF_RHEO


def bootstrap() -> Path:
    """chdir to REPO_ROOT and apply the historical sys.path order.

    insert '.', insert REPO_ROOT/jax_ib, insert REPO_ROOT/jax-cfd so the
    vendored trees beat conda site-packages (append left ``jax_ib`` on the
    bacteria install). Also insert DIFF_RHEO/src.
    """
    os.chdir(REPO_ROOT)
    # jax_ib and jax-cfd are inserted rather than appended so the vendored
    # trees take precedence over any copy installed in site-packages.
    sys.path.insert(0, ".")
    sys.path.insert(0, str(REPO_ROOT / "campaigns"))
    sys.path.insert(0, str(REPO_ROOT / "campaigns" / "battery"))
    sys.path.insert(0, str(REPO_ROOT / "jax_ib"))
    sys.path.insert(0, str(REPO_ROOT / "jax-cfd"))
    insert_diff_rheo()
    return REPO_ROOT
