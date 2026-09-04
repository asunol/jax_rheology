"""jax_rheology public facade.

Lazy: ``import jax_rheology`` is side-effect free (no JAX, no solver JIT,
no registry registration). Named exports load on attribute access
(``from jax_rheology import Simulation, geometries, models``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Simulation",
    "geometries",
    "models",
    "closures",
    "training",
    "io",
    "solvers",
    "forward",
]


def _ensure_vendored() -> None:
    """Put the vendored jax-cfd and jax_ib trees on ``sys.path``.

    Both are used in place from the repository rather than installed, so an
    editable install of this package still has to find them. Called on first
    attribute access, which keeps ``import jax_rheology`` side-effect free.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for sub in ("jax_ib", "jax-cfd"):
        tree = root / sub
        if tree.is_dir() and str(tree) not in sys.path:
            sys.path.insert(0, str(tree))


def __getattr__(name):
    import importlib
    _ensure_vendored()
    if name == "Simulation":
        from jax_rheology.api import Simulation
        return Simulation
    if name in ("geometries", "models", "closures", "training", "io",
                "solvers", "forward"):
        return importlib.import_module(f"jax_rheology.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
