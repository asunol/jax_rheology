"""Typed model constructors. Build the existing params dicts / tuples.

No JAX import. These add a named surface, not a code path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class GNFModel:
    """Named GNF (Newtonian / power-law / Carreau-Yasuda) spec."""

    name: str
    params: Tuple[float, ...]
    kind: str = "gnf"


@dataclass(frozen=True)
class MemoryModel:
    """Named memory-model spec (Giesekus / FENE-P / Oldroyd-B)."""

    name: str
    params: Mapping[str, Any]
    kind: str = "memory"


def newtonian(viscosity: float) -> GNFModel:
    """Constant-viscosity Newtonian model."""
    return GNFModel("newtonian", (float(viscosity),))


def power_law(K: float, n: float) -> GNFModel:
    """Power-law viscosity ``eta = K gammadot^{n-1}``."""
    return GNFModel("power_law", (float(K), float(n)))


def carreau_yasuda(*, eta_inf: float, eta_0: float, lam: float,
                   n: float, a: float) -> GNFModel:
    """Carreau-Yasuda shear-thinning viscosity."""
    return GNFModel(
        "carreau_yasuda",
        (float(eta_inf), float(eta_0), float(lam), float(n), float(a)),
    )


def giesekus(*, Gp: float, lam: float, nu_s: float, alpha: float) -> MemoryModel:
    """Giesekus memory model (``Gp``, ``lam``, ``nu_s``, ``alpha``)."""
    return MemoryModel(
        "giesekus",
        {"Gp": float(Gp), "lam": float(lam), "nu_s": float(nu_s),
         "alpha": float(alpha)},
    )


def fene_p(*, Gp: float, lam: float, nu_s: float, Lsq: float) -> MemoryModel:
    """FENE-P memory model (``Gp``, ``lam``, ``nu_s``, ``Lsq``)."""
    return MemoryModel(
        "fene_p",
        {"Gp": float(Gp), "lam": float(lam), "nu_s": float(nu_s),
         "Lsq": float(Lsq)},
    )
