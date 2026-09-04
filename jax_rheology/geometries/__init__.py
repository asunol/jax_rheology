"""Named geometry constructors. Specs only; no solver import."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Domain = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True)
class Constriction:
    """Constricted channel (GNF / instantaneous). Defaults match COMMON_CONFIG."""

    nx: int = 256
    ny: int = 128
    domain: Domain = ((0.0, 8.0), (0.0, 4.0))
    pressure_gradient: float = 2.5


@dataclass(frozen=True)
class Contraction:
    """Abrupt 4:1 planar contraction (memory campaigns)."""

    nx: int = 128
    ny: int = 256
    ratio: int = 4
    U: float = 0.5
    ramp_time: float = 0.7


@dataclass(frozen=True)
class Cavity:
    """Lid-driven cavity."""

    nx: int = 128
    ny: int = 128
    de: float = 0.20


@dataclass(frozen=True)
class Obstacle:
    """Single-obstacle channel (GNF)."""

    nx: int = 256
    ny: int = 128
    domain: Domain = ((0.0, 12.0), (0.0, 4.0))
    pressure_gradient: float = 2.5


@dataclass(frozen=True)
class Porous:
    """Porous array (GNF, nu0_split)."""

    nx: int = 128
    ny: int = 128
    pressure_gradient: float = 2.5
