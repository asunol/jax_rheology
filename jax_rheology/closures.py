"""Closure constructors for ``training.fit``. Specs only; no JAX, no register()."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class TBNN:
    """Memory TBNN closure (width / depth / seed match the contraction driver)."""

    width: int = 32
    depth: int = 2
    seed: int = 0
    bound_c: float = 3.0


@dataclass(frozen=True)
class MixtureOfSigmoids:
    """Instantaneous mixture-of-sigmoids viscosity (unified trainer)."""

    M: int = 6
    hidden: Sequence[int] = field(default_factory=lambda: [48, 48])
    init: str = "soft_newtonian"
