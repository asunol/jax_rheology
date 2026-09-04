"""Constitutive-model registry.

The abstraction boundary between the model-agnostic stepper /
forward-simulation code and any specific constitutive model. Nothing
here is Oldroyd-B-specific. Each model (Oldroyd-B log-conformation,
Giesekus, FENE-P, a learned TBNN closure, ...) registers a single
:class:`ConstitutiveModel` record at module-import time.

The four-part record is:

* ``state_spec`` -- declarative layout of the model's memory fields.
* ``evolution_fn`` -- pure function that advances the memory fields
  by one time step given the current velocity and parameters.
* ``stress_readout_fn`` -- pure function that returns the
  cell-centered polymer-stress triple ``(tau_xx, tau_xy, tau_yy)``
  from the current memory fields, velocity, and parameters.
* ``coupling_mode`` -- how the stress couples back to the momentum
  equation: ``'explicit_force'`` (default) or ``'implicit_block'``
  (requires ``polymer_linearization_fn``).

The evolution function signature is ``(memory_fields, velocity,
params, dt) -> memory_fields_new`` -- four positional arguments, no
PRNG key. An RNG key is not part of the type: stochastic models would
need a registry change, and none of the current models need one.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Literal, Optional, Tuple


ManifoldTag = Literal['spd', 'symmetric', 'traceless_symmetric',
                      'unconstrained', 'scalar']


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    """One memory field of a constitutive model.

    Attributes:
        name: Human-readable field label (e.g. ``'A'`` for the
            conformation tensor, ``'Q'`` for the order parameter,
            ``'Phi'`` for a scalar microstructure variable).
        components: Ordered tuple of component labels. For a 2D
            symmetric tensor this is ``('xx', 'xy', 'yy')``; for a
            2D traceless-symmetric tensor it is ``('xx', 'xy')``;
            for a scalar it is ``('',)``.
        manifold: Manifold tag used only to pick a rest state in
            :func:`get_initial_memory`. **Descriptive only** -- the
            generic layer does not enforce SPD-ness or trace-freedom;
            that is the ``evolution_fn``'s responsibility.
        offset: Grid offset for every :class:`GridVariable` of this
            field. ``(0.5, 0.5)`` is cell center.
    """

    name: str
    components: Tuple[str, ...]
    manifold: ManifoldTag
    offset: Tuple[float, ...]


StateSpec = Tuple[FieldSpec, ...]


EvolutionFn = Callable[..., Tuple[Any, ...]]
"""``(memory_fields, velocity, params, dt) -> memory_fields_new``.

Four positional arguments, no PRNG key. Pure: no side effects, no
in-place mutation. Returns a tuple of :class:`GridVariable`\\ s with
the same layout as the input ``memory_fields``.
"""

StressReadoutFn = Callable[..., Tuple[Any, Any, Any]]
"""``(memory_fields, velocity, params) -> (tau_xx, tau_xy, tau_yy)``.

Returns three cell-centered :class:`GridArray`\\ s -- the polymer
stress at offset ``(0.5, 0.5)``. 2D-specific by design; the same
shape that ``tensor_divergence`` consumes.
"""

PolymerLinearizationFn = Callable[..., Any]
"""``(memory_fields_star, candidate_u, params, dt) -> GridArrayVector``.

Optional; required only when ``coupling_mode == 'implicit_block'``.
For Oldroyd-B the closed-form linearization is
``L_p[A] u = div( G_p * dt * (A . grad u + (grad u)^T . A) )``.
Other models that opt into implicit-block coupling register their own.
"""


@dataclasses.dataclass(frozen=True)
class ConstitutiveModel:
    """Four-part record describing one constitutive model.

    Implicit-block coupling (``polymer_linearization_fn``) is opt-in
    per model; nothing in the registry assumes a particular constitutive
    law.
    """

    name: str
    state_spec: StateSpec
    evolution_fn: EvolutionFn
    stress_readout_fn: StressReadoutFn
    coupling_mode: Literal['explicit_force', 'implicit_block'] = 'explicit_force'
    polymer_linearization_fn: Optional[PolymerLinearizationFn] = None


MODEL_REGISTRY: Dict[str, ConstitutiveModel] = {}


def register(model: ConstitutiveModel) -> ConstitutiveModel:
    """Register ``model`` in :data:`MODEL_REGISTRY`.

    Raises ``ValueError`` if a model of the same name is already
    registered, or if ``coupling_mode == 'implicit_block'`` is set
    without a ``polymer_linearization_fn``. Implicit-block coupling is
    strictly opt-in per model because the linearization is
    constitutive-specific.

    Returns the model unchanged so the call can be used at module
    level next to the model definition.
    """
    if model.name in MODEL_REGISTRY:
        raise ValueError(f"Model {model.name!r} is already registered.")
    if (model.coupling_mode == 'implicit_block'
            and model.polymer_linearization_fn is None):
        raise ValueError(
            f"Model {model.name!r} sets coupling_mode='implicit_block' but "
            "does not supply a polymer_linearization_fn; variant b is "
            "strictly opt-in per model."
        )
    MODEL_REGISTRY[model.name] = model
    return model


def get_model(name: str) -> ConstitutiveModel:
    """Look up a registered model by name."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        registered = sorted(MODEL_REGISTRY)
        raise KeyError(
            f"No constitutive model registered under {name!r}. "
            f"Registered models: {registered}"
        ) from exc


def list_models() -> Tuple[str, ...]:
    """Return the names of all currently-registered models."""
    return tuple(sorted(MODEL_REGISTRY))


__all__ = (
    'ManifoldTag',
    'FieldSpec',
    'StateSpec',
    'EvolutionFn',
    'StressReadoutFn',
    'PolymerLinearizationFn',
    'ConstitutiveModel',
    'MODEL_REGISTRY',
    'register',
    'get_model',
    'list_models',
)
