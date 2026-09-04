"""Rheology-side state container.

The original ``jax_ib.base.particle_class.All_Variables`` is a 6-field
record (``particles, velocity, pressure, Drag, Step_count, MD_var``).
Log-conformation (and any other memory model) needs to attach
**generic memory fields** (the conformation tensor for Oldroyd-B,
the Q tensor for active polymers, ...) to the per-step state without
changing the upstream jax_ib library.

This module therefore *copies* the ``All_Variables`` definition into
:mod:`jax_rheology` and extends it with two additional slots:

* ``memory_fields`` -- a flat tuple of :class:`GridVariable`\\ s
  registered as a pytree child so AD / ``jax.lax.scan`` flow through
  it.
* ``memory_layout`` -- a static :class:`StateSpec` descriptor carried
  as aux_data so JIT treats it as static and never differentiates
  through it.

Both default to ``None``, so call sites that construct
``All_Variables(particles, v, p, Drag, Step_count, MD_var)``
positionally remain byte-for-byte unchanged.

The other classes (``particle``, ``particle_lista``, ``Grid1d``) and
``PyTree`` alias are re-exported from :mod:`jax_ib.base.particle_class`
unchanged, so call sites that say ``from jax_rheology import
particle_class as pc`` and then use ``pc.particle`` or ``pc.Grid1d``
get exactly the same types as before.

Every file in :mod:`jax_rheology` that previously did
``from jax_ib.base import particle_class`` now imports
``jax_rheology.particle_class`` instead, which keeps the carry-type
consistent across the scan loop. The jax_ib internals
(``jax_ib/jax_ib/base/{pressure,boundaries,particle_motion,time_stepping}.py``)
are deliberately not modified -- they still operate on the original
6-field ``jax_ib.base.particle_class.All_Variables`` and they are not
exercised on the rheology code path used by ``forward_fluid_simulation``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence, Tuple

from jax.tree_util import register_pytree_node_class

from jax_ib.base import grids
from jax_ib.base import particle_class as _ib_pc

# Re-exports so ``from jax_rheology.particle_class import ...`` and
# ``from jax_rheology import particle_class as pc; pc.<sym>`` give back
# the same types they did from the jax_ib module.
particle = _ib_pc.particle
particle_lista = _ib_pc.particle_lista
Grid1d = _ib_pc.Grid1d
PyTree = _ib_pc.PyTree


@register_pytree_node_class
@dataclasses.dataclass
class All_Variables:
    """Per-step container threaded through ``jax.lax.scan``.

    The first six fields match the jax_ib original byte-for-byte
    (same names, same order, same semantics, same pytree-flatten
    layout for those entries). The two trailing fields are new:

    Attributes:
        particles: IB geometry pytree(s).
        velocity: Face-staggered ``GridVariableVector``.
        pressure: Cell-centered ``GridVariable``.
        Drag: Per-particle drag accumulators.
        Step_count: Integer timestep counter.
        MD_var: Generic slot for MD / coupling state.
        memory_fields: Optional flat tuple of cell-centered
            :class:`GridVariable`\\ s carrying constitutive-model
            memory (the conformation tensor's three SPD components
            for Oldroyd-B, the Q tensor's two components for active
            polymers, etc.). Pytree child; flows through AD and scan.
            ``None`` for every non-viscoelastic stepper path.
        memory_layout: Optional static descriptor (a
            :class:`jax_rheology.constitutive_registry.StateSpec`)
            recording per-field names, components, manifold tags, and
            offsets. Carried as aux_data -- never differentiated
            through. The container itself never inspects it.
    """

    particles: Sequence[particle,]
    velocity: grids.GridVariableVector
    pressure: grids.GridVariable
    Drag: Sequence[Any]
    Step_count: int
    MD_var: Any
    memory_fields: Optional[Tuple[grids.GridVariable, ...]] = None
    memory_layout: Optional[Any] = None

    def tree_flatten(self):
        children = (self.particles, self.velocity, self.pressure,
                    self.Drag, self.Step_count, self.MD_var,
                    self.memory_fields,)
        aux_data = (self.memory_layout,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        memory_layout = None if aux_data is None else aux_data[0]
        return cls(*children, memory_layout=memory_layout)


__all__ = (
    'All_Variables',
    'particle',
    'particle_lista',
    'Grid1d',
    'PyTree',
)
