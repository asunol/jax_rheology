# jax_rheology/cavity_geometry.py
"""Geometry for the square lid-driven cavity.

Builds the cavity **grid**, the **zero permeability** field (there is no
immersed solid -- every wall is grid-aligned with the domain boundary),
the **lid-velocity boundary conditions** (constant "singular" lid and the
spatially-varying "regularized R1" lid profile), and the rest initial
states (``u = 0``, ``A = I``). The cavity analogue of the contraction
geometry, but **without any Brinkman penalty / SDF / logistic-smoother
machinery**.

Domain
------
``Omega = [0, L] x [0, L]`` (origin at the lower-left corner). No-slip on all
four walls; the top wall (``y = L``, the *lid*) translates tangentially in
``+x`` at peak speed ``U``. ``L = 1``, ``U = 1`` by convention.

Two lids:
  * **Singular / constant lid** ``u(x) = U`` -- a scalar Dirichlet value on
    the ``vx`` top face. This is the Newtonian benchmark forcing.
    The scalar value is imposed on *every* top-face
    node including the two corners, i.e. the **leaky-lid** convention
    (the corner node takes ``U``); published Newtonian cavity tables are
    leaky-lid singular, and the ~1% near-corner discrepancy versus
    non-leaky codes traces to this choice.
  * **Regularized R1 lid** ``u(x) = 16 U (x/L)^2 (1 - x/L)^2`` -- vanishes
    with zero slope at both top corners; this is the viscoelastic-cavity
    forcing. Its average over the lid is
    ``0.533 U`` (a cheap correctness check). The driver imposes the
    spatially-varying wall value each step via
    :class:`CavityLidBoundaryConditions`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Grid + (trivial) permeability
# ---------------------------------------------------------------------------
def make_cavity_grid(L: float = 1.0, cells_per_side: int = 128):
    """Build a uniform square :class:`grids.Grid` for ``Omega = [0, L]^2``.

    ``dx = dy = L / cells_per_side``. Mirrors
    :func:`contraction_geometry.make_contraction_grid` but square with the
    origin at the lower-left corner. An odd ``cells_per_side`` places grid
    lines exactly on the geometric centerlines ``x = L/2`` / ``y = L/2``
    (handy for point-matching published Newtonian centerline
    profiles); an even count straddles them. Either is
    accepted.
    """
    from jax_ib.base import grids  # local import: keep this module light

    n = int(cells_per_side)
    domain = ((0.0, float(L)), (0.0, float(L)))
    return grids.Grid((n, n), domain=domain)


def zero_permeability(grid) -> jnp.ndarray:
    """Cell-centered permeability field for the cavity: zero everywhere.

    The cavity has **no immersed solid** -- every wall is a grid-aligned
    domain boundary handled by the velocity BC -- so the Brinkman penalty
    term must contribute nothing. Returning ``jnp.zeros(grid.shape)``
    keeps the stepper signature (``permeability=perm_f``) unchanged while
    disabling the penalty.
    """
    return jnp.zeros(grid.shape)


# ---------------------------------------------------------------------------
# Lid velocity boundary conditions
# ---------------------------------------------------------------------------
def cavity_velocity_bc_singular(U_lid: float):
    """All-Dirichlet velocity BCs with a **constant** lid ``u = U_lid``.

    ``vx``: left/right walls ``0``, bottom wall ``0``, top lid ``U_lid``
    (leaky-lid -- the scalar value covers the corners too).
    ``vy``: zero on all four walls.

    Both components are two-sided Dirichlet on both axes, so the base
    :class:`boundaries.ConstantBoundaryConditions` (not the contraction's
    mixed-face subclass) is the correct object; the all-Dirichlet layout
    has no mixed per-face axis. Uses the module-level stable
    ``_zero_bc_fn`` for ``boundary_fn`` so the BC round-trips cleanly
    through ``jax.lax.scan``.

    Returns ``(vx_bc, vy_bc)``.
    """
    from jax_ib.base import boundaries
    from jax_rheology.core import boundaries as bnew

    D = boundaries.BCType.DIRICHLET
    types = ((D, D), (D, D))
    # values[axis] = (low_face, high_face). Axis 0 = x (left, right);
    # axis 1 = y (bottom, top). The lid is the y-high face of vx.
    vx_vals = ((0.0, 0.0), (0.0, float(U_lid)))
    vy_vals = ((0.0, 0.0), (0.0, 0.0))
    vx_bc = boundaries.ConstantBoundaryConditions(
        0.0, vx_vals, types, bnew._zero_bc_fn)
    vy_bc = boundaries.ConstantBoundaryConditions(
        0.0, vy_vals, types, bnew._zero_bc_fn)
    return vx_bc, vy_bc


def cavity_velocity_bc_regularized(grid, U_lid: float, lid_scale: float = 1.0):
    """All-Dirichlet velocity BCs with the **regularized R1** lid profile.

    ``vx``: left/right/bottom walls ``0``; top lid carries the R1 profile
    ``16 U (x/L)^2 (1 - x/L)^2`` (scaled by ``lid_scale`` for the startup
    ramp) via :class:`jax_rheology.core.boundaries.CavityLidBoundaryConditions` -- the
    scalar top value in ``bc_values`` is kept ``0`` and the profile rides
    on the extra ``lid_profile`` pytree child (see that class).
    ``vy``: zero on all four walls (a plain ``ConstantBoundaryConditions``).

    ``lid_scale`` multiplies the whole profile; the driver passes the
    ramp factor ``r(t)  in  [0, 1]`` here each step.

    Returns ``(vx_bc, vy_bc)``.
    """
    from jax_ib.base import boundaries
    from jax_rheology.core import boundaries as bnew

    D = boundaries.BCType.DIRICHLET
    types = ((D, D), (D, D))
    profile = float(lid_scale) * cavity_lid_profile(grid, U_lid)
    # Top-y scalar value stays 0.0: CavityLidBoundaryConditions._pad adds
    # 2*lid_profile to the (homogeneous) top ghost row.
    vx_vals = ((0.0, 0.0), (0.0, 0.0))
    vy_vals = ((0.0, 0.0), (0.0, 0.0))
    vx_bc = bnew.CavityLidBoundaryConditions(
        0.0, vx_vals, types, bnew._zero_bc_fn, profile)
    vy_bc = boundaries.ConstantBoundaryConditions(
        0.0, vy_vals, types, bnew._zero_bc_fn)
    return vx_bc, vy_bc


def regularized_lid_profile(x: jnp.ndarray, U_lid: float,
                            L: float = 1.0) -> jnp.ndarray:
    """R1 regularized lid ``16 U (x/L)^2 (1 - x/L)^2`` evaluated at ``x``.

    Vanishes with zero one-sided slope at ``x = 0`` and ``x = L``; peak
    ``U`` at ``x = L/2``; average over ``[0, L]`` is ``0.533 U``
    (``int0^1 16 s^2(1-s)^2 ds = 16/30``). Community-standard viscoelastic
    cavity forcing (regularized lid).
    """
    s = x / float(L)
    return 16.0 * float(U_lid) * s**2 * (1.0 - s)**2


def cavity_lid_profile(grid, U_lid: float) -> jnp.ndarray:
    """1-D R1 lid-velocity array on the ``vx`` top-face x-coordinates.

    Returns the profile ``16 U (x/L)^2 (1 - x/L)^2`` sampled at the
    x-locations of the ``vx`` faces (offset ``grid.cell_faces[0]``) along
    the top wall -- the array the driver pushes onto the top-face ``vx``
    Dirichlet value each step via :class:`CavityLidBoundaryConditions`.
    """
    L = grid.domain[0][1] - grid.domain[0][0]
    X, _ = grid.mesh(grid.cell_faces[0])
    x_top = X[:, -1]  # x-coordinates along the top row of vx faces
    return regularized_lid_profile(x_top, U_lid, L)


# ---------------------------------------------------------------------------
# Rest initial states (u = 0; A = I for the viscoelastic case)
# ---------------------------------------------------------------------------
def _zero_velocity(grid, vx_bc, vy_bc):
    """Build the rest velocity ``(vx, vy) = 0`` with the given BCs."""
    from jax_ib.base import grids as ib_grids

    zero_fn = lambda x, y: jnp.zeros_like(x + y)
    vx_arr = grid.eval_on_mesh(zero_fn, grid.cell_faces[0])
    vy_arr = grid.eval_on_mesh(zero_fn, grid.cell_faces[1])
    return (ib_grids.GridVariable(vx_arr, vx_bc),
            ib_grids.GridVariable(vy_arr, vy_bc))


def build_cavity_newtonian_state(grid, U_lid: float):
    """Rest state (``u = 0``) + zero perm + ``bc_spec`` for the Newtonian cavity.

    Newtonian: no conformation, so ``memory_fields = None``. Velocity
    starts at rest with the **singular constant-``U``** lid BC (the driver
    ramps the value from rest; for creeping/steady Newtonian the steady
    state is start-up-independent). Pressure starts at zero
    with the cavity all-Neumann pressure BC so the scan carry matches what
    the singular-Poisson projection produces each step.

    Returns ``(initial_state, perm_f, bc_spec)``.
    """
    from jax_ib.base import grids as ib_grids
    from jax_rheology.core import boundaries as bnew
    from jax_rheology.core import state as pc

    bc_spec = bnew.BCSpec.cavity(grid.ndim)

    vx_bc, vy_bc = cavity_velocity_bc_singular(U_lid)
    velocity = _zero_velocity(grid, vx_bc, vy_bc)

    pressure_var = ib_grids.GridVariable(
        ib_grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid),
        bc_spec.pressure_bc(grid))

    perm_f = zero_permeability(grid)

    initial_state = pc.All_Variables(
        particles=None,
        velocity=velocity,
        pressure=pressure_var,
        Drag=[0],
        Step_count=jnp.asarray(0),
        MD_var=[0],
        memory_fields=None,
        memory_layout=None,
    )
    return initial_state, perm_f, bc_spec


def build_cavity_viscoelastic_state(grid, U_lid: float, model):
    """Rest velocity + ``A = I`` conformation for the viscoelastic path.

    Same momentum/pressure setup as
    :func:`build_cavity_newtonian_state`, plus the constitutive
    ``memory_fields`` seeded at equilibrium ``A = I`` with the **cavity
    conformation BC** (linear-Psi extrapolation on all four walls -- every
    wall is a no-inflow wall). All memory fields are cell-centered, so no
    staggered shape bookkeeping is needed.

    The lid BC here is still the singular constant-``U`` object; the
    regularized R1 lid used by the viscoelastic benchmarks is imposed by
    the driver each step, not baked into the rest state.

    Returns ``(initial_state, perm_f, bc_spec)``.
    """
    from jax_ib.base import grids as ib_grids
    from jax_rheology.core import boundaries as bnew
    from jax_rheology.core import flow_conditions as fc
    from jax_rheology.core import state as pc

    bc_spec = bnew.BCSpec.cavity(grid.ndim)

    vx_bc, vy_bc = cavity_velocity_bc_singular(U_lid)
    velocity = _zero_velocity(grid, vx_bc, vy_bc)

    pressure_var = ib_grids.GridVariable(
        ib_grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid),
        bc_spec.pressure_bc(grid))

    conformation_bc = bc_spec.conformation_bc(grid)
    memory_fields = fc.get_initial_memory(
        grid, model.state_spec, bc=conformation_bc)

    perm_f = zero_permeability(grid)

    initial_state = pc.All_Variables(
        particles=None,
        velocity=velocity,
        pressure=pressure_var,
        Drag=[0],
        Step_count=jnp.asarray(0),
        MD_var=[0],
        memory_fields=memory_fields,
        memory_layout=model.state_spec,
    )
    return initial_state, perm_f, bc_spec
