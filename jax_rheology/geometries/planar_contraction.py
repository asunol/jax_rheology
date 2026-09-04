# jax_rheology/contraction_geometry.py
"""Geometry for the abrupt 4:1 planar contraction.

Builds the **Brinkman volume-penalty field** and the rest initial state
(``u = 0``, ``A = I``). Pure geometry / field construction -- the stepper
and operators live elsewhere.

Domain
------
``Omega = [-L_up, L_down] x [-R.H, R.H]`` with contraction ratio ``R = 4``.
Fluid occupies ``|y| <= R.H`` for ``x < 0`` (wide upstream) and
``|y| <= H`` for ``x >= 0`` (narrow downstream). The two solid penalty
blocks are ``S = {(x, y) : x >= 0, H < |y| <= R.H}``. The contraction
plane is at ``x = 0``; the characteristic length ``H`` is the
downstream half-width.

Sign convention (matches ``jax_ib.penalty.util_funs.calc_perm``)
----------------------------------------------------------------
The existing circle path forms ``G = R_circle^2 - r^2``, which is
**positive inside** the solid, then passes ``G`` to the logistic
smoother ``_logistic(G, K, w) = K . expit(G / w)`` -- which therefore
returns ``~= K`` inside the solid and ``~= 0`` in the fluid. Our box
signed-distance field follows the **same convention: ``G > 0`` inside
the solid blocks**, so it feeds the identical logistic smoother and the
identical "sum the per-block contributions" assembly that
``perm_vmap_multiple_particles`` uses.

The box shape lives here rather than in ``jax_ib``; the one-line
logistic smoother is duplicated so ``jax_ib`` stays untouched.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

# Brinkman penalty magnitude inside the solid (the ``K`` / ``Know`` of the
# circle path; ``perm`` saturates to this value where ``G >> w``). Kept as a
# module default so callers only override it deliberately.
DEFAULT_LOGISTIC_SCALE = 1.0e4

Center = Tuple[float, float]
HalfExtents = Tuple[float, float]
Block = Tuple[Center, HalfExtents]


# ---------------------------------------------------------------------------
# Low-level field primitives
# ---------------------------------------------------------------------------
def _logistic(G: jnp.ndarray, K: float, w: float) -> jnp.ndarray:
    """Logistic smoother, identical in form to the ``jax_ib`` circle path.

    ``_logistic(G, K, w) = K . expit(G / w)``. Returns ``~= K`` where
    ``G >> w`` (deep inside the solid) and ``~= 0`` where ``G << -w`` (fluid),
    with a smooth transition of spatial half-width set by ``w`` (the
    0.05->0.95 band spans ``|G| <~ 3w``, i.e. ``~= 6w`` in distance units).
    """
    return K * jax.scipy.special.expit(G / w)


def _safe_sqrt(x_sq: jnp.ndarray) -> jnp.ndarray:
    """Gradient-safe ``sqrt`` (double-where): 0 value *and* 0 grad at 0.

    Mirrors the input-safe / output-masked ``sqrt`` pattern used
    throughout the rheology stack (double-where). The geometry field
    itself is rarely differentiated, but the same array feeds the
    differentiable solver, so the guard is cheap insurance.
    """
    safe = jnp.where(x_sq > 0.0, x_sq, 1.0)
    return jnp.where(x_sq > 0.0, jnp.sqrt(safe), 0.0)


def box_sdf_positive_inside(X: jnp.ndarray, Y: jnp.ndarray,
                            center: Center, half_extents: HalfExtents,
                            ) -> jnp.ndarray:
    """Exact axis-aligned box signed-distance, **positive inside** the box.

    The standard box SDF (negative inside, positive outside) for a box
    centered at ``c`` with half-extents ``h`` is::

        q     = |p - c| - h
        d_out = ||max(q, 0)||2           # exterior distance
        d_in  = min(max(qx, qy), 0)      # interior (negative) distance
        sdf   = d_out + d_in

    We return ``G = -(d_out + d_in)`` so the field is **positive inside**
    the solid, matching the ``calc_perm`` convention (``G = R^2 - r^2 > 0``
    inside the circle). Inside, ``d_out = 0`` and ``G = -d_in >= 0`` is the
    distance to the nearest face; outside, ``G = -d_out <= 0``.
    """
    cx, cy = center
    a, b = half_extents
    qx = jnp.abs(X - cx) - a
    qy = jnp.abs(Y - cy) - b
    d_out = _safe_sqrt(jnp.maximum(qx, 0.0) ** 2 + jnp.maximum(qy, 0.0) ** 2)
    d_in = jnp.minimum(jnp.maximum(qx, qy), 0.0)
    return -(d_out + d_in)


def box_field_min_positive_inside(X: jnp.ndarray, Y: jnp.ndarray,
                                  center: Center, half_extents: HalfExtents,
                                  ) -> jnp.ndarray:
    """Simpler positive-inside box field ``G = min(a-|x-cx|, b-|y-cy|)``.

    Simpler alternative. Shares the exact rectangle zero-contour and
    interior sign with :func:`box_sdf_positive_inside` (and agrees with
    it along the faces and at the re-entrant corner); it differs only in
    the far exterior corner regions, which the logistic flattens to ~0
    anyway. No ``sqrt``, so trivially smooth. Provided for comparison /
    fallback; the exact SDF is the default for cleaner corner contours.
    """
    cx, cy = center
    a, b = half_extents
    return jnp.minimum(a - jnp.abs(X - cx), b - jnp.abs(Y - cy))


# ---------------------------------------------------------------------------
# Contraction geometry description
# ---------------------------------------------------------------------------
def contraction_domain(H: float, L_up: float, L_down: float,
                       contraction_ratio: float = 4.0,
                       ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return the full-domain rectangle ``([-L_up, L_down], [-R.H, R.H])``."""
    R = contraction_ratio
    return ((-float(L_up), float(L_down)), (-R * float(H), R * float(H)))


def contraction_blocks(H: float, L_down: float,
                       contraction_ratio: float = 4.0,
                       ) -> Tuple[Block, Block]:
    """Two solid-block specs covering ``S = {x >= 0, H < |y| <= R.H}``.

    Each block is ``(center, half_extents)``. The downstream extent runs
    from the contraction plane ``x = 0`` to the outlet ``x = L_down``
    (``cx = L_down/2``, ``a = L_down/2``); the wall thickness runs from
    ``|y| = H`` to ``|y| = R.H`` (``cy = (1+R)H/2``, ``b = (R-1)H/2``).
    """
    R = contraction_ratio
    cx = float(L_down) / 2.0
    a = float(L_down) / 2.0
    cy = (1.0 + R) * float(H) / 2.0
    b = (R - 1.0) * float(H) / 2.0
    top = ((cx, cy), (a, b))
    bottom = ((cx, -cy), (a, b))
    return (top, bottom)


def make_contraction_grid(H: float, L_up: float, L_down: float,
                          cells_per_H: float = 16.0,
                          contraction_ratio: float = 4.0):
    """Build a uniform (square-cell) :class:`grids.Grid` for the domain.

    Resolution is set by ``cells_per_H`` (cells per downstream half-width
    ``H``), giving ``dx = dy = H / cells_per_H``. ``Nx``/``Ny`` are rounded
    to the nearest integers that tile the domain.
    """
    from jax_ib.base import grids  # local import: keep field primitives light

    domain = contraction_domain(H, L_up, L_down, contraction_ratio)
    Lx = domain[0][1] - domain[0][0]
    Ly = domain[1][1] - domain[1][0]
    dx = float(H) / float(cells_per_H)
    nx = int(round(Lx / dx))
    ny = int(round(Ly / dx))
    return grids.Grid((nx, ny), domain=domain)


def build_contraction_permeability(grid,
                                   H: float,
                                   L_down: float,
                                   logistic_width: float,
                                   logistic_scale: float = DEFAULT_LOGISTIC_SCALE,
                                   contraction_ratio: float = 4.0,
                                   sdf: str = 'exact',
                                   ) -> jnp.ndarray:
    """Cell-centered Brinkman permeability (penalty) field for the contraction.

    Mirrors :func:`jax_ib.penalty.util_funs.perm_vmap_multiple_particles`:
    evaluate a positive-inside shape field per solid block, smooth each
    with the shared logistic, and **sum** the contributions. Returns the
    cell-centered field (same convention the constriction caller threads
    into the stepper as ``permeability``).

    Args:
      grid: target :class:`grids.Grid` (cell-centered field is built on it).
      H: downstream half-width (characteristic length).
      L_down: downstream length (sets block x-extent).
      logistic_width: ``w`` in ``K.expit(G/w)``; spatial transition band
        ``~= 6w`` (distance units). Smaller ``w`` => sharper (less
        regularized) corner -- a sharper corner lowers the max stable De.
      logistic_scale: ``K`` (penalty magnitude inside the solid).
      contraction_ratio: ``R`` (4 for 4:1).
      sdf: ``'exact'`` (default, :func:`box_sdf_positive_inside`) or
        ``'min'`` (:func:`box_field_min_positive_inside`).
    """
    X, Y = grid.mesh(grid.cell_center)
    shape_fn = (box_sdf_positive_inside if sdf == 'exact'
                else box_field_min_positive_inside)
    perm = jnp.zeros_like(X)
    for center, half_extents in contraction_blocks(H, L_down, contraction_ratio):
        G = shape_fn(X, Y, center, half_extents)
        perm = perm + _logistic(G, logistic_scale, logistic_width)
    return perm


def solid_mask_analytic(grid, H: float,
                        contraction_ratio: float = 4.0) -> jnp.ndarray:
    """Exact boolean solid mask ``S = {x >= 0, H < |y| <= R.H}``.

    Sharp (un-smoothed) reference used only for plotting / verifying the
    penalty field lands on the intended geometry. Not used by the solver.
    """
    X, Y = grid.mesh(grid.cell_center)
    R = contraction_ratio
    return (X >= 0.0) & (jnp.abs(Y) > float(H)) & (jnp.abs(Y) <= R * float(H))


# ---------------------------------------------------------------------------
# Initial state (u = 0, A = I) -- analogue of
# analytic_limits_validation._build_constriction_initial_state
# ---------------------------------------------------------------------------
def build_contraction_initial_state(grid,
                                    H: float,
                                    L_down: float,
                                    logistic_width: float,
                                    logistic_scale: float = DEFAULT_LOGISTIC_SCALE,
                                    contraction_ratio: float = 4.0,
                                    sdf: str = 'exact',
                                    model: Any = None,
                                    wall_conformation_bc: str = 'extrapolation',
                                    ) -> Tuple[Any, jnp.ndarray]:
    """Rest initial state (``u = 0``, ``A = I``) plus the penalty field.

    Structural analogue of
    ``analytic_limits_validation._build_constriction_initial_state`` but
    for the contraction geometry. Uses the generic wall-bounded
    placeholders (``moving_wall`` velocity, no-slip walls, conformation
    extrapolation). For the inflow/outflow ``bc_spec`` used by the
    actual contraction stepper, call
    :func:`build_contraction_newtonian_state` or
    :func:`build_contraction_viscoelastic_state`. This helper does not
    run any solver.

    If ``model`` is ``None`` the conformation ``memory_fields`` are
    skipped and only velocity/pressure are returned in the state.
    """
    # Lazy imports: building just the penalty field must not pull the
    # full solver stack.
    from jax_rheology.core import flow_conditions as fc
    from jax_rheology.core import boundaries as bnew
    from jax_rheology.core import state as pc

    velocity = fc.get_initial_velocity(
        grid, boundary_type='moving_wall', amp_shear=0.0, freq_osc=0.0)
    pressure_var = fc.get_initial_pressure(grid, velocity)

    memory_fields = None
    memory_layout = None
    if model is not None:
        memory_bc = bnew.create_conformation_bc(
            grid,
            boundary_type='moving_wall',
            wall_conformation_bc=wall_conformation_bc,
            wall_axes=(1,),
        )
        memory_fields = fc.get_initial_memory(grid, model.state_spec, bc=memory_bc)
        memory_layout = model.state_spec

    perm_f = build_contraction_permeability(
        grid, H=H, L_down=L_down, logistic_width=logistic_width,
        logistic_scale=logistic_scale, contraction_ratio=contraction_ratio,
        sdf=sdf)

    initial_state = pc.All_Variables(
        particles=None,
        velocity=velocity,
        pressure=pressure_var,
        Drag=[0],
        Step_count=jnp.asarray(0),
        MD_var=[0],
        memory_fields=memory_fields,
        memory_layout=memory_layout,
    )
    return initial_state, perm_f


# ---------------------------------------------------------------------------
# Newtonian contraction velocity/pressure BCs + rest state
# ---------------------------------------------------------------------------
def contraction_velocity_bc(U_inlet: float):
    """Velocity boundary conditions for the 4:1 contraction.

    Both components share the type layout ``x``: (Dirichlet inlet, Neumann
    outlet); ``y``: (Dirichlet wall, Dirichlet wall). Values:

      * ``vx``: inlet ``= U_inlet`` (plug), outlet zero-gradient, walls 0.
      * ``vy``: inlet ``= 0`` (``u_2 = 0``), outlet zero-gradient, walls 0.

    ``U_inlet`` here is a *constant* (impulsive start from rest). The
    cosine ramp ``U(t) = U(1-cos(pit/tau_1))/2`` is applied by the
    contraction driver, which rewrites the inlet BC each step -- the
    BE-IMEX carry has no wall-clock time of its own. For Newtonian
    creeping flow the steady state is independent of the start-up
    transient, so a constant inlet is enough for that path.

    Uses the module-level stable ``_zero_bc_fn`` for ``boundary_fn`` so the
    BC round-trips cleanly through ``jax.lax.scan``.
    """
    from jax_ib.base import boundaries
    from jax_rheology.core import boundaries as bnew

    D = boundaries.BCType.DIRICHLET
    N = boundaries.BCType.NEUMANN
    types = ((D, N), (D, D))
    vx_vals = ((float(U_inlet), 0.0), (0.0, 0.0))
    vy_vals = ((0.0, 0.0), (0.0, 0.0))
    # ContractionBoundaryConditions (not the base ConstantBoundaryConditions)
    # so the mixed Dirichlet-inlet/Neumann-outlet x-axis pads to a consistent
    # shape (see its docstring).
    vx_bc = bnew.ContractionBoundaryConditions(
        0.0, vx_vals, types, bnew._zero_bc_fn)
    vy_bc = bnew.ContractionBoundaryConditions(
        0.0, vy_vals, types, bnew._zero_bc_fn)
    return vx_bc, vy_bc


def build_contraction_newtonian_state(grid,
                                      H: float,
                                      L_down: float,
                                      U_inlet: float,
                                      logistic_width: float,
                                      logistic_scale: float = DEFAULT_LOGISTIC_SCALE,
                                      contraction_ratio: float = 4.0,
                                      sdf: str = 'exact',
                                      ):
    """Rest initial state (``u=0``) + penalty field + contraction ``bc_spec``.

    Newtonian: no conformation, so ``memory_fields=None``. The velocity
    interior starts at rest; the inlet Dirichlet value enters the solve via
    the stepper's RHS BC-lift. Pressure starts at zero with the contraction
    pressure BC (Neumann inlet, Dirichlet outlet ``p=0``, Neumann walls) so
    the scan carry matches what the projection produces each step.

    Returns ``(initial_state, perm_f, bc_spec)``.
    """
    from jax_ib.base import grids as ib_grids
    from jax_rheology.core import boundaries as bnew
    from jax_rheology.core import state as pc

    bc_spec = bnew.BCSpec.contraction(grid.ndim)

    vx_bc, vy_bc = contraction_velocity_bc(U_inlet)
    zero_fn = lambda x, y: jnp.zeros_like(x + y)
    vx_arr = grid.eval_on_mesh(zero_fn, grid.cell_faces[0])
    vy_arr = grid.eval_on_mesh(zero_fn, grid.cell_faces[1])
    velocity = (ib_grids.GridVariable(vx_arr, vx_bc),
                ib_grids.GridVariable(vy_arr, vy_bc))

    pressure_bc = bc_spec.pressure_bc(grid)
    pressure_var = ib_grids.GridVariable(
        ib_grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid),
        pressure_bc)

    perm_f = build_contraction_permeability(
        grid, H=H, L_down=L_down, logistic_width=logistic_width,
        logistic_scale=logistic_scale, contraction_ratio=contraction_ratio,
        sdf=sdf)

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


def build_contraction_viscoelastic_state(grid,
                                         H: float,
                                         L_down: float,
                                         U_inlet: float,
                                         logistic_width: float,
                                         model,
                                         logistic_scale: float = DEFAULT_LOGISTIC_SCALE,
                                         contraction_ratio: float = 4.0,
                                         sdf: str = 'exact',
                                         ):
    """Rest velocity + ``A = I`` conformation for the viscoelastic path.

    Same momentum setup as :func:`build_contraction_newtonian_state`, plus
    the constitutive ``memory_fields`` seeded at equilibrium ``A = I`` with
    the **contraction conformation BC**: inlet Dirichlet ``A = I`` (which is
    ``Psi = 0`` for the advected log-conformation and ``tau = Gp(A-I) = 0`` for
    the stress -- a single Dirichlet value of 0 serves both, see
    ``make_logconf_evolution_fn``), outlet zero-gradient (Neumann), walls
    linear extrapolation (the RheoTool default). All cell-centered, so no
    staggered shape bookkeeping is needed.

    Returns ``(initial_state, perm_f, bc_spec)``.
    """
    from jax_ib.base import grids as ib_grids
    from jax_rheology.core import boundaries as bnew
    from jax_rheology.core import flow_conditions as fc
    from jax_rheology.core import state as pc

    bc_spec = bnew.BCSpec.contraction(grid.ndim)

    vx_bc, vy_bc = contraction_velocity_bc(U_inlet)
    zero_fn = lambda x, y: jnp.zeros_like(x + y)
    vx_arr = grid.eval_on_mesh(zero_fn, grid.cell_faces[0])
    vy_arr = grid.eval_on_mesh(zero_fn, grid.cell_faces[1])
    velocity = (ib_grids.GridVariable(vx_arr, vx_bc),
                ib_grids.GridVariable(vy_arr, vy_bc))

    pressure_var = ib_grids.GridVariable(
        ib_grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid),
        bc_spec.pressure_bc(grid))

    # Conformation field at A = I with the contraction conformation BC.
    conformation_bc = bc_spec.conformation_bc(grid)
    memory_fields = fc.get_initial_memory(
        grid, model.state_spec, bc=conformation_bc)

    perm_f = build_contraction_permeability(
        grid, H=H, L_down=L_down, logistic_width=logistic_width,
        logistic_scale=logistic_scale, contraction_ratio=contraction_ratio,
        sdf=sdf)

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
