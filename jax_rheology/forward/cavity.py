# jax_rheology/cavity_forward.py
"""Forward driver for the square lid-driven cavity.

Analogue of ``contraction_forward.evolve_contraction`` but for the closed
cavity: no inflow/outflow, a **tangential lid** on the top wall (constant
"singular" lid for the Newtonian benchmark; spatially-varying "regularized
R1" lid for the viscoelastic benchmark), the singular all-Neumann pressure
solve, and zero permeability.

Two forward paths share one driver:

  * ``model is None`` -> **Newtonian** via
    ``fully_implicit_rheology_stepper`` (``model_type='newtonian'``,
    ``params = viscosity``), time-marched to steady state. This is the
    Newtonian-benchmark path.
  * ``model is not None`` -> **viscoelastic** via
    ``memory_be_imex_stepper`` (log-conformation), Re=1 creeping, with the
    regularized lid.

The lid is ramped from rest over ``ramp_time`` with a cosine ramp
``r(t) = (1 - cos(pi * clip(t/tau, 0, 1))) / 2`` (same convention as the
contraction; Newtonian tau ~ few L/U, viscoelastic tau ~ lambda).
The ramp keeps the log-conformation startup transient from spiking the
corner stress.

The BE-IMEX carry has no wall-clock time, so -- exactly as
``contraction_forward`` does for the inlet -- the ramped lid value is
rewritten onto the velocity BC each inner step via a nested scan over a
global step index.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_ib.base import advection, boundaries, grids
from jax_rheology.solvers import steppers as eqr
from jax_rheology.solvers import pressure as pressure_new
from jax_rheology.core import boundaries as bnew
from jax_rheology.geometries import cavity as cav
from jax_rheology import log_conformation as lc


def _x64():
    return jnp.float64 if jax.config.read('jax_enable_x64') else jnp.float32


def evolve_cavity(initial_state,
                  grid,
                  *,
                  density: float,
                  dt: float,
                  inner_steps: int,
                  outer_steps: int,
                  U_lid: float,
                  ramp_time: float,
                  lid_type: str = 'singular',
                  perm_f=0.0,
                  bc_spec=None,
                  model=None,
                  polymer_params: Optional[Dict[str, Any]] = None,
                  base_viscosity: float = 0.0,
                  viscosity: Optional[float] = None,
                  solver_type: str = 'bicgstab',
                  use_preconditioner: bool = False,
                  preconditioner_type: str = 'none',
                  solver_tol: Optional[float] = 1.0e-10,
                  solver_maxiter: Optional[int] = 300,
                  convection: str = 'upwind',
                  record_diagnostics: bool = False,
                  devss_viscosity: float = 0.0,
                  ) -> Tuple[Any, Dict[str, Any]]:
    """Run the cavity forward with a ramped lid; emit per-outer-step diagnostics.

    Args:
      initial_state: rest ``All_Variables`` (from
        ``cavity_geometry.build_cavity_*_state``).
      grid: square :class:`grids.Grid`.
      density, dt, inner_steps, outer_steps: time-integration controls.
      U_lid: peak lid speed.
      ramp_time: cosine-ramp time constant (few L/U Newtonian; ~lambda VE).
      lid_type: ``'singular'`` (constant U) or ``'regularized'`` (R1 profile).
      perm_f: permeability (zero field for the cavity).
      bc_spec: ``BCSpec.cavity()``.
      model: ``None`` -> Newtonian; else a ``ConstitutiveModel`` (memory path).
      polymer_params: ``{Gp, lam, ...}`` for the viscoelastic path.
      base_viscosity: solvent viscosity nu_s for the viscoelastic path.
      viscosity: total Newtonian viscosity eta0 (= U L / Re) for the
        Newtonian path (ignored when ``model`` is given).

    Returns ``(final_state, out)`` with stacked per-outer-step trajectories.
    """
    pressure_solve = pressure_new.solve_fast_diag_cavity

    # Momentum advection scheme. 'upwind' (1st order) matches the
    # contraction/viscoelastic paths and is fine at Re=1 (viscous
    # dominated). 'van_leer' (2nd-order TVD) is needed for the high-Re
    # Newtonian benchmark, where 1st-order upwind diffusion smears the
    # primary vortex and depresses |psi_min|.
    if convection == 'van_leer':
        _advect = advection.advect_van_leer
    elif convection == 'upwind':
        _advect = advection.advect_upwind
    else:
        raise ValueError(f"Unknown convection={convection!r} "
                         "(choose 'upwind' or 'van_leer').")

    def convect(v):
        return tuple(_advect(u, v, dt) for u in v)

    is_viscoelastic = model is not None

    if is_viscoelastic:
        step_fn = eqr.memory_be_imex_stepper(
            density=density, dt=dt, grid=grid, model=model,
            params=polymer_params, base_viscosity=base_viscosity,
            convect=convect, pressure_solve=pressure_solve,
            solver_type=solver_type, pressure_gradient=[0.0, 0.0],
            permeability=perm_f, U_f=0.0,
            use_preconditioner=use_preconditioner,
            preconditioner_type=preconditioner_type,
            solver_tol=solver_tol, solver_maxiter=solver_maxiter,
            bc_spec=bc_spec, devss_viscosity=devss_viscosity)
    else:
        if viscosity is None:
            raise ValueError("Newtonian cavity (model=None) requires "
                             "`viscosity` (total eta0 = U*L/Re).")
        step_fn = eqr.fully_implicit_rheology_stepper(
            density=density, viscosity=0.0, dt=dt, grid=grid,
            model_type='newtonian', params=viscosity, model=None,
            convect=convect, forcing=None, add_tbnn_residual=False,
            pressure_solve=pressure_solve, solver_type=solver_type,
            pressure_gradient=[0.0, 0.0], permeability=perm_f, U_f=0.0,
            use_preconditioner=use_preconditioner,
            preconditioner_type=preconditioner_type,
            solver_tol=solver_tol, solver_maxiter=solver_maxiter,
            bc_spec=bc_spec, devss_viscosity=devss_viscosity)

    types_v = ((boundaries.BCType.DIRICHLET, boundaries.BCType.DIRICHLET),
               (boundaries.BCType.DIRICHLET, boundaries.BCType.DIRICHLET))
    # Static base R1 profile (regularized lid only); ramp scales it.
    base_profile = (cav.cavity_lid_profile(grid, U_lid)
                    if lid_type == 'regularized' else None)
    dx, dy = grid.step

    def _ramp(t):
        if ramp_time and ramp_time > 0:
            s = jnp.clip(t / ramp_time, 0.0, 1.0)
            return (1.0 - jnp.cos(jnp.pi * s)) / 2.0
        return jnp.ones_like(t)

    def _set_lid(state, r):
        vx, vy = state.velocity
        if lid_type == 'regularized':
            profile = r * base_profile
            new_vx_bc = bnew.CavityLidBoundaryConditions(
                0.0, ((0.0, 0.0), (0.0, 0.0)), types_v, bnew._zero_bc_fn,
                profile)
        else:  # singular constant-U lid, ramped scalar
            U_t = r * U_lid
            new_vx_bc = boundaries.ConstantBoundaryConditions(
                0.0, ((0.0, 0.0), (0.0, U_t)), types_v, bnew._zero_bc_fn)
        new_vx = grids.GridVariable(vx.array, new_vx_bc)
        return dataclasses.replace(state, velocity=(new_vx, vy))

    # Normalize the initial-state lid BC to the driver's lid type (and the
    # t=0 ramp value) so the scan carry's velocity-BC pytree type is
    # identical between the initial carry and every per-step output (the
    # stepper rewraps with this same BC object). Without this, a rest
    # state built with the singular ConstantBoundaryConditions vx BC would
    # clash with the CavityLidBoundaryConditions produced each step in the
    # regularized case -> "carry pytree structure differ" from lax.scan.
    initial_state = _set_lid(initial_state, _ramp(jnp.asarray(0.0, _x64())))

    def _inner_body(state, gstep):
        r = _ramp(gstep.astype(_x64()) * dt)
        state = _set_lid(state, r)
        return step_fn(state), None

    def _kinetic_energy(state):
        u = state.velocity[0].array.data
        v = state.velocity[1].array.data
        return 0.5 * (jnp.sum(u * u) + jnp.sum(v * v)) * dx * dy

    def _diag(state, r):
        u = state.velocity[0].array.data
        v = state.velocity[1].array.data
        ke = _kinetic_energy(state)
        if is_viscoelastic:
            A_xx = state.memory_fields[0].array.data
            A_xy = state.memory_fields[1].array.data
            A_yy = state.memory_fields[2].array.data
            A_zz = state.memory_fields[3].array.data
            lam_x, lam_y, *_ = lc.eig2x2_symmetric(A_xx, A_xy, A_yy)
            min_lam = jnp.minimum(
                jnp.minimum(jnp.min(lam_x), jnp.min(lam_y)), jnp.min(A_zz))
            tau_xx, tau_xy, tau_yy = model.stress_readout_fn(
                state.memory_fields, state.velocity, polymer_params)
            any_nan = (jnp.any(jnp.isnan(u)) | jnp.any(jnp.isnan(v))
                       | jnp.any(jnp.isnan(A_xx)) | jnp.any(jnp.isnan(A_xy))
                       | jnp.any(jnp.isnan(A_yy)) | jnp.any(jnp.isnan(A_zz)))
            return (u, v, A_xx, A_xy, A_yy, A_zz,
                    tau_xx.data, tau_xy.data, tau_yy.data,
                    min_lam, jnp.max(A_xx), any_nan, ke, r)
        else:
            any_nan = jnp.any(jnp.isnan(u)) | jnp.any(jnp.isnan(v))
            return (u, v, any_nan, ke, r)

    @jax.checkpoint
    def _outer(state, o):
        gsteps = o * inner_steps + jnp.arange(inner_steps)
        state, _ = jax.lax.scan(_inner_body, state, gsteps)
        r_now = _ramp((o + 1).astype(_x64()) * inner_steps * dt)
        return state, _diag(state, r_now)

    if not record_diagnostics:
        final_state, frames = jax.lax.scan(
            _outer, initial_state, jnp.arange(outer_steps))

        if is_viscoelastic:
            keys = ('u_traj', 'v_traj', 'A_xx_traj', 'A_xy_traj', 'A_yy_traj',
                    'A_zz_traj', 'tau_xx_traj', 'tau_xy_traj', 'tau_yy_traj',
                    'min_lam_traj', 'max_Axx_traj', 'any_nan_traj', 'ke_traj',
                    'ramp_traj')
        else:
            keys = ('u_traj', 'v_traj', 'any_nan_traj', 'ke_traj', 'ramp_traj')
        out = dict(zip(keys, frames))
        return final_state, out

    if not is_viscoelastic:
        raise ValueError('record_diagnostics=True is only supported for the '
                         'viscoelastic (model is not None) path.')

    def _diag_record(state, r):
        u = state.velocity[0].array.data
        v = state.velocity[1].array.data
        ke = _kinetic_energy(state)
        A_xx = state.memory_fields[0].array.data
        A_xy = state.memory_fields[1].array.data
        A_yy = state.memory_fields[2].array.data
        A_zz = state.memory_fields[3].array.data
        lam_x, lam_y, *_ = lc.eig2x2_symmetric(A_xx, A_xy, A_yy)
        min_lam = jnp.minimum(
            jnp.minimum(jnp.min(lam_x), jnp.min(lam_y)), jnp.min(A_zz))
        tau_xx, tau_xy, tau_yy = model.stress_readout_fn(
            state.memory_fields, state.velocity, polymer_params)
        any_nan = (jnp.any(jnp.isnan(u)) | jnp.any(jnp.isnan(v))
                   | jnp.any(jnp.isnan(A_xx)) | jnp.any(jnp.isnan(A_xy))
                   | jnp.any(jnp.isnan(A_yy)) | jnp.any(jnp.isnan(A_zz)))
        flat_idx = jnp.argmax(A_xx)
        argmax_i, argmax_j = jnp.unravel_index(flat_idx, A_xx.shape)
        return (u, v, A_xx, A_xy, A_yy, A_zz,
                tau_xx.data, tau_xy.data, tau_yy.data,
                min_lam, jnp.max(A_xx), any_nan, ke, r,
                argmax_i.astype(jnp.int32), argmax_j.astype(jnp.int32))

    @jax.checkpoint
    def _outer_record(state, o):
        gsteps = o * inner_steps + jnp.arange(inner_steps)
        state, _ = jax.lax.scan(_inner_body, state, gsteps)
        r_now = _ramp((o + 1).astype(_x64()) * inner_steps * dt)
        return state, _diag_record(state, r_now)

    final_state, frames = jax.lax.scan(
        _outer_record, initial_state, jnp.arange(outer_steps))

    keys = ('u_traj', 'v_traj', 'A_xx_traj', 'A_xy_traj', 'A_yy_traj',
            'A_zz_traj', 'tau_xx_traj', 'tau_xy_traj', 'tau_yy_traj',
            'min_lam_traj', 'max_Axx_traj', 'any_nan_traj', 'ke_traj',
            'ramp_traj', 'argmaxAxx_i_traj', 'argmaxAxx_j_traj')
    out = dict(zip(keys, frames))
    out['ke_hist'] = out['ke_traj']
    out['maxAxx_hist'] = out['max_Axx_traj']
    out['minEig_hist'] = out['min_lam_traj']
    out['argmaxAxx_i_hist'] = out['argmaxAxx_i_traj']
    out['argmaxAxx_j_hist'] = out['argmaxAxx_j_traj']
    return final_state, out
