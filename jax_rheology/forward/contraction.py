# jax_rheology/contraction_forward.py
"""Forward driver for the 4:1 contraction with a time-ramped inlet.

The BE-IMEX memory stepper carry has no notion of wall-clock time, so the
cosine inlet ramp ``U(t) = U (1 - cos(pi t / tau1)) / 2`` (ramp from rest
over one relaxation time) is applied here by threading a global step
index through a nested ``scan`` and rewriting the inlet velocity BC value
each step. The driver also records per-outer-step diagnostics (velocity,
conformation components, SPD min-eigenvalue, polymer stress).

This mirrors ``analytic_limits_validation._evolve_wall_bounded_with_diagnostics``
but for the contraction BCs (inflow/outflow + walls) and with the ramp.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_ib.base import advection, grids
from jax_rheology.solvers import steppers as eqr
from jax_rheology.solvers import pressure as pressure_new
from jax_rheology.core import boundaries as bnew
from jax_rheology import log_conformation as lc


def evolve_contraction(initial_state,
                       model,
                       polymer_params: Dict[str, Any],
                       grid,
                       *,
                       density: float,
                       base_viscosity: float,
                       dt: float,
                       inner_steps: int,
                       outer_steps: int,
                       U_inlet: float,
                       ramp_time: float,
                       perm_f,
                       bc_spec,
                       solver_type: str = 'bicgstab',
                       use_preconditioner: bool = False,
                       preconditioner_type: str = 'none',
                       solver_tol: Optional[float] = 1.0e-10,
                       solver_maxiter: Optional[int] = 300,
                       devss_viscosity: float = 0.0,
                       ) -> Tuple[Any, Dict[str, Any]]:
    """Run the contraction forward with a ramped inlet; emit diagnostics.

    Returns ``(final_state, out)`` where ``out`` holds stacked per-outer-step
    trajectories: ``u_traj, v_traj, A_xx_traj, A_xy_traj, A_yy_traj,
    A_zz_traj, tau_xx_traj, tau_xy_traj, tau_yy_traj, min_lam_traj,
    max_Axx_traj, any_nan_traj``, ``U_traj`` (the ramped inlet value), and
    ``p_traj`` (the cell-centered projected pressure carried on
    ``state.pressure`` -- additive Part-0 output; see ``_diag``).
    """
    pressure_solve = pressure_new.solve_fast_diag_contraction

    def convect(v):
        return tuple(advection.advect_upwind(u, v, dt) for u in v)

    step_fn = eqr.memory_be_imex_stepper(
        density=density, dt=dt, grid=grid, model=model,
        params=polymer_params, base_viscosity=base_viscosity, convect=convect,
        pressure_solve=pressure_solve, solver_type=solver_type,
        pressure_gradient=[0.0, 0.0], permeability=perm_f, U_f=0.0,
        use_preconditioner=use_preconditioner,
        preconditioner_type=preconditioner_type,
        solver_tol=solver_tol, solver_maxiter=solver_maxiter, bc_spec=bc_spec,
        devss_viscosity=devss_viscosity)

    types_x = initial_state.velocity[0].bc.types  # ((D,N),(D,D)), static

    def _ramp(t):
        if ramp_time and ramp_time > 0:
            s = jnp.clip(t / ramp_time, 0.0, 1.0)
            return U_inlet * (1.0 - jnp.cos(jnp.pi * s)) / 2.0
        return U_inlet * jnp.ones_like(t)

    def _set_inlet(state, U_t):
        vx, vy = state.velocity
        new_vx_bc = bnew.ContractionBoundaryConditions(
            0.0, ((U_t, 0.0), (0.0, 0.0)), types_x, bnew._zero_bc_fn)
        new_vx = grids.GridVariable(vx.array, new_vx_bc)
        return dataclasses.replace(state, velocity=(new_vx, vy))

    def _inner_body(state, gstep):
        U_t = _ramp(gstep.astype(jnp.float64 if jax.config.read('jax_enable_x64')
                                 else jnp.float32) * dt)
        state = _set_inlet(state, U_t)
        return step_fn(state), None

    def _diag(state, U_t):
        A_xx = state.memory_fields[0].array.data
        A_xy = state.memory_fields[1].array.data
        A_yy = state.memory_fields[2].array.data
        A_zz = state.memory_fields[3].array.data
        lam_x, lam_y, *_ = lc.eig2x2_symmetric(A_xx, A_xy, A_yy)
        min_lam = jnp.minimum(jnp.minimum(jnp.min(lam_x), jnp.min(lam_y)),
                              jnp.min(A_zz))
        tau_xx, tau_xy, tau_yy = model.stress_readout_fn(
            state.memory_fields, state.velocity, polymer_params)
        u = state.velocity[0].array.data
        v = state.velocity[1].array.data
        # Additive (Part 0): the projected pressure IS retained on the
        # carry -- ``pressure.projection_and_update_pressure`` writes it
        # back via ``dataclasses.replace(..., pressure=New_pressure)``,
        # where ``New_pressure = q_step + old_pressure`` accumulates the
        # per-step projection potential. We expose the cell-centered
        # field as a new trajectory key ``p_traj`` WITHOUT touching any
        # existing output (velocity / A / tau are computed exactly as
        # before). The stored
        # field is the running accumulator; the instantaneous physical
        # pressure over an outer window is recovered downstream by
        # differencing consecutive ``p_traj`` snapshots (see the forward
        # discriminability driver).
        p = state.pressure.array.data
        any_nan = (jnp.any(jnp.isnan(u)) | jnp.any(jnp.isnan(v))
                   | jnp.any(jnp.isnan(A_xx)) | jnp.any(jnp.isnan(A_xy))
                   | jnp.any(jnp.isnan(A_yy)) | jnp.any(jnp.isnan(A_zz)))
        return (u, v, A_xx, A_xy, A_yy, A_zz,
                tau_xx.data, tau_xy.data, tau_yy.data,
                min_lam, jnp.max(A_xx), any_nan, U_t, p)

    @jax.checkpoint
    def _outer(state, o):
        gsteps = o * inner_steps + jnp.arange(inner_steps)
        state, _ = jax.lax.scan(_inner_body, state, gsteps)
        U_now = _ramp((o + 1).astype(jnp.float64
                                     if jax.config.read('jax_enable_x64')
                                     else jnp.float32) * inner_steps * dt)
        return state, _diag(state, U_now)

    final_state, frames = jax.lax.scan(
        _outer, initial_state, jnp.arange(outer_steps))

    keys = ('u_traj', 'v_traj', 'A_xx_traj', 'A_xy_traj', 'A_yy_traj',
            'A_zz_traj', 'tau_xx_traj', 'tau_xy_traj', 'tau_yy_traj',
            'min_lam_traj', 'max_Axx_traj', 'any_nan_traj', 'U_traj',
            'p_traj')
    out = dict(zip(keys, frames))
    return final_state, out
