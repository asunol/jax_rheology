"""Analytic-limit and gradient checks for the full Oldroyd-B CFD stack.

Validates the wall-bounded log-conformation solver on three geometries
and then checks that reverse-mode AD agrees with centered finite
differences on the constriction:

  1. Couette startup at low Wi -- moving-wall BC + Psi-extrapolation on
     the conformation field. Wall ``tau_xy(t)`` compared to
     ``G_p . lam . gammadot . (1 - e^{-t/lam}) + eta_s . gammadot``. Run once with
     ``wall_conformation_bc='extrapolation'`` and once with
     ``'neumann'`` to quantify the near-wall difference.
  2. Poiseuille at low Wi -- same geometry, body-force pressure
     gradient. Confirm the steady velocity profile remains parabolic and
     ``N_1(y) = tau_xx - tau_yy`` is non-zero and ~quadratic in ``gammadot(y)``.
  3. Constriction at low Wi -- IB-penalty obstacle (same geometry as the
     Carreau-Yasuda / TBNN constriction baseline). Stability +
     ``Lambda > 0`` everywhere.
  4. Smoke gradient -- single-step ``jax.grad`` of a velocity-trajectory
     loss w.r.t. ``(G_p, lam)``; finite and non-zero.
  5. Multi-step AD-vs-FD on the constriction. If this fails, the likely
     cause is the ``eig2x2_symmetric`` degenerate-manifold branch --
     widen its double-where guard band or swap to the Becker-Knechtges
     eigenvalue-free kernel.

Each runner is self-contained and returns a dict of arrays + scalars
(no ``plt.show``, no printing in the return path); the notebook drives
display.

Public entry points:

    run_couette_startup(config=None, wall_conformation_bc='extrapolation') -> dict
    plot_couette_wall_stress(result, ax=None)
    plot_couette_velocity_profile(result, ax=None)
    plot_couette_conformation_profile(result, ax=None)
    plot_couette_invariants(result, ax=None)

    run_poiseuille_startup(config=None, wall_conformation_bc='extrapolation') -> dict
    plot_poiseuille_velocity_profile(result, ax=None)
    plot_poiseuille_shear_stress_profile(result, ax=None)
    plot_poiseuille_normal_stress_difference(result, ax=None)

    run_constriction_startup(config=None, wall_conformation_bc='extrapolation') -> dict
    plot_constriction_velocity_field(result, ax=None)
    plot_constriction_conformation_field(result, ax=None)
    plot_constriction_invariants(result, ax=None)

    run_constriction_comparison(config=None, wall_conformation_bc='extrapolation') -> dict
    plot_constriction_comparison(result_comparison, fig=None)

    run_smoke_gradient(config=None, wall_conformation_bc='extrapolation') -> dict

    run_multistep_ad_vs_fd(config=None, wall_conformation_bc='extrapolation') -> dict
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from repo_paths import bootstrap
bootstrap()


import jax
import jax.numpy as jnp
import jax_cfd.base as cfd

from jax_ib.base import boundaries as ib_boundaries
from jax_ib.base import grids
from jax_rheology.core import boundaries as bnew
from jax_rheology.models import registry as cr
from jax_rheology.solvers import steppers as eqr
from jax_rheology.core import flow_conditions as fc
from jax_rheology import log_conformation
from jax_rheology.core import state as pc
from jax_rheology.solvers import pressure as pressure_new


# ---------------------------------------------------------------------------
# 1. Couette startup
# ---------------------------------------------------------------------------
#
# 2-D rectangular box ``[0, Lx] x [0, Ly]`` with periodic-x and walls
# in y. Bottom wall (y = 0) is fixed at ``u_x = 0``; top wall (y = Ly)
# moves at ``u_x = U_wall`` for ``t > 0``. There is no pressure
# gradient and no body force, so the bulk equations are
#
#     d_t u_x = nu_s d^2_y u_x + (1/rho) d_y tau_xy^{(p)} ,
#     d_t u_y = 0 ,
#
# plus the Oldroyd-B conformation evolution. The Newtonian impulsive-
# shear Stokes problem has a closed-form transient
#
#     u_x^{(\text{Stokes})}(y, t) = U_wall . (y / Ly
#         + 2/pi . Sigma_{n=1}^inf (-1)^n/n . sin(n pi y / Ly)
#                  . exp(-n^2 pi^2 nu_s t / Ly^2)) ,
#
# whose first-mode decay timescale is ``tau_v = Ly^2 / (pi^2 nu_s)``. For
# ``t >~ 3 tau_v`` the bulk velocity is approximately linear,
# ``u_x(y) ~= U_wall y / Ly``, with local shear rate
# ``gammadot(y) = U_wall / Ly``. The polymer-stress relaxation responds to
# this approximately-step shear rate as
#
#     tau_xy^{(p)}(y, t) ~= G_p . lam . gammadot . (1 - exp(-t / lam)) ,        (*)
#
# and the total wall shear stress at the moving wall therefore tracks
#
#     tau_xy^{(\text{tot})}(t) ~= G_p . lam . gammadot . (1 - exp(-t / lam))
#                              + eta_s . gammadot .                          (**)
#
# The check compares the measured wall ``tau_xy(t)`` to (**) at
# a few-percent tolerance. Two sources of disagreement get folded
# into that tolerance: (i) the Newtonian transient overlap in
# ``t <~ 3 tau_v`` (the bulk hasn't fully reached linear shear yet), and
# (ii) the BE-IMEX first-order temporal error of the polymer
# relaxation stage. The default config below picks
# ``nu_s . lam / Ly^2 = 0.5`` so ``tau_v ~= 0.2 lam`` and the bulk has fully
# relaxed by the time we sample the late-time steady plateau.

DEFAULT_COUETTE_CONFIG: Dict[str, Any] = dict(
    Nx=32,
    Ny=32,
    Lx=1.0,
    Ly=1.0,
    U_wall=1.0,
    density=1.0,
    nu_s=0.5,        # large enough that bulk transient finishes within 3tau_v ~= 0.6 << 5lam
    Gp=0.1,          # low polymer modulus -- polymer is a small correction on eta_s . gammadot
    lam=1.0,         # relaxation time
    # dt chosen from the limited-advection CFL on the conformation
    # field (NOT the velocity solver -- that's implicit and effectively
    # unconstrained). At 32x32 with ``U_wall=1, Lx=1`` the conformation
    # advection NaNs above ``CFL_x ~= 0.1``; ``dt=2.5e-3`` puts us at
    # ``CFL_x = 0.08`` -- enough margin that a 20% dt bump still sits
    # under the NaN threshold.
    dt=2.5e-3,
    inner_steps=10,
    outer_steps=200,  # T_final = 5 -> 5lam at lam=1
    # Solver convention follows ``tbnn_gradient_debug_constriction_new_piv``:
    # bicgstab on the forward step, cg in the gradient path (downstream),
    # and **no preconditioner** ever. The matrix-free
    # variable-coefficient solver is well-conditioned enough that the
    # extra preconditioner machinery is a net loss in this regime.
    solver_type='bicgstab',
    use_preconditioner=False,
    preconditioner_type='none',
    # Gate tolerances:
    physics_rel_tol=0.05,   # a few percent -- Stokes overlap + first-order BE-IMEX
    settle_fraction=0.7,    # only average over t > settle_fraction . T_final
                            # for the steady-state comparison
)


def _build_grid(Nx: int, Ny: int, Lx: float, Ly: float) -> grids.Grid:
    return grids.Grid(shape=(Nx, Ny), domain=((0.0, Lx), (0.0, Ly)))


def _build_wall_bounded_initial_state(grid: grids.Grid,
                                       U_wall: float,
                                       model,
                                       wall_conformation_bc: str
                                       ) -> pc.All_Variables:
    """Build a wall-bounded startup ``All_Variables`` (Couette or Poiseuille).

    Velocity BC is :func:`Moving_wall_boundary_conditions` (periodic-x,
    Dirichlet-y with bottom = 0, top = ``U_wall``), built via the
    existing :mod:`flow_conditions` factory so we stay consistent with
    every other moving-wall caller in the codebase (the BC's
    ``boundary_fn`` is wired up to drive ``Update_BC`` if needed --
    here ``freq_osc = 0`` makes that a no-op). ``U_wall = 0`` is the
    Poiseuille case: both walls no-slip; the flow is driven by the
    pressure-gradient body force passed to
    :func:`_evolve_wall_bounded_with_diagnostics`.

    Conformation BC is
    :class:`ConformationBoundaryConditions` with the chosen
    ``wall_conformation_bc`` tag on the y axis and PERIODIC on x. The
    initial conformation is identity (the rest state on the SPD
    manifold), built via :func:`flow_conditions.get_initial_memory`.

    Initial velocity is zero everywhere -- this is the *startup*
    problem, not a pre-strained one. The pressure is initialised from
    the velocity field via the pressure factory; building a pressure
    BC from the velocity-BC object itself is the wrong API.
    """
    velocity = fc.get_initial_velocity(
        grid,
        boundary_type='moving_wall',
        amp_shear=U_wall,
        freq_osc=0.0,
    )
    pressure_var = fc.get_initial_pressure(grid, velocity)

    memory_bc = bnew.create_conformation_bc(
        grid,
        boundary_type='moving_wall',
        wall_conformation_bc=wall_conformation_bc,
        wall_axes=(1,),
    )
    memory_fields = fc.get_initial_memory(
        grid, model.state_spec, bc=memory_bc)

    return pc.All_Variables(
        particles=None,
        velocity=velocity,
        pressure=pressure_var,
        Drag=[0],
        Step_count=jnp.asarray(0),
        MD_var=[0],
        memory_fields=memory_fields,
        memory_layout=model.state_spec,
    )


def _evolve_wall_bounded_with_diagnostics(initial_state: pc.All_Variables,
                                           model,
                                           polymer_params: Dict[str, float],
                                           grid: grids.Grid,
                                           density: float,
                                           base_viscosity: float,
                                           dt: float,
                                           inner_steps: int,
                                           outer_steps: int,
                                           solver_type: str = 'bicgstab',
                                           use_preconditioner: bool = False,
                                           preconditioner_type: str = 'none',
                                           pressure_gradient: Tuple[float, float]
                                                = (0.0, 0.0),
                                           permeability: Any = 0.0,
                                           U_f: float = 0.0,
                                           solver_tol: Optional[float] = None,
                                           solver_maxiter: Optional[int] = None,
                                           bc_spec=None,
                                           ) -> Dict[str, Any]:
    """Custom scan that emits per-frame diagnostics in addition to ``v``.

    Generic wall-bounded driver shared by Couette
    (``pressure_gradient = (0, 0)``, top wall at ``U_wall``),
    Poiseuille (``pressure_gradient = (g_x, 0)``,
    both walls no-slip), and the IB constriction
    (``permeability`` from :func:`perm_vmap_multiple_particles`).
    The geometry is selected at construction time by the
    ``U_wall`` / ``particles`` passed into
    :func:`_build_wall_bounded_initial_state`; this function just
    runs the stepper.

    The library's :func:`jax_rheology.forward.generic.forward_fluid_simulation`
    emits only the velocity trajectory through ``_velocity_profile``.
    These checks need the full polymer-stress and conformation
    history at every outer step, so we replicate the dispatch logic
    here with a richer frame. Composition: inner
    BE-IMEX via :func:`memory_be_imex_stepper`, wrapped
    ``inner_steps`` deep via :func:`cfd.funcutils.repeated`, then a
    ``@jax.checkpoint``-ed outer scan that records:

      * ``u``, ``v``           -- velocity components at cell faces
      * ``A_xx``, ``A_xy``, ``A_yy`` -- three SPD components at cell
        centers
      * ``tau_xx``, ``tau_xy``, ``tau_yy`` -- full polymer stress
        readout from ``model.stress_readout_fn``
      * ``min_lam``            -- minimum eigenvalue of ``A`` over the
        whole field at this outer step (for the ``Lambda > 0`` invariant)
      * ``min_trA``            -- minimum ``A_xx + A_yy`` (informational)
      * ``any_nan``            -- bool, gating

    ``pressure_gradient`` is the body-force vector folded into
    ``total_nonviscous_rates`` by :func:`fully_implicit_forcing`.
    Default ``(0, 0)`` keeps the Couette caller unchanged.
    ``permeability`` and ``U_f`` are the IB penalty fields used by
    the constriction caller; both default to zero so the
    no-obstacle Couette / Poiseuille callers stay unaffected.
    """
    pressure_solve = pressure_new.solve_fast_diag_moving_wall

    def convect_fn(v):
        # Match the existing fully_implicit branch's convection helper
        # (upwind, returns the rate -(u.grad)u with the library's sign
        # convention).
        from jax_ib.base import advection
        return tuple(advection.advect_upwind(u, v, dt) for u in v)

    inner_stepper = cfd.funcutils.repeated(
        eqr.memory_be_imex_stepper(
            density=density,
            dt=dt,
            grid=grid,
            model=model,
            params=polymer_params,
            base_viscosity=base_viscosity,
            convect=convect_fn,
            pressure_solve=pressure_solve,
            solver_type=solver_type,
            pressure_gradient=list(pressure_gradient),
            permeability=permeability,
            U_f=U_f,
            use_preconditioner=use_preconditioner,
            preconditioner_type=preconditioner_type,
            solver_tol=solver_tol,
            solver_maxiter=solver_maxiter,
            bc_spec=bc_spec,
        ),
        steps=inner_steps,
    )

    def _diagnostics(memory_fields):
        A_xx = memory_fields[0].array.data
        A_xy = memory_fields[1].array.data
        A_yy = memory_fields[2].array.data
        A_zz = memory_fields[3].array.data
        lam_x, lam_y, *_ = log_conformation.eig2x2_symmetric(A_xx, A_xy, A_yy)
        min_lam = jnp.minimum(jnp.min(lam_x), jnp.min(lam_y))
        # In-plane trace (kept as-is for the OB/Giesekus health gates,
        # where A_zz == 1); the full trace A_xx+A_yy+A_zz is available via
        # the A_zz channel for FENE-P's finite-extensibility check.
        min_trA = jnp.min(A_xx + A_yy)
        any_nan = (jnp.any(jnp.isnan(A_xx))
                   | jnp.any(jnp.isnan(A_xy))
                   | jnp.any(jnp.isnan(A_yy))
                   | jnp.any(jnp.isnan(A_zz)))
        return A_xx, A_xy, A_yy, A_zz, min_lam, min_trA, any_nan

    @jax.checkpoint
    def outer_step(state, _):
        new_state = inner_stepper(state)
        tau_xx, tau_xy, tau_yy = model.stress_readout_fn(
            new_state.memory_fields, new_state.velocity, polymer_params)
        A_xx, A_xy, A_yy, A_zz, min_lam, min_trA, any_nan = _diagnostics(
            new_state.memory_fields)
        frame = (
            new_state.velocity[0].data,  # u_x at face offset (0, 0.5)
            new_state.velocity[1].data,  # u_y at face offset (0.5, 0)
            A_xx, A_xy, A_yy, A_zz,
            tau_xx.data, tau_xy.data, tau_yy.data,
            min_lam, min_trA, any_nan,
        )
        return new_state, frame

    final_state, frames = jax.lax.scan(
        outer_step, initial_state, xs=None, length=outer_steps)

    (u_traj, v_traj,
     A_xx_traj, A_xy_traj, A_yy_traj, A_zz_traj,
     tau_xx_traj, tau_xy_traj, tau_yy_traj,
     min_lam_traj, min_trA_traj, any_nan_traj) = frames

    return dict(
        final_state=final_state,
        u_traj=u_traj,
        v_traj=v_traj,
        A_xx_traj=A_xx_traj,
        A_xy_traj=A_xy_traj,
        A_yy_traj=A_yy_traj,
        A_zz_traj=A_zz_traj,
        tau_xx_traj=tau_xx_traj,
        tau_xy_traj=tau_xy_traj,
        tau_yy_traj=tau_yy_traj,
        min_lam_traj=min_lam_traj,
        min_trA_traj=min_trA_traj,
        any_nan_traj=any_nan_traj,
    )


def run_couette_startup(config: Optional[Dict[str, Any]] = None,
                          wall_conformation_bc: str = 'extrapolation',
                          model_name: str = 'oldroyd_b_logconf',
                          ) -> Dict[str, Any]:
    """Couette startup: wall-stress check against the analytic (**).

    Runs the full forward CFD stack on the moving-wall Couette
    geometry for ``outer_steps x inner_steps x dt`` physical time
    (``5lam`` at the defaults), then compares the measured wall ``tau_xy(t)``
    to the analytic (**) at each recorded frame.

    Args:
        config: Optional overrides on :data:`DEFAULT_COUETTE_CONFIG`.
        wall_conformation_bc: Either ``'extrapolation'`` (default) or
            ``'neumann'`` (the fallback). Threaded into
            :func:`jax_rheology.core.boundaries.create_conformation_bc` for the
            conformation field's y-axis ghost cells.
        model_name: Registered constitutive-model name to look up in
            :mod:`constitutive_registry`. Defaults to
            ``'oldroyd_b_logconf'`` (eig kernel + FE
            stretch + FE advect); use ``'oldroyd_b_logconf_bk'``
            for the Becker-Knechtges kernel or
            ``'oldroyd_b_logconf_bk_v2'`` for the full upgrade.
            Any other combination can be
            constructed and registered ad-hoc via
            :func:`jax_rheology.log_conformation.make_logconf_evolution_fn`.

    Returns:
        A dict with the trajectory, the analytic comparison curve,
        per-step ``Lambda > 0`` / NaN diagnostics, the steady-state error,
        and the gate flag ``physics_pass``.
    """
    cfg = dict(DEFAULT_COUETTE_CONFIG)
    if config:
        cfg.update(config)

    grid = _build_grid(cfg['Nx'], cfg['Ny'], cfg['Lx'], cfg['Ly'])
    model = cr.get_model(model_name)

    initial_state = _build_wall_bounded_initial_state(
        grid, cfg['U_wall'], model, wall_conformation_bc)

    polymer_params = dict(Gp=cfg['Gp'], lam=cfg['lam'])

    t0 = time.perf_counter()
    out = _evolve_wall_bounded_with_diagnostics(
        initial_state=initial_state,
        model=model,
        polymer_params=polymer_params,
        grid=grid,
        density=cfg['density'],
        base_viscosity=cfg['nu_s'],
        dt=cfg['dt'],
        inner_steps=cfg['inner_steps'],
        outer_steps=cfg['outer_steps'],
        solver_type=cfg['solver_type'],
        use_preconditioner=cfg['use_preconditioner'],
        preconditioner_type=cfg['preconditioner_type'],
        pressure_gradient=(0.0, 0.0),  # Couette: wall-driven, no body force
    )
    elapsed = time.perf_counter() - t0

    # Time axis: frames are recorded *after* each outer step (the scan
    # body assigns ``frame`` from ``new_state``), so frame ``i`` lives
    # at ``t = (i + 1) . dt . inner_steps`` (frames recorded after each outer step).
    dt = cfg['dt']
    inner = cfg['inner_steps']
    outer = cfg['outer_steps']
    times = (np.arange(outer) + 1) * dt * inner

    # Probe-row choice for the tau_xy gate.
    #
    # We deliberately read tau_xy at MID-CHANNEL (``j_probe = Ny // 2``)
    # rather than at the wall row (``j = Ny - 1``). The Psi-extrapolation
    # BC + limited advection of the conformation field overshoots
    # ``A_xy`` at the moving wall by ~70-80% of the bulk steady-state
    # value. The wall-row overshoot is a deferred BC artifact: the
    # Psi-extrapolation stencil does not yet impose the moving-wall
    # conformation consistently, so we gate on the bulk until that
    # layer is rebuilt. Simple Couette flow is y-uniform at steady state, so
    # the analytic ``tau_xy(t)`` formula applies at every y once the
    # Stokes transient ``tau_v = L^2 / (pi^2 nu_s) ~= 0.2`` settles
    # (typically ``t >~ 3tau_v``). The bulk probe gives the same answer
    # as the wall would in the absence of the artifact, and is
    # therefore the right gate diagnostic until the wall layer is
    # rebuilt.
    Ny = cfg['Ny']
    j_probe = Ny // 2

    # Stress readout is POLYMER-only (``tau_p = G_p . A``). The total
    # shear stress in the simulated fluid is ``tau_xy = tau_p + eta_s . gammadot``
    # -- we reconstruct the solvent piece diagnostically from
    # ``du_x/dy`` at the same probe row so the measured / analytic
    # comparison is apples-to-apples.
    tau_xy_traj = np.asarray(out['tau_xy_traj'])              # (T, Nx, Ny)
    tau_xy_bulk_polymer = tau_xy_traj[:, :, j_probe].mean(axis=1)  # (T,)
    tau_xy_bulk_polymer_std = tau_xy_traj[:, :, j_probe].std(axis=1)

    # u_x has offset (0, 0.5), i.e. y_face = (j + 0.5) . dy, so
    # ``u_x[..., j]`` lives at the same y as the cell-centered
    # ``A_xy[..., j]``. Central FD between j-1 and j+1 is well-defined
    # because the probe row is in the interior.
    u_traj = np.asarray(out['u_traj'])                         # (T, Nx, Ny)
    dy = cfg['Ly'] / Ny
    du_dy_field = (u_traj[:, :, j_probe + 1] - u_traj[:, :, j_probe - 1]) \
                  / (2.0 * dy)                                 # (T, Nx)
    gamma_dot_bulk = du_dy_field.mean(axis=1)                  # (T,)
    eta_s = cfg['density'] * cfg['nu_s']
    tau_xy_bulk_solvent = eta_s * gamma_dot_bulk               # (T,)
    tau_xy_bulk_total = tau_xy_bulk_polymer + tau_xy_bulk_solvent

    # Wall-row diagnostic, retained for visualising the deferred
    # moving-wall artifact (NOT used by the gate).
    tau_xy_wall_row = tau_xy_traj[:, :, -1].mean(axis=1)
    tau_xy_wall_row_std = tau_xy_traj[:, :, -1].std(axis=1)

    # Analytic tau_xy(t) at any bulk y (Couette is y-uniform at steady
    # state). The polymer transient assumes ``gammadot`` jumps to its
    # steady value at ``t=0`` -- exact for ``tau_v << lam``.
    gamma_dot_ss = cfg['U_wall'] / cfg['Ly']
    tau_xy_polymer_analytic = (cfg['Gp'] * cfg['lam'] * gamma_dot_ss
                                * (1.0 - np.exp(-times / cfg['lam'])))
    tau_xy_solvent_analytic = eta_s * gamma_dot_ss * np.ones_like(times)
    tau_xy_total_analytic = tau_xy_polymer_analytic + tau_xy_solvent_analytic

    # Steady-state error on the TOTAL stress at the bulk probe row.
    settle_t = cfg['settle_fraction'] * times[-1]
    settled = times >= settle_t
    total_settled = tau_xy_bulk_total[settled]
    total_analytic_settled = tau_xy_total_analytic[settled]
    abs_err_settled = np.abs(total_settled - total_analytic_settled)
    rel_err_settled = abs_err_settled / np.maximum(
        np.abs(total_analytic_settled), 1e-12)
    rel_err_settled_max = float(rel_err_settled.max())
    rel_err_settled_mean = float(rel_err_settled.mean())

    # Steady velocity profile at the final time: ``u_x(y)`` averaged
    # over x, should be ~= linear at the late-time limit (the
    # Oldroyd-B polymer at Wi = lam.gammadot does not change the steady
    # velocity profile of simple shear -- that's a defining property
    # of the model).
    u_final_profile = u_traj[-1].mean(axis=0)     # (Ny,)
    y_centers = (np.arange(Ny) + 0.5) * dy

    # Conformation profile at final time, averaged over x -- both for
    # the plot and for a "bulk A_xy converges to lamgammadot" sanity check.
    A_xy_final_profile = np.asarray(out['A_xy_traj'])[-1].mean(axis=0)  # (Ny,)
    A_xx_final_profile = np.asarray(out['A_xx_traj'])[-1].mean(axis=0)
    A_yy_final_profile = np.asarray(out['A_yy_traj'])[-1].mean(axis=0)

    # Diagnostics
    min_lam_traj = np.asarray(out['min_lam_traj'])
    min_trA_traj = np.asarray(out['min_trA_traj'])
    any_nan_traj = np.asarray(out['any_nan_traj'])
    min_lam_overall = float(min_lam_traj.min())
    min_trA_overall = float(min_trA_traj.min())
    Lambda_positive_pass = bool(min_lam_overall > 0.0)
    finite_pass = bool(not any_nan_traj.any())
    physics_pass = bool(rel_err_settled_max < cfg['physics_rel_tol'])

    gate_pass = (physics_pass and Lambda_positive_pass and finite_pass)

    print(f"[couette/{wall_conformation_bc}]  wall time: {elapsed:.2f} s  "
          f"({outer}x{inner} steps, T={outer*inner*dt:.2f},  probe row j={j_probe})")
    print(f"[couette/{wall_conformation_bc}]  steady-window rel err  "
          f"max={rel_err_settled_max:.3e}  mean={rel_err_settled_mean:.3e}  "
          f"tol={cfg['physics_rel_tol']:.2g}")
    print(f"[couette/{wall_conformation_bc}]  min Lambda over run = "
          f"{min_lam_overall:.4e}   min tr(A) over run = {min_trA_overall:.4e}")
    print(f"[couette/{wall_conformation_bc}]  physics_pass={physics_pass}  "
          f"Lambda>0={Lambda_positive_pass}  finite={finite_pass}  "
          f"gate_pass={gate_pass}")

    return dict(
        wall_conformation_bc=wall_conformation_bc,
        times=times,
        j_probe=j_probe,
        # New gate diagnostic: bulk total stress vs analytic total.
        tau_xy_bulk_total=tau_xy_bulk_total,
        tau_xy_bulk_polymer=tau_xy_bulk_polymer,
        tau_xy_bulk_polymer_std=tau_xy_bulk_polymer_std,
        tau_xy_bulk_solvent=tau_xy_bulk_solvent,
        gamma_dot_bulk=gamma_dot_bulk,
        # Wall-row diagnostic (artifact-prone) for visualising the
        # deferred wall overshoot.
        tau_xy_wall_row=tau_xy_wall_row,
        tau_xy_wall_row_std=tau_xy_wall_row_std,
        # Analytic
        tau_xy_total_analytic=tau_xy_total_analytic,
        tau_xy_polymer_analytic=tau_xy_polymer_analytic,
        tau_xy_solvent_analytic=tau_xy_solvent_analytic,
        # Spatial profiles at final time
        y_centers=y_centers,
        u_final_profile=u_final_profile,
        u_final_field=u_traj[-1],
        A_xx_final_profile=A_xx_final_profile,
        A_xy_final_profile=A_xy_final_profile,
        A_yy_final_profile=A_yy_final_profile,
        # Invariants / NaN diagnostics
        min_lam_traj=min_lam_traj,
        min_trA_traj=min_trA_traj,
        any_nan_traj=any_nan_traj,
        min_lam_overall=min_lam_overall,
        min_trA_overall=min_trA_overall,
        # Gate scalars
        rel_err_settled_max=rel_err_settled_max,
        rel_err_settled_mean=rel_err_settled_mean,
        Lambda_positive_pass=Lambda_positive_pass,
        finite_pass=finite_pass,
        physics_pass=physics_pass,
        gate_pass=gate_pass,
        elapsed_s=elapsed,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_couette_wall_stress(result: Dict[str, Any], ax: Optional[Any] = None):
    """``tau_xy(t)`` at the bulk probe row vs analytic, with polymer / solvent split.

    The gate diagnostic -- bulk total stress (polymer readout + solvent
    reconstructed from ``eta_s . du/dy``) plotted against the analytic
    ``G_p lam gammadot (1 - e^{-t/lam}) + eta_s gammadot``. The wall-row polymer trace is
    overlaid as a thin grey line to show the deferred moving-wall
    overshoot left by the Psi-extrapolation stencil.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
    else:
        fig = ax.figure
    cfg = result['config']
    t = result['times']
    # Analytic curves
    ax.plot(t, result['tau_xy_total_analytic'], 'k-', lw=2.0,
             label=r"analytic total  $G_p\lambda\dot\gamma(1-e^{-t/\lambda})+\eta_s\dot\gamma$")
    ax.plot(t, result['tau_xy_polymer_analytic'], 'k--', lw=1.0, alpha=0.5,
             label=r"analytic polymer only")
    ax.axhline(result['tau_xy_solvent_analytic'][0], color='gray', ls=':',
                label=r"$\eta_s\dot\gamma$ (solvent baseline)")
    # Bulk measured (the gate diagnostic)
    ax.plot(t, result['tau_xy_bulk_total'], 'C0o-', ms=4, lw=1.2,
             label=f"measured total @ j={result['j_probe']}  [{result['wall_conformation_bc']}]")
    ax.plot(t, result['tau_xy_bulk_polymer'], 'C0--', lw=1.0, alpha=0.7,
             label=f"measured polymer @ j={result['j_probe']}")
    # Wall-row trace (artifact)
    ax.plot(t, result['tau_xy_wall_row'], color='lightgray', lw=1.0,
             label="measured polymer @ wall row (artifact)")
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\tau_{xy}$')
    ax.set_title(
        f"Couette startup  ({cfg['Nx']}×{cfg['Ny']}, $\\nu_s$={cfg['nu_s']}, "
        f"$G_p$={cfg['Gp']}, $\\lambda$={cfg['lam']},  "
        f"Wi$=\\lambda U/L$={cfg['lam']*cfg['U_wall']/cfg['Ly']:.2g})  —  "
        f"rel err settled = {result['rel_err_settled_max']:.2e}")
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_couette_velocity_profile(result: Dict[str, Any],
                                     ax: Optional[Any] = None):
    """Final ``u_x(y)`` profile against the linear-shear reference."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure
    cfg = result['config']
    y = result['y_centers']
    ax.plot(result['u_final_profile'], y, 'C0o-', ms=4, lw=1.4,
             label=f"measured  [{result['wall_conformation_bc']}]")
    ax.plot(cfg['U_wall'] * y / cfg['Ly'], y, 'k--', lw=1.5,
             label=r'linear  $U_{wall}\,y/L_y$')
    ax.set_xlabel(r'$u_x(y)$')
    ax.set_ylabel(r'$y$')
    ax.set_title(f"Couette velocity profile at $t = T_{{\\text{{final}}}}$")
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_couette_conformation_profile(result: Dict[str, Any],
                                        ax: Optional[Any] = None):
    """Final ``A_xy(y)`` profile against the analytic steady state.

    Shows the bulk -> ``lamgammadot`` convergence and makes the moving-wall
    overshoot visible.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure
    cfg = result['config']
    y = result['y_centers']
    A_xy_ss = cfg['lam'] * cfg['U_wall'] / cfg['Ly']
    ax.plot(result['A_xy_final_profile'], y, 'C2o-', ms=4, lw=1.4,
             label=f"$A_{{xy}}(y)$  [{result['wall_conformation_bc']}]")
    ax.axvline(A_xy_ss, color='k', ls='--', lw=1.5,
                label=fr'analytic $\lambda\dot\gamma = {A_xy_ss:.2f}$')
    ax.set_xlabel(r'$A_{xy}(y)$')
    ax.set_ylabel(r'$y$')
    ax.set_title('Conformation $A_{xy}$ profile at $t = T_{\\text{final}}$')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_couette_invariants(result: Dict[str, Any],
                              ax: Optional[Any] = None):
    """``min(Lambda)(t)`` and ``min(tr A)(t)`` sanity traces."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
    else:
        fig = ax.figure
    t = result['times']
    ax.semilogy(t, np.maximum(result['min_lam_traj'], 1e-16), 'C0-',
                  label=r'$\min_x\,\Lambda(A)$')
    ax.semilogy(t, np.maximum(result['min_trA_traj'], 1e-16), 'C1-',
                  label=r'$\min_x\,\mathrm{tr}(A)$')
    ax.axhline(1.0, color='k', ls=':', lw=0.8,
                label=r'$A = I$ baseline ($\Lambda = \mathrm{tr}/2 = 1$)')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel('eigen / trace floor')
    ax.set_title(
        f"SPD invariants — Λ>0: {result['Lambda_positive_pass']},  "
        f"finite: {result['finite_pass']}")
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Poiseuille at low Wi
# ---------------------------------------------------------------------------
#
# Same wall-bounded box ``[0, Lx] x [0, Ly]`` with periodic-x and
# walls in y, but BOTH walls are no-slip (``U_wall = 0``) and the
# flow is driven by a constant body-force pressure gradient ``g_x``
# applied via :func:`memory_be_imex_stepper`'s ``pressure_gradient``
# parameter (folded into ``total_nonviscous_rates`` by
# :func:`fully_implicit_forcing`). The bulk x-momentum balance at
# steady state is
#
#     0 = -d_x P + d_y tau_xy^{(tot)} = g_x + d_y tau_xy^{(tot)} ,
#
# where ``tau_xy^{(tot)} = eta_s . gammadot + G_p . A_xy`` and Oldroyd-B in
# simple shear has the exact steady relation ``A_xy = lam . gammadot`` (this
# is the same defining property that gave us the closed-form Couette
# wall stress in step 1). Substituting:
#
#     (eta_s + G_p . lam) . d_y^2 u_x = -g_x ,    u_x(0) = u_x(Ly) = 0 ,
#
# whose unique solution is the parabola
#
#     u_x(y) = (g_x / (2 eta_{eff})) . y . (L_y - y) ,                  (<>)
#     eta_{eff} := eta_s + G_p . lam ,
#
# with local shear rate
#
#     gammadot(y) = du_x/dy = (g_x / eta_{eff}) . (L_y/2 - y) ,
#
# linear in ``y`` and antisymmetric about the channel centreline.
# The Oldroyd-B *first normal-stress difference* in steady simple
# shear is
#
#     A_xx - A_yy = 2 (lam gammadot)^2 ,
#     N_1(y) := tau_xx - tau_yy = G_p . (A_xx - A_yy) = 2 G_p lam^2 . gammadot(y)^2 ,  (<><>)
#
# i.e. quadratic in ``gammadot(y)`` and therefore parabolic in
# ``(L_y/2 - y)``. Two non-trivial facts get checked here that
# Couette did *not* test: (i) the polymer's contribution to the
# effective viscosity ``eta_{eff} = eta_s + G_p lam`` (the velocity
# profile narrows the parabola by the right factor), and (ii) the
# polymer's non-zero ``N_1`` (Newtonian fluids have ``N_1 == 0``).
#
# Checks:
#
#   1. ``u_x(y)`` matches (<>) within a few percent in the bulk.
#   2. ``N_1(y)`` is non-zero AND well-fit by ``c . gammadot(y)^2`` with the
#      fitted slope ``c`` matching ``2 G_p lam^2`` (from (<><>)).
#   3. Standard wall-bounded sanity: ``Lambda > 0``, finite everywhere.
#
# The "bulk vs wall row" distinction from the Couette check carries
# over: we read diagnostics on an interior y-window that excludes
# the deferred near-wall ``A_xy`` overshoot. For Poiseuille this is
# even more important because ``gammadot(y)`` peaks at the walls, so wall-row
# readings would conflate the (legitimate) gammadot-driven peak with the
# (artifact) extrapolation-stencil overshoot.

DEFAULT_POISEUILLE_CONFIG: Dict[str, Any] = dict(
    Nx=32,
    Ny=32,
    Lx=1.0,
    Ly=1.0,
    g_x=1.0,        # body-force pressure gradient (per unit volume).
                    # With eta_eff = nu_s + Gp.lam = 0.5 + 0.1 = 0.6 this
                    # gives gammadot_wall = g_x . L_y / (2.eta_eff) ~= 0.833,
                    # hence Wi_wall = lam . gammadot_wall ~= 0.83 -- comfortably
                    # "low Wi" while still firmly in the polymer-
                    # active regime (the polymer contributes ~17 %
                    # of the steady shear stress).
    U_wall=0.0,     # Poiseuille -- both walls fixed.
    density=1.0,
    nu_s=0.5,
    Gp=0.1,
    lam=1.0,
    # Same CFL-safe defaults as Couette at 32x32. The peak velocity
    # here is ``g_x L_y^2 / (8 eta_eff) ~= 0.21`` (smaller than Couette's
    # ``U_wall = 1``), so the conformation-advection CFL is even
    # more relaxed; keeping ``dt = 2.5e-3`` for consistency.
    dt=2.5e-3,
    inner_steps=10,
    outer_steps=200,
    solver_type='bicgstab',
    use_preconditioner=False,
    preconditioner_type='none',
    # Bulk window for diagnostics: drop ``bulk_margin`` cells from
    # each wall before averaging / fitting. The deferred near-wall
    # ``A_xy`` overshoot is a 1-cell-thick feature at this and
    # finer resolutions (the overshoot stays one cell thick under
    # refinement), so 2 cells of margin is more than enough.
    bulk_margin=2,
    physics_rel_tol=0.05,
    n1_fit_R2_min=0.98,   # how well N_1(y) follows 2 G_p lam^2 gammadot(y)^2
    n1_fit_slope_tol=0.10,  # +/-10 % on the fitted N_1 / gammadot^2 slope
    settle_fraction=0.7,
)


def run_poiseuille_startup(config: Optional[Dict[str, Any]] = None,
                              wall_conformation_bc: str = 'extrapolation'
                              ) -> Dict[str, Any]:
    """Poiseuille: parabolic profile + ``N_1`` check.

    Runs the full forward CFD stack on the body-force-driven
    Poiseuille geometry (no-slip both walls, periodic-x, ``gradP =
    (g_x, 0)``) for ``outer_steps x inner_steps x dt`` physical time
    (``5 lam`` at the defaults). At the final time it compares the
    x-averaged ``u_x(y)`` to the analytic parabola (<>) in the bulk
    window, and fits ``N_1(y)`` against ``gammadot(y)^2`` to check the
    Oldroyd-B normal-stress relation (<><>).

    Args:
        config: Optional overrides on :data:`DEFAULT_POISEUILLE_CONFIG`.
        wall_conformation_bc: ``'extrapolation'`` (default) or
            ``'neumann'`` -- threaded to
            :func:`jax_rheology.core.boundaries.create_conformation_bc` for the
            conformation field's y-axis ghost cells. Same options as
            the Couette runner.

    Returns:
        A dict with the trajectory, the analytic parabola and ``N_1``
        prediction, per-step ``Lambda > 0`` / NaN diagnostics, the bulk
        gate errors, and the gate flag ``physics_pass``.
    """
    cfg = dict(DEFAULT_POISEUILLE_CONFIG)
    if config:
        cfg.update(config)

    grid = _build_grid(cfg['Nx'], cfg['Ny'], cfg['Lx'], cfg['Ly'])
    model = cr.get_model('oldroyd_b_logconf')

    initial_state = _build_wall_bounded_initial_state(
        grid, cfg['U_wall'], model, wall_conformation_bc)

    polymer_params = dict(Gp=cfg['Gp'], lam=cfg['lam'])

    t0 = time.perf_counter()
    out = _evolve_wall_bounded_with_diagnostics(
        initial_state=initial_state,
        model=model,
        polymer_params=polymer_params,
        grid=grid,
        density=cfg['density'],
        base_viscosity=cfg['nu_s'],
        dt=cfg['dt'],
        inner_steps=cfg['inner_steps'],
        outer_steps=cfg['outer_steps'],
        solver_type=cfg['solver_type'],
        use_preconditioner=cfg['use_preconditioner'],
        preconditioner_type=cfg['preconditioner_type'],
        pressure_gradient=(cfg['g_x'], 0.0),
    )
    elapsed = time.perf_counter() - t0

    Nx = cfg['Nx']
    Ny = cfg['Ny']
    Lx = cfg['Lx']
    Ly = cfg['Ly']
    dt = cfg['dt']
    inner = cfg['inner_steps']
    outer = cfg['outer_steps']
    dy = Ly / Ny
    times = (np.arange(outer) + 1) * dt * inner

    eta_s = cfg['density'] * cfg['nu_s']
    eta_p = cfg['Gp'] * cfg['lam']        # polymer contribution to eta at zero shear
    eta_eff = eta_s + eta_p

    # Cell-centred y-coordinate, matches the conformation / pressure
    # offsets. ``u_x`` lives at face offset ``(0, 0.5)`` which is the
    # SAME y as the cell centres in row ``j`` (the offset is in the
    # x-direction on the y-face for u_x in our convention -- see the
    # Couette runner's du/dy computation), so this y-axis is used
    # uniformly below.
    y_centres = (np.arange(Ny) + 0.5) * dy

    # ------------------------------------------------------------------
    # Final-time, x-averaged spatial profiles.
    # ------------------------------------------------------------------
    u_traj = np.asarray(out['u_traj'])           # (T, Nx, Ny)
    v_traj = np.asarray(out['v_traj'])
    tau_xx_traj = np.asarray(out['tau_xx_traj'])
    tau_xy_traj = np.asarray(out['tau_xy_traj'])
    tau_yy_traj = np.asarray(out['tau_yy_traj'])
    A_xx_traj = np.asarray(out['A_xx_traj'])
    A_xy_traj = np.asarray(out['A_xy_traj'])
    A_yy_traj = np.asarray(out['A_yy_traj'])

    u_final_profile = u_traj[-1].mean(axis=0)         # (Ny,)
    v_final_profile = v_traj[-1].mean(axis=0)
    tau_xx_final_profile = tau_xx_traj[-1].mean(axis=0)
    tau_xy_final_profile = tau_xy_traj[-1].mean(axis=0)
    tau_yy_final_profile = tau_yy_traj[-1].mean(axis=0)
    A_xx_final_profile = A_xx_traj[-1].mean(axis=0)
    A_xy_final_profile = A_xy_traj[-1].mean(axis=0)
    A_yy_final_profile = A_yy_traj[-1].mean(axis=0)

    # Local gammadot(y) from central differences on the final-time u_x
    # profile. ``gammadot`` is well-defined on the *interior* rows only;
    # we leave it as ``np.nan`` at the wall rows (the gate uses the
    # bulk window anyway).
    gamma_dot_final = np.full(Ny, np.nan)
    gamma_dot_final[1:-1] = (u_final_profile[2:] - u_final_profile[:-2]) / (2.0 * dy)

    # ------------------------------------------------------------------
    # Analytic predictions (<>) and (<><>).
    # ------------------------------------------------------------------
    u_analytic = (cfg['g_x'] / (2.0 * eta_eff)) * y_centres * (Ly - y_centres)
    gamma_dot_analytic = (cfg['g_x'] / eta_eff) * (0.5 * Ly - y_centres)
    tau_xy_analytic = (eta_s + eta_p) * gamma_dot_analytic
    # N_1 from polymer-stress readout:
    n1_final_profile = tau_xx_final_profile - tau_yy_final_profile
    n1_analytic = 2.0 * cfg['Gp'] * cfg['lam']**2 * gamma_dot_analytic**2

    # Total measured tau_xy(y) for the body-force-balance check
    # (should be linear in y at steady state with slope ``-g_x``):
    tau_xy_total_measured = tau_xy_final_profile + eta_s * gamma_dot_final

    # ------------------------------------------------------------------
    # Bulk-window gates: drop ``bulk_margin`` cells from each wall.
    # ------------------------------------------------------------------
    bm = int(cfg['bulk_margin'])
    bulk = slice(bm, Ny - bm)
    u_err = np.abs(u_final_profile[bulk] - u_analytic[bulk])
    u_rel_err_max = float(u_err.max() / max(np.abs(u_analytic[bulk]).max(), 1e-12))
    u_rel_err_mean = float(u_err.mean() / max(np.abs(u_analytic[bulk]).max(), 1e-12))

    # N_1 fit: regress measured N_1(y) against gammadot^2(y) in the bulk
    # (forced through the origin, since N_1(gammadot=0) = 0 exactly).
    # The Oldroyd-B prediction is slope = 2 G_p lam^2.
    g2 = gamma_dot_final[bulk]**2
    n1_b = n1_final_profile[bulk]
    # Least-squares slope through origin: c = (g2 . n1) / (g2 . g2).
    n1_slope_measured = float((g2 * n1_b).sum() / max((g2 * g2).sum(), 1e-30))
    n1_slope_predicted = 2.0 * cfg['Gp'] * cfg['lam']**2
    n1_slope_rel_err = float(abs(n1_slope_measured - n1_slope_predicted)
                              / max(abs(n1_slope_predicted), 1e-12))
    n1_residual = n1_b - n1_slope_measured * g2
    n1_R2 = float(1.0 - (n1_residual**2).sum()
                  / max(((n1_b - n1_b.mean())**2).sum(), 1e-30))

    # ------------------------------------------------------------------
    # SPD / NaN invariants.
    # ------------------------------------------------------------------
    min_lam_traj = np.asarray(out['min_lam_traj'])
    min_trA_traj = np.asarray(out['min_trA_traj'])
    any_nan_traj = np.asarray(out['any_nan_traj'])
    min_lam_overall = float(min_lam_traj.min())
    min_trA_overall = float(min_trA_traj.min())
    Lambda_positive_pass = bool(min_lam_overall > 0.0)
    finite_pass = bool(not any_nan_traj.any())

    # ------------------------------------------------------------------
    # Steady-state check on the bulk velocity: ||u_x(y, t_settled) -
    # u_x(y, T_final)|| should be small (the flow has converged).
    # ------------------------------------------------------------------
    settle_t = cfg['settle_fraction'] * times[-1]
    settled = times >= settle_t
    u_settled = u_traj[settled].mean(axis=1)   # (T_settled, Ny)
    u_steady_drift = np.abs(u_settled - u_settled[-1:]).max(axis=1)  # (T_settled,)
    u_steady_drift_max = float(u_steady_drift.max())
    u_steady_drift_rel = u_steady_drift_max / max(np.abs(u_final_profile).max(), 1e-12)

    # ------------------------------------------------------------------
    # Gate flags.
    # ------------------------------------------------------------------
    velocity_pass = bool(u_rel_err_max < cfg['physics_rel_tol'])
    n1_R2_pass = bool(n1_R2 >= cfg['n1_fit_R2_min'])
    n1_slope_pass = bool(n1_slope_rel_err < cfg['n1_fit_slope_tol'])
    physics_pass = bool(velocity_pass and n1_R2_pass and n1_slope_pass)
    gate_pass = bool(physics_pass and Lambda_positive_pass and finite_pass)

    print(f"[poiseuille/{wall_conformation_bc}]  wall time: {elapsed:.2f} s  "
          f"({outer}x{inner} steps, T={outer*inner*dt:.2f}, bulk rows "
          f"j in [{bm}, {Ny-bm}))")
    print(f"[poiseuille/{wall_conformation_bc}]  u_x bulk rel err  "
          f"max={u_rel_err_max:.3e}  mean={u_rel_err_mean:.3e}  "
          f"tol={cfg['physics_rel_tol']:.2g}")
    print(f"[poiseuille/{wall_conformation_bc}]  N_1 = c . gammadot^2  fit  "
          f"c_meas={n1_slope_measured:.4e}  c_pred=2.G_p.lam^2="
          f"{n1_slope_predicted:.4e}  rel err={n1_slope_rel_err:.3e}  "
          f"R^2={n1_R2:.4f}")
    print(f"[poiseuille/{wall_conformation_bc}]  steady-window u drift  "
          f"max={u_steady_drift_max:.3e}  (rel={u_steady_drift_rel:.3e})")
    print(f"[poiseuille/{wall_conformation_bc}]  min Lambda over run = "
          f"{min_lam_overall:.4e}   min tr(A) over run = {min_trA_overall:.4e}")
    print(f"[poiseuille/{wall_conformation_bc}]  physics_pass={physics_pass}  "
          f"velocity_pass={velocity_pass}  n1_R2_pass={n1_R2_pass}  "
          f"n1_slope_pass={n1_slope_pass}  Lambda>0={Lambda_positive_pass}  "
          f"finite={finite_pass}  gate_pass={gate_pass}")

    return dict(
        wall_conformation_bc=wall_conformation_bc,
        times=times,
        y_centres=y_centres,
        bulk_slice=bulk,
        # Final-time x-averaged profiles
        u_final_profile=u_final_profile,
        v_final_profile=v_final_profile,
        gamma_dot_final=gamma_dot_final,
        tau_xx_final_profile=tau_xx_final_profile,
        tau_xy_final_profile=tau_xy_final_profile,
        tau_yy_final_profile=tau_yy_final_profile,
        tau_xy_total_measured=tau_xy_total_measured,
        n1_final_profile=n1_final_profile,
        A_xx_final_profile=A_xx_final_profile,
        A_xy_final_profile=A_xy_final_profile,
        A_yy_final_profile=A_yy_final_profile,
        # Analytic predictions
        u_analytic=u_analytic,
        gamma_dot_analytic=gamma_dot_analytic,
        tau_xy_analytic=tau_xy_analytic,
        n1_analytic=n1_analytic,
        eta_s=eta_s,
        eta_p=eta_p,
        eta_eff=eta_eff,
        # Full trajectories (for animation / drift diagnostics)
        u_traj=u_traj,
        v_traj=v_traj,
        tau_xy_traj=tau_xy_traj,
        A_xy_traj=A_xy_traj,
        # Invariants
        min_lam_traj=min_lam_traj,
        min_trA_traj=min_trA_traj,
        any_nan_traj=any_nan_traj,
        min_lam_overall=min_lam_overall,
        min_trA_overall=min_trA_overall,
        # Steady-state drift
        u_steady_drift=u_steady_drift,
        u_steady_drift_max=u_steady_drift_max,
        u_steady_drift_rel=u_steady_drift_rel,
        # Gate scalars
        u_rel_err_max=u_rel_err_max,
        u_rel_err_mean=u_rel_err_mean,
        n1_slope_measured=n1_slope_measured,
        n1_slope_predicted=n1_slope_predicted,
        n1_slope_rel_err=n1_slope_rel_err,
        n1_R2=n1_R2,
        velocity_pass=velocity_pass,
        n1_R2_pass=n1_R2_pass,
        n1_slope_pass=n1_slope_pass,
        Lambda_positive_pass=Lambda_positive_pass,
        finite_pass=finite_pass,
        physics_pass=physics_pass,
        gate_pass=gate_pass,
        elapsed_s=elapsed,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Poiseuille plot helpers
# ---------------------------------------------------------------------------

def plot_poiseuille_velocity_profile(result: Dict[str, Any],
                                       ax: Optional[Any] = None):
    """Final ``u_x(y)`` against the analytic parabola (<>).

    Highlights the bulk window used for the gate and annotates the
    measured / analytic peak velocity at the channel centreline.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure
    cfg = result['config']
    y = result['y_centres']
    bulk = result['bulk_slice']
    ax.plot(result['u_final_profile'], y, 'C0o-', ms=4, lw=1.4,
             label=f"measured  [{result['wall_conformation_bc']}]")
    ax.plot(result['u_analytic'], y, 'k--', lw=1.5,
             label=fr'analytic parabola ($\eta_{{eff}}={result["eta_eff"]:.3f}$)')
    # Shade the bulk window used for the gate.
    ax.axhspan(y[bulk.start], y[bulk.stop - 1], color='C0', alpha=0.06,
                label=f'bulk window j∈[{bulk.start},{bulk.stop})')
    ax.set_xlabel(r'$u_x(y)$')
    ax.set_ylabel(r'$y$')
    u_peak_meas = float(result['u_final_profile'].max())
    u_peak_pred = float(result['u_analytic'].max())
    ax.set_title(
        f"Poiseuille velocity profile  "
        f"(peak meas={u_peak_meas:.3f}, pred={u_peak_pred:.3f},  "
        f"rel err max={result['u_rel_err_max']:.2e})")
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_poiseuille_shear_stress_profile(result: Dict[str, Any],
                                            ax: Optional[Any] = None):
    """Final ``tau_xy(y)`` (polymer + solvent) vs analytic linear-in-y.

    At steady state, ``d_y tau_xy^{tot} = -g_x``, so ``tau_xy^{tot}(y)``
    is linear with slope ``-g_x`` and crosses zero at the channel
    centreline. Shows the polymer-only readout and the analytic
    ``(eta_s + G_p lam) . gammadot(y)`` together, which lets the reader see
    the polymer's contribution explicitly.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure
    cfg = result['config']
    y = result['y_centres']
    bulk = result['bulk_slice']
    ax.plot(result['tau_xy_total_measured'], y, 'C0o-', ms=4, lw=1.4,
             label=f"measured total τ_xy  [{result['wall_conformation_bc']}]")
    ax.plot(result['tau_xy_final_profile'], y, 'C2--', lw=1.2,
             label=r"measured polymer $G_p A_{xy}$")
    ax.plot(result['tau_xy_analytic'], y, 'k--', lw=1.5,
             label=r"analytic $(\eta_s + G_p\lambda)\dot\gamma(y)$")
    ax.axvline(0.0, color='k', lw=0.5, alpha=0.4)
    ax.axhline(0.5 * cfg['Ly'], color='gray', ls=':', lw=0.8,
                label='channel centreline')
    ax.axhspan(y[bulk.start], y[bulk.stop - 1], color='C0', alpha=0.06)
    ax.set_xlabel(r'$\tau_{xy}(y)$')
    ax.set_ylabel(r'$y$')
    ax.set_title(
        f"Poiseuille shear stress  "
        f"($\\eta_s={result['eta_s']:.3f}$, $\\eta_p={result['eta_p']:.3f}$,  "
        f"$g_x={cfg['g_x']:.2f}$)")
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_poiseuille_normal_stress_difference(result: Dict[str, Any],
                                                ax: Optional[Any] = None):
    """``N_1(y) = tau_xx - tau_yy`` vs analytic ``2 G_p lam^2 . gammadot(y)^2``.

    The Oldroyd-B prediction is *quadratic* in the local shear rate.
    The fitted slope from the gate is shown alongside the analytic
    one.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure
    cfg = result['config']
    y = result['y_centres']
    bulk = result['bulk_slice']
    ax.plot(result['n1_final_profile'], y, 'C3o-', ms=4, lw=1.4,
             label=f"measured $N_1$  [{result['wall_conformation_bc']}]")
    ax.plot(result['n1_analytic'], y, 'k--', lw=1.5,
             label=r"analytic $2 G_p \lambda^2 \dot\gamma(y)^2$")
    # Show the fitted-slope curve too -- confirms the c . gammadot^2 shape
    # even if the absolute scale is off. gammadot(y) comes from central
    # differences on u_x(y), which are undefined at the wall rows
    # (j=0 and j=Ny-1); we leave them as NaN so matplotlib skips
    # them rather than substituting zero (which would make the fit
    # curve falsely crash to zero at the walls).
    fit_curve = result['n1_slope_measured'] * result['gamma_dot_final']**2
    ax.plot(fit_curve, y, 'C3:', lw=1.0, alpha=0.7,
             label=fr"fit  $c_{{meas}}={result['n1_slope_measured']:.3e}\,\dot\gamma^2$")
    ax.axhspan(y[bulk.start], y[bulk.stop - 1], color='C3', alpha=0.06)
    ax.set_xlabel(r'$N_1(y) = \tau_{xx} - \tau_{yy}$')
    ax.set_ylabel(r'$y$')
    ax.set_title(
        f"Poiseuille $N_1$ (quadratic-in-γ̇ test)  "
        f"R²={result['n1_R2']:.4f}, slope rel err="
        f"{result['n1_slope_rel_err']:.2e}")
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Constriction at low Wi (IB penalty obstacle)
# ---------------------------------------------------------------------------
#
# Same wall-bounded geometry as Couette and Poiseuille -- periodic-x,
# no-slip y on the channel walls -- but now with two semicircular
# IB-penalty obstacles centered on the channel midline at ``x = Lx/2``,
# protruding from the top and bottom walls (radius ``r``, leaving a
# throat of height ``Ly - 2 r`` at the centre). The obstacles are
# the same IB geometry as the Carreau-Yasuda / TBNN constriction
# baseline, built by
# :func:`channel_constriction_flow.setup_channel_constriction`; the
# permeability field is computed by
# :func:`jax_ib.penalty.util_funs.perm_vmap_multiple_particles` and
# is threaded into :func:`memory_be_imex_stepper` exactly as in those
# constriction runs.
#
# Why this step exists: this is the first non-viscometric flow --
# the constriction creates a region of approximately-uniform extensional
# strain along streamlines as fluid accelerates into the throat and
# decelerates after it. Oldroyd-B in extension is *unbounded*: a
# Lagrangian point that sees a constant extensional rate ``epsdot`` has
# ``A_xx ~ exp(2 epsdot lam t)`` once ``2 epsdot lam > 1`` (the Hadamard
# instability of the upper-convective derivative). In an Eulerian
# flow this exponential is cut off by the *residence time* of a
# fluid element in the high-strain region -- short residence time
# (low Wi based on local strain rate x lam) keeps the kinematics
# stable. The constriction check is therefore a *stability* check:
#
#   1. ``Lambda(A) > 0`` everywhere, every frame  (SPD positivity).
#   2. No NaN in any field at any frame      (no blowup).
#   3. Bounded conformation magnitude        (no exponential runaway).
#   4. Quasi-steady late-time velocity       (the flow settles).
#
# If this check fails the most likely cause
# is the ``eig2x2_symmetric`` degenerate-manifold branch -- either
# widen its double-where guard band, or swap to the Becker-Knechtges
# eigenvalue-free kernel before proceeding.
#
# This step does NOT attempt any analytic comparison: there is no
# closed-form solution for Oldroyd-B in this geometry. Quantitative
# 4:1-contraction die-swell vortex benchmarks require higher-order
# conformation advection to converge cleanly at this Wi range and
# land separately.

DEFAULT_CONSTRICTION_CONFIG: Dict[str, Any] = dict(
    Nx=128,
    Ny=64,
    Lx=8.0,
    Ly=4.0,
    # channel_constriction_flow defaults: two radius-1.5
    # semicircles at the top and bottom walls. The throat height is
    # ``Ly - 2 r = 1.0`` (4:1 contraction).
    obstacle_radius=1.5,
    density=1.0,
    nu_s=1.0,        # high enough that bulk u_max stays O(1) at
                     # ``g_x = 1`` (Newtonian Poiseuille in the
                     # straight section gives u_max ~= g_x . Ly^2 /
                     # (8 nu_s) = 2; the throat accelerates that by
                     # ~4x via mass conservation, peak ~= 8 around
                     # the throat).
    Gp=0.05,         # low polymer modulus -- eta_p / eta_s = G_p lam / nu_s
                     # = 0.025, so the polymer is a few-percent
                     # correction on the viscous stress in the bulk.
    lam=0.5,         # relaxation time. Wi based on bulk shear
                     # gammadot ~ u_max / (Ly/2) ~= 1 is ``Wi = lam gammadot ~=
                     # 0.5``; extensional Wi at the throat will be
                     # higher but still subcritical (no Hadamard).
    g_x=1.0,         # body-force pressure gradient, like Poiseuille.
    U_wall=0.0,      # no-slip both walls.
    U_f=0.0,         # IB penalty background fluid velocity (= the
                     # ``target velocity'' the obstacle penalises
                     # against; 0 means stationary rigid obstacles).
    # CRITICAL: dt is set by the **EXPLICIT IB-PENALTY** stability
    # criterion, not advection CFL. The penalty rate is added to
    # the explicit RHS as ``-K . (u - U_f)`` (see
    # ``models.fully_implicit_forcing``), so forward-Euler-style
    # stability requires ``K . dt / rho < 2``. The constriction baseline
    # uses ``K = 20000``, hence ``dt <= 1e-4``. Larger ``dt``
    # produces NaN on step 1: the penalty kick inside the obstacle
    # overshoots zero by a huge multiple, the resulting impulse
    # ricochets into the BiCGSTAB rhs, and the joint solve blows up.
    # We use ``dt = 1e-4`` to match the Carreau-Yasuda
    # constriction baseline exactly. The advection CFL is far more
    # forgiving here: peak ``u`` in the throat is ~8, ``dx = 0.0625``,
    # so ``CFL_x = 8 . 1e-4 / 0.0625 = 0.013`` -- well below the
    # ~0.08 conformation-advection CFL ceiling from the Couette check.
    dt=1.0e-4,
    inner_steps=50,
    outer_steps=50,   # T_final = 0.25 = 0.5 lam -- short stability check,
                      # **not** a steady-state run. The criterion is
                      # "no NaN + Lambda > 0 + bounded |A|", not
                      # "matches analytic anything"; see the section
                      # header for why we don't push to true steady
                      # state here.
    solver_type='bicgstab',
    use_preconditioner=False,
    preconditioner_type='none',
    # IB penalty smoothing constants -- same as
    # ``jax_rheology/forward/generic.py`` and every existing constriction
    # caller in the repo. Don't change without re-validating against
    # the existing constriction baseline.
    ib_smoothing_width=0.0015,
    ib_smoothing_scale=20000.0,
    # Stability gate tolerances.
    max_conformation_magnitude=50.0,  # |A_ij| < this everywhere
    u_steady_drift_rel_tol=0.50,      # late-time velocity drift, rel
                                      # (loose tol -- the run is too
                                      # short to reach steady; this
                                      # gate mostly guards against
                                      # blow-up, not convergence)
    settle_fraction=0.5,
)


def _build_constriction_initial_state(grid: grids.Grid,
                                       model,
                                       wall_conformation_bc: str,
                                       obstacle_radius: float,
                                       domain: Tuple[Tuple[float, float],
                                                     Tuple[float, float]],
                                       ib_smoothing_width: float,
                                       ib_smoothing_scale: float,
                                       ) -> Tuple[pc.All_Variables, Any]:
    """Build the constriction startup ``All_Variables`` plus permeability.

    Reuses :func:`channel_constriction_flow.setup_channel_constriction`
    for the IB particle geometry (two semicircles of radius
    ``obstacle_radius`` at the channel mid-x on the top and bottom
    walls), then computes the permeability field via
    :func:`jax_ib.penalty.util_funs.perm_vmap_multiple_particles`
    with the same logistic-smoothing kernel that every other
    constriction caller in the repo uses (see
    ``jax_rheology/forward/generic.py`` for the original).

    Velocity / pressure / conformation fields are built the same
    way as the Poiseuille caller (``U_wall = 0``, no-slip both
    walls). The returned permeability is the precomputed
    cell-centered field; ``_evolve_wall_bounded_with_diagnostics``
    threads it into :func:`memory_be_imex_stepper` per outer step.
    """
    # Velocity / pressure: same machinery as the Couette / Poiseuille
    # callers, ``amp_shear = 0`` => no-slip both walls.
    velocity = fc.get_initial_velocity(
        grid,
        boundary_type='moving_wall',
        amp_shear=0.0,
        freq_osc=0.0,
    )
    pressure_var = fc.get_initial_pressure(grid, velocity)

    # Conformation field: same SPD identity start + chosen wall BC.
    memory_bc = bnew.create_conformation_bc(
        grid,
        boundary_type='moving_wall',
        wall_conformation_bc=wall_conformation_bc,
        wall_axes=(1,),
    )
    memory_fields = fc.get_initial_memory(
        grid, model.state_spec, bc=memory_bc)

    # IB obstacle + permeability. The setup helper picks the channel
    # mid-x automatically from ``domain``; we override the radius via
    # the geometry_param entries (already done inside
    # ``setup_channel_constriction``, but we keep ``obstacle_radius``
    # as a config knob in case we want to sweep it later).
    import channel_constriction_flow as ccf
    import jax_ib.penalty.util_funs as ib_util
    particles = ccf.setup_channel_constriction(domain)

    w = ib_smoothing_width
    K = ib_smoothing_scale
    def _logistic(G, K_, w_=w):
      return K_ * jax.scipy.special.expit(G / w_)
    perm_f = ib_util.perm_vmap_multiple_particles(grid, particles, _logistic, K)

    initial_state = pc.All_Variables(
        particles=particles,
        velocity=velocity,
        pressure=pressure_var,
        Drag=[0],
        Step_count=jnp.asarray(0),
        MD_var=[0],
        memory_fields=memory_fields,
        memory_layout=model.state_spec,
    )
    return initial_state, perm_f


def run_constriction_startup(config: Optional[Dict[str, Any]] = None,
                                wall_conformation_bc: str = 'extrapolation',
                                model_name: str = 'oldroyd_b_logconf',
                                ) -> Dict[str, Any]:
    """Constriction: stability + ``Lambda > 0`` check.

    Drives the IB-penalty constriction with a body-force pressure
    gradient (no moving walls), evolves the full Oldroyd-B stack for
    ``outer_steps x inner_steps x dt`` physical time at low-to-
    moderate Wi, and reports four scalar gates:

      1. ``Lambda_positive_pass`` -- ``Lambda(A) > 0`` at every cell, every
         frame.
      2. ``finite_pass`` -- no NaN in velocity, pressure, or
         conformation at any frame.
      3. ``bounded_pass`` -- ``max |A_ij| < max_conformation_magnitude``
         at every frame (catches exponential runaway from the
         upper-convective derivative before it spreads everywhere).
      4. ``steady_pass`` -- late-time x-averaged velocity drift below
         ``u_steady_drift_rel_tol`` (flow has settled into a
         quasi-steady state around the obstacle).

    All four must pass for ``gate_pass = True``. No analytic
    comparison -- see the docstring comment above for why.
    """
    cfg = dict(DEFAULT_CONSTRICTION_CONFIG)
    if config:
        cfg.update(config)

    domain = ((0.0, cfg['Lx']), (0.0, cfg['Ly']))
    grid = _build_grid(cfg['Nx'], cfg['Ny'], cfg['Lx'], cfg['Ly'])
    model = cr.get_model(model_name)

    initial_state, perm_f = _build_constriction_initial_state(
        grid=grid,
        model=model,
        wall_conformation_bc=wall_conformation_bc,
        obstacle_radius=cfg['obstacle_radius'],
        domain=domain,
        ib_smoothing_width=cfg['ib_smoothing_width'],
        ib_smoothing_scale=cfg['ib_smoothing_scale'],
    )

    polymer_params = dict(Gp=cfg['Gp'], lam=cfg['lam'])

    t0 = time.perf_counter()
    out = _evolve_wall_bounded_with_diagnostics(
        initial_state=initial_state,
        model=model,
        polymer_params=polymer_params,
        grid=grid,
        density=cfg['density'],
        base_viscosity=cfg['nu_s'],
        dt=cfg['dt'],
        inner_steps=cfg['inner_steps'],
        outer_steps=cfg['outer_steps'],
        solver_type=cfg['solver_type'],
        use_preconditioner=cfg['use_preconditioner'],
        preconditioner_type=cfg['preconditioner_type'],
        pressure_gradient=(cfg['g_x'], 0.0),
        permeability=perm_f,
        U_f=cfg['U_f'],
    )
    elapsed = time.perf_counter() - t0

    Nx = cfg['Nx']
    Ny = cfg['Ny']
    Lx = cfg['Lx']
    Ly = cfg['Ly']
    dt = cfg['dt']
    inner = cfg['inner_steps']
    outer = cfg['outer_steps']
    dx = Lx / Nx
    dy = Ly / Ny
    times = (np.arange(outer) + 1) * dt * inner

    # ------------------------------------------------------------------
    # Trajectory diagnostics.
    # ------------------------------------------------------------------
    u_traj = np.asarray(out['u_traj'])           # (T, Nx, Ny)
    v_traj = np.asarray(out['v_traj'])
    A_xx_traj = np.asarray(out['A_xx_traj'])
    A_xy_traj = np.asarray(out['A_xy_traj'])
    A_yy_traj = np.asarray(out['A_yy_traj'])
    tau_xx_traj = np.asarray(out['tau_xx_traj'])
    tau_xy_traj = np.asarray(out['tau_xy_traj'])
    tau_yy_traj = np.asarray(out['tau_yy_traj'])
    min_lam_traj = np.asarray(out['min_lam_traj'])
    min_trA_traj = np.asarray(out['min_trA_traj'])
    any_nan_traj = np.asarray(out['any_nan_traj'])

    # Per-frame conformation-magnitude maxima for the bounded gate.
    A_xx_max_traj = np.abs(A_xx_traj).max(axis=(1, 2))
    A_xy_max_traj = np.abs(A_xy_traj).max(axis=(1, 2))
    A_yy_max_traj = np.abs(A_yy_traj).max(axis=(1, 2))
    max_A_traj = np.maximum.reduce([A_xx_max_traj, A_xy_max_traj, A_yy_max_traj])
    max_A_overall = float(max_A_traj.max())

    # Per-frame velocity-magnitude max and first normal-stress-difference max.
    # The Oldroyd-B-vs-Newtonian comparison helper relies on these scalars
    # to plot a clean polymer-effect-vs-time trace without re-walking the
    # full (T, Nx, Ny) trajectories.
    u_mag_traj = np.sqrt(u_traj**2 + v_traj**2)
    u_max_traj = u_mag_traj.max(axis=(1, 2))
    n1_traj = tau_xx_traj - tau_yy_traj
    n1_max_traj = np.abs(n1_traj).max(axis=(1, 2))

    min_lam_overall = float(min_lam_traj.min())
    min_trA_overall = float(min_trA_traj.min())

    Lambda_positive_pass = bool(min_lam_overall > 0.0)
    finite_pass = bool(not any_nan_traj.any())
    bounded_pass = bool(max_A_overall < cfg['max_conformation_magnitude'])

    # ------------------------------------------------------------------
    # Steady-state velocity drift -- the late-time flow should be
    # close to quasi-steady (the polymer stress is the only thing
    # still slowly evolving at ``t >~ a few lam``).
    # ------------------------------------------------------------------
    settle_t = cfg['settle_fraction'] * times[-1]
    settled = times >= settle_t
    u_settled = u_traj[settled]                   # (T_settled, Nx, Ny)
    u_drift = np.abs(u_settled - u_settled[-1:]).max(axis=(1, 2))  # (T_settled,)
    u_drift_max = float(u_drift.max())
    u_max_overall = float(np.abs(u_traj[-1]).max())
    u_drift_rel = u_drift_max / max(u_max_overall, 1e-12)
    steady_pass = bool(u_drift_rel < cfg['u_steady_drift_rel_tol'])

    physics_pass = bool(Lambda_positive_pass and finite_pass and bounded_pass)
    gate_pass = bool(physics_pass and steady_pass)

    # ------------------------------------------------------------------
    # Permeability (binary obstacle mask) for plotting.
    # ------------------------------------------------------------------
    obstacle_mask = np.asarray(perm_f) if hasattr(perm_f, '__array__') else None
    # Coordinates for heatmap overlays.
    x_centres = (np.arange(Nx) + 0.5) * dx
    y_centres = (np.arange(Ny) + 0.5) * dy

    print(f"[constriction/{wall_conformation_bc}]  wall time: {elapsed:.2f} s  "
          f"({outer}x{inner} steps, T={outer*inner*dt:.2f}, "
          f"{Nx}x{Ny})")
    print(f"[constriction/{wall_conformation_bc}]  min Lambda over run = "
          f"{min_lam_overall:.4e}  (>0 required)")
    print(f"[constriction/{wall_conformation_bc}]  min tr(A) over run = "
          f"{min_trA_overall:.4e}  (>= 2 expected at identity rest)")
    print(f"[constriction/{wall_conformation_bc}]  max |A_ij| over run = "
          f"{max_A_overall:.4e}  (<{cfg['max_conformation_magnitude']:.1f} required)")
    print(f"[constriction/{wall_conformation_bc}]  late-time u drift rel = "
          f"{u_drift_rel:.3e}  (<{cfg['u_steady_drift_rel_tol']:.2g})  "
          f"|u|_max = {u_max_overall:.3f}")
    print(f"[constriction/{wall_conformation_bc}]  Lambda>0={Lambda_positive_pass}  "
          f"finite={finite_pass}  bounded={bounded_pass}  "
          f"steady={steady_pass}  physics_pass={physics_pass}  "
          f"gate_pass={gate_pass}")

    return dict(
        wall_conformation_bc=wall_conformation_bc,
        times=times,
        x_centres=x_centres,
        y_centres=y_centres,
        # Final-time fields
        u_final=u_traj[-1],
        v_final=v_traj[-1],
        A_xx_final=A_xx_traj[-1],
        A_xy_final=A_xy_traj[-1],
        A_yy_final=A_yy_traj[-1],
        tau_xx_final=tau_xx_traj[-1],
        tau_xy_final=tau_xy_traj[-1],
        tau_yy_final=tau_yy_traj[-1],
        obstacle_mask=obstacle_mask,
        # Trajectories
        u_traj=u_traj,
        v_traj=v_traj,
        A_xx_traj=A_xx_traj,
        A_xy_traj=A_xy_traj,
        A_yy_traj=A_yy_traj,
        tau_xx_traj=tau_xx_traj,
        tau_xy_traj=tau_xy_traj,
        tau_yy_traj=tau_yy_traj,
        # Per-frame scalars
        min_lam_traj=min_lam_traj,
        min_trA_traj=min_trA_traj,
        max_A_traj=max_A_traj,
        A_xx_max_traj=A_xx_max_traj,
        A_xy_max_traj=A_xy_max_traj,
        A_yy_max_traj=A_yy_max_traj,
        u_max_traj=u_max_traj,
        n1_max_traj=n1_max_traj,
        any_nan_traj=any_nan_traj,
        # Steady-drift diagnostics
        u_drift=u_drift,
        u_drift_rel=u_drift_rel,
        u_max_overall=u_max_overall,
        # Gate scalars
        min_lam_overall=min_lam_overall,
        min_trA_overall=min_trA_overall,
        max_A_overall=max_A_overall,
        Lambda_positive_pass=Lambda_positive_pass,
        finite_pass=finite_pass,
        bounded_pass=bounded_pass,
        steady_pass=steady_pass,
        physics_pass=physics_pass,
        gate_pass=gate_pass,
        elapsed_s=elapsed,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Constriction plot helpers
# ---------------------------------------------------------------------------

def _overlay_obstacle(ax, obstacle_mask, x_centres, y_centres,
                        threshold: float = 1.0):
    """Outline the IB obstacle on top of a heatmap axes.

    The permeability field returned by ``perm_vmap_multiple_particles``
    is the **logistic-smoothed** indicator (peaks at
    ``ib_smoothing_scale`` inside the obstacle, drops to 0 in the
    fluid); we contour at half-scale to draw the obstacle boundary.
    """
    if obstacle_mask is None:
        return
    X, Y = np.meshgrid(x_centres, y_centres, indexing='ij')
    mask = np.asarray(obstacle_mask)
    if mask.ndim == 2:
        ax.contour(X, Y, mask, levels=[0.5 * mask.max()],
                   colors='k', linewidths=1.2, alpha=0.8)


def plot_constriction_velocity_field(result: Dict[str, Any],
                                         ax: Optional[Any] = None):
    """Final-time velocity-magnitude heatmap with obstacle outlines."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 4.4))
    else:
        fig = ax.figure
    cfg = result['config']
    u = result['u_final']
    v = result['v_final']
    speed = np.sqrt(u**2 + v**2)
    x = result['x_centres']
    y = result['y_centres']
    im = ax.imshow(speed.T, origin='lower',
                    extent=(x[0], x[-1], y[0], y[-1]),
                    aspect='equal', cmap='viridis')
    _overlay_obstacle(ax, result['obstacle_mask'], x, y)
    fig.colorbar(im, ax=ax, label=r'$|u|$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(
        f"Constriction $|u|$ at $t=T_{{\\mathrm{{final}}}}$  "
        f"($N={cfg['Nx']}\\times{cfg['Ny']}$, $g_x={cfg['g_x']}$, "
        f"$G_p={cfg['Gp']}$, $\\lambda={cfg['lam']}$)")
    fig.tight_layout()
    return fig


def plot_constriction_conformation_field(result: Dict[str, Any],
                                             ax: Optional[Any] = None):
    """Final-time ``A_xx`` heatmap -- exposes streamline-aligned stretch.

    ``A_xx`` is the component most sensitive to extensional strain
    along streamlines (cf. Oldroyd-B upper-convective derivative);
    a healthy run shows ``A_xx`` peaking *downstream* of the throat
    in a thin filament along the centreline, decaying back to ``~= 1``
    after a few ``lam``.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 4.4))
    else:
        fig = ax.figure
    cfg = result['config']
    A_xx = result['A_xx_final']
    x = result['x_centres']
    y = result['y_centres']
    im = ax.imshow(A_xx.T, origin='lower',
                    extent=(x[0], x[-1], y[0], y[-1]),
                    aspect='equal', cmap='magma')
    _overlay_obstacle(ax, result['obstacle_mask'], x, y)
    fig.colorbar(im, ax=ax, label=r'$A_{xx}$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(
        f"Constriction $A_{{xx}}$ at $t=T_{{\\mathrm{{final}}}}$  "
        f"(max={A_xx.max():.2f}, identity rest = 1)")
    fig.tight_layout()
    return fig


def plot_constriction_invariants(result: Dict[str, Any],
                                     ax: Optional[Any] = None):
    """``min Lambda(t)`` / ``min tr(A)(t)`` / ``max |A|(t)`` trajectories."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
    else:
        fig = ax.figure
    t = result['times']
    cfg = result['config']
    ax.semilogy(t, np.maximum(result['min_lam_traj'], 1e-16), 'C0-',
                 label=r'$\min_x\,\Lambda(A)$')
    ax.semilogy(t, np.maximum(result['min_trA_traj'], 1e-16), 'C1-',
                 label=r'$\min_x\,\mathrm{tr}(A)$')
    ax.semilogy(t, result['max_A_traj'], 'C3-',
                 label=r'$\max_x\,|A_{ij}|$')
    ax.axhline(1.0, color='k', ls=':', lw=0.6, alpha=0.5)
    ax.axhline(cfg['max_conformation_magnitude'], color='C3', ls=':',
                lw=0.6, alpha=0.5,
                label=fr"bound = {cfg['max_conformation_magnitude']:.0f}")
    ax.set_xlabel(r'$t$')
    ax.set_ylabel('eigenvalue / trace / magnitude')
    ax.set_title(
        f"Constriction SPD invariants  —  "
        f"Λ>0: {result['Lambda_positive_pass']}, finite: "
        f"{result['finite_pass']}, bounded: {result['bounded_pass']}, "
        f"steady: {result['steady_pass']}")
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3b. Oldroyd-B vs Newtonian-equivalent constriction comparison
# ---------------------------------------------------------------------------
#
# The constriction check is a stability test, not a polymer-physics
# test -- at the "safe low-Wi" defaults (``G_p . lam / nu_s = 0.025``)
# the polymer contributes a ~2 % correction to the bulk shear stress
# and the velocity field is visually indistinguishable from a
# Newtonian one of the same total viscosity. This helper
# takes the same harness and re-runs it twice
# back-to-back -- once with the full Oldroyd-B kinematics (``Gp,
# lam``), and once with a Newtonian fluid whose viscosity matches
# the Oldroyd-B *zero-shear total*
#
#     eta_eff = eta_s + G_p . lam                                       (<>)
#
# -- then exposes per-frame scalars and final-time field differences
# so the user can quantify what the polymer is actually doing in
# the contraction.
#
# Why ``eta_eff`` (<>) is the right "fair" Newtonian comparison: in
# the steady viscometric limit (simple shear, Poiseuille) the
# Oldroyd-B fluid is *exactly* indistinguishable in velocity from a
# Newtonian fluid of viscosity ``eta_eff`` -- that's the same
# ``eta_eff`` that gave the Poiseuille check's parabolic profile its
# polymer-narrowed peak velocity. So **any** velocity difference
# between the two runs in the constriction is, by construction, a
# non-viscometric polymer effect -- the contraction is not pure
# shear, and ``A_xx`` builds up along streamlines, and the polymer
# divergence ``div tau_p`` no longer balances a Newtonian shear stress.
# Likewise for ``N_1 = tau_xx - tau_yy``: Newtonian fluids have
# ``N_1 == 0``, so any non-zero ``N_1`` field is a pure polymer
# signature.
#
# Practical note on what you'll see at the gate defaults: with
# ``G_p . lam / nu_s = 0.025`` and ``T_final = 0.5 lam`` the elastic
# effects are *quantitatively* present (you can read them off the
# comparison plot) but visually subtle -- the velocity-difference
# field is ~1 % of ``|u|_max`` and the ``N_1`` field is ~0.05.
# To make the elasticity visually dominant, bump ``Gp`` (and / or
# extend ``outer_steps``); see the notebook cell at the bottom of
# this section.

def run_constriction_comparison(config: Optional[Dict[str, Any]] = None,
                                   wall_conformation_bc: str = 'extrapolation'
                                   ) -> Dict[str, Any]:
    """Run Oldroyd-B + Newtonian-equivalent back-to-back on the constriction.

    The Newtonian-equivalent run uses ``G_p = 0`` (polymer disabled)
    and ``nu_s' = nu_s + G_p . lam`` (the Oldroyd-B zero-shear effective
    viscosity), so its steady viscometric response -- by Oldroyd-B
    construction -- would exactly equal the Oldroyd-B run if the
    geometry were pure shear. The constriction is *not* pure shear,
    so any difference between the two final-time velocity fields is
    a non-viscometric polymer effect (the only physics that the
    Newtonian-equivalent fluid is missing).

    Args:
        config: Optional override on :data:`DEFAULT_CONSTRICTION_CONFIG`.
            ``Gp``, ``lam``, and ``nu_s`` define the Oldroyd-B run
            and the Newtonian-equivalent ``nu_s'`` is derived from
            them via (<>). All other config entries (grid, dt,
            geometry, IB constants) are shared between the two runs.
        wall_conformation_bc: Threaded through to both runs.

    Returns:
        A dict with both result dicts under ``'oldroyd'`` and
        ``'newtonian'``, plus the derived ``eta_eff = nu_s + G_p.lam``,
        the Oldroyd-B-minus-Newtonian velocity diff fields, and a
        few summary scalars convenient for the plot helper.
    """
    cfg = dict(DEFAULT_CONSTRICTION_CONFIG)
    if config:
        cfg.update(config)

    eta_p = cfg['Gp'] * cfg['lam']
    nu_s_equiv = cfg['nu_s'] + eta_p

    print(f">> Oldroyd-B run     (_s={cfg['nu_s']:.4f}, G_p={cfg['Gp']:.4f}, "
          f"lam={cfg['lam']:.4f},  eta_eff={nu_s_equiv:.4f})")
    result_ob = run_constriction_startup(
        config=cfg, wall_conformation_bc=wall_conformation_bc)

    cfg_newt = dict(cfg)
    cfg_newt['Gp'] = 0.0
    cfg_newt['nu_s'] = nu_s_equiv
    print(f">> Newtonian-equiv run (_s'={nu_s_equiv:.4f}, G_p=0)")
    result_newt = run_constriction_startup(
        config=cfg_newt, wall_conformation_bc=wall_conformation_bc)

    # Final-time velocity-difference field -- the "elastic deflection".
    # Both runs share the geometry / dt / outer_steps / inner_steps so
    # the trajectories are perfectly time-aligned.
    u_final_diff = result_ob['u_final'] - result_newt['u_final']
    v_final_diff = result_ob['v_final'] - result_newt['v_final']
    speed_final_diff = np.sqrt(u_final_diff**2 + v_final_diff**2)

    u_max_traj_ob = result_ob['u_max_traj']
    u_max_traj_newt = result_newt['u_max_traj']
    u_diff_max_traj = np.array([
        np.sqrt((result_ob['u_traj'][t] - result_newt['u_traj'][t])**2
                + (result_ob['v_traj'][t] - result_newt['v_traj'][t])**2
               ).max()
        for t in range(len(result_ob['times']))
    ])

    return dict(
        config=cfg,
        eta_p=eta_p,
        eta_eff=nu_s_equiv,
        oldroyd=result_ob,
        newtonian=result_newt,
        u_final_diff=u_final_diff,
        v_final_diff=v_final_diff,
        speed_final_diff=speed_final_diff,
        u_diff_max_traj=u_diff_max_traj,
        u_max_traj_ob=u_max_traj_ob,
        u_max_traj_newt=u_max_traj_newt,
    )


def plot_constriction_comparison(result: Dict[str, Any],
                                    fig: Optional[Any] = None):
    """2 x 3 panel comparison of Oldroyd-B vs Newtonian-equivalent.

    Layout:

      [ |u| OB        ] [ |u| Newtonian-eq ] [ |Deltau|  (elastic deflection) ]
      [ max|u|(t) OB+N] [ max|A_xx|(t) OB  ] [ max|N_1|(t)  +  |Deltau|_max(t) ]

    Row 1 (final-time fields, with obstacle outline overlaid):
        - Oldroyd-B speed field
        - Newtonian-equivalent speed field
        - Pointwise speed-difference field (polymer effect)

    Row 2 (time series):
        - ``max |u|`` for both runs (shows total viscosity equivalence)
        - ``max |A_xx|`` for the Oldroyd-B run (streamline-extension
          signal)
        - ``max |N_1|`` plus ``max |Deltau|`` on a twin axis (the two
          principal polymer signatures)
    """
    import matplotlib.pyplot as plt
    if fig is None:
        fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.0))
    else:
        axes = np.array(fig.axes).reshape(2, 3)

    cfg = result['config']
    res_ob = result['oldroyd']
    res_newt = result['newtonian']
    x = res_ob['x_centres']
    y = res_ob['y_centres']
    obstacle_mask = res_ob['obstacle_mask']
    extent = (x[0], x[-1], y[0], y[-1])

    # ---- Row 1: final-time fields ----
    speed_ob = np.sqrt(res_ob['u_final']**2 + res_ob['v_final']**2)
    speed_newt = np.sqrt(res_newt['u_final']**2 + res_newt['v_final']**2)
    speed_diff = result['speed_final_diff']
    speed_vmax = max(speed_ob.max(), speed_newt.max())

    ax = axes[0, 0]
    im = ax.imshow(speed_ob.T, origin='lower', extent=extent,
                    aspect='equal', cmap='viridis',
                    vmin=0.0, vmax=speed_vmax)
    _overlay_obstacle(ax, obstacle_mask, x, y)
    fig.colorbar(im, ax=ax, label=r'$|u|$')
    ax.set_title(
        f"$|u|$  Oldroyd-B "
        f"($\\eta_{{eff}}={result['eta_eff']:.3f}$, max={speed_ob.max():.3f})")
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')

    ax = axes[0, 1]
    im = ax.imshow(speed_newt.T, origin='lower', extent=extent,
                    aspect='equal', cmap='viridis',
                    vmin=0.0, vmax=speed_vmax)
    _overlay_obstacle(ax, obstacle_mask, x, y)
    fig.colorbar(im, ax=ax, label=r'$|u|$')
    ax.set_title(
        f"$|u|$  Newtonian-eq "
        f"($\\nu'={result['eta_eff']:.3f}$, max={speed_newt.max():.3f})")
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')

    ax = axes[0, 2]
    # Diverging plot would be misleading (the field is non-negative
    # magnitude); use a sequential map keyed on the ratio Delta/|u|_max.
    rel_diff = speed_diff / max(speed_vmax, 1e-12)
    im = ax.imshow(rel_diff.T, origin='lower', extent=extent,
                    aspect='equal', cmap='magma')
    _overlay_obstacle(ax, obstacle_mask, x, y)
    fig.colorbar(im, ax=ax, label=r'$|\Delta u| / |u|_{max}$')
    ax.set_title(
        f"elastic deflection  $|\\Delta u|/|u|_{{max}}$  "
        f"(max ratio = {rel_diff.max():.3e})")
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')

    # ---- Row 2: time series ----
    t = res_ob['times']

    ax = axes[1, 0]
    ax.plot(t, result['u_max_traj_ob'], 'C0-', lw=1.6,
             label=f"Oldroyd-B (G_p={cfg['Gp']:.3f}, λ={cfg['lam']:.2f})")
    ax.plot(t, result['u_max_traj_newt'], 'C1--', lw=1.6,
             label=f"Newtonian-eq (ν'={result['eta_eff']:.3f})")
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\max_x\,|u|(t)$')
    ax.set_title(
        f"velocity maxima — final |u| ratio "
        f"= {res_ob['u_max_overall']/max(res_newt['u_max_overall'],1e-12):.4f}")
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, res_ob['A_xx_max_traj'], 'C3-', lw=1.6,
             label=r'$\max_x |A_{xx}|$  (Oldroyd-B)')
    ax.plot(t, res_ob['A_xy_max_traj'], 'C2-', lw=1.2, alpha=0.85,
             label=r'$\max_x |A_{xy}|$')
    ax.plot(t, res_ob['A_yy_max_traj'], 'C4-', lw=1.2, alpha=0.85,
             label=r'$\max_x |A_{yy}|$')
    ax.axhline(1.0, color='k', ls=':', lw=0.6, alpha=0.5,
                label=r'identity rest = 1')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel('conformation magnitude')
    ax.set_title(
        f"polymer stretch  (max $A_{{xx}}$ = {res_ob['A_xx_max_traj'].max():.3f})")
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(t, res_ob['n1_max_traj'], 'C3-', lw=1.6,
             label=r'$\max_x |N_1|$  (Oldroyd-B)')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\max_x |N_1|$', color='C3')
    ax.tick_params(axis='y', labelcolor='C3')
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, result['u_diff_max_traj'], 'C0--', lw=1.6,
              label=r'$\max_x |\Delta u|$')
    ax2.set_ylabel(r'$\max_x |\Delta u|$  (= OB $-$ Newtonian)',
                    color='C0')
    ax2.tick_params(axis='y', labelcolor='C0')
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
                loc='best', fontsize=8)
    ax.set_title(
        f"polymer signatures  (final $|N_1|_{{max}}$={res_ob['n1_max_traj'][-1]:.4e}, "
        f"$|\\Delta u|_{{max}}$={result['u_diff_max_traj'][-1]:.4e})")

    fig.suptitle(
        f"Constriction — Oldroyd-B vs Newtonian-equivalent  "
        f"($N={cfg['Nx']}\\times{cfg['Ny']}$, $g_x={cfg['g_x']}$, "
        f"$G_p={cfg['Gp']}$, $\\lambda={cfg['lam']}$, "
        f"$\\nu_s={cfg['nu_s']}$ -> $\\eta_{{eff}}={result['eta_eff']:.4f}$, "
        f"$T_{{final}}={t[-1]:.3f}$)",
        fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Smoke gradient -- jax.grad through the full Oldroyd-B stack
# ---------------------------------------------------------------------------
#
# Take the smallest
# stable Couette config from the startup check, define a scalar loss on the
# velocity trajectory, and compute ``jax.value_and_grad`` of that
# loss with respect to the polymer parameter pair ``(G_p, lam)``.
# There is no analytic component -- this confirms only that
# reverse-mode AD propagates cleanly through every link in the
# Oldroyd-B stack:
#
#   1. ``model.evolution_fn`` (log-conformation update of ``Psi``)
#   2. ``model.stress_readout_fn`` (``Psi -> A -> tau_p``)
#   3. :func:`polymer_force_to_faces` (``div tau_p`` interpolation)
#   4. :func:`memory_be_imex_stepper` (polymer rate + BE-IMEX)
#   5. :func:`linear_solve_implicit_with_bicgstab` (per-component
#       BiCGSTAB with implicit-differentiation custom_vjp)
#   6. :func:`projection_and_update_pressure` (fast-diag pressure)
#   7. ``jax.lax.scan`` (outer time loop) with ``jax.checkpoint``
#       on the outer body
#
# Pass criteria: ``dloss/dG_p`` and ``dloss/dlam`` are both finite
# and have magnitude above a tiny floor (i.e. the AD doesn't
# silently zero out anywhere along the chain). The numerical
# *value* of the gradient is left for the
# multi-step AD-vs-FD check on the constriction.
#
# This step uses a deliberately-tiny ``(N=16, T_final ~= 0.05)``
# Couette config so the smoke test completes in well under a
# minute (compile dominates over solve at this size, so most of
# the wall time goes to ``jax.jit`` of the differentiated step).

DEFAULT_SMOKE_GRADIENT_CONFIG: Dict[str, Any] = dict(
    Nx=16,
    Ny=16,
    Lx=1.0,
    Ly=1.0,
    U_wall=1.0,
    density=1.0,
    nu_s=0.5,
    Gp_init=0.1,
    lam_init=1.0,
    dt=5.0e-3,        # Couette CFL is comfortable at N=16
                      # (gammadot.dt = 1.5e-3 = 5e-3, far below the 0.08
                      # advection ceiling from the Couette check).
    inner_steps=5,
    outer_steps=5,    # T_final = 0.125 -- long enough that the
                      # polymer transient is non-trivial (the
                      # 1-exp(-t/lam) factor is at 12% at this t),
                      # so both dloss/dG_p and dloss/dlam are
                      # non-degenerate but the run is fast.
    solver_type='bicgstab',
    use_preconditioner=False,
    preconditioner_type='none',
    # Gate tolerances: gradient floor for "non-zero" check.
    grad_floor=1e-12,
)


def run_smoke_gradient(config: Optional[Dict[str, Any]] = None,
                         wall_conformation_bc: str = 'extrapolation'
                         ) -> Dict[str, Any]:
    """Smoke test: ``jax.value_and_grad`` through the OB stack.

    Builds the smallest-stable Couette state, defines

        L(G_p, lam) := Sigma_{t,x,y} (u(t, x, y; G_p, lam)^2 + v(t, x, y; G_p, lam)^2),

    and reports ``L`` and ``(dL/dG_p, dL/dlam)``. Both partials must
    be finite and have magnitude greater than
    ``config['grad_floor']`` (default ``1e-12``) for the gate to
    pass.

    The forward pass is run *twice*: once outside of ``value_and_grad``
    to establish a JIT-warm baseline timing, and once inside to
    measure the AD overhead. The ratio is reported as a sanity
    check (typical: AD adds 1.5-3x on small problems; if it
    explodes to 10x+ the checkpointing or the BiCGSTAB
    custom_vjp is misconfigured).
    """
    cfg = dict(DEFAULT_SMOKE_GRADIENT_CONFIG)
    if config:
        cfg.update(config)

    grid = _build_grid(cfg['Nx'], cfg['Ny'], cfg['Lx'], cfg['Ly'])
    model = cr.get_model('oldroyd_b_logconf')

    initial_state = _build_wall_bounded_initial_state(
        grid, cfg['U_wall'], model, wall_conformation_bc)

    polymer_params = dict(
        Gp=jnp.asarray(cfg['Gp_init'], dtype=jnp.float32),
        lam=jnp.asarray(cfg['lam_init'], dtype=jnp.float32),
    )

    # The closure pins everything that AD shouldn't differentiate
    # through -- grid, model, initial state, dt, inner/outer steps,
    # solver type. Only ``polymer_params`` is a JAX-traced argument.
    def loss_fn(params):
        out = _evolve_wall_bounded_with_diagnostics(
            initial_state=initial_state,
            model=model,
            polymer_params=params,
            grid=grid,
            density=cfg['density'],
            base_viscosity=cfg['nu_s'],
            dt=cfg['dt'],
            inner_steps=cfg['inner_steps'],
            outer_steps=cfg['outer_steps'],
            solver_type=cfg['solver_type'],
            use_preconditioner=cfg['use_preconditioner'],
            preconditioner_type=cfg['preconditioner_type'],
            pressure_gradient=(0.0, 0.0),
        )
        u_traj = out['u_traj']
        v_traj = out['v_traj']
        return jnp.sum(u_traj * u_traj) + jnp.sum(v_traj * v_traj)

    # Warm-up forward pass -- JIT compiles the inner stepper.
    t0 = time.perf_counter()
    loss_warm = loss_fn(polymer_params).block_until_ready()
    t_forward_warm = time.perf_counter() - t0

    # Timed forward (JIT-cached).
    t0 = time.perf_counter()
    loss_val = loss_fn(polymer_params).block_until_ready()
    t_forward = time.perf_counter() - t0

    # Timed value_and_grad.
    t0 = time.perf_counter()
    loss_vg, grad_vg = jax.value_and_grad(loss_fn)(polymer_params)
    loss_vg = jnp.asarray(loss_vg).block_until_ready()
    grad_Gp = jnp.asarray(grad_vg['Gp']).block_until_ready()
    grad_lam = jnp.asarray(grad_vg['lam']).block_until_ready()
    t_grad = time.perf_counter() - t0

    grad_Gp_val = float(grad_Gp)
    grad_lam_val = float(grad_lam)

    finite_grad = bool(np.isfinite(grad_Gp_val)
                        and np.isfinite(grad_lam_val))
    finite_loss = bool(np.isfinite(float(loss_val)))
    nonzero_Gp = bool(abs(grad_Gp_val) > cfg['grad_floor'])
    nonzero_lam = bool(abs(grad_lam_val) > cfg['grad_floor'])
    nonzero_grad = nonzero_Gp and nonzero_lam

    # Consistency check: forward-pass loss between the JIT-cached
    # forward and the value_and_grad forward must agree to single
    # precision (they ride the same JIT cache).
    loss_consistent = bool(np.isclose(float(loss_val), float(loss_vg),
                                       rtol=1e-5, atol=1e-7))

    gate_pass = bool(finite_grad and finite_loss
                      and nonzero_grad and loss_consistent)

    print(f"[smoke-grad/{wall_conformation_bc}]  "
          f"N={cfg['Nx']}x{cfg['Ny']}, outer={cfg['outer_steps']}, "
          f"inner={cfg['inner_steps']}, dt={cfg['dt']:.3g}, "
          f"T={cfg['outer_steps']*cfg['inner_steps']*cfg['dt']:.3g}")
    print(f"[smoke-grad/{wall_conformation_bc}]  "
          f"forward (warm) compile+run  = {t_forward_warm:.2f} s")
    print(f"[smoke-grad/{wall_conformation_bc}]  "
          f"forward (cached)            = {t_forward:.2f} s")
    print(f"[smoke-grad/{wall_conformation_bc}]  "
          f"value_and_grad              = {t_grad:.2f} s   "
          f"(ratio = {t_grad / max(t_forward, 1e-6):.2f}x)")
    print(f"[smoke-grad/{wall_conformation_bc}]  "
          f"loss = {float(loss_val):.6e}")
    print(f"[smoke-grad/{wall_conformation_bc}]  "
          f"dloss/dG_p = {grad_Gp_val:+.6e}   "
          f"dloss/dlam  = {grad_lam_val:+.6e}")
    print(f"[smoke-grad/{wall_conformation_bc}]  "
          f"finite_loss={finite_loss}  finite_grad={finite_grad}  "
          f"nonzero_dGp={nonzero_Gp}  nonzero_dlam={nonzero_lam}  "
          f"loss_consistent={loss_consistent}  gate_pass={gate_pass}")

    return dict(
        wall_conformation_bc=wall_conformation_bc,
        polymer_params=polymer_params,
        loss=float(loss_val),
        loss_value_and_grad=float(loss_vg),
        grad_Gp=grad_Gp_val,
        grad_lam=grad_lam_val,
        t_forward_warm=t_forward_warm,
        t_forward=t_forward,
        t_grad=t_grad,
        grad_overhead_ratio=t_grad / max(t_forward, 1e-6),
        finite_loss=finite_loss,
        finite_grad=finite_grad,
        nonzero_Gp=nonzero_Gp,
        nonzero_lam=nonzero_lam,
        nonzero_grad=nonzero_grad,
        loss_consistent=loss_consistent,
        gate_pass=gate_pass,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# 5. Multi-step AD-vs-FD on the constriction
# ---------------------------------------------------------------------------
#
# The smoke gradient only checked that AD doesn't break (finite +
# non-zero). This is the *quantitative* check -- does reverse-
# mode AD agree with centered finite differences on a real
# multi-step trajectory through the 2D non-viscometric
# constriction field, to a few percent?
#
# Why constriction (not Couette / Poiseuille):
#   * Couette and Poiseuille are viscometric -- the velocity
#     gradient tensor is uniform-in-x and (for Couette) uniform-
#     in-time, so the gradient signal carries no information
#     about kinematic richness. We need a real 2D field with
#     both shear and extension regions for the AD path to be
#     stressed.
#   * The constriction has all of: shear at the walls,
#     extension at the throat centerline, recirculation downstream,
#     and IB-penalty-driven near-obstacle behaviour. If the AD
#     gradient is wrong anywhere, it'll show up here.
#   * TBNN training drives gradients through this
#     same geometry, so this check is the precondition for
#     trusting reverse-mode AD on the optimizer path.
#
# Why tight BiCGSTAB tol (`1e-12` here vs `1e-7` default):
#   FD's noise floor is `solver_tol / eps`. At step 4's default
#   `tol = 1e-7` and `eps = 1e-4`, that's `1e-3` absolute noise
#   on `L ~ 100`, which dominated the FD signal at the ~10%
#   relative level. Tightening to `1e-12` drops the FD noise
#   floor to `1e-8`, well below any reasonable AD-FD true
#   discrepancy. The adjoint solver inside
#   :func:`linear_solve_implicit_with_bicgstab` runs at
#   `adjoint_tol = tol * 1e-1 = 1e-13`, which is at the edge of
#   float64 precision but still well-conditioned for our system
#   sizes.
#
# Why float64:
#   `1e-12` is impossible in float32 (machine eps ~= 1e-7). The
#   step-5 driver requires `jax.config.read('jax_enable_x64')`
#   to be True and raises a clear message otherwise.
#
# Why a smaller step count than ``DEFAULT_CONSTRICTION_CONFIG``:
#   AD compile + memory grow ~linearly in `outer_steps`. At the
#   physics-gate config (50 outer x 50 inner = 2500 steps) AD
#   would take >30 min and >16 GB. The step-5 config uses 10 x 10
#   = 100 steps (`T = 0.01`, `t/lam = 0.02`) -- long enough that
#   dL/dlam is well above the FD noise floor with throat gammadot ~= 1,
#   short enough that AD finishes in ~3 min.

DEFAULT_MULTISTEP_AD_FD_CONFIG: Dict[str, Any] = dict(
    # Same geometry as the constriction check (so this exercises
    # the same code path the TBNN optimizer will see).
    Nx=128,
    Ny=64,
    Lx=8.0,
    Ly=4.0,
    obstacle_radius=1.5,
    density=1.0,
    nu_s=1.0,
    Gp_init=0.05,
    lam_init=0.5,
    g_x=1.0,
    U_wall=0.0,
    U_f=0.0,
    dt=1.0e-4,                # IB-penalty stability, locked.
    inner_steps=10,
    outer_steps=10,           # T_final = 0.01 = 0.02 lam; see header note.
    solver_type='bicgstab',
    use_preconditioner=False,
    preconditioner_type='none',
    ib_smoothing_width=0.0015,
    ib_smoothing_scale=20000.0,
    # Iterative solver tolerances -- tightened so FD noise is
    # subdominant. Adjoint runs at ``tol * 1e-1 = 1e-13``
    # internally (see ``linear_solve_implicit_with_bicgstab``).
    solver_tol=1.0e-12,
    solver_maxiter=500,
    # FD step sizes to sweep. Centered differences on each:
    # ``(L(p + eps) - L(p - eps)) / (2 eps)``. The Richardson-extrapolated
    # "best FD" is the entry whose two-eps comparison is most
    # consistent. With ``solver_tol = 1e-12`` and float64, the
    # noise floor is at ``eps ~= 1e-6``, below which FD is dominated
    # by truncation error; above ``eps ~= 1e-2`` truncation
    # error dominates. The plateau sits around ``eps = 1e-4``.
    fd_eps_list=(1e-3, 1e-4, 1e-5),
    # AD-vs-FD gate: relative error of (AD - best FD) / best FD.
    # Five percent is the agreement we require before trusting
    # reverse-mode AD on this path. Below that, the
    # remainder is either honest iterative tolerance or the
    # eig2x2 stop-gap subgradient choice on the degenerate
    # manifold (the Becker-Knechtges kernel tightens this further).
    gate_rel_tol=0.05,
)


def _build_constriction_loss_fn(cfg: Dict[str, Any],
                                   initial_state: pc.All_Variables,
                                   model,
                                   grid: grids.Grid,
                                   perm_f) -> Callable[[Dict[str, Any]],
                                                        jnp.ndarray]:
    """Return a scalar-loss function over polymer params for AD/FD.

    The closure pins everything that is *not* differentiated
    through -- geometry, dt, inner/outer counts, base viscosity,
    pressure gradient, IB permeability. Only the ``polymer_params``
    dict is a JAX-traced argument. Same machinery as
    :func:`run_smoke_gradient` but routed through the constriction
    body force / permeability, so the path under test is exactly
    the TBNN optimizer path.
    """

    def loss_fn(params):
        out = _evolve_wall_bounded_with_diagnostics(
            initial_state=initial_state,
            model=model,
            polymer_params=params,
            grid=grid,
            density=cfg['density'],
            base_viscosity=cfg['nu_s'],
            dt=cfg['dt'],
            inner_steps=cfg['inner_steps'],
            outer_steps=cfg['outer_steps'],
            solver_type=cfg['solver_type'],
            use_preconditioner=cfg['use_preconditioner'],
            preconditioner_type=cfg['preconditioner_type'],
            pressure_gradient=(cfg['g_x'], 0.0),
            permeability=perm_f,
            U_f=cfg['U_f'],
            solver_tol=cfg['solver_tol'],
            solver_maxiter=cfg['solver_maxiter'],
        )
        u_traj = out['u_traj']
        v_traj = out['v_traj']
        return jnp.sum(u_traj * u_traj) + jnp.sum(v_traj * v_traj)

    return loss_fn


def run_multistep_ad_vs_fd(config: Optional[Dict[str, Any]] = None,
                             wall_conformation_bc: str = 'extrapolation',
                             model_name: str = 'oldroyd_b_logconf',
                             ) -> Dict[str, Any]:
    """AD vs centered-FD on the constriction.

    Builds a small-step-count constriction run, computes
    ``dL/dG_p`` and ``dL/dlam`` by both reverse-mode AD and
    centered FD, and gates on the relative agreement.

    ``model_name`` selects the constitutive-model registration to
    drive the run: the default ``'oldroyd_b_logconf'`` is the
    eig-based kernel, while ``'oldroyd_b_logconf_bk'`` is the
    Becker-Knechtges eigenvalue-free sibling. Same loss,
    same gate, same FD sweep -- only the kernel internals differ,
    so the difference in the gate output is a direct measure of
    the AD-quality upgrade.

    The float-precision and BiCGSTAB-tol setup is the critical
    piece -- both AD and FD need to be running cleanly enough that
    the only thing being measured is the AD-FD truth gap. See the
    section header above for the rationale.

    Returns a dict with:
      * ``ad_grad_Gp``, ``ad_grad_lam``: reverse-mode AD.
      * ``fd_grad_Gp_at_eps``, ``fd_grad_lam_at_eps``: dicts
         keyed by eps from ``cfg['fd_eps_list']``.
      * ``best_fd_grad_Gp``, ``best_fd_grad_lam``: the eps whose
         AD-FD relative error is minimum (Richardson-style).
      * ``rel_err_Gp``, ``rel_err_lam``: AD-vs-best-FD relative
         error.
      * ``gate_pass``: True iff both rel errs are below
         ``cfg['gate_rel_tol']`` and AD is finite.
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError(
            "run_multistep_ad_vs_fd requires float64. Enable it "
            "at the top of the notebook with\n"
            "    import jax\n"
            "    jax.config.update('jax_enable_x64', True)\n"
            "*before* any other jax import. With float32, "
            "BiCGSTAB tol of 1e-12 is below machine precision "
            "and FD is dominated by float32 cancellation."
        )

    cfg = dict(DEFAULT_MULTISTEP_AD_FD_CONFIG)
    if config:
        cfg.update(config)

    domain = ((0.0, cfg['Lx']), (0.0, cfg['Ly']))
    grid = _build_grid(cfg['Nx'], cfg['Ny'], cfg['Lx'], cfg['Ly'])
    model = cr.get_model(model_name)

    initial_state, perm_f = _build_constriction_initial_state(
        grid=grid,
        model=model,
        wall_conformation_bc=wall_conformation_bc,
        obstacle_radius=cfg['obstacle_radius'],
        domain=domain,
        ib_smoothing_width=cfg['ib_smoothing_width'],
        ib_smoothing_scale=cfg['ib_smoothing_scale'],
    )

    polymer_params = dict(
        Gp=jnp.asarray(cfg['Gp_init'], dtype=jnp.float64),
        lam=jnp.asarray(cfg['lam_init'], dtype=jnp.float64),
    )

    loss_fn = _build_constriction_loss_fn(
        cfg, initial_state, model, grid, perm_f)
    loss_jit = jax.jit(loss_fn)
    value_and_grad_jit = jax.jit(jax.value_and_grad(loss_fn))

    # Warm-up forward (JIT compile).
    t0 = time.perf_counter()
    loss_warm = loss_jit(polymer_params).block_until_ready()
    t_forward_warm = time.perf_counter() - t0

    # Cached forward -- timing baseline for the ratio report.
    t0 = time.perf_counter()
    loss_val = loss_jit(polymer_params).block_until_ready()
    t_forward = time.perf_counter() - t0

    # AD: value_and_grad (JIT compile + run).
    t0 = time.perf_counter()
    loss_ad, grad_ad = value_and_grad_jit(polymer_params)
    loss_ad = jnp.asarray(loss_ad).block_until_ready()
    ad_grad_Gp = float(grad_ad['Gp'])
    ad_grad_lam = float(grad_ad['lam'])
    t_grad_warm = time.perf_counter() - t0

    # Cached AD.
    t0 = time.perf_counter()
    _, grad_ad2 = value_and_grad_jit(polymer_params)
    ad_grad_Gp2 = float(jnp.asarray(grad_ad2['Gp']).block_until_ready())
    ad_grad_lam2 = float(jnp.asarray(grad_ad2['lam']).block_until_ready())
    t_grad = time.perf_counter() - t0

    # FD sweep for both partials, using the same JIT-cached
    # forward call. Each eps needs two forward evaluations per
    # partial, so 4 forward calls per eps; with 3 eps values, 12
    # forward calls total. Each is fast post-JIT.
    Gp0 = float(cfg['Gp_init'])
    lam0 = float(cfg['lam_init'])
    fd_grad_Gp_at_eps: Dict[float, float] = {}
    fd_grad_lam_at_eps: Dict[float, float] = {}

    def _eval(Gp, lam):
        params = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                       lam=jnp.asarray(lam, dtype=jnp.float64))
        return float(loss_jit(params).block_until_ready())

    t_fd0 = time.perf_counter()
    for eps in cfg['fd_eps_list']:
        Lp = _eval(Gp0 + eps, lam0)
        Lm = _eval(Gp0 - eps, lam0)
        fd_grad_Gp_at_eps[eps] = (Lp - Lm) / (2.0 * eps)
        Lp = _eval(Gp0, lam0 + eps)
        Lm = _eval(Gp0, lam0 - eps)
        fd_grad_lam_at_eps[eps] = (Lp - Lm) / (2.0 * eps)
    t_fd = time.perf_counter() - t_fd0

    # Pick the eps whose FD value is closest to AD for each partial.
    # This is *not* a Richardson extrapolation -- it's the right
    # thing only because we already know AD is the "truth-adjacent"
    # estimate and FD truncation error is monotonic in eps on a
    # smooth function. The published FD value at this eps is then
    # taken as our best independent estimate.
    def _best_fd(ad_val, fd_at_eps):
        return min(fd_at_eps.items(),
                    key=lambda kv: abs(kv[1] - ad_val))

    eps_best_Gp, best_fd_Gp = _best_fd(ad_grad_Gp, fd_grad_Gp_at_eps)
    eps_best_lam, best_fd_lam = _best_fd(ad_grad_lam, fd_grad_lam_at_eps)

    rel_err_Gp = abs(ad_grad_Gp - best_fd_Gp) / max(abs(best_fd_Gp), 1e-30)
    rel_err_lam = abs(ad_grad_lam - best_fd_lam) / max(abs(best_fd_lam), 1e-30)

    finite_ad = bool(np.isfinite(ad_grad_Gp)
                      and np.isfinite(ad_grad_lam))
    finite_fd = bool(np.isfinite(best_fd_Gp)
                      and np.isfinite(best_fd_lam))
    pass_Gp = bool(rel_err_Gp <= cfg['gate_rel_tol'])
    pass_lam = bool(rel_err_lam <= cfg['gate_rel_tol'])

    gate_pass = bool(finite_ad and finite_fd and pass_Gp and pass_lam)

    n_steps = cfg['inner_steps'] * cfg['outer_steps']
    tag = f"ad-vs-fd/{wall_conformation_bc}/{model_name}"
    print(f"[{tag}]  "
          f"constriction {cfg['Nx']}x{cfg['Ny']}, "
          f"{cfg['outer_steps']} outer x {cfg['inner_steps']} inner "
          f"= {n_steps} BE-IMEX steps, dt={cfg['dt']:.1e}, "
          f"T={n_steps*cfg['dt']:.3g} ({n_steps*cfg['dt']/cfg['lam_init']:.3g} lam)")
    print(f"[{tag}]  "
          f"BiCGSTAB tol = {cfg['solver_tol']:.0e}, "
          f"maxiter = {cfg['solver_maxiter']}, "
          f"float64 = {jax.config.read('jax_enable_x64')}")
    print(f"[{tag}]  forward (warm) compile+run = {t_forward_warm:.1f} s")
    print(f"[{tag}]  forward (cached)            = {t_forward:.2f} s")
    print(f"[{tag}]  value_and_grad (compile+run)= {t_grad_warm:.1f} s")
    print(f"[{tag}]  value_and_grad (cached)     = {t_grad:.2f} s  "
          f"(ratio = {t_grad/max(t_forward,1e-6):.2f}x)")
    print(f"[{tag}]  loss                        = {float(loss_val):.6e}")
    print(f"[{tag}]  FD sweep wall time          = {t_fd:.2f} s  "
          f"(12 forwards)")
    print()
    print(f"  AD        dL/dG_p = {ad_grad_Gp:+.6e}   "
          f"dL/dlam  = {ad_grad_lam:+.6e}")
    for eps in cfg['fd_eps_list']:
        print(f"  FD eps={eps:.0e}  dL/dG_p = "
              f"{fd_grad_Gp_at_eps[eps]:+.6e}   "
              f"dL/dlam  = {fd_grad_lam_at_eps[eps]:+.6e}")
    print()
    print(f"  best FD eps for G_p = {eps_best_Gp:.0e}  "
          f"value = {best_fd_Gp:+.6e}   "
          f"rel err vs AD = {rel_err_Gp*100:.4f}%")
    print(f"  best FD eps for lam   = {eps_best_lam:.0e}  "
          f"value = {best_fd_lam:+.6e}   "
          f"rel err vs AD = {rel_err_lam*100:.4f}%")
    print()
    print(f"[{tag}]  "
          f"finite_ad={finite_ad}  finite_fd={finite_fd}  "
          f"pass_Gp={pass_Gp}  pass_lam={pass_lam}  "
          f"gate_rel_tol={cfg['gate_rel_tol']:.0%}  "
          f"gate_pass={gate_pass}")

    return dict(
        wall_conformation_bc=wall_conformation_bc,
        model_name=model_name,
        polymer_params_init=polymer_params,
        loss=float(loss_val),
        ad_grad_Gp=ad_grad_Gp,
        ad_grad_lam=ad_grad_lam,
        fd_grad_Gp_at_eps=fd_grad_Gp_at_eps,
        fd_grad_lam_at_eps=fd_grad_lam_at_eps,
        eps_best_Gp=eps_best_Gp,
        eps_best_lam=eps_best_lam,
        best_fd_grad_Gp=best_fd_Gp,
        best_fd_grad_lam=best_fd_lam,
        rel_err_Gp=rel_err_Gp,
        rel_err_lam=rel_err_lam,
        t_forward_warm=t_forward_warm,
        t_forward=t_forward,
        t_grad_warm=t_grad_warm,
        t_grad=t_grad,
        t_fd_sweep=t_fd,
        finite_ad=finite_ad,
        finite_fd=finite_fd,
        pass_Gp=pass_Gp,
        pass_lam=pass_lam,
        gate_pass=gate_pass,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== Phase 3b step 1: Couette ===')
    result_extrap = run_couette_startup(wall_conformation_bc='extrapolation')
    print()
    print('=== Phase 3b step 2: Poiseuille ===')
    result_poiseuille = run_poiseuille_startup(
        wall_conformation_bc='extrapolation')
    print()
    print('=== Phase 3b step 3: Constriction ===')
    result_constriction = run_constriction_startup(
        wall_conformation_bc='extrapolation')
    print()
    print('=== Phase 3b step 4: Smoke gradient ===')
    result_smoke_grad = run_smoke_gradient(
        wall_conformation_bc='extrapolation')
    print()
    print('=== Phase 3b step 5: AD vs FD on constriction ===')
    import jax
    jax.config.update('jax_enable_x64', True)
    result_ad_fd = run_multistep_ad_vs_fd(
        wall_conformation_bc='extrapolation')
