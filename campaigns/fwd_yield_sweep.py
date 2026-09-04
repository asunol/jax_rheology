#!/usr/bin/env python
"""Forward-only yield-transition sweep: Saramito truth + forced-yield TBNN demo.

Diagnoses numerics/signal (Q vs g_x, plug formation, steady convergence) and
proves TBNN mobility expressivity by injecting the Saramito yield law through
the TBNN forward path (demo-only; no training path touched).

Partition: real solves on gpu_test; aggregation is login-node only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

import jax_rheology.models.registry as cr
import jax_rheology.solvers.steppers as eqr
import jax_rheology.log_conformation as lc
import jax_rheology.models.tbnn_memory as tb
import jax_cfd.base.grids as grids
import jax_rheology.solvers.pressure as pressure_new
import analytic_limits_validation as p3b
import visco_families as vf

# ---------------------------------------------------------------------------
# Fixed truth and grid
# ---------------------------------------------------------------------------
TRUTH_GP = 3.2
TRUTH_LAM = 0.7
TRUTH_NUS = 0.8
TRUTH_TAU_Y = 1.45
G_CRIT = TRUTH_TAU_Y / 1.0          # H = Ly/2 = 1.0

SWEEP_GX_ALL = (0.5, 1.0, 1.3, 1.45, 1.6, 1.8, 2.5, 4.0, 6.0)
JOB_A_GX = (0.5, 1.0, 1.3, 1.45)
JOB_B_GX = (1.6, 1.8, 2.5, 4.0, 6.0)

CONV_RTOL = 1e-3
CONV_WINDOW_LAM = 1.0
T_MAX_LAM = 15.0
YIELD_THRESH = 1e-3
GDOT_PLUG_FRAC = 0.05


def _base_cfg() -> Dict[str, Any]:
    cfg = dict(vf.DEFAULT_CHANNEL_CONFIG)
    cfg.update(Ny=64, Nx=32, solver_tol=1e-8, nu_s=TRUTH_NUS,
               Gp_init=TRUTH_GP, lam_init=TRUTH_LAM,
               inner_steps=10, dt=2.5e-3)
    return cfg


def _outer_dt(cfg: Dict[str, Any]) -> float:
    return cfg['dt'] * cfg['inner_steps']


def _steps_per_lam(cfg: Dict[str, Any], lam: float = TRUTH_LAM) -> int:
    return max(1, int(round(lam / _outer_dt(cfg))))


def _max_outer_steps(cfg: Dict[str, Any]) -> int:
    return int(round(T_MAX_LAM * TRUTH_LAM / _outer_dt(cfg)))


# ---------------------------------------------------------------------------
# Demo-only forced-yield TBNN mobility (forward flag; not used in training)
# ---------------------------------------------------------------------------

def _forced_yield_relaxation_fn(A_xx, A_xy, A_yy, A_zz, velocity, dt, params):
    """Bypass the network mobility head: m0 = kappa_y, m1 = 0, A* = I."""
    del velocity
    lam = lc._params_get(params, 'lam')
    Gp = lc._params_get(params, 'Gp')
    tau_y = lc._params_get(params, 'tau_y')
    kappa_y = tb.saramito_kappa_y(A_xx, A_xy, A_yy, A_zz, Gp, tau_y)
    zero = jnp.zeros_like(A_xy)
    one = jnp.ones_like(A_xy)
    return lc._affine_exponential_relaxation_step(
        A_xx, A_xy, A_yy, A_zz,
        kappa_y, zero, kappa_y, kappa_y,
        one, zero, one, one,
        dt, lam)


def _make_forced_yield_tbnn_model() -> cr.ConstitutiveModel:
    """TBNN forward path with injected yield law; stress readout unchanged."""
    base = cr.get_model('tbnn_potential_logconf_bk_v2')
    evo = lc.make_logconf_evolution_fn(
        psi_kernel='bk', uc_method='analytic', advect_method='rk2',
        relaxation_fn=_forced_yield_relaxation_fn)
    return dataclasses.replace(base, name='_fwd_yield_demo_tbnn',
                               evolution_fn=evo)


def _build_channel_with_model(cfg: Dict[str, Any], model: cr.ConstitutiveModel):
    grid = p3b._build_grid(cfg['Nx'], cfg['Ny'], cfg['Lx'], cfg['Ly'])
    init_state = p3b._build_wall_bounded_initial_state(
        grid, 0.0, model, 'extrapolation')
    return grid, model, init_state, 0.0


def _make_chunk_runner(cfg: Dict[str, Any], model, polymer_params: Dict,
                       grid: grids.Grid, perm_f: float, g_x: float):
    """JIT-compiled outer-step chunk (same diagnostics driver as the analytic-limits wall-bounded scan)."""
    from jax_ib.base import advection
    import jax_cfd.base as cfd

    pressure_solve = pressure_new.solve_fast_diag_moving_wall

    def convect_fn(v):
        return tuple(advection.advect_upwind(u, v, cfg['dt']) for u in v)

    inner_stepper = cfd.funcutils.repeated(
        eqr.memory_be_imex_stepper(
            density=cfg['density'],
            dt=cfg['dt'],
            grid=grid,
            model=model,
            params=polymer_params,
            base_viscosity=cfg['nu_s'],
            convect=convect_fn,
            pressure_solve=pressure_solve,
            solver_type=cfg['solver_type'],
            pressure_gradient=[g_x, 0.0],
            permeability=perm_f,
            U_f=cfg['U_f'],
            use_preconditioner=cfg['use_preconditioner'],
            preconditioner_type=cfg['preconditioner_type'],
            solver_tol=cfg['solver_tol'],
            solver_maxiter=cfg['solver_maxiter'],
        ),
        steps=cfg['inner_steps'],
    )

    def _diagnostics(memory_fields):
        A_xx = memory_fields[0].array.data
        A_xy = memory_fields[1].array.data
        A_yy = memory_fields[2].array.data
        A_zz = memory_fields[3].array.data
        lam_x, lam_y, *_ = lc.eig2x2_symmetric(A_xx, A_xy, A_yy)
        min_lam = jnp.minimum(jnp.min(lam_x), jnp.min(lam_y))
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
            new_state.velocity[0].data,
            new_state.velocity[1].data,
            A_xx, A_xy, A_yy, A_zz,
            tau_xx.data, tau_xy.data, tau_yy.data,
            min_lam, min_trA, any_nan,
        )
        return new_state, frame

    def run_chunk(state, n_steps: int):
        final_state, frames = jax.lax.scan(outer_step, state, xs=None,
                                           length=n_steps)
        (u_traj, v_traj,
         A_xx_traj, A_xy_traj, A_yy_traj, A_zz_traj,
         tau_xx_traj, tau_xy_traj, tau_yy_traj,
         min_lam_traj, min_trA_traj, any_nan_traj) = frames
        return dict(
            final_state=final_state,
            u_traj=u_traj, v_traj=v_traj,
            A_xx_traj=A_xx_traj, A_xy_traj=A_xy_traj,
            A_yy_traj=A_yy_traj, A_zz_traj=A_zz_traj,
            tau_xx_traj=tau_xx_traj, tau_xy_traj=tau_xy_traj,
            tau_yy_traj=tau_yy_traj,
            min_lam_traj=min_lam_traj, min_trA_traj=min_trA_traj,
            any_nan_traj=any_nan_traj,
        )

    return jax.jit(run_chunk, static_argnums=(1,))


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def _u_profile_and_Q(u_final: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    y = (np.arange(Ny) + 0.5) * dy
    u_prof = np.asarray(u_final).mean(axis=0)
    Q = float(np.trapz(u_prof, y))
    return u_prof, Q


def _plug_halfwidth(u_prof: np.ndarray, cfg: Dict[str, Any]) -> float:
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    gdot = np.zeros(Ny)
    gdot[1:-1] = (u_prof[2:] - u_prof[:-2]) / (2.0 * dy)
    gmax = max(float(np.abs(gdot).max()), 1e-30)
    thresh = GDOT_PLUG_FRAC * gmax
    low_shear = np.abs(gdot) < thresh
    jc = Ny // 2
    if not low_shear[jc]:
        return 0.0
    lo = jc
    while lo - 1 >= 0 and low_shear[lo - 1]:
        lo -= 1
    hi = jc
    while hi + 1 < Ny and low_shear[hi + 1]:
        hi += 1
    return 0.5 * (hi - lo + 1) * dy


def _yielded_fraction(Axx, Axy, Ayy, Azz, Gp: float, tau_y: float,
                      closure: str) -> float:
    if closure == 'saramito':
        ky = np.asarray(tb.saramito_kappa_y(
            jnp.asarray(Axx), jnp.asarray(Axy),
            jnp.asarray(Ayy), jnp.asarray(Azz), Gp, tau_y))
        return float((ky > YIELD_THRESH).mean())
    # forced-yield TBNN: m0 = kappa_y
    ky = np.asarray(tb.saramito_kappa_y(
        jnp.asarray(Axx), jnp.asarray(Axy),
        jnp.asarray(Ayy), jnp.asarray(Azz), Gp, tau_y))
    return float((ky > YIELD_THRESH).mean())


def _measure_from_out(out: Dict[str, Any], cfg: Dict[str, Any], g_x: float,
                      closure: str, Q_hist: List[float],
                      steady_converged: bool, conv_ratio: float,
                      stop_reason: str, T_final: float,
                      walltime: float, n_outer: int) -> Dict[str, Any]:
    u_final = np.asarray(out['u_traj'][-1])
    u_prof, Q = _u_profile_and_Q(u_final, cfg)
    u_max = float(u_prof.max())

    Axx = np.asarray(out['A_xx_traj'][-1]).reshape(-1)
    Axy = np.asarray(out['A_xy_traj'][-1]).reshape(-1)
    Ayy = np.asarray(out['A_yy_traj'][-1]).reshape(-1)
    Azz = np.asarray(out['A_zz_traj'][-1]).reshape(-1)

    plug_hw = _plug_halfwidth(u_prof, cfg)
    y_p_theory = TRUTH_TAU_Y / g_x if g_x > G_CRIT else None
    yp_rel = (abs(plug_hw - y_p_theory) / max(y_p_theory, 1e-30)
              if y_p_theory is not None and plug_hw > 0 else None)

    nan_arr = np.asarray(out['any_nan_traj'])
    nan_step = int(np.argmax(nan_arr)) if nan_arr.any() else None

    min_eigA = float(np.asarray(out['min_lam_traj']).min())
    trA = (np.asarray(out['A_xx_traj'][-1])
           + np.asarray(out['A_yy_traj'][-1])
           + np.asarray(out['A_zz_traj'][-1]))
    max_trA = float(trA.max())

    y = ((np.arange(cfg['Ny']) + 0.5) * (cfg['Ly'] / cfg['Ny'])).tolist()

    return dict(
        closure=closure,
        g_x=g_x,
        Gp=TRUTH_GP, lam=TRUTH_LAM, nu_s=TRUTH_NUS, tau_y=TRUTH_TAU_Y,
        g_crit=G_CRIT,
        Q=Q, u_max=u_max,
        plug_halfwidth=plug_hw,
        y_p_theory=y_p_theory,
        yp_rel=yp_rel,
        yielded_fraction=_yielded_fraction(
            Axx, Axy, Ayy, Azz, TRUTH_GP, TRUTH_TAU_Y, closure),
        steady_converged=steady_converged,
        conv_ratio=conv_ratio,
        stop_reason=stop_reason,
        T_final=T_final,
        n_outer=n_outer,
        min_eigA=min_eigA,
        max_trA=max_trA,
        nan_step=nan_step,
        any_nan=bool(nan_arr.any()),
        walltime_s=walltime,
        Ny=cfg['Ny'], Nx=cfg['Nx'], dt=cfg['dt'],
        solver_tol=cfg['solver_tol'],
        y=y,
        u_profile=u_prof.tolist(),
        Q_hist=Q_hist,
    )


# ---------------------------------------------------------------------------
# Steady-state forward integration
# ---------------------------------------------------------------------------

def _evolve_to_steady(cfg: Dict[str, Any], model, polymer_params: Dict,
                      grid, init_state, perm_f: float, g_x: float,
                      closure: str) -> Dict[str, Any]:
    chunk_steps = _steps_per_lam(cfg)
    max_outer = _max_outer_steps(cfg)
    run_chunk = _make_chunk_runner(cfg, model, polymer_params, grid,
                                   perm_f, g_x)

    state = init_state
    keys = ('u_traj', 'v_traj', 'A_xx_traj', 'A_xy_traj', 'A_yy_traj',
            'A_zz_traj', 'tau_xx_traj', 'tau_xy_traj', 'tau_yy_traj',
            'min_lam_traj', 'min_trA_traj', 'any_nan_traj')
    acc: Dict[str, List] = {k: [] for k in keys}
    Q_hist: List[float] = []
    n_outer = 0
    steady_converged = False
    conv_ratio = float('inf')
    stop_reason = 'T_max'

    t0 = time.perf_counter()
    while n_outer < max_outer:
        n_this = min(chunk_steps, max_outer - n_outer)
        chunk = run_chunk(state, n_this)
        state = chunk['final_state']
        for k in keys:
            acc[k].append(np.asarray(chunk[k]))
        for ui in np.asarray(chunk['u_traj']):
            _, Qi = _u_profile_and_Q(ui, cfg)
            Q_hist.append(Qi)
        n_outer += n_this

        if np.asarray(chunk['any_nan_traj']).any():
            stop_reason = 'nan'
            break

        if len(Q_hist) > chunk_steps:
            conv_ratio = (abs(Q_hist[-1] - Q_hist[-1 - chunk_steps])
                          / max(abs(Q_hist[-1]), 1e-9))
            if conv_ratio < CONV_RTOL:
                steady_converged = True
                stop_reason = 'steady'
                break

    walltime = time.perf_counter() - t0
    T_final = n_outer * _outer_dt(cfg)

    out: Dict[str, Any] = {'final_state': state}
    for k in keys:
        out[k] = np.concatenate(acc[k], axis=0) if acc[k] else np.array([])

    meta = _measure_from_out(
        out, cfg, g_x, closure, Q_hist, steady_converged, conv_ratio,
        stop_reason, T_final, walltime, n_outer)
    out['metrics'] = meta
    return out


def _polymer_params(closure: str) -> Dict[str, Any]:
    p = dict(Gp=jnp.asarray(TRUTH_GP, dtype=jnp.float64),
             lam=jnp.asarray(TRUTH_LAM, dtype=jnp.float64))
    if closure in ('saramito', 'tbnn_yield'):
        p['tau_y'] = jnp.asarray(TRUTH_TAU_Y, dtype=jnp.float64)
    if closure == 'tbnn_yield':
        theta, _ = tb.init_tbnn_theta(jax.random.PRNGKey(0),
                                      bound_c=tb.TBNN_DEFAULT_BOUND_C)
        p['theta'] = theta
        p['tbnn_bound_c'] = float(tb.TBNN_DEFAULT_BOUND_C)
    return p


def _model_for_closure(closure: str):
    if closure == 'saramito':
        return cr.get_model('saramito_logconf_bk_v2')
    if closure == 'tbnn_yield':
        return _make_forced_yield_tbnn_model()
    raise ValueError(f"unknown closure {closure}")


def run_one(g_x: float, closure: str, out_dir: str) -> Dict[str, Any]:
    cfg = _base_cfg()
    cfg['g_x'] = g_x
    model = _model_for_closure(closure)
    grid, model, init_state, perm_f = _build_channel_with_model(cfg, model)
    params = _polymer_params(closure)

    print(f"  [{closure}] g_x={g_x}  Ny={cfg['Ny']}  "
          f"window={_steps_per_lam(cfg)} outer  T_max={T_MAX_LAM}lam ...",
          flush=True)
    out = _evolve_to_steady(cfg, model, params, grid, init_state, perm_f,
                            g_x, closure)
    m = out['metrics']
    print(f"    Q={m['Q']:.6e}  u_max={m['u_max']:.4e}  "
          f"plug_hw={m['plug_halfwidth']:.4f}  yielded={m['yielded_fraction']:.2%}  "
          f"conv={m['steady_converged']} ({m['stop_reason']}, "
          f"ratio={m['conv_ratio']:.2e}, T={m['T_final']:.3f})  "
          f"min_eigA={m['min_eigA']:.3e}  nan={m['any_nan']}  "
          f"wall={m['walltime_s']:.1f}s", flush=True)

    prefix = 'saramito' if closure == 'saramito' else 'tbnn_yield'
    gx_tag = f"{g_x:g}".replace('.', 'p')
    path = os.path.join(out_dir, f"{prefix}_gx{gx_tag}.json")
    with open(path, 'w') as f:
        json.dump(m, f, indent=2)
    np.savez(os.path.join(out_dir, f"{prefix}_gx{gx_tag}_profile.npz"),
             y=np.asarray(m['y']), u_profile=np.asarray(m['u_profile']))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', choices=('A', 'B'), required=True,
                    help='A: low/near-yield gx; B: above-yield gx')
    ap.add_argument('--out-dir', type=str, default='./work/fwd_sweep')
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(ap)

    os.makedirs(args.out_dir, exist_ok=True)
    gx_list = JOB_A_GX if args.job == 'A' else JOB_B_GX
    closures = ('saramito', 'tbnn_yield')

    print(f"=== fwd_yield_sweep job {args.job}  gx={gx_list}  "
          f"closures={closures}  out={args.out_dir} ===", flush=True)
    t0 = time.perf_counter()
    results = []
    for g_x in gx_list:
        for closure in closures:
            results.append(run_one(g_x, closure, args.out_dir))
    print(f"=== job {args.job} done  wall={time.perf_counter()-t0:.1f}s ===",
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
