#!/usr/bin/env python
"""Check the elastoviscoplastic training targets before launching a fit.

Regenerates Saramito truth (velocity + Q) at each ablation forcing at the FIT
horizon T~=3lam, checks numerics/plug health, computes lam0 from OB-init losses.
"""
from __future__ import annotations

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

import analytic_limits_validation as p3b
import visco_families as vf
import visco_tbnn as vt
from jax_rheology.models import tbnn_memory as tb

TRUTH_GP = 3.2
TRUTH_LAM = 0.7
TRUTH_NUS = 0.8
TRUTH_TAU_Y = 1.45
G_CRIT = TRUTH_TAU_Y / 1.0
FORCINGS = (1.3, 1.8, 2.5, 4.0)
Q_EPS = 1e-9


def _cfg():
    cfg = dict(vf.DEFAULT_CHANNEL_CONFIG)
    cfg.update(Ny=64, Nx=32, outer_steps=84, solver_tol=1e-8,
               nu_s=TRUTH_NUS, Gp_init=TRUTH_GP, lam_init=TRUTH_LAM)
    return cfg


def _flow_rate_Q(u_traj, cfg):
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    y = (jnp.arange(Ny, dtype=jnp.float64) + 0.5) * dy
    u_prof = jnp.mean(u_traj[-1], axis=0)
    return float(jnp.trapz(u_prof, y))


def _plug_halfwidth_gdot(u_prof, cfg, frac=0.05):
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    gdot = np.zeros(Ny)
    gdot[1:-1] = (u_prof[2:] - u_prof[:-2]) / (2.0 * dy)
    gmax = max(float(np.abs(gdot).max()), 1e-30)
    low = np.abs(gdot) < frac * gmax
    jc = Ny // 2
    if not low[jc]:
        return 0.0
    lo = hi = jc
    while lo - 1 >= 0 and low[lo - 1]:
        lo -= 1
    while hi + 1 < Ny and low[hi + 1]:
        hi += 1
    return 0.5 * (hi - lo + 1) * dy


def _truth_forward(cfg, g_x):
    grid, model, state, perm = vt._build_geometry(
        dict(cfg, g_x=g_x), 'saramito_logconf_bk_v2', 'channel')
    pp = dict(Gp=jnp.asarray(TRUTH_GP, dtype=jnp.float64),
              lam=jnp.asarray(TRUTH_LAM, dtype=jnp.float64),
              tau_y=jnp.asarray(TRUTH_TAU_Y, dtype=jnp.float64))
    return p3b._evolve_wall_bounded_with_diagnostics(
        initial_state=state, model=model, polymer_params=pp, grid=grid,
        density=cfg['density'], base_viscosity=TRUTH_NUS, dt=cfg['dt'],
        inner_steps=cfg['inner_steps'], outer_steps=cfg['outer_steps'],
        solver_type=cfg['solver_type'],
        use_preconditioner=cfg['use_preconditioner'],
        preconditioner_type=cfg['preconditioner_type'],
        pressure_gradient=(g_x, 0.0), permeability=perm, U_f=cfg['U_f'],
        solver_tol=cfg['solver_tol'], solver_maxiter=cfg['solver_maxiter'])


def _model_forward(cfg, g_x, fit, model_name='tbnn_potential_free_logconf_bk_v2'):
    grid, model, state, perm = vt._build_geometry(
        dict(cfg, g_x=g_x), model_name, 'channel')
    pp = dict(Gp=jnp.asarray(1.0, dtype=jnp.float64),
              lam=jnp.asarray(1.0, dtype=jnp.float64),
              theta=fit['theta'], tbnn_bound_c=tb.TBNN_DEFAULT_BOUND_C,
              tbnn_kappa=1.0)
    nu = float(fit['nu_s'])
    return p3b._evolve_wall_bounded_with_diagnostics(
        initial_state=state, model=model, polymer_params=pp, grid=grid,
        density=cfg['density'], base_viscosity=nu, dt=cfg['dt'],
        inner_steps=cfg['inner_steps'], outer_steps=cfg['outer_steps'],
        solver_type=cfg['solver_type'],
        use_preconditioner=cfg['use_preconditioner'],
        preconditioner_type=cfg['preconditioner_type'],
        pressure_gradient=(g_x, 0.0), permeability=perm, U_f=cfg['U_f'],
        solver_tol=cfg['solver_tol'], solver_maxiter=cfg['solver_maxiter'])


def main():
    out_dir = './work/evp_channel_ablation'
    os.makedirs(out_dir, exist_ok=True)
    cfg = _cfg()
    T = cfg['outer_steps'] * cfg['inner_steps'] * cfg['dt']
    print(f"=== ablation target precheck  T={T:.3f}={T/TRUTH_LAM:.2f}lam  "
          f"forcings={FORCINGS} ===", flush=True)

    forcings_out = {}
    degraded = False
    t0 = time.time()
    for gx in FORCINGS:
        out = jax.jit(lambda g=gx: _truth_forward(cfg, g))()
        out['u_traj'].block_until_ready()
        u = np.asarray(out['u_traj'][-1]).mean(axis=0)
        Q = _flow_rate_Q(out['u_traj'], cfg)
        plug_hw = _plug_halfwidth_gdot(u, cfg)
        y_p = TRUTH_TAU_Y / gx
        nan_arr = np.asarray(out['any_nan_traj'])
        any_nan = bool(nan_arr.any())
        min_lam = float(np.asarray(out['min_lam_traj']).min())
        yf = vt.saramito_yielded_fraction(dict(cfg, g_x=gx), TRUTH_GP, TRUTH_LAM,
                                          TRUTH_TAU_Y, TRUTH_NUS, geometry='channel')
        ok = (not any_nan) and min_lam > 0
        if gx >= 4.0 - 1e-9:
            # Anchor forcing: plug should be ~formed by T~=3lam (B4-reduced bar).
            if plug_hw > 0:
                yp_rel = abs(plug_hw - y_p) / y_p
                ok = ok and yp_rel < 0.25 and Q > 0.2
        elif gx < G_CRIT:
            # Sub-yield arrest anchor: |Q| small at T~=3lam (transient; not 15lam steady).
            ok = ok and abs(Q) < 0.2
        else:
            # Mid forcings at T~=3lam: plug not fully developed -- require yield
            # signal + forward flow direction only (no plug-width bar).
            ok = ok and Q > -0.05 and yf['yielded_fraction'] > 0.08
        if not ok:
            degraded = True
        forcings_out[f"{gx:g}"] = dict(
            g_x=gx, Q_truth=Q, plug_halfwidth=plug_hw,
            y_p_theory=y_p if gx > G_CRIT else None,
            yielded_fraction=float(yf['yielded_fraction']),
            any_nan=any_nan, min_eigA=min_lam, healthy=ok)
        print(f"  g_x={gx:g}: Q={Q:.4e} plug_hw={plug_hw:.3f} "
              f"y_p={y_p:.3f} yielded={yf['yielded_fraction']:.1%} "
              f"min_eigA={min_lam:.3e} nan={any_nan} healthy={ok}", flush=True)

    if degraded:
        print("HALT: degraded target(s) -- do not launch fits.", flush=True)
        path = os.path.join(out_dir, 'ablation_targets.json')
        with open(path, 'w') as f:
            json.dump(dict(healthy=False, forcings=forcings_out), f, indent=2)
        return 2

    # lam0 from OB-init losses (gauge-fixed: Gp=lam=1, nu_s=1)
    theta0, _ = tb.init_tbnn_theta(jax.random.PRNGKey(0),
                                   bound_c=tb.TBNN_DEFAULT_BOUND_C,
                                   anchored=False, mobility='relu_annealed')
    fit_init = {'theta': theta0,
                'nu_s': jnp.asarray(1.0, dtype=jnp.float64)}
    L_vel_sum = 0.0
    L_Q_sum = 0.0
    for gx in FORCINGS:
        td = forcings_out[f"{gx:g}"]
        out_t = _truth_forward(cfg, gx)
        u_t = out_t['u_traj']; v_t = out_t['v_traj']
        out_m = _model_forward(cfg, gx, fit_init)
        L_vel = float(jnp.sum((out_m['u_traj'] - u_t) ** 2)
                      + jnp.sum((out_m['v_traj'] - v_t) ** 2))
        Q_m = _flow_rate_Q(out_m['u_traj'], cfg)
        rel = (Q_m - td['Q_truth']) / max(abs(td['Q_truth']), Q_EPS)
        L_Q = float(rel ** 2)
        L_vel_sum += L_vel
        L_Q_sum += L_Q
        print(f"  init g_x={gx:g}: L_vel={L_vel:.4e} L_Q={L_Q:.4e}", flush=True)
    lambda0 = L_vel_sum / max(L_Q_sum, Q_EPS)
    print(f"lam0 = SigmaL_vel/SigmaL_Q = {L_vel_sum:.4e}/{L_Q_sum:.4e} = {lambda0:.6e}",
          flush=True)

    result = dict(
        healthy=True, lambda0=float(lambda0),
        L_vel_init_sum=float(L_vel_sum), L_Q_init_sum=float(L_Q_sum),
        cfg=dict(Ny=cfg['Ny'], Nx=cfg['Nx'], outer_steps=cfg['outer_steps'],
                 dt=cfg['dt'], solver_tol=cfg['solver_tol'],
                 T_final=T, T_lam=T / TRUTH_LAM),
        truth=dict(Gp=TRUTH_GP, lam=TRUTH_LAM, nu_s=TRUTH_NUS, tau_y=TRUTH_TAU_Y),
        forcings=forcings_out, walltime_s=time.time() - t0)
    path = os.path.join(out_dir, 'ablation_targets.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"PASS -- wrote {path}  wall={result['walltime_s']:.1f}s", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
