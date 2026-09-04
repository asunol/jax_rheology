#!/usr/bin/env python
"""Elastoviscoplastic baseline diagnostics at the production resolution.

Six checks on the truth solve, run before committing to a fit:

  1. x-invariance of the channel solution
  2. insensitivity to domain size at fixed grid spacing
  3. plug resolution at each training drive
  4. horizon choice: plug width at three versus seven relaxation times
  5. insensitivity to the solver tolerance
  6. yielded fraction is neither ~0% nor ~100%, so the drive is informative

A seventh, timing, is a separate job and is only recorded as a placeholder
here. The targets JSON is regenerated at the same config. A failing check
means the configuration is wrong, not that the check needs loosening.
"""
from __future__ import annotations
import json
import os
import sys
import time

from repo_paths import bootstrap, REPO_ROOT
bootstrap()

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import analytic_limits_validation as p3b
import visco_families as vf
import visco_tbnn as vt
from jax_rheology.models import tbnn_memory as tb

OUT = 'work/evp_baseline'
os.makedirs(OUT, exist_ok=True)

TRUTH = dict(Gp=3.2, lam=0.7, nu_s=0.8, tau_y=1.45)
DRIVES = (1.8, 2.5, 4.0)          # production drives (no 1.3)
Q_EPS = 1e-9


def locked_cfg(**over):
    """Final locked config: DEFAULT_CHANNEL + Ny=128, outer=84, tol=1e-8."""
    cfg = dict(vf.DEFAULT_CHANNEL_CONFIG)
    cfg.update(Nx=32, Ny=128, Lx=1.0, Ly=2.0, dt=2.5e-3, inner_steps=10,
               outer_steps=84, solver_tol=1e-8,
               nu_s=TRUTH['nu_s'], Gp_init=TRUTH['Gp'], lam_init=TRUTH['lam'])
    cfg.update(over)
    return cfg


def truth_forward(cfg, g_x):
    grid, model, state, perm = vt._build_geometry(
        dict(cfg, g_x=g_x), 'saramito_logconf_bk_v2', 'channel')
    pp = dict(Gp=jnp.asarray(TRUTH['Gp']), lam=jnp.asarray(TRUTH['lam']),
              tau_y=jnp.asarray(TRUTH['tau_y']))
    return p3b._evolve_wall_bounded_with_diagnostics(
        initial_state=state, model=model, polymer_params=pp, grid=grid,
        density=cfg['density'], base_viscosity=TRUTH['nu_s'], dt=cfg['dt'],
        inner_steps=cfg['inner_steps'], outer_steps=cfg['outer_steps'],
        solver_type=cfg['solver_type'],
        use_preconditioner=cfg['use_preconditioner'],
        preconditioner_type=cfg['preconditioner_type'],
        pressure_gradient=(g_x, 0.0), permeability=perm, U_f=cfg['U_f'],
        solver_tol=cfg['solver_tol'], solver_maxiter=cfg['solver_maxiter'])


def flow_rate_Q(u_traj, cfg):
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    y = (jnp.arange(Ny, dtype=jnp.float64) + 0.5) * dy
    u_prof = jnp.mean(u_traj[-1], axis=0)
    return float(jnp.trapz(u_prof, y))


def plug_halfwidth_gdot(u_prof, cfg, frac=0.05):
    """LEGACY kinematic flat-core detector. Prefer plug_halfwidth_yield."""
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    gdot = np.zeros(Ny)
    gdot[1:-1] = (u_prof[2:] - u_prof[:-2]) / (2.0 * dy)
    gmax = max(float(np.abs(gdot).max()), 1e-30)
    low = np.abs(gdot) < frac * gmax
    jc = Ny // 2
    if not low[jc]:
        return 0.0, gdot, gmax
    lo = hi = jc
    while lo - 1 >= 0 and low[lo - 1]:
        lo -= 1
    while hi + 1 < Ny and low[hi + 1]:
        hi += 1
    return 0.5 * (hi - lo + 1) * dy, gdot, gmax


def plug_halfwidth_yield(Axx, Axy, Ayy, Azz, cfg, Gp=None, tau_y=None):
    """Plug half-width = central UNyielded band (|tau_d| <= tau_y). Gate-6 / B4 ruler."""
    from jax_rheology.models import tbnn_memory as tb
    Gp = TRUTH['Gp'] if Gp is None else Gp
    tau_y = TRUTH['tau_y'] if tau_y is None else tau_y
    dy = cfg['Ly'] / cfg['Ny']
    Aj = (jnp.asarray(Axx), jnp.asarray(Axy), jnp.asarray(Ayy), jnp.asarray(Azz))
    td = np.asarray(tb.saramito_tau_d_norm(*Aj, Gp))
    uny = td <= tau_y
    jc = cfg['Ny'] // 2
    if not uny[jc]:
        return 0.0, td, uny
    lo = hi = jc
    while lo - 1 >= 0 and uny[lo - 1]:
        lo -= 1
    while hi + 1 < cfg['Ny'] and uny[hi + 1]:
        hi += 1
    return 0.5 * (hi - lo + 1) * dy, td, uny


def x_rel_std(field, scale):
    """max over (t,y) of std_x / GLOBAL scale; field (T,Nx,Ny) or (Nx,Ny)."""
    a = np.asarray(field)
    if a.ndim == 2:
        a = a[None]
    std = np.std(a, axis=1)
    return float(np.max(std) / max(float(scale), 1e-30))


# ---------------------------------------------------------------------------
# Gate 1 -- x-invariance
# ---------------------------------------------------------------------------
def gate1_xinv():
    print('=== gate1: x-invariance (g_x=4, full T) ===', flush=True)
    cfg = locked_cfg()
    out = jax.jit(lambda: truth_forward(cfg, 4.0))()
    out['u_traj'].block_until_ready()
    u = np.asarray(out['u_traj']); v = np.asarray(out['v_traj'])
    Axx = np.asarray(out['A_xx_traj']); Axy = np.asarray(out['A_xy_traj'])
    Ayy = np.asarray(out['A_yy_traj']); Azz = np.asarray(out['A_zz_traj'])
    jc = u.shape[-1] // 2
    u_cl = float(np.max(np.abs(u[:, :, jc].mean(axis=1))))
    A_scale = float(np.max(np.stack([
        np.abs(Axx - 1.0), np.abs(Axy), np.abs(Ayy - 1.0), np.abs(Azz - 1.0)])))
    # GLOBAL scales: u_cl for velocities, max|A-I| for conformation; thr=1e-7
    thr = 1e-7
    rels = {
        'u': x_rel_std(u, u_cl), 'v': x_rel_std(v, u_cl),
        'A_xx': x_rel_std(Axx, A_scale), 'A_xy': x_rel_std(Axy, A_scale),
        'A_yy': x_rel_std(Ayy, A_scale), 'A_zz': x_rel_std(Azz, A_scale),
    }
    any_nan = bool(np.asarray(out['any_nan_traj']).any())
    ok = (not any_nan) and all(r < thr for r in rels.values())
    print(f'  any_nan={any_nan}  u_cl={u_cl:.3e} A_scale={A_scale:.3e}',
          flush=True)
    print(f'  rel_std_x/global: '
          + ' '.join(f'{k}={r:.3e}' for k, r in rels.items()), flush=True)
    print(f'  p_inst: N/A (channel evolve has no p_traj; body-force p~-g_x x)',
          flush=True)
    print(f'  GATE1: {"PASS" if ok else "FAIL"} (thr={thr:.0e})', flush=True)
    return dict(ok=ok, rels=rels, any_nan=any_nan, u_cl=u_cl, A_scale=A_scale,
                thr=thr, p_inst='N/A')


# ---------------------------------------------------------------------------
# Gate 2 -- domain-size Lx=2 Nx=64 vs Lx=1 Nx=32
# ---------------------------------------------------------------------------
def gate2_domain():
    print('=== gate2: domain-size Lx=2/Nx=64 vs Lx=1/Nx=32 (same dx) ===',
          flush=True)
    cfg1 = locked_cfg(Lx=1.0, Nx=32)
    cfg2 = locked_cfg(Lx=2.0, Nx=64)
    out1 = jax.jit(lambda: truth_forward(cfg1, 4.0))()
    out2 = jax.jit(lambda: truth_forward(cfg2, 4.0))()
    out1['u_traj'].block_until_ready(); out2['u_traj'].block_until_ready()
    # y-profiles: mean over x of final frame
    u1 = np.asarray(out1['u_traj'][-1]).mean(axis=0)
    u2 = np.asarray(out2['u_traj'][-1]).mean(axis=0)
    A1 = np.asarray(out1['A_xx_traj'][-1]).mean(axis=0)
    A2 = np.asarray(out2['A_xx_traj'][-1]).mean(axis=0)
    du = float(np.max(np.abs(u1 - u2)) / (np.max(np.abs(u1)) + 1e-30))
    dA = float(np.max(np.abs(A1 - A2)) / (np.max(np.abs(A1)) + 1e-30))
    thr = 1e-7  # ~=10x solver_tol; same bar as gate 1
    ok = du < thr and dA < thr
    print(f'  max_rel |u1-u2|={du:.3e}  |Axx1-Axx2|={dA:.3e}', flush=True)
    print(f'  GATE2: {"PASS" if ok else "FAIL"} (thr={thr:.0e})', flush=True)
    return dict(ok=ok, du=du, dA=dA, thr=thr)


# ---------------------------------------------------------------------------
# Gate 3 -- B4 plug-resolution at Ny=128, all three drives
# ---------------------------------------------------------------------------
def gate3_plug():
    print('=== gate3: B4 plug-resolution @ Ny=128 (3 drives) ===', flush=True)
    cfg = locked_cfg()
    dy = cfg['Ly'] / cfg['Ny']
    rows = {}
    all_ok = True
    for gx in DRIVES:
        out = jax.jit(lambda g=gx: truth_forward(cfg, g))()
        out['u_traj'].block_until_ready()
        u = np.asarray(out['u_traj'][-1]).mean(axis=0)
        Axx = np.asarray(out['A_xx_traj'][-1]).mean(axis=0)
        Axy = np.asarray(out['A_xy_traj'][-1]).mean(axis=0)
        Ayy = np.asarray(out['A_yy_traj'][-1]).mean(axis=0)
        Azz = np.asarray(out['A_zz_traj'][-1]).mean(axis=0)
        plug, td, uny = plug_halfwidth_yield(Axx, Axy, Ayy, Azz, cfg)
        # kinematic flat-core (legacy) for diagnosis only
        plug_g, gdot, gmax = plug_halfwidth_gdot(u, cfg, frac=0.05)
        any_nan = bool(np.asarray(out['any_nan_traj']).any())
        jc = cfg['Ny'] // 2
        inside = (float(np.abs(gdot[jc]) / gmax) if gmax > 0 else 0.0)
        y_p = TRUTH['tau_y'] / gx
        hw_from_yf = (1.0 - float((~uny).mean())) * (cfg['Ly'] / 2.0)
        plug_ok = plug > 0 and abs(plug - y_p) <= 2 * dy and (not any_nan)
        shoulders = float(np.max(td)) > TRUTH['tau_y']
        ok = plug_ok and shoulders
        all_ok = all_ok and ok
        rows[f'{gx:g}'] = dict(
            plug_hw=plug, plug_hw_gdot_legacy=plug_g, y_p=y_p, dy=dy,
            hw_from_yf=hw_from_yf, inside_gdot_frac=inside,
            any_nan=any_nan, shoulders=shoulders, ok=ok)
        print(f'  g_x={gx}: plug_hw={plug:.5f} (= {plug/dy:.2f} dy) '
              f'y_p={y_p:.4f} (1-yf)H={hw_from_yf:.5f} '
              f'gdot_legacy={plug_g:.5f} nan={any_nan} ok={ok}', flush=True)
    print(f'  GATE3: {"PASS" if all_ok else "FAIL"}  dy={dy}', flush=True)
    return dict(ok=all_ok, dy=dy, drives=rows)


# ---------------------------------------------------------------------------
# Gate 4 -- T-choice: 3lam vs 7lam plug width within 1.dy
# ---------------------------------------------------------------------------
def gate4_T():
    print('=== gate4: T-choice 3lam (outer=84) vs 7lam (outer=200) @ g_x=4 ===',
          flush=True)
    cfg3 = locked_cfg(outer_steps=84)    # T=2.1 = 3lam
    cfg7 = locked_cfg(outer_steps=200)   # T=5.0 ~= 7lam
    dy = cfg3['Ly'] / cfg3['Ny']
    out3 = jax.jit(lambda: truth_forward(cfg3, 4.0))()
    out7 = jax.jit(lambda: truth_forward(cfg7, 4.0))()
    out3['u_traj'].block_until_ready(); out7['u_traj'].block_until_ready()
    u3 = np.asarray(out3['u_traj'][-1]).mean(axis=0)
    u7 = np.asarray(out7['u_traj'][-1]).mean(axis=0)
    def _plug(out, cfg):
        return plug_halfwidth_yield(
            np.asarray(out['A_xx_traj'][-1]).mean(0),
            np.asarray(out['A_xy_traj'][-1]).mean(0),
            np.asarray(out['A_yy_traj'][-1]).mean(0),
            np.asarray(out['A_zz_traj'][-1]).mean(0), cfg)[0]
    p3 = _plug(out3, cfg3); p7 = _plug(out7, cfg7)
    uc3 = float(u3[cfg3['Ny'] // 2]); uc7 = float(u7[cfg7['Ny'] // 2])
    dplug = abs(p3 - p7)
    ok = dplug <= dy + 1e-15
    print(f'  plug 3lam={p3:.5f}  7lam={p7:.5f}  |Delta|={dplug:.5f}  1.dy={dy:.5f}',
          flush=True)
    print(f'  u_cl 3lam={uc3:.5f}  7lam={uc7:.5f}', flush=True)
    print(f'  GATE4: {"PASS" if ok else "FAIL"}', flush=True)
    return dict(ok=ok, plug_3lam=p3, plug_7lam=p7, dplug=dplug, dy=dy,
                u_cl_3=uc3, u_cl_7=uc7)


# ---------------------------------------------------------------------------
# Gate 5 -- tol 1e-8 vs 1e-12
# ---------------------------------------------------------------------------
def gate5_tol():
    print('=== gate5: tol 1e-8 vs 1e-12 @ g_x=4 ===', flush=True)
    cfg8 = locked_cfg(solver_tol=1e-8)
    cfg12 = locked_cfg(solver_tol=1e-12)
    out8 = jax.jit(lambda: truth_forward(cfg8, 4.0))()
    out12 = jax.jit(lambda: truth_forward(cfg12, 4.0))()
    out8['u_traj'].block_until_ready(); out12['u_traj'].block_until_ready()
    u8 = np.asarray(out8['u_traj'][-1]); u12 = np.asarray(out12['u_traj'][-1])
    du = float(np.max(np.abs(u8 - u12)) / (np.max(np.abs(u12)) + 1e-30))
    Q8 = flow_rate_Q(out8['u_traj'], cfg8)
    Q12 = flow_rate_Q(out12['u_traj'], cfg12)
    dQ = abs(Q8 - Q12) / (abs(Q12) + 1e-30)
    # discrimination scale: loss sees O(1e-2) relative Q errors typically;
    # require far below that -- use 1e-4 relative as a conservative bar
    ok = du < 1e-4 and dQ < 1e-4
    print(f'  max_rel |u|={du:.3e}  rel|Q|={dQ:.3e}  Q8={Q8:.6e} Q12={Q12:.6e}',
          flush=True)
    print(f'  GATE5: {"PASS" if ok else "FAIL"}', flush=True)
    return dict(ok=ok, du=du, dQ=dQ, Q8=Q8, Q12=Q12)


# ---------------------------------------------------------------------------
# Gate 6 -- yielded fraction not ~0% or ~100%
# ---------------------------------------------------------------------------
def gate6_yielded():
    print('=== gate6: yielded-fraction guard (3 drives) ===', flush=True)
    cfg = locked_cfg()
    rows = {}
    all_ok = True
    for gx in DRIVES:
        yf = vt.saramito_yielded_fraction(
            dict(cfg, g_x=gx), TRUTH['Gp'], TRUTH['lam'], TRUTH['tau_y'],
            TRUTH['nu_s'], geometry='channel')
        frac = float(yf['yielded_fraction'])
        ok = 0.05 < frac < 0.95
        all_ok = all_ok and ok
        rows[f'{gx:g}'] = dict(yielded_fraction=frac, ok=ok)
        print(f'  g_x={gx}: yielded={frac:.1%}  ok={ok}', flush=True)
    print(f'  GATE6: {"PASS" if all_ok else "FAIL"}', flush=True)
    return dict(ok=all_ok, drives=rows)


# ---------------------------------------------------------------------------
# Targets-json regen at Ny=128 (production drives only)
# ---------------------------------------------------------------------------
def regen_targets():
    print('=== regen targets-json @ Ny=128 ===', flush=True)
    cfg = locked_cfg()
    T = cfg['outer_steps'] * cfg['inner_steps'] * cfg['dt']
    forcings_out = {}
    t0 = time.time()
    for gx in DRIVES:
        out = jax.jit(lambda g=gx: truth_forward(cfg, g))()
        out['u_traj'].block_until_ready()
        u = np.asarray(out['u_traj'][-1]).mean(axis=0)
        Q = flow_rate_Q(out['u_traj'], cfg)
        plug, _, _ = plug_halfwidth_yield(
            np.asarray(out['A_xx_traj'][-1]).mean(0),
            np.asarray(out['A_xy_traj'][-1]).mean(0),
            np.asarray(out['A_yy_traj'][-1]).mean(0),
            np.asarray(out['A_zz_traj'][-1]).mean(0), cfg)
        any_nan = bool(np.asarray(out['any_nan_traj']).any())
        min_lam = float(np.asarray(out['min_lam_traj']).min())
        yf = vt.saramito_yielded_fraction(
            dict(cfg, g_x=gx), TRUTH['Gp'], TRUTH['lam'], TRUTH['tau_y'],
            TRUTH['nu_s'], geometry='channel')
        forcings_out[f'{gx:g}'] = dict(
            g_x=gx, Q_truth=Q, plug_halfwidth=plug,
            y_p_theory=TRUTH['tau_y'] / gx,
            yielded_fraction=float(yf['yielded_fraction']),
            any_nan=any_nan, min_eigA=min_lam, healthy=(not any_nan and min_lam > 0))
        print(f'  g_x={gx}: Q={Q:.6e} plug={plug:.5f} '
              f'yielded={yf["yielded_fraction"]:.1%}', flush=True)

    # lam0 from OB-init (gauge-fixed Gp=lam=1, nu_s=1) -- same protocol as
    # ablation_targets_precheck, but only over the 3 production drives.
    theta0, _ = tb.init_tbnn_theta(jax.random.PRNGKey(0),
                                   bound_c=tb.TBNN_DEFAULT_BOUND_C,
                                   anchored=False, mobility='relu_annealed')
    fit_init = {'theta': theta0, 'nu_s': jnp.asarray(1.0)}
    L_vel_sum = L_Q_sum = 0.0
    for gx in DRIVES:
        td = forcings_out[f'{gx:g}']
        out_t = truth_forward(cfg, gx)
        u_t, v_t = out_t['u_traj'], out_t['v_traj']
        grid, model, state, perm = vt._build_geometry(
            dict(cfg, g_x=gx), 'tbnn_potential_free_logconf_bk_v2', 'channel')
        pp = dict(Gp=jnp.asarray(1.0), lam=jnp.asarray(1.0),
                  theta=fit_init['theta'], tbnn_bound_c=tb.TBNN_DEFAULT_BOUND_C,
                  tbnn_kappa=1.0)
        out_m = p3b._evolve_wall_bounded_with_diagnostics(
            initial_state=state, model=model, polymer_params=pp, grid=grid,
            density=cfg['density'], base_viscosity=1.0, dt=cfg['dt'],
            inner_steps=cfg['inner_steps'], outer_steps=cfg['outer_steps'],
            solver_type=cfg['solver_type'],
            use_preconditioner=cfg['use_preconditioner'],
            preconditioner_type=cfg['preconditioner_type'],
            pressure_gradient=(gx, 0.0), permeability=perm, U_f=cfg['U_f'],
            solver_tol=cfg['solver_tol'], solver_maxiter=cfg['solver_maxiter'])
        L_vel = float(jnp.sum((out_m['u_traj'] - u_t) ** 2)
                      + jnp.sum((out_m['v_traj'] - v_t) ** 2))
        Q_m = flow_rate_Q(out_m['u_traj'], cfg)
        rel = (Q_m - td['Q_truth']) / max(abs(td['Q_truth']), Q_EPS)
        L_Q = float(rel ** 2)
        L_vel_sum += L_vel; L_Q_sum += L_Q
        print(f'  init g_x={gx}: L_vel={L_vel:.4e} L_Q={L_Q:.4e}', flush=True)
    lambda0 = L_vel_sum / max(L_Q_sum, Q_EPS)
    result = dict(
        healthy=True, lambda0=float(lambda0),
        L_vel_init_sum=float(L_vel_sum), L_Q_init_sum=float(L_Q_sum),
        cfg=dict(Ny=cfg['Ny'], Nx=cfg['Nx'], Lx=cfg['Lx'], Ly=cfg['Ly'],
                 outer_steps=cfg['outer_steps'], dt=cfg['dt'],
                 inner_steps=cfg['inner_steps'], solver_tol=cfg['solver_tol'],
                 T_final=T, T_lam=T / TRUTH['lam']),
        truth=TRUTH, forcings=forcings_out, walltime_s=time.time() - t0)
    path = os.path.join(OUT, 'final_targets.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  wrote {path}  lambda0={lambda0:.6e}', flush=True)
    return result


def main():
    print(f'[baseline] device={jax.devices()}', flush=True)
    gates = {}
    gates['1_xinv'] = gate1_xinv()
    gates['2_domain'] = gate2_domain()
    gates['3_plug'] = gate3_plug()
    gates['4_T'] = gate4_T()
    gates['5_tol'] = gate5_tol()
    gates['6_yielded'] = gate6_yielded()
    targets = regen_targets()
    # gate 7 (timing) is a separate job -- placeholder here
    gates['7_timing'] = dict(ok=None, note='see timing-probe job')
    all_ok = all(g.get('ok') for k, g in gates.items() if k != '7_timing')
    summary = dict(gates=gates, targets_lambda0=targets['lambda0'],
                   overall_pass=bool(all_ok),
                   config=targets['cfg'])
    path = os.path.join(OUT, 'phase0_gates.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[baseline] OVERALL={'PASS' if all_ok else 'FAIL'} -> {path}",
          flush=True)
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
