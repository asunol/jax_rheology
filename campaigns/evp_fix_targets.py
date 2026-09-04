#!/usr/bin/env python
"""Generate the truth targets, loss weights, and initial scalars for each arm.

Truth targets and BR-init depend on the drive set AND the horizon, so there
are four of each: {A, B} x {3 lambda, 7 lambda}. Everything is generated
through the EXISTING target-generation path -- ``ablation_targets_precheck``'s
``_truth_forward`` / ``_flow_rate_Q`` / ``_model_forward`` and the runner's own
``_br_init_from_targets`` -- so the pipeline is the same one that produced
``tbnn_evp_data/channel_ablation/ablation_targets.json`` for v2_prod2. Nothing
is reimplemented.

Per (drive set, horizon) it writes a targets JSON carrying:
  Q_truth, Q_scale      per drive (Q_scale = max_t |Q_truth(t)|)
  W_vel                 per drive: sum over trajectory AND cells of
                        (u_truth^2 + v_truth^2) -- the frozen weight base
  w_vel                 per drive: W_max / W_i, the ratio weight
  lambda0               = (sum_i w_i L_vel,i) / (sum_i L_Q,i) at OB-init,
                        i.e. v2_prod2's lambda0 definition with the weighting
                        applied. v2_prod2 used lambda_q = 1.0 x lambda0
                        (G4: the "multiple" is exactly 1), so the runner is
                        launched with --lambda-q equal to this number.
  br_init               BR scalars from THIS arm's own truth flow rates

Consistency check: drive set A at 3 lambda regenerates the config v2_prod2
ran, so its Q_truth must reproduce
``v2_prod2/batch_metrics.json::Q_truth`` to all 16 digits.

gpu_test. Eight truth forwards + eight OB-init forwards per arm-pair; minutes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from repo_paths import bootstrap, REPO_ROOT
bootstrap()

import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                       # noqa: E402
import numpy as np                                            # noqa: E402

import ablation_targets_precheck as atp                       # noqa: E402
import visco_opt_tbnn_evp_run as vevp                         # noqa: E402
import visco_tbnn as vt                                       # noqa: E402
from jax_rheology.models import tbnn_memory as tb  #  noqa: E402

OUT_DIR = './reference_values'
V2_BM = './tbnn_evp_data/v2_yield/v2_prod2/batch_metrics.json'

DRIVE_SETS = {'A': (1.8, 2.5, 4.0), 'B': (1.6, 1.8, 2.5)}
HORIZONS = {'3lam': 84, '7lam': 200}
Q_EPS = atp.Q_EPS


def _log(m):
    print(m, flush=True)


def _cfg_for(outer: int) -> dict:
    """atp._cfg() is the production config (DEFAULT_CHANNEL_CONFIG + Ny=64,
    Nx=32, solver_tol=1e-8); only the horizon moves per arm."""
    cfg = atp._cfg()
    cfg['outer_steps'] = int(outer)
    return cfg


def one_arm(dset: str, hkey: str) -> dict:
    drives = DRIVE_SETS[dset]
    outer = HORIZONS[hkey]
    cfg = _cfg_for(outer)
    T = cfg['outer_steps'] * cfg['inner_steps'] * cfg['dt']
    _log(f"\n{'='*70}\n=== arm-pair {dset}/{hkey}: drives={drives} "
         f"outer={outer} T={T:.4f}={T/atp.TRUTH_LAM:.3f}lam ===\n{'='*70}")
    _log(f"  cfg: Nx={cfg['Nx']} Ny={cfg['Ny']} Lx={cfg['Lx']} Ly={cfg['Ly']} "
         f"dt={cfg['dt']} inner={cfg['inner_steps']} tol={cfg['solver_tol']}")

    t0 = time.time()
    forcings = {}
    truths = {}
    for gx in drives:
        out = jax.jit(lambda g=gx: atp._truth_forward(cfg, g))()
        out['u_traj'].block_until_ready()
        u_t, v_t = out['u_traj'], out['v_traj']
        # Q_truth through the SAME reduction the precheck used (final-step
        # x-average, trapezoid in y). Q_scale is max_t |Q(t)|, the runner's
        # relative-Q denominator.
        Q = atp._flow_rate_Q(u_t, cfg)
        Q_traj = np.asarray(vevp._flow_rate_Q_traj(u_t, cfg))
        Q_scale = float(np.max(np.abs(Q_traj)))
        W = float(jnp.sum(u_t ** 2) + jnp.sum(v_t ** 2))
        u_prof = np.asarray(u_t[-1]).mean(axis=0)
        plug = atp._plug_halfwidth_gdot(u_prof, cfg)
        nan = bool(np.asarray(out['any_nan_traj']).any())
        min_lam = float(np.asarray(out['min_lam_traj']).min())
        yf = vt.saramito_yielded_fraction(
            dict(cfg, g_x=gx), atp.TRUTH_GP, atp.TRUTH_LAM, atp.TRUTH_TAU_Y,
            atp.TRUTH_NUS, geometry='channel')
        forcings[f'{gx:g}'] = dict(
            g_x=gx, Q_truth=Q, Q_scale=Q_scale, W_vel=W,
            plug_halfwidth=plug, y_p_theory=atp.TRUTH_TAU_Y / gx,
            yielded_fraction=float(yf['yielded_fraction']),
            any_nan=nan, min_eigA=min_lam,
            healthy=bool(not nan and min_lam > 0))
        truths[gx] = (u_t, v_t)
        _log(f"  g_x={gx:<4g} Q_truth={Q:+.16e}  Q_scale={Q_scale:.6e}  "
             f"W_vel={W:.6e}  plug={plug:.4f}  yielded={yf['yielded_fraction']:.1%}  "
             f"min_eigA={min_lam:.3e} nan={nan}")

    # ---- ratio weights -------------------------------------------------
    W_max = max(forcings[f'{g:g}']['W_vel'] for g in drives)
    for gx in drives:
        f = forcings[f'{gx:g}']
        f['w_vel'] = W_max / f['W_vel']
    _log(f"  W_max = {W_max:.6e}")
    for gx in drives:
        _log(f"  w_i(g_x={gx:g}) = {forcings[f'{gx:g}']['w_vel']:.6f}")

    # ---- lambda0 under the weighting -----------------------------------
    # Same OB-init definition v2_prod2's lambda0 used (atp: V1 unanchored
    # relu_annealed theta at Gp=lam=nu_s=1, model
    # tbnn_potential_free_logconf_bk_v2), so the only things that move are the
    # weighting, the drive set and the horizon. That path is yield_mode='off'
    # and is untouched by the Part-1 closure change.
    theta0, _ = tb.init_tbnn_theta(jax.random.PRNGKey(0),
                                   bound_c=tb.TBNN_DEFAULT_BOUND_C,
                                   anchored=False, mobility='relu_annealed')
    fit_init = {'theta': theta0, 'nu_s': jnp.asarray(1.0, dtype=jnp.float64)}
    Lv_w = Lv_raw = LQ = 0.0
    for gx in drives:
        f = forcings[f'{gx:g}']
        u_t, v_t = truths[gx]
        out_m = atp._model_forward(cfg, gx, fit_init)
        L_vel = float(jnp.sum((out_m['u_traj'] - u_t) ** 2)
                      + jnp.sum((out_m['v_traj'] - v_t) ** 2))
        Q_m = atp._flow_rate_Q(out_m['u_traj'], cfg)
        rel = (Q_m - f['Q_truth']) / max(abs(f['Q_truth']), Q_EPS)
        L_Q = float(rel ** 2)
        f['L_vel_init'] = L_vel
        f['L_vel_init_weighted'] = f['w_vel'] * L_vel
        f['L_Q_init'] = L_Q
        Lv_raw += L_vel
        Lv_w += f['w_vel'] * L_vel
        LQ += L_Q
        _log(f"  init g_x={gx:<4g} L_vel={L_vel:.6e} -> w*L_vel="
             f"{f['w_vel']*L_vel:.6e}   L_Q={L_Q:.6e}")
    lambda0_new = Lv_w / max(LQ, Q_EPS)
    lambda0_unw = Lv_raw / max(LQ, Q_EPS)
    _log(f"  lambda0_new = (sum w_i L_vel,i)/(sum L_Q,i) = {Lv_w:.6e}/{LQ:.6e}"
         f" = {lambda0_new:.10e}")
    _log(f"  (unweighted, for reference: {lambda0_unw:.10e})")

    # ---- BR-init from THIS arm's own truth flow rates -------------------
    Q_by_gx = {gx: forcings[f'{gx:g}']['Q_truth'] for gx in drives}
    br = vevp._br_init_from_targets(list(drives), Q_by_gx)
    prov = br['prov']
    _log(f"  BR-init: Gp0={br['Gp0']:.6f} lam0={br['lam0']:.6f} "
         f"nu_s0={br['nu_s0']:.6f} tau_y0={br['tau_y0']:.6f}  "
         f"(slope={prov['slope']:.6f} g_c0={prov.get('g_c0', float('nan')):.6f} "
         f"from Q({prov['forcing_mid']:g}), Q({prov['forcing_hi']:g}))")

    res = dict(
        healthy=all(f['healthy'] for f in forcings.values()),
        drive_set=dset, drive_set_values=list(drives), horizon=hkey,
        lambda0=float(lambda0_new), lambda0_unweighted=float(lambda0_unw),
        lambda_q_multiple=1.0,
        lambda_q=float(lambda0_new),   # multiple x lambda0_new, multiple == 1
        L_vel_init_sum_weighted=float(Lv_w),
        L_vel_init_sum_unweighted=float(Lv_raw),
        L_Q_init_sum=float(LQ), W_max=float(W_max),
        cfg=dict(Ny=cfg['Ny'], Nx=cfg['Nx'], Lx=cfg['Lx'], Ly=cfg['Ly'],
                 outer_steps=cfg['outer_steps'], dt=cfg['dt'],
                 inner_steps=cfg['inner_steps'],
                 solver_tol=cfg['solver_tol'], T_final=T,
                 T_lam=T / atp.TRUTH_LAM),
        truth=dict(Gp=atp.TRUTH_GP, lam=atp.TRUTH_LAM, nu_s=atp.TRUTH_NUS,
                   tau_y=atp.TRUTH_TAU_Y),
        forcings=forcings,
        br_init=dict(Gp0=float(br['Gp0']), lam0=float(br['lam0']),
                     nu_s0=float(br['nu_s0']), tau_y0=float(br['tau_y0']),
                     prov=prov),
        yield_pref_floor=float(tb._YIELD_PREF_FLOOR),
        walltime_s=time.time() - t0)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'evp_targets_geom{dset}_{hkey}.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)
    _log(f"  wrote {path}  ({res['walltime_s']:.0f}s)")
    return res


def consistency_check(res_A3: dict) -> dict:
    """Drive set A at 3 lambda regenerates the config v2_prod2 ran, so its
    Q_truth must reproduce v2_prod2/batch_metrics.json::Q_truth EXACTLY."""
    _log(f"\n{'='*70}\n=== Part 4 consistency check: A/3lam vs v2_prod2 "
         f"batch_metrics Q_truth ===\n{'='*70}")
    ref = json.load(open(V2_BM))['Q_truth']
    rows, ok = [], True
    for k, q_ref in ref.items():
        q_new = res_A3['forcings'][k]['Q_truth']
        bit = bool(np.float64(q_new) == np.float64(q_ref))
        rel = abs(q_new - q_ref) / max(abs(q_ref), 1e-300)
        ok = ok and bit
        rows.append(dict(g_x=k, Q_v2prod2=q_ref, Q_regen=q_new,
                         bit_identical=bit, rel=rel))
        _log(f"  g_x={k:<4s} v2_prod2={q_ref!r}")
        _log(f"           regen   ={q_new!r}   bit_identical={bit}  "
             f"rel={rel:.3e}")
    _log(f"  CONSISTENCY {'PASS (16/16 digits)' if ok else 'FAIL'}")
    return dict(rows=rows, pass_=ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arms', default='A/3lam,A/7lam,B/3lam,B/7lam')
    from jax_rheology.io.config import parse_with_config
    a, _cfg = parse_with_config(p)
    _log(f"[setup] device = {jax.devices()}")
    _log(f"[setup] _YIELD_PREF_FLOOR={tb._YIELD_PREF_FLOOR!r}")
    out = {}
    for spec in [s.strip() for s in a.arms.split(',') if s.strip()]:
        dset, hkey = spec.split('/')
        out[spec] = one_arm(dset, hkey)
    if 'A/3lam' in out:
        chk = consistency_check(out['A/3lam'])
        out['consistency_check'] = chk
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, 'evp_targets_consistency.json'), 'w') as f:
            json.dump(chk, f, indent=2)
        if not chk['pass_']:
            _log('\n*** Q_truth did NOT reproduce to 16 digits: the target '
                 'pipeline changed. STOPPING. ***')
            return 2

    _log(f"\n{'='*70}\n=== SUMMARY: w_i and lambda_q per arm-pair ===\n{'='*70}")
    for spec, r in out.items():
        if spec == 'consistency_check':
            continue
        w = {k: round(v['w_vel'], 6) for k, v in r['forcings'].items()}
        _log(f"  {spec:<9s} w_i={w}  lambda_q={r['lambda_q']:.6e}  "
             f"BR=({r['br_init']['Gp0']:.3f},{r['br_init']['lam0']:.3f},"
             f"{r['br_init']['nu_s0']:.3f},{r['br_init']['tau_y0']:.3f})")
    with open(os.path.join(OUT_DIR, 'evp_targets_summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
