#!/usr/bin/env python
"""Evaluate the eight elastoviscoplastic training arms on one fixed protocol.

Every arm is scored identically, at a FIXED horizon with NO early stopping.
Per-closure early stopping is what made the original flow curve compare the
two closures at different times, so ``efd.run_forward(..., early_stop=False)``
is used throughout and the truth is run under the identical protocol.

Per arm:
  * flow curve over the full ladder, truth-simulated vs learned-simulated at
    the SAME horizon
  * held-out drives g_x = 1.3 (arrest; taken to 30 lambda), 1.45
    (near-critical, never trained), 6.0 (extrapolation, and whether the NaN
    at 8.04 lambda is gone)
  * recovery: SIGNED relative error on Gp, lam, nu_s, tau_y
  * plug, KINEMATIC: flat-core half-width from |du/dy| and the residual
    |du/dy| in the core, against the truth's. The stress-ruler IoU is
    recorded as a SECONDARY diagnostic only -- the pre-fix closure reported
    81 % of the half-channel unyielded while shearing throughout, so mask
    agreement is not evidence of a plug.
  * health: min eig(A), max|tau_d|, NaN, per drive
  * the w_i and lambda_q the arm actually used

Truth forwards are shared across arms (one per drive) since the truth does not
depend on the arm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from repo_paths import bootstrap, REPO_ROOT
bootstrap()

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                       # noqa: E402

import ablation_targets_precheck as atp                       # noqa: E402
import evp_forward_diag as efd                                # noqa: E402
from jax_rheology.models import registry as cr  #  noqa: E402
from jax_rheology.models import tbnn_memory as tb  #  noqa: E402
from visco_opt_tbnn_run import theta_from_named_arrays        # noqa: E402

ROOT = REPO_ROOT
FIT_DIR = ROOT / 'work/evp_channel'
OUT = ROOT / 'work/evp_eval'
TARGETS = ROOT / 'reference_values'

TRUTH = dict(Gp=3.2, lam=0.7, nu_s=0.8, tau_y=1.45)
LADDER = (0.5, 1.0, 1.3, 1.45, 1.6, 1.8, 2.5, 4.0, 6.0)
HELD_OUT = (1.3, 1.45, 6.0)
EVAL_LAM = 15.0        # fixed evaluation horizon for the whole ladder
ARREST_LAM = 30.0      # sub-yield drives get the long tail
SUBYIELD = (0.5, 1.0, 1.3)

ARMS = [f'evp_fix_{d}_{h}_{i}' for d in ('A', 'B')
        for h in ('3lam', '7lam') for i in ('agn', 'br')]


def _log(m):
    print(m, flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


# ---------------------------------------------------------------------------
def load_arm_closure(run: str):
    """Frozen closure for one fitted arm. Mirrors
    evp_learned_flowcurve._load_learned but on an arbitrary run directory, so
    the checkpoint's own recorded flags (anchored/mobility/yield_mode) drive
    the registration rather than an assumption."""
    ck = FIT_DIR / run / 'theta_checkpoint.npz'
    z = np.load(ck, allow_pickle=False)
    heads = [str(h) for h in z['ckpt_heads']]
    nlayers = {h: int(n) for h, n in zip(heads, z['ckpt_nlayers'])}
    theta = theta_from_named_arrays(z, heads, nlayers)
    v = {'Gp': float(z['ckpt_Gp_fit']), 'lam': float(z['ckpt_lam_fit']),
         'nu_s': float(z['ckpt_nu_s']), 'tau_y': float(z['ckpt_tau_y_fit']),
         'bound_c': float(z['ckpt_bound_c']),
         'kappa': float(z['ckpt_kappa_final']),
         'yield_mode': str(z['ckpt_yield_mode']),
         'mobility': str(z['ckpt_mobility']),
         'anchored': bool(z['ckpt_anchored'])}
    v['yield_pref_floor'] = (float(z['ckpt_yield_pref_floor'])
                             if 'ckpt_yield_pref_floor' in z else None)
    params = {'Gp': jnp.asarray(v['Gp'], dtype=jnp.float64),
              'lam': jnp.asarray(v['lam'], dtype=jnp.float64),
              'tau_y': jnp.asarray(v['tau_y'], dtype=jnp.float64),
              'theta': theta, 'tbnn_bound_c': v['bound_c'],
              'tbnn_kappa': v['kappa']}
    return dict(name=run,
                model=cr.get_model('tbnn_potential_yield_logconf_bk_v2'),
                params=params, Gp=v['Gp'], lam=v['lam'], nu_s=v['nu_s'],
                tau_y=v['tau_y']), v


def kinematic_plug(u_prof: np.ndarray, cfg, frac: float = 0.05):
    """Flat-core half-width and residual |du/dy| in the core.

    The half-width reuses the production kinematic detector
    ``ablation_targets_precheck._plug_halfwidth_gdot`` (|du/dy| below `frac` of
    its max, contiguous about the centreline). The residual shear rate is the
    mean |du/dy| over that core, which is the number that actually says
    whether the core is a plug or merely a gentle parabola: a smooth parabola
    with no plug still registers a nonzero flat-core half-width under any
    threshold-based detector, but its core |du/dy| does not vanish.
    """
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    hw = atp._plug_halfwidth_gdot(u_prof, cfg, frac=frac)
    gdot = np.zeros(Ny)
    gdot[1:-1] = (u_prof[2:] - u_prof[:-2]) / (2.0 * dy)
    jc = Ny // 2
    n_core = max(int(round(hw / dy)), 1)
    lo, hi = max(jc - n_core, 0), min(jc + n_core + 1, Ny)
    core = np.abs(gdot[lo:hi])
    umax = max(float(np.abs(u_prof).max()), 1e-300)
    return dict(halfwidth=float(hw),
                core_absdudy_mean=float(core.mean()) if core.size else 0.0,
                core_absdudy_max=float(core.max()) if core.size else 0.0,
                # normalised: |du/dy|_core / (u_max / H), dimensionless
                core_shear_norm=float(core.mean() / (umax / (0.5 * Ly))),
                gdot_absmax=float(np.abs(gdot).max()))


def run_ladder(closure, cfg, tag: str, out_dir: Path, drives=LADDER):
    """One closure across the ladder at the FIXED horizon, no early stopping."""
    spl = efd.steps_per_lam(cfg)
    res = {}
    for gx in drives:
        n_lam = ARREST_LAM if gx in SUBYIELD else EVAL_LAM
        n_outer = int(round(n_lam * spl))
        out, meta = efd.run_forward(closure, gx, cfg, n_outer,
                                    label=f'{tag}_gx{efd.tag(gx)}',
                                    early_stop=False)
        u_last = out['u_prof'][-1]
        kin = kinematic_plug(u_last, cfg)
        res[f'{gx:g}'] = dict(
            g_x=gx, T_lam=n_lam, Q=float(out['Q'][-1]),
            Q_at=dict(),
            conv_ratio=(float(out['conv_ratio'][-1])
                        if np.isfinite(out['conv_ratio'][-1]) else None),
            kinematic=kin,
            stress_plug_halfwidth=float(out['plug'][-1]),
            yielded_frac=float(out['yielded_frac'][-1]),
            unyielded_mask=out['unyielded'][-1].tolist(),
            min_eig=float(out['min_eig'].min()),
            max_td=float(out['td_max'].max()),
            any_nan=bool(out['any_nan'].any()),
            nan_at_lam=(float(np.argmax(out['any_nan']) + 1) / spl
                        if out['any_nan'].any() else None),
            stop_reason=meta['stop_reason'], wall_s=meta['walltime_s'])
        for l in (3, 7, 15, 30):
            if l <= n_lam:
                i = min(int(round(l * spl)) - 1, len(out['Q']) - 1)
                res[f'{gx:g}']['Q_at'][f'{l}'] = float(out['Q'][i])
        np.savez_compressed(out_dir / f'{tag}_gx{efd.tag(gx)}.npz',
                            Q=out['Q'], u_prof=out['u_prof'],
                            plug=out['plug'], yielded_frac=out['yielded_frac'],
                            unyielded=out['unyielded'], min_eig=out['min_eig'],
                            td_max=out['td_max'], t=out['t'],
                            conv_ratio=out['conv_ratio'],
                            meta=json.dumps(_jsonable(meta)))
        r = res[f'{gx:g}']
        _log(f"    g_x={gx:<5g} Q={r['Q']:+.6e}  kin_plug={kin['halfwidth']:.4f} "
             f"core|du/dy|={kin['core_absdudy_mean']:.3e} "
             f"(norm {kin['core_shear_norm']:.3e})  "
             f"stress_plug={r['stress_plug_halfwidth']:.4f} "
             f"min_eig={r['min_eig']:.3e} max|td|={r['max_td']:.3f} "
             f"nan={r['any_nan']}  {r['wall_s']:.0f}s")
    return res


def iou(a, b):
    a = np.asarray(a, dtype=bool); b = np.asarray(b, dtype=bool)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--skip-truth', action='store_true')
    from jax_rheology.io.config import parse_with_config
    a, _cfg = parse_with_config(p)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[setup] device = {jax.devices()}")
    _log(f"[setup] _YIELD_PREF_FLOOR={tb._YIELD_PREF_FLOOR!r}")
    cfg = efd.prod_cfg()
    _log(f"[setup] eval cfg Nx={cfg['Nx']} Ny={cfg['Ny']} dt={cfg['dt']} "
         f"inner={cfg['inner_steps']} tol={cfg['solver_tol']}  "
         f"FIXED horizon {EVAL_LAM}lam ({ARREST_LAM}lam sub-yield), "
         f"no early stopping")

    # ---- truth, once, shared across arms ------------------------------
    tpath = a.out_dir / 'truth_ladder.json'
    if tpath.exists() and a.skip_truth:
        truth = json.load(open(tpath))
        _log(f"[truth] reusing {tpath}")
    else:
        _log('\n=== truth ladder (shared) ===')
        t0 = time.time()
        truth = run_ladder(efd.truth_closure(), cfg, 'truth', a.out_dir)
        with open(tpath, 'w') as f:
            json.dump(_jsonable(truth), f, indent=2)
        _log(f"[truth] {time.time()-t0:.0f}s -> {tpath}")

    # ---- per arm --------------------------------------------------------
    all_res = {'truth': truth, 'config': _jsonable(dict(
        Nx=cfg['Nx'], Ny=cfg['Ny'], Lx=cfg['Lx'], Ly=cfg['Ly'], dt=cfg['dt'],
        inner_steps=cfg['inner_steps'], solver_tol=cfg['solver_tol'],
        eval_lam=EVAL_LAM, arrest_lam=ARREST_LAM, ladder=list(LADDER),
        held_out=list(HELD_OUT), early_stop=False)), 'arms': {}}
    for run in [s.strip() for s in a.arms.split(',') if s.strip()]:
        d = FIT_DIR / run
        if not (d / 'theta_checkpoint.npz').exists():
            _log(f"\n=== {run}: NO CHECKPOINT (skipped) ===")
            all_res['arms'][run] = dict(status='no_checkpoint')
            continue
        _log(f"\n=== {run} ===")
        closure, v = load_arm_closure(run)
        bm = json.load(open(d / 'batch_metrics.json')) if (
            d / 'batch_metrics.json').exists() else {}
        rec = {k: dict(fit=v[k], truth=TRUTH[k],
                       signed_rel=(v[k] - TRUTH[k]) / TRUTH[k])
               for k in ('Gp', 'lam', 'nu_s', 'tau_y')}
        _log(f"  recovery: " + '  '.join(
            f"{k}={rec[k]['fit']:.4f} ({rec[k]['signed_rel']:+.2%})"
            for k in ('Gp', 'lam', 'nu_s', 'tau_y')))
        _log(f"  w_i={bm.get('vel_weights')}  lambda_q={bm.get('lambda_q')}  "
             f"pref_floor={v['yield_pref_floor']}  "
             f"converged={bm.get('converged')} n_grads={bm.get('n_grads')} "
             f"s/grad={bm.get('s_per_grad')}")
        lad = run_ladder(closure, cfg, run, a.out_dir)
        # flow curve + secondary IoU against the shared truth
        fc = {}
        for k, r in lad.items():
            t = truth[k]
            qt = t['Q']
            fc[k] = dict(
                g_x=r['g_x'], Q_truth=qt, Q_learned=r['Q'],
                Q_rel=(r['Q'] - qt) / (abs(qt) if abs(qt) > 1e-12 else 1.0),
                Q_abs_ratio=abs(r['Q']) / max(abs(qt), 1e-300),
                kin_plug_truth=t['kinematic']['halfwidth'],
                kin_plug_learned=r['kinematic']['halfwidth'],
                kin_plug_rel=((r['kinematic']['halfwidth']
                               - t['kinematic']['halfwidth'])
                              / t['kinematic']['halfwidth']
                              if t['kinematic']['halfwidth'] > 0 else None),
                core_shear_truth=t['kinematic']['core_shear_norm'],
                core_shear_learned=r['kinematic']['core_shear_norm'],
                iou_secondary=iou(r['unyielded_mask'], t['unyielded_mask']),
                held_out=bool(r['g_x'] in HELD_OUT))
        all_res['arms'][run] = dict(
            status='ok', scalars=v, recovery=rec,
            vel_weights=bm.get('vel_weights'), vel_W=bm.get('vel_W'),
            lambda_q=bm.get('lambda_q'), lambda_q0=bm.get('lambda_q0'),
            converged=bm.get('converged'), n_grads=bm.get('n_grads'),
            s_per_grad=bm.get('s_per_grad'), wall_opt_s=bm.get('wall_opt_s'),
            loss_init=bm.get('loss_init'), loss_final=bm.get('loss_final'),
            br_init=bm.get('br_init'), ladder=lad, flow_curve=fc)
        with open(a.out_dir / 'eval_summary.json', 'w') as f:
            json.dump(_jsonable(all_res), f, indent=2)

    with open(a.out_dir / 'eval_summary.json', 'w') as f:
        json.dump(_jsonable(all_res), f, indent=2)
    _log(f"\n[done] -> {a.out_dir / 'eval_summary.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
