#!/usr/bin/env python
"""Evaluate the 5-seed A_3lam_agn ensemble.

Common protocol from evp_fix_eval.py (15lam ladder, 30lam sub-yield, no early
stopping) plus NaN-ladder drives {4.5, 5.0, 5.5, 6.0}. Reports the five seeds
as one ensemble -- original (theta_seed=0) is seed 1, not a reference.
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

import evp_forward_diag as efd
import evp_fix_eval as ee

OUT = Path('work/evp_seed_eval')
SEEDS = [
    ('evp_fix_A_3lam_agn', 1, 0),      # run name, ensemble #, theta_seed
    ('evp_fix_A_3lam_agn_s2', 2, 2),
    ('evp_fix_A_3lam_agn_s3', 3, 3),
    ('evp_fix_A_3lam_agn_s4', 4, 4),
    ('evp_fix_A_3lam_agn_s5', 5, 5),
]
# union of eval ladder and NaN ladder
LADDER = (0.5, 1.0, 1.3, 1.45, 1.6, 1.8, 2.5, 4.0, 4.5, 5.0, 5.5, 6.0)
NAN_DRIVES = (4.5, 5.0, 5.5, 6.0)
TRUTH = ee.TRUTH


def _log(m):
    print(m, flush=True)


def well_posed_ceiling(lad):
    """Highest g_x such that every NaN-ladder drive up to it is finite
    (contiguous from below; a hole at 5.5 with finite 6 does not count as 6)."""
    ceil = None
    for gx in NAN_DRIVES:
        r = lad[f'{gx:g}']
        me = r.get('min_eig')
        if (r.get('any_nan') or me is None
                or not np.isfinite(me) or me <= 1e-8):
            break
        ceil = gx
    return ceil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=Path, default=OUT)
    from jax_rheology.io.config import parse_with_config
    a, _cfg = parse_with_config(p)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    _log(f'[setup] devices={jax.devices()}')
    import jax_rheology.models.tbnn_memory as tb
    _log(f'[setup] _YIELD_PREF_FLOOR={tb._YIELD_PREF_FLOOR!r}')
    cfg = efd.prod_cfg()
    _log(f"[setup] Nx={cfg['Nx']} Ny={cfg['Ny']} dt={cfg['dt']} "
         f"FIXED {ee.EVAL_LAM}lam ({ee.ARREST_LAM}lam sub-yield), "
         f"ladder={list(LADDER)}")

    # truth once
    tpath = a.out_dir / 'truth_ladder.json'
    if tpath.exists():
        truth = json.load(open(tpath))
        _log(f'[truth] reusing {tpath}')
    else:
        _log('\n=== truth ===')
        t0 = time.time()
        # temporarily point ee SUBYIELD / run_ladder drives
        truth = ee.run_ladder(efd.truth_closure(), cfg, 'truth', a.out_dir,
                              drives=LADDER)
        with open(tpath, 'w') as f:
            json.dump(ee._jsonable(truth), f, indent=2)
        _log(f'[truth] {time.time()-t0:.0f}s')

    all_res = dict(
        config=ee._jsonable(dict(
            Nx=cfg['Nx'], Ny=cfg['Ny'], dt=cfg['dt'],
            eval_lam=ee.EVAL_LAM, arrest_lam=ee.ARREST_LAM,
            ladder=list(LADDER), nan_drives=list(NAN_DRIVES),
            early_stop=False)),
        truth=truth, seeds={})

    for run, ens_id, theta_seed in SEEDS:
        _log(f'\n=== ens#{ens_id} {run} (theta_seed={theta_seed}) ===')
        closure, v = ee.load_arm_closure(run)
        bm = {}
        bmp = ee.FIT_DIR / run / 'batch_metrics.json'
        if bmp.exists():
            bm = json.load(open(bmp))
        rec = {k: dict(fit=v[k], truth=TRUTH[k],
                       signed_rel=(v[k] - TRUTH[k]) / TRUTH[k])
               for k in ('Gp', 'lam', 'nu_s', 'tau_y')}
        _log('  recovery: ' + '  '.join(
            f"{k}={rec[k]['fit']:.4f} ({rec[k]['signed_rel']:+.2%})"
            for k in rec))

        lad = ee.run_ladder(closure, cfg, run, a.out_dir, drives=LADDER)
        fc = {}
        for k, r in lad.items():
            t = truth[k]
            qt = t['Q']
            fc[k] = dict(
                g_x=r['g_x'], Q_truth=qt, Q_learned=r['Q'],
                Q_rel=((r['Q'] - qt) / abs(qt) if abs(qt) > 1e-12 else None),
                kin_plug_learned=r['kinematic']['halfwidth'],
                core_shear_learned=r['kinematic']['core_shear_norm'],
                core_shear_truth=t['kinematic']['core_shear_norm'],
                any_nan=r['any_nan'],
                first_nan_T_lam=r['nan_at_lam'],
                min_eig=r['min_eig'], max_td=r['max_td'],
                held_out=bool(r['g_x'] in ee.HELD_OUT
                              or r['g_x'] in NAN_DRIVES))
        ceil = well_posed_ceiling(lad)
        q30 = abs(lad['1.3']['Q_at'].get('30', lad['1.3']['Q']))
        all_res['seeds'][f's{ens_id}'] = dict(
            run=run, ensemble_id=ens_id, theta_seed=theta_seed,
            status='ok', scalars=v, recovery=rec,
            converged=bm.get('converged'), n_grads=bm.get('n_grads'),
            loss_final=bm.get('loss_final'), s_per_grad=bm.get('s_per_grad'),
            ladder=lad, flow_curve=fc,
            well_posed_gx_ceiling=ceil,
            arrest_Q30_abs=q30,
            Q_rel_gx5=fc['5']['Q_rel'],
            core_shear_gx1p8=fc['1.8']['core_shear_learned'],
        )
        _log(f"  ceiling={ceil}  |Q|(1.3,30lam)={q30:.3e}  "
             f"Q_rel@5={fc['5']['Q_rel']}  "
             f"core_shear@1.8={fc['1.8']['core_shear_learned']:.4f}")
        with open(a.out_dir / 'seed_eval_summary.json', 'w') as f:
            json.dump(ee._jsonable(all_res), f, indent=2)

    # ensemble aggregate
    seeds = [all_res['seeds'][f's{i}'] for i in range(1, 6)]
    ens = {}
    for k in ('Gp', 'lam', 'nu_s', 'tau_y'):
        arr = np.array([s['recovery'][k]['signed_rel'] for s in seeds])
        ens[f'{k}_signed_rel'] = dict(mean=float(arr.mean()),
                                      std=float(arr.std(ddof=1)),
                                      values=arr.tolist())
    ens['arrest_Q30_abs'] = [s['arrest_Q30_abs'] for s in seeds]
    ens['well_posed_ceiling'] = [s['well_posed_gx_ceiling'] for s in seeds]
    ens['n_reach_gx6'] = int(sum(
        1 for c in ens['well_posed_ceiling'] if c is not None and c >= 6.0))
    q5 = np.array([s['Q_rel_gx5'] for s in seeds], dtype=float)
    ens['Q_rel_gx5'] = dict(mean=float(np.nanmean(q5)),
                            std=float(np.nanstd(q5, ddof=1)),
                            values=q5.tolist())
    cs = np.array([s['core_shear_gx1p8'] for s in seeds])
    ens['core_shear_gx1p8'] = dict(
        mean=float(cs.mean()), std=float(cs.std(ddof=1)),
        values=cs.tolist(),
        truth=float(truth['1.8']['kinematic']['core_shear_norm']))
    all_res['ensemble'] = ens

    with open(a.out_dir / 'seed_eval_summary.json', 'w') as f:
        json.dump(ee._jsonable(all_res), f, indent=2)

    _log('\n=== ensemble (n=5) ===')
    for k in ('Gp', 'lam', 'nu_s', 'tau_y'):
        e = ens[f'{k}_signed_rel']
        _log(f"  {k} signed rel: {100*e['mean']:+.4f}% +/- {100*e['std']:.4f}%")
    _log(f"  arrest |Q|(1.3,30lam): {ens['arrest_Q30_abs']}")
    _log(f"  ceilings: {ens['well_posed_ceiling']}  "
         f"n_reach_gx6={ens['n_reach_gx6']}/5")
    _log(f"  Q_rel@5: mean={100*ens['Q_rel_gx5']['mean']:+.3f}% +/- "
         f"{100*ens['Q_rel_gx5']['std']:.3f}%  values="
         f"{[None if v!=v else f'{100*v:+.3f}%' for v in ens['Q_rel_gx5']['values']]}")
    _log(f"  core_shear@1.8: {ens['core_shear_gx1p8']['mean']:.4f} +/- "
         f"{ens['core_shear_gx1p8']['std']:.4f}  "
         f"(truth {ens['core_shear_gx1p8']['truth']:.4f})")
    _log(f'\n[done] -> {a.out_dir / "seed_eval_summary.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
