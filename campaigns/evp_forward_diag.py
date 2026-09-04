#!/usr/bin/env python
"""EVP yield-stress forward-only diagnostics.

Forward only. No fitting. No scalar is changed except the deliberate yield-
stress perturbation of the truth in the sensitivity mode.

Design principle: each forward is run ONCE to the long horizon and the
diagnostics are reduced at EVERY outer step. Every horizon question is then a
slice of the saved trajectory, at no extra compute. Nothing reads only [-1].

Modes
  --mode b0      reproduce a single drive (g_x=4) under both the sweep and the
                 production configuration, to confirm the two agree
  --mode ladder  run truth and the frozen learned closure over the nine-drive
                 ladder, so their flow curves can be compared
  --mode sens    sensitivity of the truth to the yield stress, at
                 tau_y x {0.95, 1.05}
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
import jax.numpy as jnp

import evp_baseline_diag as phase0
import evp_learned_flowcurve as elf
import fwd_yield_sweep as fys
import visco_opt_tbnn_evp_run as vevp
from jax_rheology.models import registry as cr
from jax_rheology.models import tbnn_memory as tb

ROOT = REPO_ROOT
OUT = ROOT / 'work/evp_forward_diag'

TRUTH = dict(Gp=3.2, lam=0.7, nu_s=0.8, tau_y=1.45)
LAM_T = TRUTH['lam']

# Ladder (prompt B1). Sub-yield drives get the long tail.
LADDER = (0.5, 1.0, 1.3, 1.45, 1.6, 1.8, 2.5, 4.0, 6.0)
SUBYIELD = (0.5, 1.0, 1.3)

T_LONG_LAM = 15.0          # g_x >= 1.45
PROD_OUTER = 84            # A1-resolved production horizon (T = 2.1 = 3 lam)


def tag(v: float) -> str:
    return f'{v:g}'.replace('.', 'p').replace('-', 'm')


# ---------------------------------------------------------------------------
# Config -- the A1-resolved PRODUCTION config
# ---------------------------------------------------------------------------
def prod_cfg() -> dict:
    """fys._base_cfg() is byte-identical to the A1-resolved production config:
    DEFAULT_CHANNEL_CONFIG + Ny=64, Nx=32, solver_tol=1e-8, inner=10, dt=2.5e-3.
    outer_steps is set per run by the caller."""
    return dict(fys._base_cfg())


def steps_per_lam(cfg) -> int:
    return fys._steps_per_lam(cfg)          # 28 at the production config


def outer_dt(cfg) -> float:
    return fys._outer_dt(cfg)               # 0.025


# ---------------------------------------------------------------------------
# Closure builders
# ---------------------------------------------------------------------------
def truth_closure(tau_y_scale: float = 1.0):
    tau_y = TRUTH['tau_y'] * tau_y_scale
    params = dict(Gp=jnp.asarray(TRUTH['Gp'], dtype=jnp.float64),
                  lam=jnp.asarray(TRUTH['lam'], dtype=jnp.float64),
                  tau_y=jnp.asarray(tau_y, dtype=jnp.float64))
    return dict(name='truth' if tau_y_scale == 1.0 else f'truth_ty{tau_y_scale:g}',
                model=cr.get_model('saramito_logconf_bk_v2'), params=params,
                Gp=TRUTH['Gp'], tau_y=tau_y, nu_s=TRUTH['nu_s'],
                lam=TRUTH['lam'])


def learned_closure():
    """Frozen v2_prod2 theta + fitted scalars, gradients off."""
    params, values = elf._load_learned()
    return dict(name='learned',
                model=cr.get_model('tbnn_potential_yield_logconf_bk_v2'),
                params=params, Gp=values['Gp'], tau_y=values['tau_y'],
                nu_s=values['nu_s'], lam=values['lam'],
                bound_c=values['bound_c'], kappa=values['kappa'])


# ---------------------------------------------------------------------------
# Per-outer-step reduction
# ---------------------------------------------------------------------------
def _reduce_chunk(chunk, cfg, Gp: float, tau_y: float):
    """Reduce one jitted chunk to per-outer-step diagnostics.

    Q(t) uses the PRODUCTION _flow_rate_Q_traj (x-average, trapezoid in y).
    The plug/yield ruler is the gate-6 / B4 stress criterion
    phase0.plug_halfwidth_yield applied to the x-averaged conformation, with
    the CLOSURE'S OWN (Gp, tau_y) -- the one-ruler contract.
    """
    Q = np.asarray(vevp._flow_rate_Q_traj(chunk['u_traj'], cfg))   # (T,)
    u_prof = np.asarray(jnp.mean(chunk['u_traj'], axis=1))          # (T, Ny)
    Axx = np.asarray(jnp.mean(chunk['A_xx_traj'], axis=1))
    Axy = np.asarray(jnp.mean(chunk['A_xy_traj'], axis=1))
    Ayy = np.asarray(jnp.mean(chunk['A_yy_traj'], axis=1))
    Azz = np.asarray(jnp.mean(chunk['A_zz_traj'], axis=1))

    n = Q.shape[0]
    plug = np.empty(n)
    yf = np.empty(n)
    uny = np.empty((n, cfg['Ny']), dtype=bool)
    td_max = np.empty(n)
    for i in range(n):
        p, td, u_ = phase0.plug_halfwidth_yield(
            Axx[i], Axy[i], Ayy[i], Azz[i], cfg, Gp=Gp, tau_y=tau_y)
        plug[i] = p
        uny[i] = u_
        yf[i] = float((~u_).mean())
        td_max[i] = float(np.max(td))

    return dict(Q=Q, u_prof=u_prof, plug=plug, yielded_frac=yf, unyielded=uny,
                td_max=td_max,
                min_eig=np.asarray(chunk['min_lam_traj']),
                min_trA=np.asarray(chunk['min_trA_traj']),
                any_nan=np.asarray(chunk['any_nan_traj']).astype(bool),
                td_profile_final=np.asarray(
                    tb.saramito_tau_d_norm(jnp.asarray(Axx[-1]),
                                           jnp.asarray(Axy[-1]),
                                           jnp.asarray(Ayy[-1]),
                                           jnp.asarray(Azz[-1]), Gp)))


def run_forward(closure: dict, g_x: float, cfg: dict, n_outer: int, *,
                label: str, early_stop: bool = False):
    """One forward to n_outer outer steps, diagnostics at EVERY outer step.

    early_stop=True reproduces the fwd_yield_sweep / evp_learned_flowcurve
    convergence protocol (used only by the B0 gate); the ladder runs
    the full horizon so every horizon can be sliced.
    """
    cfg = dict(cfg)
    cfg['nu_s'] = closure['nu_s']            # base viscosity is a closure scalar
    cfg['g_x'] = float(g_x)
    chunk_steps = steps_per_lam(cfg)

    grid, model, init_state, perm = fys._build_channel_with_model(
        cfg, closure['model'])
    run_chunk = fys._make_chunk_runner(cfg, model, closure['params'], grid,
                                       perm, float(g_x))

    acc = {k: [] for k in ('Q', 'u_prof', 'plug', 'yielded_frac', 'unyielded',
                           'td_max', 'min_eig', 'min_trA', 'any_nan')}
    state = init_state
    done = 0
    stop_reason = 'T_max'
    t0 = time.perf_counter()
    td_final = None
    while done < n_outer:
        n_this = min(chunk_steps, n_outer - done)
        chunk = run_chunk(state, n_this)
        state = chunk['final_state']
        red = _reduce_chunk(chunk, cfg, closure['Gp'], closure['tau_y'])
        td_final = red.pop('td_profile_final')
        for k in acc:
            acc[k].append(red[k])
        done += n_this
        if red['any_nan'].any():
            stop_reason = 'nan'
            break
        if early_stop and done > chunk_steps:
            Qc = np.concatenate(acc['Q'])
            cr_ = (abs(Qc[-1] - Qc[-1 - chunk_steps])
                   / max(abs(Qc[-1]), 1e-9))
            if cr_ < fys.CONV_RTOL:
                stop_reason = 'steady'
                break
    wall = time.perf_counter() - t0

    out = {k: np.concatenate(acc[k], axis=0) for k in acc}
    nT = out['Q'].shape[0]
    t = (np.arange(1, nT + 1)) * outer_dt(cfg)

    # conv_ratio(t): relative change in Q over a one-lambda window
    Q = out['Q']
    conv = np.full(nT, np.nan)
    if nT > chunk_steps:
        conv[chunk_steps:] = (np.abs(Q[chunk_steps:] - Q[:-chunk_steps])
                              / np.maximum(np.abs(Q[chunk_steps:]), 1e-9))

    meta = dict(
        label=label, closure=closure['name'], g_x=float(g_x),
        Gp=float(closure['Gp']), lam=float(closure['lam']),
        nu_s=float(closure['nu_s']), tau_y=float(closure['tau_y']),
        Nx=int(cfg['Nx']), Ny=int(cfg['Ny']), Lx=float(cfg['Lx']),
        Ly=float(cfg['Ly']), dt=float(cfg['dt']),
        inner_steps=int(cfg['inner_steps']), solver_tol=float(cfg['solver_tol']),
        n_outer=int(nT), outer_dt=float(outer_dt(cfg)),
        steps_per_lam=int(chunk_steps),
        T_final=float(t[-1]), T_over_lam=float(t[-1] / LAM_T),
        stop_reason=stop_reason, walltime_s=float(wall),
        s_per_outer=float(wall / max(nT, 1)),
        Q_final=float(Q[-1]),
        conv_ratio_final=(float(conv[-1]) if np.isfinite(conv[-1]) else None),
        any_nan=bool(out['any_nan'].any()),
        min_eig_min=float(out['min_eig'].min()),
        y_p_analytic=(float(TRUTH['tau_y'] / g_x) if g_x > TRUTH['tau_y'] else None),
        # protocol fingerprint: the initial state B1 and B2 must share
        init_u_absmax=float(np.abs(np.asarray(init_state.velocity[0].data)).max()),
        init_v_absmax=float(np.abs(np.asarray(init_state.velocity[1].data)).max()),
        init_A_fingerprint=[
            float(np.asarray(init_state.memory_fields[i].array.data).sum())
            for i in range(4)],
    )
    out['t'] = t
    out['conv_ratio'] = conv
    out['td_profile_final'] = td_final
    return out, meta


def save_run(out, meta, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f'{name}.npz',
                        **{k: v for k, v in out.items()})
    (out_dir / f'{name}.json').write_text(json.dumps(meta, indent=2,
                                                     sort_keys=True))
    print(f"  [{name}] Q={meta['Q_final']:.8e} T={meta['T_over_lam']:.2f}lam "
          f"n_outer={meta['n_outer']} conv={meta['conv_ratio_final']} "
          f"stop={meta['stop_reason']} nan={meta['any_nan']} "
          f"{meta['s_per_outer']*1e3:.1f} ms/outer  wall={meta['walltime_s']:.1f}s",
          flush=True)


# ---------------------------------------------------------------------------
# B0 -- reproduction gate
# ---------------------------------------------------------------------------
def mode_b0(out_dir: Path):
    print('=== B0 reproduction gate @ g_x=4.0 ===', flush=True)
    print('NOTE (A1): the reported +0.040% / plug / IoU numbers were produced by\n'
          'evp_learned_flowcurve.py at phase0.locked_cfg() -> Ny=128 with the\n'
          '15-lambda convergence protocol, NOT at the Ny=64 / T=3lam production\n'
          'fit config. Both are therefore run here.', flush=True)
    res = {}
    for which, cfg, n_outer, es in (
            ('sweep_Ny128', phase0.locked_cfg(), fys._max_outer_steps(
                phase0.locked_cfg()), True),
            ('prod_Ny64_3lam', prod_cfg(), PROD_OUTER, False)):
        print(f"\n-- {which}: Ny={cfg['Ny']} n_outer_max={n_outer} "
              f"early_stop={es}", flush=True)
        pair = {}
        for closure in (truth_closure(), learned_closure()):
            out, meta = run_forward(closure, 4.0, cfg, n_outer,
                                    label=f'b0_{which}', early_stop=es)
            save_run(out, meta, out_dir, f'b0_{which}_{closure["name"]}')
            pair[closure['name']] = (out, meta)
        (to, tm) = pair['truth']
        (lo, lm) = pair['learned']
        q_err = (lm['Q_final'] - tm['Q_final']) / abs(tm['Q_final']) * 100.0
        # honest plug comparison: each closure's own ruler on its OWN field
        iou = float((~to['unyielded'][-1] & ~lo['unyielded'][-1]).sum()
                    / max((~to['unyielded'][-1] | ~lo['unyielded'][-1]).sum(), 1))
        plug_t = float(to['plug'][-1]); plug_l = float(lo['plug'][-1])
        plug_rel = abs(plug_l - plug_t) / max(plug_t, 1e-30) * 100.0
        res[which] = dict(
            Q_truth=tm['Q_final'], Q_learned=lm['Q_final'], Q_err_pct=q_err,
            plug_truth=plug_t, plug_learned=plug_l, plug_rel_pct=plug_rel,
            iou_simulated=iou, y_p_analytic=tm['y_p_analytic'],
            T_over_lam_truth=tm['T_over_lam'],
            T_over_lam_learned=lm['T_over_lam'],
            s_per_outer_truth=tm['s_per_outer'],
            s_per_outer_learned=lm['s_per_outer'],
            walltime_truth_s=tm['walltime_s'],
            walltime_learned_s=lm['walltime_s'])
        r = res[which]
        print(f"\n  >> {which}: Q_truth={r['Q_truth']:.8e} "
              f"Q_learned={r['Q_learned']:.8e}  Q err = {q_err:+.4f} %")
        print(f"     plug truth={plug_t:.5f} learned={plug_l:.5f} "
              f"(analytic y_p={r['y_p_analytic']}) plug_rel={plug_rel:.3f} %  "
              f"IoU(simulated,own-ruler)={iou:.4f}")
    (out_dir / 'b0_gate.json').write_text(json.dumps(res, indent=2))
    print(f"\n[b0] wrote {out_dir/'b0_gate.json'}")


# ---------------------------------------------------------------------------
# B1 + B2 -- truth and frozen learned ladder
# ---------------------------------------------------------------------------
def _n_outer_for(g_x: float, cfg: dict) -> int:
    spl = steps_per_lam(cfg)
    if g_x in SUBYIELD:
        # 10x the production horizon or 30 lambda, whichever is LONGER
        return max(10 * PROD_OUTER, int(round(30.0 * LAM_T / outer_dt(cfg))))
    return int(round(T_LONG_LAM * LAM_T / outer_dt(cfg)))


def mode_ladder(out_dir: Path, only=None):
    cfg = prod_cfg()
    print('=== B1 (truth) + B2 (frozen learned) ladder @ PRODUCTION config ===')
    print(f"  Nx={cfg['Nx']} Ny={cfg['Ny']} dt={cfg['dt']} "
          f"inner={cfg['inner_steps']} tol={cfg['solver_tol']} "
          f"steps_per_lam={steps_per_lam(cfg)}", flush=True)
    closures = [truth_closure(), learned_closure()]
    for g_x in LADDER:
        if only is not None and g_x not in only:
            continue
        n_outer = _n_outer_for(g_x, cfg)
        print(f"\n-- g_x={g_x:g}  n_outer={n_outer} "
              f"= {n_outer*outer_dt(cfg)/LAM_T:.1f} lam", flush=True)
        for closure in closures:
            out, meta = run_forward(closure, g_x, cfg, n_outer,
                                    label='ladder')
            save_run(out, meta, out_dir, f"{closure['name']}_gx{tag(g_x)}")


# ---------------------------------------------------------------------------
# B3 -- yield-stress sensitivity (TRUTH only)
# ---------------------------------------------------------------------------
def mode_sens(out_dir: Path, only=None):
    cfg = prod_cfg()
    print('=== B3 tau_y sensitivity: TRUTH at tau_y x {0.95, 1.05} ===')
    for g_x in LADDER:
        if only is not None and g_x not in only:
            continue
        n_outer = _n_outer_for(g_x, cfg)
        print(f"\n-- g_x={g_x:g}  n_outer={n_outer}", flush=True)
        for scale in (0.95, 1.05):
            closure = truth_closure(scale)
            out, meta = run_forward(closure, g_x, cfg, n_outer, label='sens')
            save_run(out, meta, out_dir, f'truth_ty{tag(scale)}_gx{tag(g_x)}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True,
                   choices=('b0', 'ladder', 'sens'))
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--drives', type=str, default=None,
                   help='comma list to restrict the ladder (debug/split)')
    from jax_rheology.io.config import parse_with_config
    a, _cfg = parse_with_config(p)
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError('float64 must be enabled')
    print(f'[setup] devices = {jax.devices()}', flush=True)
    if os.environ.get('SLURM_JOB_ID') and jax.devices()[0].platform != 'gpu':
        raise RuntimeError(
            f'batch job but JAX sees {jax.devices()} -- refusing to run a '
            f'GPU-budgeted forward sweep on CPU (JAX_PLATFORMS='
            f'{os.environ.get("JAX_PLATFORMS")!r})')
    only = None
    if a.drives:
        only = tuple(float(x) for x in a.drives.split(','))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.mode == 'b0':
        mode_b0(a.out_dir)
    elif a.mode == 'ladder':
        mode_ladder(a.out_dir, only)
    else:
        mode_sens(a.out_dir, only)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
