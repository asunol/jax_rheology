#!/usr/bin/env python
"""Full agnostic TBNN fit on the 4:1 contraction (stretch-weighted loss).

Continuation of the identifiability work. This wires the *existing* agnostic
training schemes (s1 / s1b / s4 -- the alternating L-BFGS scalar re-solve +
persistent-Adam theta blocks from ``visco_opt_tbnn_run.py``, reproduced here
VERBATIM) to:

  * the 4:1 contraction forward path (``jax_rheology.forward.contraction.
    evolve_contraction`` with the ramped inlet), the same contraction
    forward path used to evaluate a trained TBNN; and
  * the **stretch-weighted velocity loss** from the identifiability
    diagnostics (weight = local tr A above equilibrium, from the TRUTH
    trajectory, normalized to mean 1 -- the same reweighting that gave the
    ~3x alpha-SE sharpening in the FIM).

It does NOT modify the working scheme code or the contraction solver: the
scheme functions are copied unchanged (only ``_forward`` / ``loss_fn`` /
truth-trajectory generation point at the contraction), and the checkpoint /
flatten helpers are imported from ``visco_opt_tbnn_run``.

Agnostic protocol (unchanged): the fitted TBNN runs at the declared gauge
Gp = lam = 1 at init (fit jointly), nu_s fit from a neutral 1.0; truth is
``giesekus_logconf_bk_v2`` (alpha=0.30) or ``fene_p_logconf_bk_v2`` (L^2=12)
with true scalars Gp=3.2, lam=0.7, nu_s=0.8.

Usage (one of the six runs):
  python visco_opt_tbnn_contraction_run.py --truth-model giesekus --scheme s1 \
      --run-name ctr_g3_s1 --out-dir ./work/contraction_train
  # memory smoke (2 blocks, prints GPU footprint, then exits):
  python visco_opt_tbnn_contraction_run.py --truth-model fene_p --scheme s1 \
      --mem-smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

from repo_paths import bootstrap, REPO_ROOT
bootstrap()

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jax_rheology.models import registry as cr
from jax_rheology.geometries import planar_contraction as cg
from jax_rheology.forward import contraction as cf
from jax_rheology.models import tbnn_memory as tb  #  registers the TBNN model
# reuse (import, do NOT duplicate) the cross-env checkpoint helpers
from visco_opt_tbnn_run import (theta_to_named_arrays, theta_from_named_arrays,
                                _save_and_verify_checkpoint)

TRUTH_NAME = {'giesekus': 'giesekus_logconf_bk_v2',
              'fene_p': 'fene_p_logconf_bk_v2'}
TBNN_NAME = 'tbnn_potential_logconf_bk_v2'

# --- wall-tap pressure observable (additive) -------------------------------
# Four fixed PHYSICAL wall taps (converged at >=4x per pressure_yconv_report):
#   p1,p2 off the wide wall (y=RH-offset); p3,p4 off the narrow wall.
# p1-referenced differences dp2=p2-p1, dp3=p3-p1 (entry drop), dp4=p4-p1.
TAPS_PHYS = [(-2.917, 3.583), (-0.750, 3.583), (0.750, 0.583), (6.083, 0.583)]


def bilinear_idx(xc, yc, x, y):
    """Cell indices + bilinear weights for physical (x,y) on cell-center axes."""
    i = int(np.clip(np.searchsorted(xc, x) - 1, 0, len(xc) - 2))
    j = int(np.clip(np.searchsorted(yc, y) - 1, 0, len(yc) - 2))
    tx = float((x - xc[i]) / (xc[i + 1] - xc[i]))
    ty = float((y - yc[j]) / (yc[j + 1] - yc[j]))
    return i, j, tx, ty


def _p_inst_jax(p_traj, factor):
    """Instantaneous pressure from the Chorin accumulator: differenced snapshots
    times rho/(inner*dt). Never the accumulator itself."""
    prev = jnp.concatenate([jnp.zeros_like(p_traj[:1]), p_traj[:-1]], axis=0)
    return factor * (p_traj - prev)


def _tap_series_jax(p_inst, idx):
    i, j, tx, ty = idx
    return ((1 - tx) * (1 - ty) * p_inst[:, i, j]
            + tx * (1 - ty) * p_inst[:, i + 1, j]
            + (1 - tx) * ty * p_inst[:, i, j + 1]
            + tx * ty * p_inst[:, i + 1, j + 1])


def dp_from_ptraj(p_traj, tap_idx, factor):
    """p1-referenced (dp2,dp3,dp4) trajectories, JAX-differentiable. (outer,3)."""
    pi = _p_inst_jax(p_traj, factor)
    p = [_tap_series_jax(pi, idx) for idx in tap_idx]
    return jnp.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]], axis=1)


def _campaign_config_hash(args, U_list):
    """Stable short hash of the RATE-INDEPENDENT knobs that determine each
    rate's truth + per-tap pressure norms, so ONE norms.json can be shared by
    all campaign jobs (single-rate and dual). U_list is deliberately excluded
    from the hash (each job verifies only the per-rate entries it uses); it is
    still recorded in the config dict for provenance."""
    import hashlib
    keys = ['truth_model', 'truth_gp', 'truth_lam', 'truth_nus', 'truth_lsq',
            'truth_alpha', 'H', 'ratio', 'L_up', 'L_down', 'cells_per_H',
            'nx', 'ny', 'density', 'dt', 'inner', 'outer', 'ramp_time',
            'solver_tol', 'solver_maxiter', 'no_stretch_weight', 'loss_weight',
            'roi_a', 'roi_c', 'roi_ell', 'roi_sigma_y', 'roi_kappa', 'roi_xc',
            'n_sub']
    d = {k: getattr(args, k) for k in keys}
    d['taps_phys'] = TAPS_PHYS
    blob = json.dumps(d, sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()[:16]
    d_full = dict(d)
    d_full['U_list'] = [float(u) for u in U_list]
    return h, d_full


# --- full-state checkpoint / resume (additive; pickled numpy pytrees) -------
def _tree_to_np(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _tree_to_jnp(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def _save_train_ckpt(run_dir, obj):
    """Atomic pickle write of the full resumable training state."""
    path = os.path.join(run_dir, 'train_ckpt.pkl')
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def _load_train_ckpt(run_dir):
    path = os.path.join(run_dir, 'train_ckpt.pkl')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[resume] failed to read {path}: {e}; starting fresh.", flush=True)
        return None


def roi_weight(grid, x_c, y_c, ubar, lam, a=0.5, c=1.75, ell=0.4,
               sigma_y=0.4, kappa=4.0, normalize=True):
    """Static, truth-free, theta-free geometric ROI weight on the die
    centerline, offset downstream by one convective relaxation length
    Delta=ubar*lam. Floor 1, plateau ~1+kappa. Returns (w, w_raw) shape (Nx,Ny)."""
    X, Y = grid.mesh(grid.cell_center)
    X = jnp.asarray(X, jnp.float64); Y = jnp.asarray(Y, jnp.float64)
    Delta = ubar * lam
    x_on = x_c - a
    x_off = x_c + c * Delta
    S = lambda z: 1.0 / (1.0 + jnp.exp(-z))
    psi_x = S((X - x_on) / ell) * S((x_off - X) / ell)
    psi_y = jnp.exp(-((Y - y_c) ** 2) / (2.0 * sigma_y ** 2))
    w_raw = 1.0 + kappa * psi_x * psi_y
    w = w_raw / jnp.mean(w_raw) if normalize else w_raw
    return w, w_raw


def _save_roi_png(grid, w, path, x_off):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dom = grid.domain
    ext = [dom[0][0], dom[0][1], dom[1][0], dom[1][1]]
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(np.asarray(w).T, origin='lower', extent=ext, aspect='equal',
                   cmap='magma')
    ax.axvline(0.0, color='c', ls='--', lw=0.8)
    ax.set_title(f'ROI weight (band x_off={x_off:.2f}H)')
    ax.set_xlabel('x/H'); ax.set_ylabel('y/H'); plt.colorbar(im, ax=ax, label='w')
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a memory (conformation-tensor) TBNN closure on "
                    "contraction-flow velocity data.",
        epilog="Usual invocation, which sets every option below from a config "
               "file:\n"
               "  python experiments/contraction_train.py "
               "--config experiments/configs/giesekus.yaml\n"
               "The individual flags exist so a config can be overridden for "
               "one run; see REPRODUCE.md for the published settings.")
    p.add_argument('--truth-model', choices=['giesekus', 'fene_p'],
                   default='giesekus')
    p.add_argument('--scheme', choices=['s1', 's1b', 's4'], default='s1')
    # truth scalars (identical to the bumps training)
    p.add_argument('--truth-gp', type=float, default=3.2)
    p.add_argument('--truth-lam', type=float, default=0.7)
    p.add_argument('--truth-nus', type=float, default=0.8)
    p.add_argument('--truth-alpha', type=float, default=0.3)
    p.add_argument('--truth-lsq', type=float, default=12.0)
    # agnostic init (gauge Gp=lam=1, neutral nu_s=1)
    p.add_argument('--gp-init', type=float, default=1.0)
    p.add_argument('--lam-init', type=float, default=1.0)
    p.add_argument('--nus-init', type=float, default=1.0)
    # theta / scalar optimizer knobs (bumps defaults)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--scalar-lr2', type=float, default=2e-3)
    p.add_argument('--stage1-ftol', type=float, default=1e-9)
    p.add_argument('--scalar-bound-lo', type=float, default=0.02)
    p.add_argument('--scalar-bound-hi', type=float, default=20.0)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--bound-c', type=float, default=tb.TBNN_DEFAULT_BOUND_C)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--time-budget-s', type=float, default=39000.0)
    # contraction geometry / integration (matches the FIM diagnostic config)
    p.add_argument('--H', type=float, default=1.0)
    p.add_argument('--ratio', type=float, default=4.0)
    p.add_argument('--L-up', type=float, default=6.0)
    p.add_argument('--L-down', type=float, default=12.0)
    p.add_argument('--cells-per-H', type=float, default=6.0)
    p.add_argument('--U', type=float, default=0.5,
                   help='single inlet rate (default). Overridden by --U-list '
                        'when that is set to more than one value.')
    p.add_argument('--U-list', type=str, default=None,
                   help='comma-separated inlet rates for multi-flow loss, e.g. '
                        '"0.5,4". Each rate gets its own truth + ROI + pressure '
                        'term; losses are scale-normalized so the high-U rate '
                        'does not dominate. Default: single --U (bit-identical).')
    p.add_argument('--density', type=float, default=1.0)
    p.add_argument('--dt', type=float, default=1.0e-4)
    p.add_argument('--inner', type=int, default=50)
    p.add_argument('--outer', type=int, default=200)
    p.add_argument('--ramp-time', type=float, default=0.7)
    p.add_argument('--solver-tol', type=float, default=1.0e-10)
    p.add_argument('--solver-maxiter', type=int, default=300)
    p.add_argument('--no-stretch-weight', action='store_true',
                   help='ablation: uniform velocity loss (default is '
                        'stretch-weighted, the diagnostics recommendation).')
    # loss weight: stretch (ground-truth trA*, default), roi (geometric ROI band), uniform
    p.add_argument('--loss-weight', choices=['stretch', 'roi', 'uniform'],
                   default='stretch',
                   help='velocity-MSE weight. stretch=ground-truth trA*-3 (default); '
                        'roi=static geometric ROI band on the die centerline; '
                        'uniform=ones.')
    p.add_argument('--roi-a', type=float, default=0.5)
    p.add_argument('--roi-c', type=float, default=1.75)
    p.add_argument('--roi-ell', type=float, default=0.4)
    p.add_argument('--roi-sigma-y', type=float, default=0.4)
    p.add_argument('--roi-kappa', type=float, default=4.0)
    p.add_argument('--roi-xc', type=float, default=0.0,
                   help='contraction-plane x (this geometry: 0).')
    # optional beta gauge: if set, overrides truth Gp/nu_s at fixed eta0, lam
    p.add_argument('--beta', type=float, default=None,
                   help='solvent fraction; sets nu_s=beta*eta0, Gp=(1-beta)*eta0/lam.')
    p.add_argument('--eta0', type=float, default=3.04,
                   help='total viscosity for the --beta gauge.')
    # explicit grid override (production 128x256 has NON-square cells: finer in
    # y -- cannot come from cells_per_H, so pass nx/ny directly. Default path
    # (nx=ny=None) uses make_contraction_grid and stays bit-identical.)
    p.add_argument('--nx', type=int, default=None,
                   help='explicit x cells (with --ny; overrides cells-per-H).')
    p.add_argument('--ny', type=int, default=None,
                   help='explicit y cells (with --nx; overrides cells-per-H).')
    # additive wall-tap pressure observable in the loss
    p.add_argument('--w-p', type=float, default=0.0,
                   help='pressure-term weight. 0 (default) => velocity-only, '
                        'bit-identical. Overridden by --w-p-scale if set.')
    p.add_argument('--w-p-scale', type=float, default=None,
                   help='if set, w_p = scale * w_bal, where w_bal (=Lv0/R0 at '
                        'init) balances L_p/L_v~1; recomputed at the actual '
                        '(dt,inner). Use 0.1/1/10 for the bracket.')
    p.add_argument('--n-sub', type=int, default=8,
                   help='number of sub-sampled post-ramp instants for the '
                        'sparse wall-tap pressure observable.')
    p.add_argument('--dump-norms', type=str, default=None,
                   help='Target-generation mode: write truth + per-tap pressure norms + '
                        'w_bal + dual alpha scales at this config, write the JSON '
                        'to this path (with config hash + health arrays), then '
                        'exit before optimization. Single source of truth for the '
                        'campaign.')
    p.add_argument('--norms-json', type=str, default=None,
                   help='verify the internally-computed truth norms / w_bal '
                        'against this pre-generated norms.json (config-hash + '
                        'per-tap value cross-check) at start; HALT on mismatch. '
                        'Provenance guard for production jobs.')
    p.add_argument('--grad-probe', action='store_true',
                   help='diagnostic: build truth + evaluate the loss and its '
                        'reverse-mode gradient ONCE at init (exact production '
                        'forward+backward), print finiteness + truth max trA, '
                        'then exit before optimization. For sweeping the '
                        'reverse-mode-stable rate ceiling.')
    p.add_argument('--ckpt-every', type=int, default=25,
                   help='checkpoint the full training state every N theta steps '
                        '(also at every block boundary).')
    p.add_argument('--out-dir', type=str, default='./work/contraction_train')
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--mem-smoke', action='store_true',
                   help='run a tiny slice (2 short blocks), print the GPU '
                        'memory footprint + grad sanity, then exit.')
    # -- FENE7 curriculum (additive; OFF => bit-identical to the proven path) --
    p.add_argument('--curriculum-ulist', type=str, default=None,
                   help='FENE-P two-rate curriculum: comma rate list, e.g. "0.5,4". '
                        'Stage 1 trains on the FIRST rate only; after '
                        '--curriculum-gate-after completed block-pairs a '
                        'forward envelope gate at the remaining rate(s) '
                        'activates the full multi-flow dual (per-rate alpha_v). '
                        'Truth is built for ALL listed rates. OFF (None) keeps '
                        'the existing single/multi-rate path unchanged.')
    p.add_argument('--curriculum-gate-after', type=int, default=1,
                   help='gate after this many completed block-pairs (default 1: '
                        'after the first L-BFGS scalar solve + first theta '
                        'block).')
    p.add_argument('--curriculum-gate-lam-margin', type=float, default=1.5,
                   help='forward-gate margin variant: also require a finite '
                        'forward at lam x this factor (default 1.5).')
    p.add_argument('--curriculum-max-pairs', type=int, default=4,
                   help='cap on stage-1 block-pairs before the curriculum stops and writes '
                        'a GATE_FAILED marker file in the run directory '
                        '(default 4).')
    # FENE8 one-axis change: per-rate velocity-loss rebalancing.
    # DEFAULT legacy => bit-identical to the pre-flag path (P0/legacy arm).
    p.add_argument('--rate-balance', choices=('legacy', 'equal'),
                   default='legacy',
                   help='legacy (default): alpha_v = 1/su2 in multi-flow '
                        '(unchanged). equal: rescale those factors by static '
                        'constants from the ROI weight field and truth '
                        'velocity so each rate has equal effective '
                        'velocity-loss mass (PR9b mass = alpha_v * sum(w)).')
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(p)
    return args


def main():
    args = parse_args()
    if args.beta is not None:               # gauge by solvent fraction
        nu_b = args.beta * args.eta0
        Gp_b = (1.0 - args.beta) * args.eta0 / args.truth_lam
        print(f"[gauge] beta={args.beta:.4f} eta0={args.eta0} -> "
              f"truth Gp={Gp_b:.4f} nu_s={nu_b:.4f} (was {args.truth_gp},{args.truth_nus})")
        args.truth_gp, args.truth_nus = Gp_b, nu_b
    if args.run_name is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f'ctr_{args.truth_model}_{args.scheme}_{stamp}'
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, 'DONE')) and not args.mem_smoke:
        print(f"[resume] {run_dir}/DONE present -- run already complete; exit.",
              flush=True)
        return 0
    print(f"[setup] device = {jax.devices()}")
    print(f"[setup] run dir = {run_dir}")
    bound_c = float(args.bound_c)
    Gp_t, lam_t, nu_t = args.truth_gp, args.truth_lam, args.truth_nus

    # -- build the contraction grid + truth/TBNN states ----------------------
    H, R = args.H, args.ratio
    L_up, L_down = args.L_up * H, args.L_down * H
    if args.nx is not None and args.ny is not None:
        # explicit (possibly non-square) grid, e.g. production 128x256; the wall
        # smear tracks the wall-NORMAL cell (dy), matching the sizing runs.
        from jax_ib.base import grids as _grids
        domain = cg.contraction_domain(H, L_up, L_down, R)
        grid = _grids.Grid((int(args.nx), int(args.ny)), domain=domain)
        dx, dy = grid.step
        wlog = 0.5 * dy
        print(f"[grid] explicit {args.nx}x{args.ny} dx={dx:.4g} dy={dy:.4g} "
              f"wlog=0.5*dy={wlog:.4g}")
    else:
        grid = cg.make_contraction_grid(H, L_up, L_down,
                                        cells_per_H=args.cells_per_H,
                                        contraction_ratio=R)
        dx, _dy = grid.step
        wlog = 0.5 * dx
    truth_model = cr.get_model(TRUTH_NAME[args.truth_model])
    tbnn_model = cr.get_model(TBNN_NAME)
    # multi-flow: --U-list overrides --U; single-element list keeps the
    # existing single-rate path bit-identical (no per-rate reweighting).
    if args.U_list:
        U_list = [float(x) for x in args.U_list.split(',') if x.strip()]
        if not U_list:
            raise SystemExit('[HALT] --U-list parsed empty')
    else:
        U_list = [float(args.U)]
    # FENE7 curriculum: build truth for ALL staged rates; stage 1 trains on the
    # first rate only (active_rates), activation appends the rest. multi_flow
    # True => per-rate alpha_v=1/su2 (the dual normalization) throughout.
    curriculum = args.curriculum_ulist is not None
    if curriculum:
        cur_list = [float(x) for x in args.curriculum_ulist.split(',')
                    if x.strip()]
        if len(cur_list) < 2:
            raise SystemExit('[HALT] --curriculum-ulist needs >=2 rates '
                             '(e.g. "0.5,4").')
        U_list = cur_list
    multi_flow = len(U_list) > 1
    # BC / initial state: U_inlet only sets the impulsive BC constant; the
    # ramped inlet used in evolve_contraction is the U_inlet kwarg, so one
    # state is fine for all rates (ramp overrides). Use U_list[0] for BC.
    U0 = U_list[0]
    truth_state, perm_f, bc_spec = cg.build_contraction_viscoelastic_state(
        grid, H=H, L_down=L_down, U_inlet=U0, logistic_width=wlog,
        model=truth_model, contraction_ratio=R)
    tbnn_state, perm_t, bc_t = cg.build_contraction_viscoelastic_state(
        grid, H=H, L_down=L_down, U_inlet=U0, logistic_width=wlog,
        model=tbnn_model, contraction_ratio=R)
    De = lam_t * R * U0 / H
    T_final = args.outer * args.inner * args.dt
    print(f"[setup] truth={args.truth_model} scheme={args.scheme} "
          f"Gp={Gp_t} lam={lam_t} nu_s={nu_t} "
          f"{'alpha=%.3f' % args.truth_alpha if args.truth_model=='giesekus' else 'Lsq=%.1f' % args.truth_lsq}")
    _lw = 'uniform' if args.no_stretch_weight else args.loss_weight
    print(f"[setup] contraction U_list={U_list} multi_flow={multi_flow} "
          f"De(U0)={De:.3g} grid={grid.shape} "
          f"steps={args.inner*args.outer} T={T_final:.3g}={T_final/lam_t:.2f}lam "
          f"loss_weight={_lw} beta={nu_t/(nu_t+Gp_t*lam_t):.3f}")

    def _evolve(state, model, params, nu, U):
        _final, out = cf.evolve_contraction(
            state, model, params, grid, density=args.density,
            base_viscosity=nu, dt=args.dt, inner_steps=args.inner,
            outer_steps=args.outer, U_inlet=U, ramp_time=args.ramp_time,
            perm_f=perm_f, bc_spec=bc_spec, solver_type='bicgstab',
            solver_tol=args.solver_tol, solver_maxiter=args.solver_maxiter)
        return out

    # -- per-rate truth + ROI/stretch weight + pressure truth ------------------
    pressure_on = (args.w_p_scale is not None) or (float(args.w_p) > 0.0)
    factor = args.density / (args.inner * args.dt)
    tap_idx = None
    if pressure_on:
        Xc, Yc = grid.mesh(grid.cell_center)
        xc = np.asarray(Xc)[:, 0]; yc = np.asarray(Yc)[0, :]
        tap_idx = [bilinear_idx(xc, yc, x, y) for (x, y) in TAPS_PHYS]
        print(f"[pressure] taps(phys)={TAPS_PHYS} factor=rho/(inner*dt)="
              f"{factor:.4g}", flush=True)

    truth_pp = dict(Gp=jnp.asarray(Gp_t, jnp.float64),
                    lam=jnp.asarray(lam_t, jnp.float64))
    if args.truth_model == 'giesekus':
        truth_pp['alpha'] = jnp.asarray(args.truth_alpha, jnp.float64)
    else:
        truth_pp['Lsq'] = jnp.asarray(args.truth_lsq, jnp.float64)

    rate_data = {}
    health_by_U = {}          # norms-dump path only: KE/max_Axx/psi_min/dp3 vs t
    lw = 'uniform' if args.no_stretch_weight else args.loss_weight
    for U in U_list:
        print(f"[truth] generating contraction truth @ U={U} ...", flush=True)
        t0 = time.time()
        out_truth = jax.jit(lambda U=U: _evolve(
            truth_state, truth_model, truth_pp, nu_t, U))()
        u_truth_U = out_truth['u_traj']
        v_truth_U = out_truth['v_traj']
        u_truth_U.block_until_ready()
        trA = (np.asarray(out_truth['A_xx_traj'])
               + np.asarray(out_truth['A_yy_traj'])
               + np.asarray(out_truth['A_zz_traj']))
        if lw == 'roi':
            ubar = R * U
            w2d, w_raw = roi_weight(grid, x_c=args.roi_xc, y_c=0.0, ubar=ubar,
                                    lam=lam_t, a=args.roi_a, c=args.roi_c,
                                    ell=args.roi_ell, sigma_y=args.roi_sigma_y,
                                    kappa=args.roi_kappa)
            w_raw_np = np.asarray(w_raw)
            finite = bool(np.all(np.isfinite(np.asarray(w2d))))
            floor_ok = abs(float(w_raw_np.min()) - 1.0) < 1e-6
            peak = float(w_raw_np.max())
            peak_ok = peak > 1.0 + 0.5 * args.roi_kappa
            X, Y = grid.mesh(grid.cell_center)
            X = np.asarray(X); Y = np.asarray(Y)
            ic = np.unravel_index(np.argmin((X - args.roi_xc) ** 2
                                            + (Y - args.H) ** 2), X.shape)
            corner = float(w_raw_np[ic])
            corner_ok = corner < 1.2
            x_off = args.roi_xc + args.roi_c * ubar * lam_t
            print(f"[roi U={U}] band x_on={args.roi_xc-args.roi_a:.2f} "
                  f"x_off={x_off:.2f}H Delta={ubar*lam_t:.3f} finite={finite} "
                  f"min_raw={w_raw_np.min():.4f} peak={peak:.3f} "
                  f"corner(xc,H)={corner:.3f}", flush=True)
            if not (finite and floor_ok and peak_ok and corner_ok):
                raise SystemExit(f"[HALT] ROI weight sanity failed @U={U}: "
                                 f"finite={finite} floor_ok={floor_ok} "
                                 f"peak_ok={peak_ok} corner_ok={corner_ok}")
            _save_roi_png(grid, w2d,
                          os.path.join(run_dir, f'roi_weight_U{U:g}.png'), x_off)
            w_U = jnp.asarray(w2d, jnp.float64)
        elif lw == 'uniform':
            w_U = jnp.ones_like(u_truth_U)
        else:
            stretch = np.clip(trA - 3.0, 0.0, None)
            mw = stretch.mean()
            w_U = jnp.asarray(stretch / mw if mw > 0 else np.ones_like(stretch),
                              jnp.float64)
        su2 = float(jnp.sum(w_U * (u_truth_U ** 2 + v_truth_U ** 2))
                    / (2.0 * jnp.sum(w_U)))
        dp_truth_U = None
        sub_j_U = None
        sp2 = 1.0
        tap_norm_U = None
        tap_w_U = None
        if pressure_on:
            dp_truth_U = jax.lax.stop_gradient(
                dp_from_ptraj(out_truth['p_traj'], tap_idx, factor))
            tframe = (np.arange(args.outer) + 1) * args.inner * args.dt
            post = np.where(tframe > args.ramp_time)[0]
            if len(post) < args.n_sub:
                post = np.arange(args.outer)
            sub = np.unique(np.clip(
                np.linspace(post[0], post[-1], args.n_sub).round().astype(int),
                0, args.outer - 1))
            sub_j_U = jnp.asarray(sub)
            sp2 = float(np.mean(np.asarray(dp_truth_U)[sub] ** 2))
            # per-tap normalization (paper doctrine, one level below alpha_p):
            # each tap difference k in {dp2,dp3,dp4} is normalized by the SQUARE
            # of its max-over-(post-ramp)-time |truth|. Truth-only, computed once,
            # stored in metadata; near-zero taps (|.|<1e-8) are dropped (w=0).
            dp_np = np.asarray(dp_truth_U)
            tap_norm_U = np.max(np.abs(dp_np[post]), axis=0)          # (3,)
            tap_w_np = np.zeros(3, dtype=np.float64)
            dropped = []
            for _k in range(3):
                if tap_norm_U[_k] < 1e-8:
                    dropped.append(('dp2', 'dp3', 'dp4')[_k])
                else:
                    tap_w_np[_k] = 1.0 / (tap_norm_U[_k] ** 2)
            tap_w_U = jnp.asarray(tap_w_np, jnp.float64)
            print(f"[pressure U={U}] n_sub={len(sub)} sub={sub.tolist()} "
                  f"su2={su2:.4g} sp2={sp2:.4g}", flush=True)
            print(f"[pressure U={U}] per-tap |truth|_max(dp2,dp3,dp4)="
                  f"{[float(v) for v in tap_norm_U]} tap_w={tap_w_np.tolist()}"
                  + (f"  DROPPED={dropped}" if dropped else ""), flush=True)
        print(f"[truth U={U}] forward={time.time()-t0:.1f}s "
              f"max|u|={float(jnp.max(jnp.abs(u_truth_U))):.3f} "
              f"max trA={trA.max():.3f}", flush=True)
        # multi-flow: normalize each rate by its own sigma^2 so U=4 does not
        # dominate; single-flow: alpha=1 keeps the existing unnormalized MSE.
        # --rate-balance equal (applied after the per-rate loop) may rescale
        # alpha_v; legacy leaves these values untouched.
        if multi_flow:
            alpha_v = 1.0 / max(su2, 1e-300)
            alpha_p = 1.0 / max(sp2, 1e-300)
        else:
            alpha_v = 1.0
            alpha_p = 1.0
        roi_mass = float(jnp.sum(w_U))
        vel_mass = float(jnp.sum(w_U * (u_truth_U ** 2 + v_truth_U ** 2)))
        if args.dump_norms is not None:
            tf = (np.arange(args.outer) + 1) * args.inner * args.dt
            uu = np.asarray(u_truth_U); vv = np.asarray(v_truth_U)
            Axx_t = np.asarray(out_truth['A_xx_traj'])
            Ayy_t = np.asarray(out_truth['A_yy_traj'])
            Azz_t = np.asarray(out_truth['A_zz_traj'])
            diagmin = np.minimum(np.minimum(Axx_t, Ayy_t), Azz_t)
            health_by_U[U] = dict(
                t=tf.tolist(),
                ke=(0.5 * np.mean(uu ** 2 + vv ** 2, axis=(1, 2))).tolist(),
                max_Axx=Axx_t.max(axis=(1, 2)).tolist(),
                psi_min=diagmin.min(axis=(1, 2)).tolist(),   # SPD-diag proxy
                dp3=(np.asarray(dp_truth_U)[:, 1].tolist()
                     if dp_truth_U is not None else [0.0] * args.outer),
                finite=bool(np.all(np.isfinite(uu))
                            and np.all(np.isfinite(Axx_t))))
        rate_data[U] = dict(u_truth=u_truth_U, v_truth=v_truth_U, w=w_U,
                            dp_truth=dp_truth_U, sub_j=sub_j_U,
                            su2=su2, sp2=sp2,
                            alpha_v=alpha_v, alpha_p=alpha_p,
                            tap_norm=tap_norm_U, tap_w=tap_w_U,
                            max_trA=float(trA.max()),
                            roi_mass=roi_mass, vel_mass=vel_mass,
                            alpha_v_legacy=float(alpha_v),
                            balance_scale=1.0)

    # --rate-balance equal: rescale legacy alpha_v so effective velocity-loss
    # masses alpha_v * roi_mass are equal across rates. Constants are static
    # (ROI weight + truth velocity only; no conformation). legacy = no-op.
    rate_balance_info = dict(mode=args.rate_balance, scales={}, masses={},
                             mass_ratio_legacy=None, mass_ratio_balanced=None)
    if args.rate_balance == 'equal':
        if not multi_flow or len(U_list) < 2:
            raise SystemExit(
                "[HALT] --rate-balance equal requires multi-flow (>=2 rates).")
        # Effective mass (PR9b): alpha_v_legacy * sum(w). vel_mass recorded for
        # provenance (su2 = vel_mass / (2*roi_mass)); equalization target is
        # the PR9b ROI-weight mass that measured ~114:1 under legacy.
        masses = {U: float(rate_data[U]['alpha_v_legacy']
                           * rate_data[U]['roi_mass'])
                  for U in U_list}
        # Geometric-mean target keeps the product of alphas stable.
        log_sum = sum(math.log(max(m, 1e-300)) for m in masses.values())
        target = math.exp(log_sum / len(masses))
        for U in U_list:
            scale = target / max(masses[U], 1e-300)
            rate_data[U]['balance_scale'] = float(scale)
            rate_data[U]['alpha_v'] = float(
                rate_data[U]['alpha_v_legacy'] * scale)
            rate_balance_info['scales'][f'{U:g}'] = float(scale)
            rate_balance_info['masses'][f'{U:g}'] = dict(
                roi_mass=float(rate_data[U]['roi_mass']),
                vel_mass=float(rate_data[U]['vel_mass']),
                su2=float(rate_data[U]['su2']),
                alpha_v_legacy=float(rate_data[U]['alpha_v_legacy']),
                alpha_v=float(rate_data[U]['alpha_v']),
                eff_mass_legacy=float(masses[U]),
                eff_mass=float(rate_data[U]['alpha_v']
                               * rate_data[U]['roi_mass']),
            )
        m0 = masses[U_list[0]]
        m1 = masses[U_list[1]]
        rate_balance_info['mass_ratio_legacy'] = float(m0 / max(m1, 1e-300))
        mb0 = rate_data[U_list[0]]['alpha_v'] * rate_data[U_list[0]]['roi_mass']
        mb1 = rate_data[U_list[1]]['alpha_v'] * rate_data[U_list[1]]['roi_mass']
        rate_balance_info['mass_ratio_balanced'] = float(mb0 / max(mb1, 1e-300))
        print(f"[rate-balance] mode=equal scales={rate_balance_info['scales']} "
              f"mass_ratio_legacy(U0:U1)="
              f"{rate_balance_info['mass_ratio_legacy']:.6g} "
              f"mass_ratio_balanced="
              f"{rate_balance_info['mass_ratio_balanced']:.6g}",
              flush=True)
        # Versioned norms file for balanced runs (legacy norms untouched).
        bal_norms = os.path.join(run_dir, 'norms_v3_ratebalance_equal.json')
        cfg_hash_bal, cfg_dict_bal = _campaign_config_hash(args, U_list)
        bal_payload = dict(
            config_hash=cfg_hash_bal, config=cfg_dict_bal,
            rate_balance=rate_balance_info,
            parent_norms=args.norms_json,
            per_rate={
                f'{U:g}': dict(
                    U=float(U),
                    su2=float(rate_data[U]['su2']),
                    sp2=float(rate_data[U]['sp2']),
                    roi_mass=float(rate_data[U]['roi_mass']),
                    vel_mass=float(rate_data[U]['vel_mass']),
                    alpha_v_legacy=float(rate_data[U]['alpha_v_legacy']),
                    balance_scale=float(rate_data[U]['balance_scale']),
                    alpha_v=float(rate_data[U]['alpha_v']),
                    alpha_p=float(rate_data[U]['alpha_p']),
                    tap_norm=([float(v) for v in rate_data[U]['tap_norm']]
                              if rate_data[U]['tap_norm'] is not None else None),
                    tap_w=([float(v) for v in np.asarray(rate_data[U]['tap_w'])]
                           if rate_data[U]['tap_w'] is not None else None),
                )
                for U in U_list
            },
        )
        with open(bal_norms, 'w') as f:
            json.dump(bal_payload, f, indent=2)
        rate_balance_info['norms_json'] = bal_norms
        print(f"[rate-balance] wrote {bal_norms}", flush=True)
    elif args.rate_balance == 'legacy':
        print("[rate-balance] mode=legacy (no rescale; bit-identical path)",
              flush=True)
        bal_norms = None
    else:
        raise SystemExit(f"[HALT] unknown --rate-balance {args.rate_balance}")

    # back-compat aliases used by later logging (primary rate)
    u_truth = rate_data[U0]['u_truth']
    v_truth = rate_data[U0]['v_truth']
    w = rate_data[U0]['w']
    dp_truth = rate_data[U0]['dp_truth']
    sub_j = rate_data[U0]['sub_j']

    def _gp_of(fit):
        return jnp.maximum(fit['Gp'], 1e-4)

    def _lam_of(fit):
        return jnp.maximum(fit['lam'], 1e-4)

    def _nu_of(fit):
        return jnp.maximum(fit['nu_s'], 1e-4)

    def _tbnn_out(fit, U):
        params = {'Gp': _gp_of(fit), 'lam': _lam_of(fit),
                  'theta': fit['theta'], 'tbnn_bound_c': bound_c}
        return _evolve(tbnn_state, tbnn_model, params, _nu_of(fit), U)

    # -- FENE7 curriculum: mutable active-rate set + forward envelope gate -----
    # active_rates == U_list unless curriculum is ON, in which case stage 1 uses
    # only the first rate and activation appends the rest. The loss functions
    # iterate active_rates, so with curriculum OFF the code path is unchanged.
    active_rates = [U0] if curriculum else list(U_list)
    _Xg, _Yg = grid.mesh(grid.cell_center)
    _xg = np.asarray(_Xg)[:, 0]
    _yg = np.asarray(_Yg)[0, :]

    def _gate_locus(x, y):
        ay = abs(y)
        if x <= -L_up + 1.0 and ay >= R * H - 1.0:
            return 'inlet_corner'
        if abs(x) <= 0.7 and (H - 0.4) <= ay <= (H + 0.6):
            return 'reentrant_lip'
        if -0.2 <= x <= 1.5 and ay <= H:
            return 'throat'
        return 'upstream' if x < 0 else 'downstream'

    def _forward_gate(cur_fit, U_gate, lam_margin):
        """Forward-only screen of (theta,scalars) at U_gate: as-is and
        lam x lam_margin. PASS = both finite through T. Returns (pass, info)."""
        info = {'U_gate': float(U_gate)}
        ok_all = True
        for tag, lam_fac in (('asis', 1.0),
                             (f'lamx{lam_margin:g}', float(lam_margin))):
            try:
                params = {'Gp': _gp_of(cur_fit),
                          'lam': _lam_of(cur_fit) * lam_fac,
                          'theta': cur_fit['theta'], 'tbnn_bound_c': bound_c}
                out = jax.jit(lambda p=params: _evolve(
                    tbnn_state, tbnn_model, p, _nu_of(cur_fit), U_gate))()
                out['u_traj'].block_until_ready()
                anan = np.asarray(out['any_nan_traj']).astype(bool)
                trA = (np.asarray(out['A_xx_traj']) + np.asarray(out['A_yy_traj'])
                       + np.asarray(out['A_zz_traj']))
                flat = trA.reshape(trA.shape[0], -1)
                fin_frames = (~anan) & np.all(np.isfinite(flat), axis=1)
                fin_idx = np.where(fin_frames)[0]
                lf = int(fin_idx[-1]) if fin_idx.size else 0
                finite = bool((not anan.any()) and np.all(np.isfinite(trA)))
                pk = np.unravel_index(int(np.nanargmax(trA[lf])), trA[lf].shape)
                px, py = float(_xg[pk[0]]), float(_yg[pk[1]])
                info[tag] = dict(
                    finite=finite, max_trA=float(np.nanmax(trA[lf])),
                    first_nan_t=(float((int(np.where(anan)[0][0]) + 1)
                                       * args.inner * args.dt)
                                 if anan.any() else None),
                    peak_x=px, peak_y=py, locus=_gate_locus(px, py))
            except Exception as e:  # record-don't-fail: a crash is a gate FAIL
                finite = False
                info[tag] = dict(finite=False, err=repr(e)[:150])
            ok_all = ok_all and finite
        info['PASS'] = bool(ok_all)
        return bool(ok_all), info

    curr_path = os.path.join(run_dir, 'curriculum.json')
    curr_log = dict(enabled=bool(curriculum),
                    curriculum_ulist=(list(U_list) if curriculum else None),
                    gate_after=int(args.curriculum_gate_after),
                    gate_lam_margin=float(args.curriculum_gate_lam_margin),
                    max_pairs=int(args.curriculum_max_pairs),
                    stage1_rate=float(U0), activated=False,
                    activation_theta_step=None, gate_attempts=[])
    if curriculum and not args.mem_smoke and os.path.exists(curr_path):
        try:
            with open(curr_path) as _f:
                curr_log = json.load(_f)
            if curr_log.get('activated'):
                active_rates[:] = list(U_list)   # resume already-activated dual
            print(f"[curriculum] resumed curriculum.json activated="
                  f"{curr_log.get('activated')} "
                  f"act_step={curr_log.get('activation_theta_step')}", flush=True)
        except Exception as _e:
            print(f"[curriculum] failed to reload {curr_path}: {_e}", flush=True)

    def _save_curr():
        if curriculum:
            with open(curr_path, 'w') as _f:
                json.dump(curr_log, _f, indent=2, default=float)

    theta0, _cfgth = tb.init_tbnn_theta(jax.random.PRNGKey(args.seed),
                                        width=args.width, depth=args.depth,
                                        bound_c=bound_c)
    fit = {'theta': theta0,
           'Gp': jnp.asarray(args.gp_init, jnp.float64),
           'lam': jnp.asarray(args.lam_init, jnp.float64),
           'nu_s': jnp.asarray(args.nus_init, jnp.float64)}

    w_p = float(args.w_p)
    w_bal = None
    w_p_by_U = {U: w_p for U in U_list}
    if pressure_on and args.w_p_scale is not None:
        w_bal_by_U = {}
        for U in U_list:
            rd = rate_data[U]
            out0 = jax.jit(lambda fit, U=U: _tbnn_out(fit, U))(fit)
            du0 = out0['u_traj'] - rd['u_truth']
            dv0 = out0['v_traj'] - rd['v_truth']
            Lv0i = float(jnp.sum(rd['w'] * du0 * du0)
                         + jnp.sum(rd['w'] * dv0 * dv0))
            Lv0i *= rd['alpha_v']
            dp0 = dp_from_ptraj(out0['p_traj'], tap_idx, factor)
            r0 = (dp0 - rd['dp_truth'])[rd['sub_j']]
            R0i = float(jnp.sum(r0 * r0 * rd['tap_w']))   # per-tap normalized
            wb = Lv0i / max(R0i, 1e-300)
            w_bal_by_U[U] = wb
            w_p_by_U[U] = float(args.w_p_scale) * wb
            print(f"[pressure U={U}] init Lv0(scaled)={Lv0i:.6e} "
                  f"R0(scaled)={R0i:.6e} w_bal={wb:.6g} "
                  f"scale={args.w_p_scale} => w_p={w_p_by_U[U]:.6g} "
                  f"(alpha_v={rd['alpha_v']:.4g} alpha_p={rd['alpha_p']:.4g})",
                  flush=True)
        w_bal = w_bal_by_U[U0]
        w_p = w_p_by_U[U0]
    elif pressure_on:
        for U in U_list:
            w_p_by_U[U] = w_p
    if pressure_on:
        print(f"[pressure] ACTIVE multi_flow={multi_flow} w_p_by_U="
              f"{{{', '.join(f'{U:g}:{w_p_by_U[U]:.4g}' for U in U_list)}}}",
              flush=True)
    else:
        print("[pressure] OFF (velocity-only, bit-identical path)", flush=True)

    # -- optional norms dump / production norms verification -------------------
    cfg_hash, cfg_dict = _campaign_config_hash(args, U_list)
    print(f"[config] campaign hash = {cfg_hash}", flush=True)
    if args.dump_norms is not None:
        if not pressure_on or w_bal is None:
            raise SystemExit("[HALT] --dump-norms requires pressure on with "
                             "--w-p-scale 1.0 so w_bal_new is derived.")
        norms = dict(config_hash=cfg_hash, config=cfg_dict,
                     taps_phys=TAPS_PHYS, p_factor=float(factor),
                     n_sub=int(args.n_sub), per_rate={})
        for U in U_list:
            rd = rate_data[U]
            norms['per_rate'][f'{U:g}'] = dict(
                U=float(U),
                tap_norm=[float(v) for v in rd['tap_norm']],
                tap_w=[float(v) for v in np.asarray(rd['tap_w'])],
                su2=float(rd['su2']), sp2=float(rd['sp2']),
                alpha_v=float(rd['alpha_v']),
                w_bal=float(w_bal_by_U[U]))
        odir = os.path.dirname(os.path.abspath(args.dump_norms))
        os.makedirs(odir, exist_ok=True)
        with open(args.dump_norms, 'w') as f:
            json.dump(norms, f, indent=2)
        hp = args.dump_norms.replace('.json', '') + '_health.json'
        with open(hp, 'w') as f:
            json.dump({f'{U:g}': health_by_U[U] for U in U_list}, f)
        print(f"[dump-norms] wrote {args.dump_norms} (+_health.json) "
              f"hash={cfg_hash}; exiting before optimization.", flush=True)
        return 0
    if args.norms_json is not None:
        with open(args.norms_json) as f:
            ref = json.load(f)
        if ref.get('config_hash') != cfg_hash:
            raise SystemExit(
                f"[HALT] norms.json hash {ref.get('config_hash')} != this job "
                f"{cfg_hash} -- config drift; refusing to run.")
        for U in U_list:
            if f'{U:g}' not in ref['per_rate']:
                raise SystemExit(
                    f"[HALT] rate U={U:g} absent from {args.norms_json}.")
            if rate_data[U]['tap_norm'] is None:
                # pressure-off (w_p=0) job: config hash already verified above;
                # no per-tap norms to cross-check.
                continue
            got = np.asarray([float(v) for v in rate_data[U]['tap_norm']])
            exp = np.asarray(ref['per_rate'][f'{U:g}']['tap_norm'])
            if not np.allclose(got, exp, rtol=1e-6, atol=1e-12):
                raise SystemExit(
                    f"[HALT] U={U} tap_norm {got.tolist()} != norms.json "
                    f"{exp.tolist()} -- truth regen drift; refusing to run.")
            # w_bal is flow-mode dependent (single vs multi); each job re-derives
            # it deterministically, so this is informational, not a gate.
            if pressure_on and args.w_p_scale is not None:
                wb_got = float(w_bal_by_U[U])
                wb_exp = float(ref['per_rate'][f'{U:g}'].get('w_bal', float('nan')))
                print(f"[norms-json] U={U:g} w_bal(this job, flow-mode)={wb_got:.6g}"
                      f"  recorded(dump)={wb_exp:.6g}", flush=True)
        print(f"[norms-json] verified against {args.norms_json} "
              f"hash={cfg_hash} (rate-indep config + per-tap norms match).",
              flush=True)

    def loss_fn(fit):
        total = 0.0
        for U in active_rates:
            rd = rate_data[U]
            out = _tbnn_out(fit, U)
            du = out['u_traj'] - rd['u_truth']
            dv = out['v_traj'] - rd['v_truth']
            Lv = (jnp.sum(rd['w'] * du * du) + jnp.sum(rd['w'] * dv * dv)
                  ) * rd['alpha_v']
            term = Lv
            if pressure_on and w_p_by_U[U] > 0.0:
                dp = dp_from_ptraj(out['p_traj'], tap_idx, factor)
                r = (dp - rd['dp_truth'])[rd['sub_j']]
                term = term + w_p_by_U[U] * jnp.sum(r * r * rd['tap_w'])
            total = total + term
        return total / float(len(active_rates))

    def _loss_vp(fit):
        Lv_sum = 0.0
        Lp_sum = 0.0
        split = {}                     # per-rate (raw_SSE, norm_SSE) per tap
        for U in active_rates:
            rd = rate_data[U]
            out = _tbnn_out(fit, U)
            du = out['u_traj'] - rd['u_truth']
            dv = out['v_traj'] - rd['v_truth']
            Lv = (jnp.sum(rd['w'] * du * du) + jnp.sum(rd['w'] * dv * dv)
                  ) * rd['alpha_v']
            Lv_sum = Lv_sum + Lv
            if pressure_on:
                dp = dp_from_ptraj(out['p_traj'], tap_idx, factor)
                r = (dp - rd['dp_truth'])[rd['sub_j']]
                raw = jnp.sum(r * r, axis=0)          # (3,) raw per-tap SSE
                norm = raw * rd['tap_w']              # (3,) normalized per-tap
                Lp_sum = Lp_sum + w_p_by_U[U] * jnp.sum(norm)
                split[U] = (raw, norm)
        n = float(len(active_rates))
        return Lv_sum / n, Lp_sum / n, split
    # Rebuildable jit holder: the loss closures read the mutable `active_rates`,
    # so at curriculum activation we re-jit (the traced rate-loop changes length
    # exactly once). With curriculum OFF nothing is ever rebuilt.
    J = {}
    J['eval_vp'] = jax.jit(_loss_vp) if pressure_on else None
    J['vg'] = jax.jit(jax.value_and_grad(loss_fn))
    eval_vp = J['eval_vp']            # alias for back-compat references
    vg = J['vg']
    print("[opt] warm-compiling value_and_grad (reverse-mode through the "
          "unrolled contraction solve) ...", flush=True)
    t0 = time.time()
    L0, g0 = vg(fit)
    L0 = float(L0)
    jax.block_until_ready(g0)
    gnorm = float(jnp.sqrt(sum(jnp.sum(x * x) for x in
                               jax.tree_util.tree_leaves(g0))))
    print(f"[opt] vag warm = {time.time()-t0:.1f}s  loss(init)={L0:.6e}  "
          f"|grad|={gnorm:.3e}  init Gp={float(_gp_of(fit)):.3f} "
          f"lam={float(_lam_of(fit)):.3f} nu_s={float(_nu_of(fit)):.3f}",
          flush=True)
    _report_mem('after vag warm')
    if args.grad_probe:
        # reverse-mode stability characterization: report the init loss/grad
        # finiteness (the exact production forward+backward) and each rate's
        # truth stretch proximity to the FENE Lsq, then exit cleanly (0) even
        # when NaN -- so a shell can sweep rates without aborting.
        finite = bool(np.isfinite(L0) and np.isfinite(gnorm))
        trA_str = " ".join(f"U{U:g}:maxtrA={rate_data[U].get('max_trA', float('nan')):.4f}"
                           for U in U_list)
        print(f"[GRAD-PROBE] U_list={U_list} loss_init={L0:.6e} "
              f"grad_norm={gnorm:.6e} finite={finite} Lsq={args.truth_lsq} "
              f"{trA_str}", flush=True)
        return 0
    # -- resume detection (full-state checkpoint; disabled for mem-smoke).
    # Loaded BEFORE the init-finiteness gate: an already-activated curriculum
    # resume warm-compiles vg on the FRESH init theta at the dual rate set, and
    # a fresh OB theta NaNs at the activated U=4 -- a FALSE positive, since the
    # trained (finite-when-saved) theta is restored in the driver. Only a
    # genuine FRESH start (no checkpoint) may abort on non-finite init.
    resume = None if args.mem_smoke else _load_train_ckpt(run_dir)
    if not np.isfinite(L0) or not np.isfinite(gnorm):
        if resume is None:
            raise SystemExit("[FATAL] non-finite init loss/grad -- aborting.")
        print("[resume] fresh-init loss/grad non-finite -- expected for an "
              "activated-curriculum resume (fresh OB theta at the dual U); the "
              "checkpointed theta is restored in the driver. Skipping abort.",
              flush=True)

    def _scalars_str(fit):
        return (f"Gp={float(_gp_of(fit)):.4f} lam={float(_lam_of(fit)):.4f} "
                f"nu_s={float(_nu_of(fit)):.4f}")

    if resume is not None:
        print(f"[resume] train_ckpt.pkl: block={resume['b']} "
              f"i_theta={resume['i_theta']} scalars_done={resume['scalars_done']} "
              f"gstep={resume['gstep']} best={resume['best_loss']:.6e} "
              f"hist_len={len(resume['loss_hist'])}", flush=True)

    # =====================================================================
    # Scheme machinery -- reproduced VERBATIM from visco_opt_tbnn_run.py
    # (only the loss_fn/vg above changed). multi_transform label, theta &
    # scalar optaxes, persistent-Adam alternating driver, L-BFGS scalar solve.
    # =====================================================================
    def _label(params):
        lab = {'theta': jax.tree_util.tree_map(lambda _: 'theta', params['theta'])}
        for k in ('Gp', 'lam', 'nu_s'):
            if k in params:
                lab[k] = 'scalars'
        return lab

    def _theta_opt_p(peak, warmup, n_steps):
        sched = optax.warmup_cosine_decay_schedule(
            init_value=peak * 0.05, peak_value=peak,
            warmup_steps=max(1, warmup),
            decay_steps=max(warmup + 1, n_steps), end_value=peak * 0.02)
        chain = []
        if args.clip > 0:
            chain.append(optax.clip_by_global_norm(args.clip))
        chain.append(optax.adam(sched))
        return optax.chain(*chain)

    def _theta_opt(n_steps):
        return _theta_opt_p(args.lr, args.warmup, n_steps)

    progress_path = os.path.join(run_dir, 'progress.csv')
    press_path = os.path.join(run_dir, 'pressure.csv')
    if resume is None:
        with open(progress_path, 'w') as pf:
            # additive columns (grad_norm,reject,wall_s); loss+scalars unchanged
            pf.write("stage,step,loss,Gp,lam,nu_s,grad_norm,reject,wall_s\n")
            pf.write("0,0,%.6e,%.6f,%.6f,%.6f,%.6e,0,0.0\n" % (
                L0, float(_gp_of(fit)), float(_lam_of(fit)), float(_nu_of(fit)),
                gnorm))
        if pressure_on:
            with open(press_path, 'w') as pf:
                cols = ["stage", "step", "L_velocity", "L_pressure", "active_n"]
                for U in U_list:
                    for k in (2, 3, 4):
                        cols += [f"rawSSE_dp{k}_U{U:g}", f"normSSE_dp{k}_U{U:g}"]
                pf.write(",".join(cols) + "\n")
        loss_hist = [(0, L0, float(_nu_of(fit)))]
    else:
        loss_hist = [tuple(x) for x in resume['loss_hist']]
    t_opt = time.time()

    # L-BFGS-B scalar solve at the CURRENT theta (jit shared over (svec, theta)).
    def _loss_vec_th(svec, th):
        return loss_fn({'theta': th, 'Gp': svec[0], 'lam': svec[1],
                        'nu_s': svec[2]})
    J['vag_scalars'] = jax.jit(jax.value_and_grad(_loss_vec_th, argnums=0))

    def _rebuild_active_jits():
        """Re-jit the loss/grad closures after active_rates changes (curriculum
        activation). Called at most once per run."""
        J['vg'] = jax.jit(jax.value_and_grad(loss_fn))
        J['eval_vp'] = jax.jit(_loss_vp) if pressure_on else None
        J['vag_scalars'] = jax.jit(jax.value_and_grad(_loss_vec_th, argnums=0))

    def _lbfgs_scalars(cur_fit, maxiter, ftol, label):
        th = cur_fit['theta']
        x0 = np.array([float(_gp_of(cur_fit)), float(_lam_of(cur_fit)),
                       float(_nu_of(cur_fit))], dtype=np.float64)

        def _obj(x):
            val, g = J['vag_scalars'](jnp.asarray(x, dtype=jnp.float64), th)
            fv, fg = float(val), np.asarray(g, dtype=np.float64)
            loss_hist.append((len(loss_hist), fv, float(x[2])))
            gn = float(np.linalg.norm(fg)) if np.all(np.isfinite(fg)) else float('nan')
            with open(progress_path, 'a') as pf:
                pf.write(f"{label},{len(loss_hist)-1},{fv:.6e},"
                         f"{x[0]:.6f},{x[1]:.6f},{x[2]:.6f},{gn:.6e},0,0.0\n")
            return fv, fg

        res = minimize(_obj, x0, jac=True, method='L-BFGS-B',
                       bounds=[(args.scalar_bound_lo, args.scalar_bound_hi)] * 3,
                       options=dict(maxiter=maxiter, ftol=ftol, gtol=1e-10))
        nf = {'theta': th,
              'Gp': jnp.asarray(res.x[0], dtype=jnp.float64),
              'lam': jnp.asarray(res.x[1], dtype=jnp.float64),
              'nu_s': jnp.asarray(res.x[2], dtype=jnp.float64)}
        return nf, float(res.fun), int(res.nfev), int(res.nit)

    # alternating driver (s1/s1b/s4): persistent theta Adam state (cosine spans
    # the WHOLE run, never reset per block) + a fresh L-BFGS scalar re-solve
    # each block (full re-equilibration at the current theta).
    def _alternating(n_theta_block, n_scalar_iters, n_blocks, scalar_ftol):
        total_theta = n_theta_block * n_blocks
        topt = optax.multi_transform(
            {'theta': _theta_opt(total_theta),
             'scalars': optax.set_to_zero()}, _label)

        # tstep reads J['vg'] so a curriculum activation (which re-jits vg) is
        # picked up by rebuilding tstep with the SAME topt (persistent Adam
        # state -- optimizer/schedule never reset). Extra outputs (grad finite
        # flag, grad norm) drive the trajectory-health step guard; on a finite
        # step the applied update is identical to the proven path.
        def _make_tstep():
            @jax.jit
            def tstep(f, st):
                L, g = J['vg'](f)
                leaves = jax.tree_util.tree_leaves(g)
                gfin = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))
                                          for x in leaves]))
                gn = jnp.sqrt(sum(jnp.sum(x * x) for x in leaves))
                upd, st2 = topt.update(g, st, f)
                return optax.apply_updates(f, upd), st2, L, gfin, gn
            return tstep
        TS = {'tstep': _make_tstep()}

        # resume ONLY if the recipe matches (schedule/structure identical)
        _res = resume
        if _res is not None and tuple(_res.get('recipe', ())) != \
                (n_theta_block, n_scalar_iters, n_blocks):
            print("[resume] recipe mismatch; ignoring checkpoint, fresh start.",
                  flush=True)
            _res = None
        if _res is not None:
            cur = _tree_to_jnp(_res['fit'])
            tstate = _tree_to_jnp(_res['tstate'])
            best = [float(_res['best_loss']), _tree_to_jnp(_res['best_fit'])]
            gstep = int(_res['gstep'])
            b_start, i_start = int(_res['b']), int(_res['i_theta'])
            scalars_done0 = bool(_res['scalars_done'])
        else:
            cur = fit
            tstate = topt.init(fit)
            best = [float('inf'), cur]
            gstep = 0
            b_start, i_start, scalars_done0 = 0, 0, False

        state = {'cur': cur, 'tstate': tstate, 'gstep': gstep}
        _step_times = []                # per-tstep wall times (timing only)

        def _ckpt(b, i_theta, scalars_done, archive_block=None):
            obj = dict(fit=_tree_to_np(state['cur']),
                       tstate=_tree_to_np(state['tstate']),
                       best_fit=_tree_to_np(best[1]), best_loss=float(best[0]),
                       b=int(b), i_theta=int(i_theta),
                       scalars_done=bool(scalars_done), gstep=int(state['gstep']),
                       loss_hist=[list(x) for x in loss_hist],
                       recipe=(n_theta_block, n_scalar_iters, n_blocks),
                       w_p=float(w_p))
            _save_train_ckpt(run_dir, obj)
            # per-block archive for response-overlay / ridge diagnostics
            if archive_block is not None:
                import shutil
                src = os.path.join(run_dir, 'train_ckpt.pkl')
                dst = os.path.join(run_dir, f'ckpt_block{int(archive_block)}.pkl')
                shutil.copy2(src, dst)

        def _log_components(tag):
            if J['eval_vp'] is None:
                return
            Lv, Lp, split = J['eval_vp'](state['cur'])
            row = [tag, str(len(loss_hist) - 1),
                   f"{float(Lv):.6e}", f"{float(Lp):.6e}",
                   str(len(active_rates))]
            for U in U_list:      # inactive rates (curriculum stage 1) -> zeros
                if U in split:
                    raw, norm = split[U]
                    raw = np.asarray(raw); norm = np.asarray(norm)
                    for k in range(3):
                        row += [f"{float(raw[k]):.6e}", f"{float(norm[k]):.6e}"]
                else:
                    row += ["0.0", "0.0"] * 3
            with open(press_path, 'a') as pf:
                pf.write(",".join(row) + "\n")

        reject_run = [0]            # consecutive non-finite theta steps
        last_gn = [float('nan')]    # last accepted grad norm (for activation log)

        def _activate(b, theta_step):
            """Curriculum activation: append remaining rates, re-jit loss/grad
            (persistent Adam state + schedule preserved), reset best to track
            the dual loss, log grad-norm before/after."""
            gn_before = last_gn[0]
            active_rates[:] = list(U_list)
            _rebuild_active_jits()
            TS['tstep'] = _make_tstep()
            La, ga = J['vg'](state['cur'])
            gn_after = float(jnp.sqrt(sum(jnp.sum(x * x) for x in
                                          jax.tree_util.tree_leaves(ga))))
            best[0] = float('inf'); best[1] = state['cur']   # re-baseline
            curr_log['activated'] = True
            curr_log['activation_theta_step'] = int(theta_step)
            curr_log['grad_norm_before_activation'] = (
                None if not np.isfinite(gn_before) else float(gn_before))
            curr_log['grad_norm_after_activation'] = gn_after
            curr_log['loss_after_activation'] = float(La)
            _save_curr()
            print(f"[curriculum] ACTIVATED dual {list(U_list)} @ theta "
                  f"{theta_step} (block {b}); loss->dual  |g|_before="
                  f"{gn_before:.3e} |g|_after={gn_after:.3e} "
                  f"L_dual={float(La):.6e}", flush=True)

        completed = True
        for b in range(b_start, n_blocks):
            if not (b == b_start and scalars_done0):
                nf_fit, lsc, nf, ni = _lbfgs_scalars(
                    state['cur'], n_scalar_iters, scalar_ftol, f'sc{b}')
                sc_ok = (np.isfinite(lsc) and
                         all(np.isfinite(float(nf_fit[k]))
                             for k in ('Gp', 'lam', 'nu_s')))
                if sc_ok:
                    state['cur'] = nf_fit
                    if lsc < best[0]:
                        best = [lsc, state['cur']]
                    print(f"  [{args.scheme} blk {b}] L-BFGS scalars nfev={nf} "
                          f"nit={ni} loss={lsc:.6e}  {_scalars_str(state['cur'])}"
                          f"  [{time.time()-t_opt:.0f}s]", flush=True)
                else:
                    print(f"  [{args.scheme} blk {b}] L-BFGS scalars NON-FINITE "
                          f"(loss={lsc}); rejecting solve, keeping "
                          f"{_scalars_str(state['cur'])}", flush=True)
                _ckpt(b, 0, True)
            istart = i_start if b == b_start else 0
            for _i in range(istart, n_theta_block):
                prev = state['cur']
                _t_step0 = time.time()
                newf, newst, L, gfin, gn = TS['tstep'](prev, state['tstate'])
                Lf = float(L)                       # forces block_until_ready
                _dt_step = time.time() - _t_step0
                _step_times.append(_dt_step)
                step_ok = np.isfinite(Lf) and bool(gfin)
                if step_ok:
                    state['cur'], state['tstate'] = newf, newst
                    reject_run[0] = 0
                    last_gn[0] = float(gn)
                    state['gstep'] += 1
                    loss_hist.append((len(loss_hist), Lf, float(_nu_of(prev))))
                    if Lf < best[0]:
                        best = [Lf, prev]
                    if state['gstep'] % 25 == 0:
                        with open(progress_path, 'a') as pf:
                            pf.write(f"th{b},{len(loss_hist)-1},{Lf:.6e},"
                                     f"{float(_gp_of(prev)):.6f},"
                                     f"{float(_lam_of(prev)):.6f},"
                                     f"{float(_nu_of(prev)):.6f},"
                                     f"{float(gn):.6e},0,{_dt_step:.4f}\n")
                        print(f"  [{args.scheme} blk {b}] theta step "
                              f"{state['gstep']}/{total_theta} loss={Lf:.6e} "
                              f"best={best[0]:.4e} {_scalars_str(prev)} "
                              f"[{time.time()-t_opt:.0f}s]", flush=True)
                    if (_i + 1) % max(1, args.ckpt_every) == 0:
                        _ckpt(b, _i + 1, True)
                        _log_components(f'th{b}')
                else:
                    # trajectory-health guard: revert theta to best-so-far,
                    # keep the (un-poisoned) pre-step optimizer state, continue.
                    reject_run[0] += 1
                    state['cur'] = best[1]
                    with open(progress_path, 'a') as pf:
                        pf.write(f"th{b},{len(loss_hist)-1},{Lf:.6e},"
                                 f"{float(_gp_of(prev)):.6f},"
                                 f"{float(_lam_of(prev)):.6f},"
                                 f"{float(_nu_of(prev)):.6f},"
                                 f"{float(gn):.6e},1,{_dt_step:.4f}\n")
                    print(f"  [{args.scheme} blk {b}] STEP-GUARD REJECT "
                          f"#{reject_run[0]} (loss={Lf} gfin={bool(gfin)}); "
                          f"reverted to best={best[0]:.4e}", flush=True)
                    if reject_run[0] >= 3:
                        print(f"  [{args.scheme}] 3 consecutive rejections; "
                              "checkpointing and stopping.", flush=True)
                        _ckpt(b, _i + 1, True)
                        with open(os.path.join(run_dir, 'STEP_GUARD_STOP'),
                                  'w') as f:
                            f.write(f"3 consecutive non-finite theta steps @ "
                                    f"block {b} step {state['gstep']}\n")
                        return (best[0], best[1], False)
                if b == b_start and _i == istart:
                    _report_mem('after first theta step')
            # block boundary: archive train_ckpt -> ckpt_block{b}.pkl
            _ckpt(b + 1, 0, False, archive_block=b)
            # -- curriculum forward envelope gate (after completed block-pair) --
            if curriculum and not curr_log['activated']:
                kpair = b + 1                      # 1-based completed block-pairs
                if kpair >= args.curriculum_gate_after:
                    to_activate = [u for u in U_list if u not in active_rates]
                    Ug = to_activate[-1] if to_activate else U_list[-1]
                    print(f"[curriculum] gate attempt @ pair {kpair} "
                          f"(theta {state['gstep']}) U_gate={Ug} ...", flush=True)
                    passed, info = _forward_gate(
                        state['cur'], Ug, args.curriculum_gate_lam_margin)
                    info['block_pair'] = int(kpair)
                    info['theta_step'] = int(state['gstep'])
                    curr_log['gate_attempts'].append(info)
                    _save_curr()
                    print(f"[curriculum] gate @ pair {kpair}: PASS={passed} "
                          f"{info.get('asis')} {info.get(f'lamx{args.curriculum_gate_lam_margin:g}')}",
                          flush=True)
                    if passed:
                        _activate(b, state['gstep'])
                    elif kpair >= args.curriculum_max_pairs:
                        print(f"[curriculum] GATE_FAILED after {kpair} pairs "
                              "(cap reached); checkpoint + stop.", flush=True)
                        with open(os.path.join(run_dir, 'GATE_FAILED'),
                                  'w') as f:
                            json.dump(curr_log, f, indent=2, default=float)
                        return (best[0], best[1], False)
                    # else: FAIL but under cap -> run one more block, re-gate
            if time.time() - t_opt > args.time_budget_s:
                print(f"  [{args.scheme}] time budget hit at block {b}; stopping "
                      "(checkpoint saved; resubmit to resume).", flush=True)
                completed = False
                break
        if len(_step_times) >= 2:
            steady = _step_times[1:]           # drop first (compile-inflated)
            print(f"  [{args.scheme}] fwd+bwd theta-step time: "
                  f"first={_step_times[0]:.1f}s steady median="
                  f"{float(np.median(steady)):.1f}s (n={len(steady)})", flush=True)
        return (best[0], best[1], completed)

    recipe = {'s1': (100, 40, 6), 's1b': (50, 25, 12),
              's4': (25, 20, 24)}[args.scheme]
    if args.mem_smoke:
        recipe = (6, 2, 1)   # tiny: 1 block x (2 L-BFGS + 6 theta) for timing
        if curriculum:
            # 3 blocks so the smoke exercises stage-1 -> gate/activate -> dual
            recipe = (3, 2, 3)
        print(f"[mem-smoke] tiny recipe {recipe}; will exit after.")
    n_theta_block, n_scalar_iters, n_blocks = recipe
    print(f"[{args.scheme}] alternating: {n_blocks}x[L-BFGS(scalars,"
          f"{n_scalar_iters}) + Adam(theta,{n_theta_block})]  "
          f"(theta cosine spans {n_theta_block*n_blocks})", flush=True)
    L_final, fit, run_completed = _alternating(
        n_theta_block, n_scalar_iters, n_blocks, args.stage1_ftol)
    if not run_completed and not args.mem_smoke:
        print(f"[opt] time budget reached before completion; checkpoint saved. "
              f"Resubmit the SAME job to resume. loss so far={L_final:.6e}",
              flush=True)

    Gp_fit, lam_fit, nu_fit = (float(_gp_of(fit)), float(_lam_of(fit)),
                               float(_nu_of(fit)))
    _report_mem('end of training')
    print(f"[opt] done {time.time()-t_opt:.0f}s  loss {L0:.3e} -> {L_final:.3e}"
          f"  ({L0/max(L_final,1e-300):.2e}x)  recovered {_scalars_str(fit)} "
          f"(truth Gp={Gp_t} lam={lam_t} nu_s={nu_t})", flush=True)

    if args.mem_smoke:
        print("[mem-smoke] forward+reverse-mode fit ran without OOM/NaN; "
              "running finalization (ckpt+heads) to validate the full path.")

    # -- checkpoint (+ reload self-check) using the imported helper -----------
    class _A:    # the helper reads .width/.depth/.bound_c/.truth_model
        pass
    ahelp = _A()
    ahelp.width, ahelp.depth = args.width, args.depth
    ahelp.bound_c, ahelp.truth_model = args.bound_c, args.truth_model
    ckpt_ok, ckpt_path = _save_and_verify_checkpoint(
        run_dir, fit['theta'], fit, ahelp, Gp_fit, lam_fit, nu_fit, L_final,
        loss_fn, True)
    print(f"[ckpt] reload self-check: {'PASS' if ckpt_ok else 'FAIL'} -> {ckpt_path}")

    # -- recovered nonlinearity readout from the heads -----------------------
    #   Giesekus: m1 head ~ alpha (m0 = 1-alpha, m1 = alpha).
    #   FENE-P:   phi_tau curvature carries finite extensibility; record m1~0.
    Axx = np.asarray(out_truth['A_xx_traj'][-1]).reshape(-1)
    Axy = np.asarray(out_truth['A_xy_traj'][-1]).reshape(-1)
    Ayy = np.asarray(out_truth['A_yy_traj'][-1]).reshape(-1)
    Azz = np.asarray(out_truth['A_zz_traj'][-1]).reshape(-1)
    x1, x2, x3 = tb.tbnn_invariant_features(jnp.asarray(Axx), jnp.asarray(Axy),
                                            jnp.asarray(Ayy), jnp.asarray(Azz))
    X = jnp.stack([x1, x2, x3], axis=1)
    heads = tb.tbnn_heads(fit['theta'], X, bound_c)
    m1 = np.asarray(heads[4])
    # weight the head readout toward the high-stretch cells (where alpha lives)
    sval = np.clip((Axx + Ayy + Azz) - 3.0, 0.0, None)
    hi = sval > np.quantile(sval, 0.9) if sval.max() > 0 else np.ones_like(sval, bool)
    m1_hi = float(np.median(m1[hi])) if hi.any() else float(np.median(m1))
    m1_med = float(np.median(m1))
    alpha_truth = args.truth_alpha if args.truth_model == 'giesekus' else float('nan')
    print(f"[recover] m1 head: median={m1_med:.4f}  high-stretch median={m1_hi:.4f}"
          f"  (Giesekus alpha truth={alpha_truth})", flush=True)

    # -- save loss history + recovered-params summary ------------------------
    hist = np.array(loss_hist, dtype=np.float64)
    np.savez_compressed(os.path.join(run_dir, 'arrays.npz'),
                        loss_history=hist, loss_init=np.float64(L0),
                        loss_final=np.float64(L_final),
                        m1_cloud=m1, x1=np.asarray(x1), x3=np.asarray(x3),
                        stretch=sval)
    summary = dict(
        run_name=args.run_name, truth_model=args.truth_model,
        scheme=args.scheme, geometry='contraction',
        stretch_weighted=(not args.no_stretch_weight),
        loss_weight=('uniform' if args.no_stretch_weight else args.loss_weight),
        beta=nu_t / (nu_t + Gp_t * lam_t),
        De=De, grid=list(grid.shape), steps=args.inner * args.outer,
        T_over_lam=T_final / lam_t,
        loss_init=L0, loss_final=L_final,
        loss_reduction=L0 / max(L_final, 1e-300),
        Gp_fit=Gp_fit, Gp_truth=Gp_t, Gp_relerr=abs(Gp_fit - Gp_t) / Gp_t,
        lam_fit=lam_fit, lam_truth=lam_t, lam_relerr=abs(lam_fit - lam_t) / lam_t,
        nu_s_fit=nu_fit, nu_s_truth=nu_t, nu_s_relerr=abs(nu_fit - nu_t) / nu_t,
        m1_median=m1_med, m1_high_stretch_median=m1_hi,
        alpha_truth=alpha_truth, ckpt_ok=bool(ckpt_ok),
        max_trA_truth=float(trA.max()),
        max_Axx_truth=float(np.asarray(out_truth['A_xx_traj']).max()),
        Lsq_truth=(args.truth_lsq if args.truth_model == 'fene_p' else float('nan')),
        pressure_on=bool(pressure_on), w_p=float(w_p),
        U_list=list(U_list), multi_flow=bool(multi_flow),
        w_p_by_U={str(U): float(w_p_by_U[U]) for U in U_list},
        rate_scales={str(U): dict(su2=float(rate_data[U]['su2']),
                                  sp2=float(rate_data[U]['sp2']),
                                  alpha_v=float(rate_data[U]['alpha_v']),
                                  alpha_p=float(rate_data[U]['alpha_p']),
                                  alpha_v_legacy=float(rate_data[U]['alpha_v_legacy']),
                                  balance_scale=float(rate_data[U]['balance_scale']),
                                  roi_mass=float(rate_data[U]['roi_mass']),
                                  vel_mass=float(rate_data[U]['vel_mass']),
                                  tap_norm=([float(v) for v in rate_data[U]['tap_norm']]
                                            if rate_data[U]['tap_norm'] is not None else None),
                                  tap_w=([float(v) for v in np.asarray(rate_data[U]['tap_w'])]
                                         if rate_data[U]['tap_w'] is not None else None))
                    for U in U_list},
        norms_json=(args.norms_json if args.norms_json is not None else None),
        rate_balance=rate_balance_info,
        w_p_scale=(float(args.w_p_scale) if args.w_p_scale is not None else None),
        w_bal=(float(w_bal) if w_bal is not None else None),
        n_sub=int(args.n_sub), taps_phys=TAPS_PHYS, p_factor=float(factor),
        run_completed=bool(run_completed),
        curriculum=(curr_log if curriculum else None),
        args=vars(args))
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    # quick per-run loss curve
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].semilogy(hist[:, 0], hist[:, 1])
    ax[0].set_xlabel('loss eval'); ax[0].set_ylabel('stretch-weighted loss')
    ax[0].set_title(f'{args.run_name}: loss '
                    f'{L0:.2e}->{L_final:.2e}')
    ax[0].grid(True, which='both', alpha=0.3)
    ax[1].semilogy(hist[:, 0], hist[:, 2])
    ax[1].axhline(nu_t, color='k', ls='--', label='nu_s truth')
    ax[1].set_xlabel('loss eval'); ax[1].set_ylabel('nu_s')
    ax[1].set_title('nu_s trajectory'); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'loss_curve.png'), dpi=140)
    plt.close(fig)
    print(f"[done] {args.run_name}: loss {L0:.3e}->{L_final:.3e}  "
          f"Gp={Gp_fit:.3f}/{Gp_t} lam={lam_fit:.3f}/{lam_t} "
          f"nu_s={nu_fit:.3f}/{nu_t}  m1(hi)={m1_hi:.3f}", flush=True)
    if run_completed and not args.mem_smoke:
        with open(os.path.join(run_dir, 'DONE'), 'w') as f:
            f.write(f"completed {datetime.now().isoformat()} "
                    f"loss={L_final:.6e}\n")
        print(f"[done] wrote {run_dir}/DONE (run complete).", flush=True)
    return 0


def _report_mem(tag):
    try:
        st = jax.devices()[0].memory_stats()
        if st:
            cur = st.get('bytes_in_use', 0) / 1e9
            peak = st.get('peak_bytes_in_use', 0) / 1e9
            lim = st.get('bytes_limit', 0) / 1e9
            print(f"[mem:{tag}] in_use={cur:.2f}GB peak={peak:.2f}GB "
                  f"limit={lim:.2f}GB", flush=True)
    except Exception as e:
        print(f"[mem:{tag}] unavailable ({e})", flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
