#!/usr/bin/env python
"""Fit the yield-capable memory TBNN to an elastoviscoplastic truth.

Fits the UNANCHORED, YIELD-CAPABLE TBNN (registered model
tbnn_potential_free_logconf_bk_v2, = anchored=False, mobility=relu_annealed)
to a hard-coded Saramito Bingham EVP truth (saramito_logconf_bk_v2) on the
constriction. The annealing temperature kappa is swept 1 -> 0.1 -> 0.02 on a
schedule (static float in params['tbnn_kappa'], never a pytree leaf); the
kappa history is saved and NO gate is evaluated mid-anneal.

Deliverable figure: the learned min-eig(m0 I + m1 A) over the visited
(x1, x3) invariant plane with its (near-)zero-set, overlaid with the true
Saramito yield locus |tau_d|(x) = tau_y (closed form in (tau, p2)).

P3-G4 gate (record-don't-fail by design): loss decreases; the learned 0D
flow-curve intercept is within ~10-15% of tau_y; the learned mobility
zero-set qualitatively tracks the true yield locus inside the visited cloud.
Quantitative tightness is a research question -- recorded, not failed on.

Yielded-fraction guard (trap checklist): the truth yielded fraction is
logged BEFORE fitting; if ~0% or ~100% the data cannot identify the yield
surface -- adjust the forcing g_x (the script warns).

Partition: FULL fit => gpu,seas_gpu ONLY. Never submit to gpu_test.

Kernel-restart note: importing tbnn_closure registers the three toggle
models + saramito_logconf_bk_v2 once; rerun in a FRESH process after edits
(cr.register refuses duplicates).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

from repo_paths import bootstrap
bootstrap()

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from jax.scipy.integrate import trapezoid as jax_trapezoid

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import analytic_limits_validation as p3b  # noqa: E402
import visco_families as vf  # noqa: E402
import visco_tbnn as vt  # noqa: E402
from jax_rheology.models import registry as cr  #  noqa: E402,F401
from jax_rheology import log_conformation as lc  #  noqa: E402
from jax_rheology.models import tbnn_memory as tb  #  noqa: E402
from visco_opt_tbnn_run import (theta_to_named_arrays,  # noqa: E402
                                theta_from_named_arrays)

MODEL_NAME = 'tbnn_potential_free_logconf_bk_v2'   # (anchored=False, relu_annealed)
V2_MODEL_NAME = 'tbnn_potential_yield_logconf_bk_v2'  # V2 yield scalar
TRUTH_NAME = 'saramito_logconf_bk_v2'
_Q_EPS = 1e-9   # relative flow-rate denominator floor


def _flow_rate_Q(u_traj, cfg):
    """Scalar flow rate Q = trapz(mean_x u(y), dy) at the final outer step."""
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    y = (jnp.arange(Ny, dtype=jnp.float64) + 0.5) * dy
    u_prof = jnp.mean(u_traj[-1], axis=0)
    return jax_trapezoid(u_prof, y)


def _flow_rate_Q_traj(u_traj, cfg):
    """Q(t) at every outer step: trapz(mean_x u(y,t), dy). Shape (T,)."""
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    y = (jnp.arange(Ny, dtype=jnp.float64) + 0.5) * dy
    u_prof = jnp.mean(u_traj, axis=1)  # (T, Ny)
    # trapz along y for each t
    return jax.vmap(lambda up: jax_trapezoid(up, y))(u_prof)


def _loss_components(out, u_truth, v_truth, Q_truth, cfg, Q_scale=None):
    """Velocity SSE + relative-Q squared error for one forcing.

    Denominator is ``Q_scale = max_t |Q_truth(t)|`` (never final-time /
    instantaneous Q -- that caused the lam0~=172 pathology when Q(T)~=0).
    Residual still compares final-time model Q to final-time truth Q.

    ``L_vel`` is an ABSOLUTE sum of squares and is returned unweighted. Every
    caller multiplies it by that drive's frozen ratio weight
    ``w_i = W_max / W_i`` before summing across drives (``make_loss``,
    ``make_eval_parts_jit``, ``_full_parts``); ``L_Q`` is already relative and
    is not reweighted. Without ``w_i`` a drive's share of the gradient scales
    as the square of its velocity scale, so the highest drive dominates.
    """
    L_vel = (jnp.sum((out['u_traj'] - u_truth) ** 2)
             + jnp.sum((out['v_traj'] - v_truth) ** 2))
    Q_mod = _flow_rate_Q(out['u_traj'], cfg)
    denom = jnp.maximum(jnp.abs(Q_truth), _Q_EPS) if Q_scale is None else (
        jnp.maximum(Q_scale, _Q_EPS))
    rel = (Q_mod - Q_truth) / denom
    L_Q = rel ** 2
    return L_vel, L_Q


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a memory TBNN closure with a yield stress on "
                    "elastoviscoplastic channel-flow data.",
        epilog="Usual invocation, which sets every option below from a config "
               "file:\n"
               "  python experiments/evp_train.py "
               "--config experiments/configs/evp_channel_geomA_3lam.yaml\n"
               "The individual flags exist so a config can be overridden for "
               "one run; see REPRODUCE.md for the published settings.")
    # Truth (Saramito Bingham EVP). Pick tau_y moderate so a meaningful
    # fraction of cells yields (logged; adjust g_x if degenerate).
    p.add_argument('--truth-gp', type=float, default=3.2)
    p.add_argument('--truth-lam', type=float, default=0.7)
    p.add_argument('--truth-nus', type=float, default=0.8)
    p.add_argument('--truth-tau-y', type=float, default=1.0,
                   help='Saramito yield stress (moderate: partial yielding).')
    p.add_argument('--geometry', choices=['constriction', 'channel'],
                   default='constriction',
                   help='Flow geometry. "constriction" (default) = the '
                        'existing P3-G4 (IB object, short transient horizon). '
                        '"channel" = pressure-driven flat-wall plane channel '
                        '(Buckingham-Reiner plug makes tau_y a first-order '
                        'velocity feature; developed to steady state).')
    p.add_argument('--g-x', type=float, default=None,
                   help='Body-force drive (single forcing). Default: 8.0 '
                        '(constriction) / DEFAULT_CHANNEL_CONFIG g_x (channel). '
                        'Ignored when --g-x-list is set.')
    p.add_argument('--g-x-list', type=str, default=None,
                   help='Comma-separated body forces for multi-forcing ablation '
                        '(e.g. "1.3,1.8,2.5,4.0"). Overrides --g-x.')
    p.add_argument('--lambda-q', type=float, default=0.0,
                   help='Weight on the relative flow-rate loss Sigma L_Q,i (0 = '
                        'velocity-only; scales the RELATIVE-Q error).')
    p.add_argument('--targets-json', type=str, default=None,
                   help='Precheck targets from Sec.2 (Q_truth per forcing, lambda0). '
                        'Ensures identical denominators across ablation runs.')
    p.add_argument('--outer-steps', type=int, default=None,
                   help='Default: 300 (constriction) / channel config (200). '
                        'Reduced-config channel batch uses T~3lam (outer~84).')
    p.add_argument('--inner-steps', type=int, default=None)
    p.add_argument('--Ny', type=int, default=None,
                   help='Override grid Ny (reduced-config channel batch: 64). '
                        'Coarser dy -> ~4x smaller implicit diffusion number '
                        '(the biggest cost lever).')
    p.add_argument('--Nx', type=int, default=None,
                   help='Override grid Nx (final prod: 16; x-invariance certified).')
    p.add_argument('--fit-solver-tol', type=float, default=None,
                   help='Override solver_tol for the FIT (reduced batch: 1e-8). '
                        'A fit is not an AD-vs-FD gate, so 1e-12 is not needed; '
                        'fewer Krylov iters/substep. Truth is computed at the '
                        'SAME tol so the target is consistent.')
    # Agnostic scalar protocol (fit Gp, lam, nu_s; Gp/lam are gauge dirs).
    p.add_argument('--gp-init', type=float, default=1.0)
    p.add_argument('--lam-init', type=float, default=1.0)
    p.add_argument('--nus-init', type=float, default=1.0)
    p.add_argument('--unit-test', action='store_true',
                   help='Hold Gp,lam,nu_s at truth; fit theta only.')
    p.add_argument('--gauge-fixed', action='store_true',
                   help='Declared-gauge protocol: PIN Gp=lam=1 '
                        '(--gp-gauge/--lam-gauge) and fit ONLY nu_s + theta. '
                        'REQUIRED for developed geometries (the channel): '
                        'fitting (Gp,lam) there is a flat gauge that diverges '
                        'to the bounds -> conformation runaway -> NaN. The '
                        'network learns the modulus/rate scales itself; the '
                        'plug width (a velocity feature) is gauge-invariant.')
    p.add_argument('--gp-gauge', type=float, default=1.0)
    p.add_argument('--lam-gauge', type=float, default=1.0)
    # kappa annealing schedule: preset name ('base'|'slowtail') OR a
    # comma-separated list of floats (back-compat). 'slowtail' adds a 0.01
    # block AND gives low-kappa blocks (<=0.05) --tail-step-mult x the steps
    # (the low-kappa regime is where the annealed ReLU sharpens the yield
    # surface; this axis tests whether more time there tightens the intercept).
    p.add_argument('--kappa-schedule', type=str, default='1.0,0.3,0.1,0.05,0.02',
                   help="'base' | 'slowtail' | comma-separated floats.")
    p.add_argument('--tail-step-mult', type=int, default=2,
                   help='slowtail: steps multiplier for kappa<=0.05 blocks.')
    p.add_argument('--init', choices=['ob', 'random'], default='ob',
                   help="Legacy theta init flag. Prefer --theta-init. "
                        "'ob' = OB warm start; 'random' = random theta. "
                        "If --theta-init is left at default 'ob' and --init "
                        "random is passed, random wins (backward compat).")
    p.add_argument('--theta-init', choices=['ob', 'giesekus', 'random'],
                   default='ob',
                   help="theta init mode (additive; default 'ob' is the "
                        "existing path, bit-identical). 'giesekus' = OB "
                        "potential + Giesekus(alpha=0.3) mobility biases; "
                        "'random' = standard MLP init on all heads.")
    p.add_argument('--theta-init-scale', type=float, default=1.0,
                   help='Scale on --theta-init random (1.0 = standard init).')
    p.add_argument('--freeze-theta', action='store_true',
                   help='alt_mode: fit scalars only; skip all theta Adam '
                        'blocks (frozen at the chosen --theta-init).')
    p.add_argument('--theta-seed', type=int, default=0,
                   help='Seed for --theta-init random/giesekus (and legacy '
                        '--init random). OB init uses --seed.')
    p.add_argument('--timing-probe', type=int, default=0,
                   help='If >0: time this many value_and_grads at the reduced '
                        'config (gauge-fixed) and exit (no fit). gpu_test probe.')
    p.add_argument('--adam-steps-per-kappa', type=int, default=150)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--scalar-lr2', type=float, default=2e-3)
    # Stage-1 scalar solve (theta frozen at OB, kappa=1): well-conditioned.
    p.add_argument('--stage1-maxiter', type=int, default=120)
    p.add_argument('--stage1-ftol', type=float, default=1e-9)
    p.add_argument('--stage1-vel-only', action='store_true',
                   help='Stage-1 uses velocity-only loss (lam_Q=0); fallback if '
                        'scalars rail at bounds during Buckingham-Reiner regression.')
    p.add_argument('--yield-mode', choices=['off', 'scalar'], default='off',
                   help="'scalar' selects V2 (anchored+softplus+yield prefactor); "
                        'no kappa anneal.')
    p.add_argument('--tau-y-init', type=float, default=None,
                   help='Initial tau_y for V2 theta-block (default 1.0 in yield mode).')
    p.add_argument('--no-br-init', action='store_true',
                   help='alt_mode: skip Buckingham-Reiner data-driven init and '
                        'start all four scalars at the neutral 1.0 (gp/lam/nus/'
                        'tau-y-init). Init-robustness axis (farther from truth).')
    p.add_argument('--scalar-bound-lo', type=float, default=0.02)
    p.add_argument('--scalar-bound-hi', type=float, default=20.0)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-shear', type=int, default=12)
    p.add_argument('--time-budget-s', type=float, default=36000.0)
    p.add_argument('--wall-time-s', type=float, default=None,
                   help='SLURM wall (s); exit if time_budget_s < 0.9xwall.')
    p.add_argument('--resume', action='store_true',
                   help='Resume alt_mode from train_ckpt.pkl if present '
                        '(chained 48h jobs). Exits 0 immediately if DONE exists.')
    p.add_argument('--out-dir', type=str, default='./work/evp_channel')
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--no-eval', action='store_true')
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(p)
    return args


def _br_init_from_targets(g_x_list, Q_by_gx, *, H=1.0, beta0=0.25, lam0=1.0,
                          tau_y_lo=0.5, tau_y_hi=2.5):
    """Buckingham-Reiner data-driven init for the four scalars from the target
    flow rates Q(g_x) -- NEVER from the truth column. Uses the two LARGEST
    forcings (most-yielded, closest to the Newtonian high-shear asymptote):

        slope   = (Q(hi) - Q(mid)) / (hi - mid)          # dQ/dg_x
        eta_eff = (2/3) H^3 / slope                        # channel Q-g_x slope
        g_c     = hi - Q(hi)/slope                          # x-intercept (yield)
        tau_y0  = clip(g_c * H, tau_y_lo, tau_y_hi)
        nu_s0   = beta0 * eta_eff  (beta0 solvent fraction prior)
        Gp0     = (eta_eff - nu_s0) / lam0  (lam0 prior)

    Returns the init quadruple + all regression inputs for provenance."""
    gs = sorted(float(g) for g in g_x_list)
    hi, mid = gs[-1], gs[-2]
    Q_hi, Q_mid = float(Q_by_gx[hi]), float(Q_by_gx[mid])
    slope = (Q_hi - Q_mid) / (hi - mid)
    prov = dict(method='buckingham_reiner', forcing_hi=hi, forcing_mid=mid,
                Q_hi=Q_hi, Q_mid=Q_mid, slope=slope, H=H, beta0=beta0,
                lam0=lam0, tau_y_lo=tau_y_lo, tau_y_hi=tau_y_hi)
    if not np.isfinite(slope) or slope <= 0:
        prov.update(note='slope<=0 or non-finite; neutral fallback',
                    eta_eff0=float('nan'), g_c0=float('nan'),
                    Gp0=1.0, lam0_out=1.0, nu_s0=0.8, tau_y0=1.0)
        return dict(Gp0=1.0, lam0=1.0, nu_s0=0.8, tau_y0=1.0, prov=prov)
    eta_eff0 = (2.0 / 3.0) * H ** 3 / slope
    g_c0 = hi - Q_hi / slope
    tau_y0 = float(np.clip(g_c0 * H, tau_y_lo, tau_y_hi))
    nu_s0 = beta0 * eta_eff0
    Gp0 = (eta_eff0 - nu_s0) / lam0
    prov.update(eta_eff0=eta_eff0, g_c0=g_c0, Gp0=Gp0, nu_s0=nu_s0,
                tau_y0=tau_y0, lam0_out=lam0)
    return dict(Gp0=Gp0, lam0=lam0, nu_s0=nu_s0, tau_y0=tau_y0, prov=prov)


def main():
    args = parse_args()
    yield_mode = str(args.yield_mode)
    yield_scalar = (yield_mode == 'scalar')
    model_name = V2_MODEL_NAME if yield_scalar else MODEL_NAME
    # alt_mode = V2 corrected: un-gauged (fit all four scalars) + Hookean-map
    # criterion + s1 alternation (scalar L-BFGS <-> theta Adam). This is the
    # post-v2_prod path; the OLD V2 (gauge-fixed, nu_s-only stage-1, tau_y in
    # theta-Adam) is kept for --gauge-fixed back-compat.
    alt_mode = bool(yield_scalar and not args.gauge_fixed)
    # OLD V2 gauge-fixed: stage-1 is nu_s-only velocity loss. alt_mode stage-1
    # uses the FULL vel+Q loss over all four scalars (theta=OB == generator).
    stage1_vel_only = bool(args.stage1_vel_only or (yield_scalar and args.gauge_fixed))
    tau_y_init = (1.0 if args.tau_y_init is None else float(args.tau_y_init)
                  if yield_scalar else 0.0)
    if args.wall_time_s is not None:
        wall_need = 0.9 * float(args.wall_time_s)
        if args.time_budget_s < wall_need:
            raise SystemExit(
                f"time_budget_s={args.time_budget_s:g} < 0.9xwall="
                f"{wall_need:g} (wall_time_s={args.wall_time_s:g})")
    geometry = args.geometry
    # Geometry-specific base config + defaults. Channel: flat-wall plane
    # channel (DEFAULT_CHANNEL_CONFIG, developed to steady state). Constriction:
    # the existing P3-G4 (DEFAULT_MULTISTEP_AD_FD_CONFIG, short IB-locked dt).
    if geometry == 'channel':
        cfg = dict(vf.DEFAULT_CHANNEL_CONFIG)
        default_gx = cfg['g_x']
        outer_steps = args.outer_steps if args.outer_steps is not None else cfg['outer_steps']
    else:
        cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
        default_gx = 8.0
        outer_steps = args.outer_steps if args.outer_steps is not None else 300
    cfg['outer_steps'] = outer_steps
    if args.g_x_list:
        g_x_list = [float(x.strip()) for x in args.g_x_list.split(',') if x.strip()]
    else:
        g_x = args.g_x if args.g_x is not None else default_gx
        g_x_list = [g_x]
    cfg['g_x'] = g_x_list[0]
    args.g_x = g_x_list[0]
    args.g_x_list_resolved = g_x_list
    multi_forcing = len(g_x_list) > 1
    lambda_q = float(args.lambda_q)
    targets_meta = {}
    if args.targets_json:
        with open(args.targets_json) as f:
            targets_meta = json.load(f)
    lambda_q0 = float(targets_meta.get('lambda0', float('nan')))
    if args.inner_steps is not None:
        cfg['inner_steps'] = args.inner_steps
    if args.Ny is not None:
        cfg['Ny'] = int(args.Ny)         # reduced-config channel batch (64)
    if args.Nx is not None:
        cfg['Nx'] = int(args.Nx)
    if args.fit_solver_tol is not None:
        cfg['solver_tol'] = float(args.fit_solver_tol)   # fit + truth, consistent

    # kappa schedule: preset name or comma list; slowtail weights low-kappa.
    # V2 (yield_mode='scalar'): NO kappa anneal -- single block at kappa=1.
    _KAPPA_PRESETS = {'base': [1.0, 0.3, 0.1, 0.05, 0.02],
                      'slowtail': [1.0, 0.3, 0.1, 0.05, 0.02, 0.01]}
    if yield_scalar:
        kappa_schedule = [1.0]
        slowtail = False
    elif args.kappa_schedule in _KAPPA_PRESETS:
        kappa_schedule = list(_KAPPA_PRESETS[args.kappa_schedule])
        slowtail = (args.kappa_schedule == 'slowtail')
    else:
        kappa_schedule = [float(k) for k in args.kappa_schedule.split(',')]
        slowtail = False

    # Scalar bounds. alt_mode (un-gauged) fits all four with the recorded
    # bounds: Gp[0.2,10], lam[0.1,5], nu_s[0.1,5], tau_y[0,3]. OLD gauge-fixed
    # V2 fits nu_s+tau_y only (Gp=lam pinned).
    if yield_scalar:
        gp_bound = (0.2, 10.0)
        lam_bound = (0.1, 5.0)
        nus_bound = (0.1, 5.0)
        tau_y_bound = (0.0, 3.0)
    else:
        gp_bound = (args.scalar_bound_lo, args.scalar_bound_hi)
        lam_bound = (args.scalar_bound_lo, args.scalar_bound_hi)
        nus_bound = (args.scalar_bound_lo, args.scalar_bound_hi)
        tau_y_bound = None

    if args.run_name is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode = 'unit' if args.unit_test else 'agn'
        args.run_name = (f'tbnn_evp_{geometry}_{mode}_ty{args.truth_tau_y:g}'
                         f'_T{outer_steps}_{stamp}')
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    done_path = os.path.join(run_dir, 'DONE')
    if args.resume and os.path.isfile(done_path):
        print(f"[resume] {done_path} present -- already complete; exit 0.")
        return 0
    print(f"[setup] device = {jax.devices()}")
    print(f"[setup] geometry = {geometry}   run dir = {run_dir}")
    print(f"[setup] grid Nx={cfg.get('Nx')} Ny={cfg.get('Ny')} "
          f"outer={cfg['outer_steps']} tol={cfg['solver_tol']}")
    Gp_t, lam_t, nu_t, tau_y_t = (args.truth_gp, args.truth_lam,
                                  args.truth_nus, args.truth_tau_y)
    bound_c = tb.TBNN_DEFAULT_BOUND_C
    # kappa_schedule + slowtail already resolved above (preset or comma list).
    agnostic = not args.unit_test
    T_final = cfg['outer_steps'] * cfg['inner_steps'] * cfg['dt']
    print(f"[setup] model={model_name}  yield_mode={yield_mode}")
    if yield_scalar:
        print(f"[setup] V2 stage-1: nu_s only, vel-only (tau_y init={tau_y_init:g} "
              f"in theta-block)")
    print(f"[setup] truth=Saramito Gp={Gp_t} lam={lam_t} nu_s={nu_t} "
          f"tau_y={tau_y_t} g_x_list={g_x_list} T={T_final:.4f}={T_final/lam_t:.3f}lam")
    print(f"[setup] lambda_Q={lambda_q:g}  lambda0(precheck)={lambda_q0:g}  "
          f"multi_forcing={multi_forcing}")
    print(f"[setup] kappa schedule = {kappa_schedule}  "
          f"({'V2 single-block' if yield_scalar else 'never gate mid-anneal'})")
    if yield_scalar and args.time_budget_s < 36000:
        print(f"[setup] WARNING: V2 production fits expect time_budget_s ~41000 "
              f"(got {args.time_budget_s:g})")

    # --- yielded-fraction guard (log BEFORE fitting) ---
    for gx in g_x_list:
        cgx = dict(cfg); cgx['g_x'] = gx
        yf = vt.saramito_yielded_fraction(cgx, Gp_t, lam_t, tau_y_t, nu_t,
                                          geometry=geometry)
        print(f"[yield] g_x={gx:g} yielded fraction = {yf['yielded_fraction']:.1%}  "
              f"(|tau_d| q50={yf['td_q50']:.3f} q99={yf['td_q99']:.3f})")
        if yf['yielded_fraction'] < 0.02 or yf['yielded_fraction'] > 0.98:
            print(f"[yield] WARNING g_x={gx:g}: yielded {yf['yielded_fraction']:.1%} "
                  f"~0% or ~100% -- yield surface may be unidentifiable.")
    yf = vt.saramito_yielded_fraction(
        dict(cfg, g_x=g_x_list[-1]), Gp_t, lam_t, tau_y_t, nu_t, geometry=geometry)

    grid, truth_model, truth_state, truth_perm = vt._build_geometry(
        cfg, TRUTH_NAME, geometry)
    _, tbnn_model, tbnn_state, tbnn_perm = vt._build_geometry(
        cfg, model_name, geometry)

    def _forward(state, model, perm, params, nu, g_x_force):
        return p3b._evolve_wall_bounded_with_diagnostics(
            initial_state=state, model=model, polymer_params=params, grid=grid,
            density=cfg['density'], base_viscosity=nu, dt=cfg['dt'],
            inner_steps=cfg['inner_steps'], outer_steps=cfg['outer_steps'],
            solver_type=cfg['solver_type'],
            use_preconditioner=cfg['use_preconditioner'],
            preconditioner_type=cfg['preconditioner_type'],
            pressure_gradient=(g_x_force, 0.0), permeability=perm, U_f=cfg['U_f'],
            solver_tol=cfg['solver_tol'], solver_maxiter=cfg['solver_maxiter'])

    print("[truth] generating Saramito truth trajectories ...")
    t0 = time.time()
    truth_pp = {'Gp': jnp.asarray(Gp_t, dtype=jnp.float64),
                'lam': jnp.asarray(lam_t, dtype=jnp.float64),
                'tau_y': jnp.asarray(tau_y_t, dtype=jnp.float64)}
    truth_forcings = []
    truth_npz_paths = {}
    for gx in g_x_list:
        gx_key = f"{gx:g}"
        out_t = jax.jit(lambda g=gx: _forward(
            truth_state, truth_model, truth_perm, truth_pp, nu_t, g))()
        u_t = out_t['u_traj']; v_t = out_t['v_traj']
        u_t.block_until_ready()
        Q_t = float(_flow_rate_Q(u_t, cfg))
        Q_traj = np.asarray(_flow_rate_Q_traj(u_t, cfg))
        # Relative-Q denominator: max_t |Q_truth(t)| (never final-time Q).
        Q_scale = float(np.max(np.abs(Q_traj)))
        # Prefer targets-json Q_truth for residual target if present, but
        # ALWAYS use max_t|Q| for the scale (targets may store final Q only).
        Q_from_file = None
        if targets_meta.get('forcings', {}).get(gx_key):
            Q_from_file = float(targets_meta['forcings'][gx_key]['Q_truth'])
            # Prefer Q_scale from targets if recorded
            if 'Q_scale' in targets_meta['forcings'][gx_key]:
                Q_scale = float(targets_meta['forcings'][gx_key]['Q_scale'])
        Q_use = Q_from_file if Q_from_file is not None else Q_t
        # --- per-drive velocity-loss weight (ratio weighting) --------------
        # W_i = sum over trajectory AND cells of (u_truth^2 + v_truth^2): the
        # absolute scale L_vel,i would carry if the model predicted zero. It
        # is a pure function of the TRUTH targets, so it is a frozen constant
        # with no gradient. Prefer the value in the targets JSON (computed
        # once by the target-generation path) and only fall back to computing
        # it here, so that the fit cannot silently re-derive its own weights.
        W_from_file = None
        if targets_meta.get('forcings', {}).get(gx_key, {}).get('W_vel'):
            W_from_file = float(targets_meta['forcings'][gx_key]['W_vel'])
        W_here = float(jnp.sum(u_t ** 2) + jnp.sum(v_t ** 2))
        W_use = W_from_file if W_from_file is not None else W_here
        tpath = os.path.join(run_dir, f'truth_traj_gx{gx_key.replace(".", "p")}.npz')
        np.savez_compressed(
            tpath,
            u=np.asarray(u_t), v=np.asarray(v_t),
            A_xx=np.asarray(out_t['A_xx_traj']),
            A_xy=np.asarray(out_t['A_xy_traj']),
            A_yy=np.asarray(out_t['A_yy_traj']),
            A_zz=np.asarray(out_t['A_zz_traj']),
            Q_traj=Q_traj, Q_final=Q_t, Q_scale=Q_scale, g_x=gx)
        truth_npz_paths[gx_key] = tpath
        truth_forcings.append(dict(
            g_x=gx, out_truth=out_t, u_truth=u_t, v_truth=v_t,
            Q_truth=jnp.asarray(Q_use, dtype=jnp.float64),
            Q_scale=jnp.asarray(Q_scale, dtype=jnp.float64),
            Q_traj=Q_traj, W_vel=W_use, W_vel_local=W_here))
        print(f"  [truth] g_x={gx:g} max|u|={float(jnp.max(jnp.abs(u_t))):.3e}  "
              f"Q_final={Q_t:.6e}  Q_use={Q_use:.6e}  "
              f"Q_scale=max_t|Q|={Q_scale:.6e}  "
              f"W_vel={W_use:.6e}{'' if W_from_file is None else ' (targets)'}",
              flush=True)
    print(f"[truth] forward warm = {time.time()-t0:.1f}s")

    # === ratio weights on the per-drive velocity loss =====================
    # w_i = W_max / W_i, so the drive with the LARGEST velocity scale keeps
    # weight 1 and the quieter drives are lifted to match it. The unweighted
    # loss made each drive's share scale as the square of its velocity scale,
    # which put ~98% of the gradient on g_x=4 -- exactly inverted against
    # |d ln Q / d ln tau_y|, which is largest at the LOWEST drive. Dividing by
    # W_i outright would instead shrink the total by ~1e4 and invalidate every
    # tuned hyperparameter (Adam lr, stage1_ftol, L-BFGS gtol); the ratio form
    # leaves the total near its old magnitude so they all carry over untouched.
    W_max = max(td['W_vel'] for td in truth_forcings)
    for td in truth_forcings:
        td['w_vel'] = jnp.asarray(W_max / td['W_vel'], dtype=jnp.float64)
    vel_weights = {f"{td['g_x']:g}": float(td['w_vel']) for td in truth_forcings}
    vel_W = {f"{td['g_x']:g}": float(td['W_vel']) for td in truth_forcings}
    print("[loss-weights] ratio weighting w_i = W_max/W_i  (W_max="
          f"{W_max:.6e})")
    for td in truth_forcings:
        print(f"  [loss-weights] g_x={td['g_x']:g}  W_i={td['W_vel']:.6e}  "
              f"w_i={float(td['w_vel']):.6f}", flush=True)
    out_truth = truth_forcings[-1]['out_truth']   # primary for single-forcing compat

    # Gauge-fixed protocol: Gp and lam are EXACT gauge
    # directions of the closure (Gp absorbable into phi, lam into the
    # mobility). In a DEVELOPED flow (the channel, run to steady state) the
    # OB-init parabola depends on the scalars only through eta_eff = nu_s +
    # Gp*lam, so fitting (Gp, lam) is a FLAT gauge that L-BFGS wanders to the
    # bounds (-> lam=15, conformation runaway -> NaN, as the first channel run
    # showed). Fixing Gp=lam=1 removes that degeneracy; the network learns the
    # modulus/rate scales itself. nu_s stays genuinely identifiable.
    gauge_fixed = bool(args.gauge_fixed)
    gp_gauge, lam_gauge = float(args.gp_gauge), float(args.lam_gauge)
    Gp_init = args.gp_init if agnostic else Gp_t
    lam_init = args.lam_init if agnostic else lam_t
    nus_init = args.nus_init if agnostic else nu_t
    fit_scalars = agnostic and not gauge_fixed   # fit (Gp, lam, nu_s)?

    # --- Buckingham-Reiner data-driven init (alt_mode only) ---------------
    # Derive (Gp, lam, nu_s, tau_y)^0 from the target Q(g_x) slope+intercept,
    # NOT from the truth column. Provenance logged to config.json.
    br_init = None
    if alt_mode and args.no_br_init:
        # Neutral 1s init (init-robustness axis): all four scalars start at the
        # --gp/lam/nus/tau-y-init defaults (1.0), NOT the BR estimate.
        br_init = dict(method='neutral_ones', note='--no-br-init: all scalars '
                       'start at 1.0', Gp_init_clipped=float(Gp_init),
                       lam_init_clipped=float(lam_init),
                       nu_s_init_clipped=float(nus_init),
                       tau_y_init_clipped=float(tau_y_init))
        print(f"[init] alt_mode NEUTRAL 1s init (--no-br-init): "
              f"Gp0={Gp_init:.4f} lam0={lam_init:.4f} nu_s0={nus_init:.4f} "
              f"tau_y0={tau_y_init:.4f}  (truth 3.2/0.7/0.8/1.45 NOT used)")
    elif alt_mode:
        Q_by_gx = {td['g_x']: float(td['Q_truth']) for td in truth_forcings}
        _br = _br_init_from_targets(g_x_list, Q_by_gx)
        br_init = _br['prov']
        Gp_init = float(np.clip(_br['Gp0'], gp_bound[0], gp_bound[1]))
        lam_init = float(np.clip(_br['lam0'], lam_bound[0], lam_bound[1]))
        nus_init = float(np.clip(_br['nu_s0'], nus_bound[0], nus_bound[1]))
        tau_y_init = float(np.clip(_br['tau_y0'], tau_y_bound[0], tau_y_bound[1]))
        br_init['Gp_init_clipped'] = Gp_init
        br_init['lam_init_clipped'] = lam_init
        br_init['nu_s_init_clipped'] = nus_init
        br_init['tau_y_init_clipped'] = tau_y_init
        print(f"[BR-init] slope={_br['prov']['slope']:.4f} "
              f"eta_eff0={_br['prov'].get('eta_eff0', float('nan')):.4f} "
              f"g_c0={_br['prov'].get('g_c0', float('nan')):.4f}")
        print(f"[BR-init] -> Gp0={Gp_init:.4f} lam0={lam_init:.4f} "
              f"nu_s0={nus_init:.4f} tau_y0={tau_y_init:.4f}  "
              f"(from Q({_br['prov']['forcing_mid']:g})={_br['prov']['Q_mid']:.4f}, "
              f"Q({_br['prov']['forcing_hi']:g})={_br['prov']['Q_hi']:.4f}; "
              f"truth 3.2/0.7/0.8/1.45 NOT used; Bingham-based known-biased init only)")
        br_init['note'] = ('Bingham/Buckingham-Reiner based; known-biased; '
                           'init only -- not a physics claim')

    def _gp_of(fit):
        if gauge_fixed:
            return jnp.asarray(gp_gauge, dtype=jnp.float64)
        return jnp.maximum(fit['Gp'], 1e-4) if agnostic else jnp.asarray(Gp_t, dtype=jnp.float64)

    def _lam_of(fit):
        if gauge_fixed:
            return jnp.asarray(lam_gauge, dtype=jnp.float64)
        return jnp.maximum(fit['lam'], 1e-4) if agnostic else jnp.asarray(lam_t, dtype=jnp.float64)

    def _nu_of(fit):
        if gauge_fixed:
            return jnp.maximum(fit['nu_s'], 1e-4)
        return jnp.maximum(fit['nu_s'], 1e-4) if agnostic else jnp.asarray(nu_t, dtype=jnp.float64)

    def _tau_y_of(fit):
        if yield_scalar:
            return jnp.clip(fit['tau_y'], tau_y_bound[0], tau_y_bound[1])
        raise KeyError('tau_y only exists in yield_mode=scalar')

    stage1_lambda_q = 0.0 if stage1_vel_only else lambda_q

    def make_loss(kappa, lam_q=lambda_q):
        """Loss over the pytree `fit` at a FIXED (static) kappa."""
        def loss_fn(fit):
            params = {'Gp': _gp_of(fit), 'lam': _lam_of(fit),
                      'theta': fit['theta'], 'tbnn_bound_c': bound_c,
                      'tbnn_kappa': float(kappa)}
            if yield_scalar:
                params['tau_y'] = _tau_y_of(fit)
            L_vel_sum = jnp.asarray(0.0, dtype=jnp.float64)
            L_Q_sum = jnp.asarray(0.0, dtype=jnp.float64)
            for td in truth_forcings:
                out = _forward(tbnn_state, tbnn_model, tbnn_perm, params,
                               _nu_of(fit), td['g_x'])
                L_vel, L_Q = _loss_components(
                    out, td['u_truth'], td['v_truth'], td['Q_truth'], cfg,
                    Q_scale=td.get('Q_scale'))
                L_vel_sum = L_vel_sum + td['w_vel'] * L_vel
                L_Q_sum = L_Q_sum + L_Q
            return L_vel_sum + lam_q * L_Q_sum
        return loss_fn

    def make_eval_parts_jit(kappa, lam_q=lambda_q):
        """JIT loss-component logger; kappa pinned as concrete float."""
        kf = float(kappa)

        def _eval_parts(fit):
            params = {'Gp': _gp_of(fit), 'lam': _lam_of(fit),
                      'theta': fit['theta'], 'tbnn_bound_c': bound_c,
                      'tbnn_kappa': kf}
            if yield_scalar:
                params['tau_y'] = _tau_y_of(fit)
            L_vel_sum = jnp.asarray(0.0, dtype=jnp.float64)
            L_Q_sum = jnp.asarray(0.0, dtype=jnp.float64)
            for td in truth_forcings:
                out = _forward(tbnn_state, tbnn_model, tbnn_perm, params,
                               _nu_of(fit), td['g_x'])
                L_vel, L_Q = _loss_components(
                    out, td['u_truth'], td['v_truth'], td['Q_truth'], cfg,
                    Q_scale=td.get('Q_scale'))
                L_vel_sum = L_vel_sum + td['w_vel'] * L_vel
                L_Q_sum = L_Q_sum + L_Q
            L_tot = L_vel_sum + lam_q * L_Q_sum
            return L_tot, L_vel_sum, L_Q_sum

        return jax.jit(_eval_parts)

    # theta init. --theta-init is additive; default 'ob' dispatches to the
    # exact pre-existing init_tbnn_theta path (bit-identity, G-init-1).
    # Legacy: --init random with --theta-init left at default 'ob' still
    # selects random.
    theta_init_mode = args.theta_init
    if args.init == 'random' and args.theta_init == 'ob':
        theta_init_mode = 'random'

    if theta_init_mode == 'random':
        # Random on all heads. Prefer the closure helper (supports scale);
        # scale=1.0 matches the legacy inline init_mlp path bit-for-bit
        # when yield_mode/anchored flags are ignored by init_mlp itself.
        if yield_scalar:
            theta0, _ = tb.init_tbnn_theta_random(
                jax.random.PRNGKey(args.theta_seed), width=args.width,
                depth=args.depth, bound_c=bound_c, anchored=True,
                mobility='softplus', yield_mode='scalar',
                scale=float(args.theta_init_scale))
        else:
            theta0, _ = tb.init_tbnn_theta_random(
                jax.random.PRNGKey(args.theta_seed), width=args.width,
                depth=args.depth, bound_c=bound_c, anchored=False,
                mobility='relu_annealed',
                scale=float(args.theta_init_scale))
        print(f"[init] RANDOM theta (theta-init=random, seed={args.theta_seed}, "
              f"scale={args.theta_init_scale:g})")
    elif theta_init_mode == 'giesekus':
        if not yield_scalar:
            raise SystemExit('--theta-init giesekus requires --yield-mode scalar')
        theta0, _ = tb.init_tbnn_theta_giesekus(
            jax.random.PRNGKey(args.theta_seed), width=args.width,
            depth=args.depth, bound_c=bound_c, anchored=True,
            mobility='softplus', yield_mode='scalar', alpha=0.3)
        print(f"[init] GIESEKUS(alpha=0.3) mobility + OB potential "
              f"(theta-seed={args.theta_seed}); scalars untouched")
    else:
        # --- OB path: EXACTLY the pre-existing code. Do not edit. ---
        if yield_scalar:
            theta0, _ = tb.init_tbnn_theta(
                jax.random.PRNGKey(args.seed), width=args.width,
                depth=args.depth, bound_c=bound_c, anchored=True,
                mobility='softplus', yield_mode='scalar')
            print("[init] OB warm start (V2 anchored Tier-1 + yield scalar)")
        else:
            theta0, _ = tb.init_tbnn_theta(jax.random.PRNGKey(args.seed),
                                           width=args.width, depth=args.depth,
                                           bound_c=bound_c, anchored=False,
                                           mobility='relu_annealed')
            print("[init] OB warm start (theta -> Oldroyd-B at kappa=1)")
    fit = {'theta': theta0}
    if gauge_fixed:
        fit['nu_s'] = jnp.asarray(nus_init, dtype=jnp.float64)  # Gp=lam=gauge (fixed)
        if yield_scalar:
            fit['tau_y'] = jnp.asarray(tau_y_init, dtype=jnp.float64)
    elif agnostic:
        fit['Gp'] = jnp.asarray(Gp_init, dtype=jnp.float64)
        fit['lam'] = jnp.asarray(lam_init, dtype=jnp.float64)
        fit['nu_s'] = jnp.asarray(nus_init, dtype=jnp.float64)
        if yield_scalar:                      # alt_mode: fit tau_y too
            fit['tau_y'] = jnp.asarray(tau_y_init, dtype=jnp.float64)

    L0 = float(jax.jit(make_loss(kappa_schedule[0]))(fit))
    eval_parts_init = make_eval_parts_jit(kappa_schedule[0])
    _, L_vel0, L_Q0 = eval_parts_init(fit)
    L_vel0 = float(L_vel0); L_Q0 = float(L_Q0)
    print(f"[opt] loss(init, kappa={kappa_schedule[0]}) = {L0:.6e}  "
          f"SigmaL_vel={L_vel0:.6e}  SigmaL_Q={L_Q0:.6e}  (lam_Q={lambda_q:g})")

    # --- timing probe: N value_and_grads at this (reduced) config, then exit ---
    if args.timing_probe > 0:
        vg_probe = jax.jit(jax.value_and_grad(make_loss(kappa_schedule[0])))
        _v, _g = vg_probe(fit); jax.block_until_ready(_v)   # warm (compile)
        import time as _t
        ts = []
        for _ in range(int(args.timing_probe)):
            t0 = _t.perf_counter()
            _v, _g = vg_probe(fit); jax.block_until_ready(_v)
            ts.append(_t.perf_counter() - t0)
        spg = float(np.median(ts))
        # total grads for a full run of THIS schedule (stage1 ~ maxiter + blocks)
        n_blocks_est = sum((args.tail_step_mult if (slowtail and k <= 0.05) else 1)
                           * args.adam_steps_per_kappa for k in kappa_schedule)
        est_grads = args.stage1_maxiter + n_blocks_est
        print(f"[timing-probe] {args.timing_probe} grads: median={spg:.2f}s/grad "
              f"(min={min(ts):.2f} max={max(ts):.2f})")
        print(f"[timing-probe] est grads/fit={est_grads} => est wall="
              f"{spg*est_grads/3600.0:.2f}h  (target <=3-4h; flag={spg*est_grads>4.2*3600})")
        return 0

    eval_parts_s1 = make_eval_parts_jit(1.0)

    progress_path = os.path.join(run_dir, 'progress.csv')
    scalars_path = os.path.join(run_dir, 'scalars.csv')
    train_ckpt_path = os.path.join(run_dir, 'train_ckpt.pkl')
    # progress: total + per-drive L_vel/L_Q + scalars + grad_norm + wall
    gx_cols = []
    for gx in g_x_list:
        tag = f'{gx:g}'.replace('.', 'p')
        gx_cols += [f'L_vel_gx{tag}', f'L_Q_gx{tag}']
    hdr = ("stage,kappa,step,loss,L_vel,L_Q," + ','.join(gx_cols)
           + ",Gp,lam,nu_s")
    if yield_scalar:
        hdr += ",tau_y"
    hdr += ",grad_norm,wall_s"
    # On resume: append; else rewrite headers
    prog_mode = 'a' if (args.resume and os.path.isfile(progress_path)) else 'w'
    scal_mode = 'a' if (args.resume and os.path.isfile(scalars_path)) else 'w'
    if prog_mode == 'w':
        with open(progress_path, 'w') as pf:
            pf.write(hdr + "\n")
    if scal_mode == 'w':
        with open(scalars_path, 'w') as sf:
            sf.write("stage,step,Gp,lam,nu_s,tau_y,loss,L_vel,L_Q,wall_s\n")
    loss_hist = []          # (global_step, loss, L_vel, L_Q, nu_s, kappa[, tau_y])
    kappa_hist = []         # (global_step, kappa)
    tau_y_hist = []         # (global_step, tau_y) for V2 headline diagnostic
    gstep = [0]
    t_opt = time.time()

    def _log(stage, kappa, Lf, Lvf, LQf, f, grad_norm=float('nan'),
             per_drive=None):
        ty = float(_tau_y_of(f)) if yield_scalar else float('nan')
        loss_hist.append((gstep[0], Lf, Lvf, LQf, float(_nu_of(f)), float(kappa),
                          ty) if yield_scalar else
                         (gstep[0], Lf, Lvf, LQf, float(_nu_of(f)), float(kappa)))
        if yield_scalar:
            tau_y_hist.append((gstep[0], ty))
        if per_drive is None:
            per_drive = [(float('nan'), float('nan'))] * len(g_x_list)
        wall = time.time() - t_opt
        with open(progress_path, 'a') as pf:
            row = (f"{stage},{kappa:g},{gstep[0]},{Lf:.6e},{Lvf:.6e},{LQf:.6e}")
            for lv, lq in per_drive:
                row += f",{lv:.6e},{lq:.6e}"
            row += (f",{float(_gp_of(f)):.6f},{float(_lam_of(f)):.6f},"
                    f"{float(_nu_of(f)):.6f}")
            if yield_scalar:
                row += f",{ty:.6f}"
            row += f",{grad_norm:.6e},{wall:.1f}"
            pf.write(row + "\n")

    def _log_scalars(stage, f, Lf=float('nan'), Lvf=float('nan'),
                     LQf=float('nan')):
        ty = float(_tau_y_of(f)) if yield_scalar else float('nan')
        with open(scalars_path, 'a') as sf:
            sf.write(
                f"{stage},{gstep[0]},{float(_gp_of(f)):.8f},"
                f"{float(_lam_of(f)):.8f},{float(_nu_of(f)):.8f},{ty:.8f},"
                f"{Lf:.6e},{Lvf:.6e},{LQf:.6e},{time.time()-t_opt:.1f}\n")

    def _save_train_ckpt(fit, stage, cycle, extra=None):
        """Rolling mid-fit checkpoint for chained 48h resume."""
        import pickle

        def _to_np(tree):
            if isinstance(tree, dict):
                return {k: _to_np(v) for k, v in tree.items()}
            if hasattr(tree, 'shape'):
                return np.asarray(tree)
            return tree

        obj = {
            'stage': stage, 'cycle': cycle, 'gstep': int(gstep[0]),
            'fit': _to_np(fit),
            'wall_s': time.time() - t_opt,
            'extra': extra or {},
        }
        tmp = train_ckpt_path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, train_ckpt_path)
        print(f"[ckpt] train_ckpt -> {train_ckpt_path} "
              f"(stage={stage} cycle={cycle} gstep={gstep[0]})", flush=True)

    def _load_train_ckpt():
        import pickle
        if not os.path.isfile(train_ckpt_path):
            return None
        with open(train_ckpt_path, 'rb') as f:
            obj = pickle.load(f)

        def _to_jnp(tree):
            if isinstance(tree, dict):
                return {k: _to_jnp(v) for k, v in tree.items()}
            if hasattr(tree, 'shape'):
                return jnp.asarray(tree, dtype=jnp.float64)
            return tree

        fit_r = _to_jnp(obj['fit'])
        gstep[0] = int(obj.get('gstep', 0))
        print(f"[resume] loaded {train_ckpt_path}: stage={obj.get('stage')} "
              f"cycle={obj.get('cycle')} gstep={gstep[0]}", flush=True)
        return obj, fit_r

    def _scalars_str(f):
        if gauge_fixed:
            s = (f"Gp={gp_gauge:g}(gauge) lam={lam_gauge:g}(gauge) "
                 f"nu_s={float(_nu_of(f)):.3f}")
            if yield_scalar:
                s += f" tau_y={float(_tau_y_of(f)):.3f}"
            return s
        if not agnostic:
            return "scalars@truth"
        s = (f"Gp={float(_gp_of(f)):.3f} lam={float(_lam_of(f)):.3f} "
             f"nu_s={float(_nu_of(f)):.3f}")
        if yield_scalar:
            s += f" tau_y={float(_tau_y_of(f)):.3f}"
        return s

    # ---- Stage 1: L-BFGS on the DIMENSIONAL scalars, theta frozen at OB ----
    # Gauge-fixed: fit ONLY nu_s (1-D, well-conditioned; Gp=lam pinned). Full
    # agnostic: fit (Gp, lam, nu_s) -- degenerate in a developed flow (see the
    # gauge note above), so use gauge-fixed for the channel. A non-finite
    # L-BFGS eval is mapped to a large finite value so scipy backtracks instead
    # of poisoning its curvature.
    _BIG = 1e12
    if gauge_fixed:
        loss1 = make_loss(1.0, lam_q=stage1_lambda_q)
        th_ob = fit['theta']
        ty_frozen = fit.get('tau_y')

        def _loss_nu(nu, th):
            d = {'theta': th, 'nu_s': nu[0]}
            if yield_scalar:
                d['tau_y'] = ty_frozen
            return loss1(d)
        vag_sc = jax.jit(jax.value_and_grad(_loss_nu, argnums=0))

        def _obj(x):
            val, g = vag_sc(jnp.asarray(x, dtype=jnp.float64), th_ob)
            fv = float(val); gv = np.asarray(g, dtype=np.float64)
            gstep[0] += 1
            fit_tmp = {'theta': th_ob, 'nu_s': jnp.asarray(x[0])}
            if yield_scalar:
                fit_tmp['tau_y'] = fit['tau_y']
            _, lvf, lqf = make_eval_parts_jit(1.0, lam_q=stage1_lambda_q)(fit_tmp)
            _log('stage1', 1.0, fv, float(lvf), float(lqf), fit_tmp)
            if not (np.isfinite(fv) and np.all(np.isfinite(gv))):
                return _BIG, np.zeros_like(gv)
            return fv, gv

        s1_bounds = [nus_bound if yield_scalar else
                     (args.scalar_bound_lo, args.scalar_bound_hi)]
        print(f"[stage1] L-BFGS on nu_s only (gauge-fixed Gp={gp_gauge:g} "
              f"lam={lam_gauge:g}, theta frozen OB"
              f"{', tau_y frozen' if yield_scalar else ''}); "
              f"lam_Q_stage1={stage1_lambda_q:g}  maxiter={args.stage1_maxiter}")
        res = minimize(_obj, np.array([nus_init]), jac=True, method='L-BFGS-B',
                       bounds=s1_bounds,
                       options=dict(maxiter=args.stage1_maxiter,
                                    ftol=args.stage1_ftol, gtol=1e-10))
        fit['nu_s'] = jnp.asarray(res.x[0], dtype=jnp.float64)
        print(f"[stage1] done nfev={res.nfev} loss={float(res.fun):.6e} "
              f"{_scalars_str(fit)} (truth nu_s={nu_t}"
              f"{f' tau_y={tau_y_t}' if yield_scalar else ''})")
    elif agnostic and not alt_mode:
        loss1 = make_loss(1.0)

        def _loss_vec(svec, th):
            return loss1({'theta': th, 'Gp': svec[0], 'lam': svec[1],
                          'nu_s': svec[2]})
        vag_sc = jax.jit(jax.value_and_grad(_loss_vec, argnums=0))
        th_ob = fit['theta']
        x0 = np.array([Gp_init, lam_init, nus_init], dtype=np.float64)

        def _obj(x):
            val, g = vag_sc(jnp.asarray(x, dtype=jnp.float64), th_ob)
            fv = float(val); gv = np.asarray(g, dtype=np.float64)
            gstep[0] += 1
            fit_tmp = {'theta': th_ob, 'Gp': jnp.asarray(x[0]),
                       'lam': jnp.asarray(x[1]), 'nu_s': jnp.asarray(x[2])}
            _, lvf, lqf = eval_parts_s1(fit_tmp)
            _log('stage1', 1.0, fv, float(lvf), float(lqf), fit_tmp)
            if not (np.isfinite(fv) and np.all(np.isfinite(gv))):
                return _BIG, np.zeros_like(gv)
            return fv, gv

        print(f"[stage1] L-BFGS scalars (theta frozen OB, kappa=1); "
              f"maxiter={args.stage1_maxiter}")
        res = minimize(_obj, x0, jac=True, method='L-BFGS-B',
                       bounds=[(args.scalar_bound_lo, args.scalar_bound_hi)] * 3,
                       options=dict(maxiter=args.stage1_maxiter,
                                    ftol=args.stage1_ftol, gtol=1e-10))
        fit['Gp'] = jnp.asarray(res.x[0], dtype=jnp.float64)
        fit['lam'] = jnp.asarray(res.x[1], dtype=jnp.float64)
        fit['nu_s'] = jnp.asarray(res.x[2], dtype=jnp.float64)
        print(f"[stage1] done nfev={res.nfev} loss={float(res.fun):.6e} "
              f"{_scalars_str(fit)} (truth Gp={Gp_t} lam={lam_t} nu_s={nu_t})")

    # ---- Stage 2: kappa-annealed Adam on theta (+ slow scalars) ----
    def _label(params):
        lab = {'theta': jax.tree_util.tree_map(lambda _: 'theta', params['theta'])}
        if yield_scalar and gauge_fixed:
            if 'nu_s' in params:
                lab['nu_s'] = 'frozen'   # stage-1 estimate; no Adam updates
            if 'tau_y' in params:
                lab['tau_y'] = 'scalars'
            return lab
        for k in ('Gp', 'lam', 'nu_s'):
            if k in params:
                lab[k] = 'scalars'
        if yield_scalar and 'tau_y' in params:
            lab['tau_y'] = 'scalars'
        return lab

    def _clip_tau_y(f):
        if yield_scalar and 'tau_y' in f:
            f = {**f, 'tau_y': jnp.clip(f['tau_y'], tau_y_bound[0], tau_y_bound[1])}
        return f

    def _theta_opt(n_steps):
        sched = optax.warmup_cosine_decay_schedule(
            init_value=args.lr * 0.05, peak_value=args.lr,
            warmup_steps=max(1, args.warmup),
            decay_steps=max(args.warmup + 1, n_steps), end_value=args.lr * 0.02)
        chain = []
        if args.clip > 0:
            chain.append(optax.clip_by_global_norm(args.clip))
        chain.append(optax.adam(sched))
        return optax.chain(*chain)

    def _scalar_opt(n_steps):
        sched = optax.warmup_cosine_decay_schedule(
            init_value=args.scalar_lr2 * 0.5, peak_value=args.scalar_lr2,
            warmup_steps=max(1, args.warmup // 2),
            decay_steps=max(args.warmup + 1, n_steps), end_value=args.scalar_lr2 * 0.02)
        return optax.adam(sched)

    best = (float('inf'), fit, kappa_schedule[-1])
    L_final = L0
    n_blk_base = args.adam_steps_per_kappa
    nan_stop = False

    # ==================================================================
    # alt_mode: un-gauged V2 s1 alternation (Hookean criterion). Replaces
    # the kappa loop below (which no-ops when alt_mode). Structure:
    #   stage-1: 4-scalar L-BFGS with theta=OB frozen, FULL vel+Q loss
    #            (theta=OB => K=A-I => model class == Saramito generator,
    #             so all four scalars should land near truth here);
    #   then N_CYCLES x [ theta-block Adam (scalars frozen)
    #                     -> scalar L-BFGS re-solve (theta frozen) ].
    # tau_y lives ONLY in the scalar stages (never in the theta Adam group).
    # ==================================================================
    if alt_mode:
        scalar_bounds = [gp_bound, lam_bound, nus_bound, tau_y_bound]
        scalar_keys = ('Gp', 'lam', 'nu_s', 'tau_y')

        def _full_parts(d, theta_p):
            params = {'Gp': _gp_of(d), 'lam': _lam_of(d), 'theta': theta_p,
                      'tbnn_bound_c': bound_c, 'tbnn_kappa': 1.0,
                      'tau_y': _tau_y_of(d)}
            Lv = jnp.asarray(0.0, dtype=jnp.float64)
            Lq = jnp.asarray(0.0, dtype=jnp.float64)
            per_lv = []
            per_lq = []
            for tdi in truth_forcings:
                out = _forward(tbnn_state, tbnn_model, tbnn_perm, params,
                               _nu_of(d), tdi['g_x'])
                lv, lq = _loss_components(out, tdi['u_truth'], tdi['v_truth'],
                                          tdi['Q_truth'], cfg,
                                          Q_scale=tdi.get('Q_scale'))
                lv = tdi['w_vel'] * lv          # ratio weighting (frozen)
                Lv = Lv + lv
                Lq = Lq + lq
                per_lv.append(lv)
                per_lq.append(lq)
            # stack per-drive for logging (no extra forwards)
            return (Lv + lambda_q * Lq,
                    (Lv, Lq, jnp.stack(per_lv), jnp.stack(per_lq)))

        def _solve_scalars(fit_in, stage_label, maxiter):
            th = fit_in['theta']

            def _loss_svec(svec):
                d = {'theta': th, 'Gp': svec[0], 'lam': svec[1],
                     'nu_s': svec[2], 'tau_y': svec[3]}
                return _full_parts(d, th)
            vag = jax.jit(jax.value_and_grad(_loss_svec, has_aux=True))

            def _obj(x):
                (val, (lv, lq, plv, plq)), g = vag(
                    jnp.asarray(x, dtype=jnp.float64))
                fv = float(val); gv = np.asarray(g, dtype=np.float64)
                gstep[0] += 1
                ftmp = {'theta': th, 'Gp': jnp.asarray(x[0]),
                        'lam': jnp.asarray(x[1]), 'nu_s': jnp.asarray(x[2]),
                        'tau_y': jnp.asarray(x[3])}
                pd = list(zip([float(x) for x in plv],
                              [float(x) for x in plq]))
                _log(stage_label, 1.0, fv, float(lv), float(lq), ftmp,
                     grad_norm=float(np.linalg.norm(gv)), per_drive=pd)
                if not (np.isfinite(fv) and np.all(np.isfinite(gv))):
                    return _BIG, np.zeros_like(gv)
                return fv, gv

            x0 = np.array([float(fit_in['Gp']), float(fit_in['lam']),
                           float(fit_in['nu_s']), float(fit_in['tau_y'])],
                          dtype=np.float64)
            res = minimize(_obj, x0, jac=True, method='L-BFGS-B',
                           bounds=scalar_bounds,
                           options=dict(maxiter=maxiter, ftol=args.stage1_ftol,
                                        gtol=1e-10))
            out = dict(fit_in)
            for k, v in zip(scalar_keys, res.x):
                out[k] = jnp.asarray(v, dtype=jnp.float64)
            return out, res

        def _theta_block(fit_in, n_steps, cyc):
            th0 = fit_in['theta']
            scalars = {k: fit_in[k] for k in scalar_keys}   # FROZEN

            def _loss_theta(theta_p):
                d = {'theta': theta_p, **scalars}
                return _full_parts(d, theta_p)
            vag = jax.jit(jax.value_and_grad(_loss_theta, has_aux=True))
            opt = _theta_opt(n_steps)
            opt_state = opt.init(th0)

            @jax.jit
            def _step(theta_p, ostate):
                (L, (lv, lq, plv, plq)), g = vag(theta_p)
                gnorm = optax.global_norm(g)
                upd, ostate = opt.update(g, ostate, theta_p)
                theta_p = optax.apply_updates(theta_p, upd)
                return theta_p, ostate, L, lv, lq, plv, plq, gnorm

            th = th0
            L_last = float('inf')
            print(f"  [c{cyc}] theta-block: {n_steps} Adam steps "
                  f"(scalars frozen)", flush=True)
            for it in range(1, n_steps + 1):
                th_new, opt_state, L, lv, lq, plv, plq, gnorm = _step(
                    th, opt_state)
                Lf = float(L)
                gstep[0] += 1
                if not np.isfinite(Lf):
                    print(f"  [c{cyc}] step {gstep[0]} loss=NaN/inf -- "
                          f"diverged; NaN early-stop.", flush=True)
                    return {'theta': th, **scalars}, True, L_last
                th = th_new
                L_last = Lf
                pd = list(zip([float(x) for x in plv],
                              [float(x) for x in plq]))
                _log(f'c{cyc}', 1.0, Lf, float(lv), float(lq),
                     {'theta': th, **scalars},
                     grad_norm=float(gnorm), per_drive=pd)
                if it % 25 == 0 or it == 1:
                    print(f"  [c{cyc}] step {gstep[0]} loss={Lf:.6e} "
                          f"[{time.time()-t_opt:.0f}s]", flush=True)
                if time.time() - t_opt > args.time_budget_s:
                    print(f"  [c{cyc}] time budget hit inside theta-block.")
                    break
            return {'theta': th, **scalars}, False, L_last

        # --- Resume from train_ckpt if requested --------------------------
        resume_state = None
        start_cycle = 0
        skip_stage1 = False
        if args.resume:
            loaded = _load_train_ckpt()
            if loaded is not None:
                resume_state, fit = loaded
                skip_stage1 = True
                st = resume_state.get('stage', '')
                if st == 'stage1' or st.startswith('resolve'):
                    start_cycle = int(resume_state.get('cycle', 0))
                    if st.startswith('resolve'):
                        start_cycle = int(resume_state.get('cycle', 0)) + 1
                elif st.startswith('c'):
                    # mid theta-block: restart that cycle from scalars
                    start_cycle = int(resume_state.get('cycle', 0))
                print(f"[resume] skip_stage1={skip_stage1} "
                      f"start_cycle={start_cycle}", flush=True)

        # --- Stage 1: 4-scalar L-BFGS, theta=OB frozen, full vel+Q ---------
        if not skip_stage1:
            print("[alt] V2 un-gauged alternating (Hookean criterion). "
                  "Stage-1 scalar solve at theta=OB (model class == generator).",
                  flush=True)
            fit, res1 = _solve_scalars(fit, 'stage1', args.stage1_maxiter)
            q = tuple(float(fit[k]) for k in scalar_keys)
            print(f"[stage1] >>> RECOVERED QUADRUPLE (theta=OB): Gp={q[0]:.4f} "
                  f"lam={q[1]:.4f} nu_s={q[2]:.4f} tau_y={q[3]:.4f}  "
                  f"(truth 3.2/0.7/0.8/1.45; loss={float(res1.fun):.4e}) <<<",
                  flush=True)
            ty_rel_s1 = abs(q[3] - tau_y_t) / tau_y_t
            if ty_rel_s1 <= 0.10:
                print(f"[stage1] tau_y within {ty_rel_s1:.1%} of 1.45 -- ON TRACK.")
            else:
                print(f"[stage1] FLAG: tau_y off by {ty_rel_s1:.1%} (>10%); "
                      f"run continues (theta blocks may still help).")
            drift_rows = [('stage1',) + q]
            _log_scalars('stage1', fit, float(res1.fun))
            _archive_stage_ckpt(run_dir, fit, 'stage1')
            _save_train_ckpt(fit, 'stage1', -1)
        else:
            q = tuple(float(fit[k]) for k in scalar_keys)
            drift_rows = [('stage1_resumed',) + q]
            print(f"[resume] continuing from quadruple Gp={q[0]:.4f} "
                  f"lam={q[1]:.4f} nu_s={q[2]:.4f} tau_y={q[3]:.4f}",
                  flush=True)

        # --- N_CYCLES x (theta-block -> scalar re-solve) -------------------
        # --freeze-theta: scalars only (control). Skip every theta Adam block.
        N_CYCLES = 0 if args.freeze_theta else 4
        if args.freeze_theta:
            print("[alt] --freeze-theta: skipping all theta Adam blocks; "
                  "scalars-only control at the chosen theta init.",
                  flush=True)
        n_theta = int(args.adam_steps_per_kappa)
        resolve_maxiter = max(20, args.stage1_maxiter // 2)
        for cyc in range(start_cycle, N_CYCLES):
            if time.time() - t_opt > args.time_budget_s:
                print(f"[alt] time budget hit before cycle {cyc}; stopping.")
                _save_train_ckpt(fit, f'pre_c{cyc}', cyc - 1)
                break
            fit, nan_stop, _ = _theta_block(fit, n_theta, cyc)
            _save_train_ckpt(fit, f'c{cyc}', cyc)
            if nan_stop:
                break
            prev_q = tuple(float(fit[k]) for k in scalar_keys)
            fit, res_r = _solve_scalars(fit, f'resolve{cyc}', resolve_maxiter)
            new_q = tuple(float(fit[k]) for k in scalar_keys)
            drift = tuple(new_q[i] - prev_q[i] for i in range(4))
            drift_rows.append((f'resolve{cyc}',) + new_q)
            print(f"[resolve{cyc}] Gp={new_q[0]:.4f} lam={new_q[1]:.4f} "
                  f"nu_s={new_q[2]:.4f} tau_y={new_q[3]:.4f}  "
                  f"drift(dGp,dlam,dnu,dtau)=({drift[0]:+.3f},{drift[1]:+.3f},"
                  f"{drift[2]:+.3f},{drift[3]:+.3f})", flush=True)
            if abs(drift[3]) > 0.10 * tau_y_t:
                print(f"[resolve{cyc}] FLAG: |dtau_y|={abs(drift[3]):.3f} "
                      f"(>10% of {tau_y_t}) -- theta may be eating the yield "
                      f"surface; scalars re-absorbing it.")
            _log_scalars(f'resolve{cyc}', fit, float(res_r.fun))
            _archive_stage_ckpt(run_dir, fit, f'resolve{cyc}')
            _save_train_ckpt(fit, f'resolve{cyc}', cyc)
            if time.time() - t_opt > args.time_budget_s:
                break

        eval_final_alt = make_eval_parts_jit(1.0)
        L_alt, _, _ = eval_final_alt(fit)
        L_final = float(L_alt)
        best = (L_final, fit, 1.0)
        np.savez_compressed(
            os.path.join(run_dir, 'scalar_drift.npz'),
            quad=np.array([r[1:] for r in drift_rows], dtype=np.float64),
            labels=np.array([r[0] for r in drift_rows]))
        _archive_stage_ckpt(run_dir, fit, 'final')
        _log_scalars('final', fit, L_final)
        # Only mark train_ckpt as 'final' if all cycles completed; else leave
        # last resolve/c* tag so --resume continues.
        alt_complete = (not nan_stop) and (
            any(r[0] == f'resolve{N_CYCLES-1}' for r in drift_rows)
            or (start_cycle >= N_CYCLES))
        if alt_complete:
            _save_train_ckpt(fit, 'final', N_CYCLES - 1)
        else:
            _save_train_ckpt(fit, 'incomplete', max(start_cycle, 0))
            print("[alt] INCOMPLETE (time budget / early stop) -- "
                  "DONE will NOT be written; chain resume will continue.",
                  flush=True)
        # stash for DONE gate below
        args._alt_complete = alt_complete

    for bi, kappa in enumerate([] if alt_mode else kappa_schedule):
        if nan_stop:
            break
        # slowtail: low-kappa blocks (<=0.05) get tail_step_mult x the steps
        # (the ReLU-sharpening regime; tests whether more time there tightens
        # the yield surface). base schedule: uniform.
        n_blk = (n_blk_base * int(args.tail_step_mult)
                 if (slowtail and kappa <= 0.05) else n_blk_base)
        kappa_hist.append((gstep[0], float(kappa)))
        loss_k = make_loss(kappa)
        vg = jax.jit(jax.value_and_grad(loss_k))
        if agnostic or (yield_scalar and gauge_fixed):
            transforms = {'theta': _theta_opt(n_blk), 'scalars': _scalar_opt(n_blk)}
            if yield_scalar and gauge_fixed:
                transforms['frozen'] = optax.set_to_zero()
            opt = optax.multi_transform(transforms, _label)
        else:
            opt = _theta_opt(n_blk)
        opt_state = opt.init(fit)

        @jax.jit
        def step(fit, opt_state):
            L, g = vg(fit)
            upd, opt_state = opt.update(g, opt_state, fit)
            fit_new = optax.apply_updates(fit, upd)
            if yield_scalar:
                fit_new = _clip_tau_y(fit_new)
            return fit_new, opt_state, L

        eval_parts_k = make_eval_parts_jit(kappa)
        is_last = (bi == len(kappa_schedule) - 1)
        print(f"  [k{bi} kappa={kappa:g}] block: {n_blk} steps", flush=True)
        for it in range(1, n_blk + 1):
            prev = fit
            fit, opt_state, L = step(fit, opt_state)
            Lf = float(L)
            gstep[0] += 1
            # NaN early-stop: a diverged forward poisons all subsequent steps
            # (and burns hours churning NaN). Stop, keep the last good `best`.
            if not np.isfinite(Lf):
                print(f"  [k{bi} kappa={kappa:g}] step {gstep[0]} loss=NaN/inf "
                      f"-- diverged; NaN early-stop (keeping best={best[0]:.4e}).",
                      flush=True)
                fit = prev
                nan_stop = True
                break
            L_final = Lf
            _, lvf, lqf = eval_parts_k(prev)
            _log(f'k{bi}', kappa, Lf, float(lvf), float(lqf), prev)
            # Track best ONLY within the final kappa (losses across kappa are
            # not comparable -- m0 changes with kappa; never gate mid-anneal).
            if is_last and Lf < best[0]:
                best = (Lf, prev, kappa)
            if it % 25 == 0 or it == 1:
                print(f"  [k{bi} kappa={kappa:g}] step {gstep[0]} loss={Lf:.6e} "
                      f"{_scalars_str(prev)} [{time.time()-t_opt:.0f}s]", flush=True)
            if time.time() - t_opt > args.time_budget_s:
                print(f"  [k{bi}] time budget hit; stopping."); break
        print(f"[k{bi}] kappa={kappa:g} done  loss={L_final:.6e}  {_scalars_str(fit)}")
        if time.time() - t_opt > args.time_budget_s:
            break

    if best[0] < float('inf'):
        L_final, fit, kappa_final = best
    else:
        kappa_final = kappa_schedule[-1]
    theta = fit['theta']
    Gp_fit, lam_fit, nu_fit = (float(_gp_of(fit)), float(_lam_of(fit)),
                               float(_nu_of(fit)))
    eval_parts_final = make_eval_parts_jit(kappa_final)
    _, L_vel_final, L_Q_final = eval_parts_final(fit)
    L_vel_final = float(L_vel_final); L_Q_final = float(L_Q_final)
    wall_opt = time.time() - t_opt
    n_grads = int(gstep[0])
    spg = wall_opt / max(n_grads, 1)
    loss_red0 = L0 / max(L_final, 1e-300)
    converged_run = bool((not nan_stop) and loss_red0 > 10.0)
    print(f"[opt] done {wall_opt:.0f}s  loss {L0:.3e} -> {L_final:.3e} "
          f"({loss_red0:.2e}x)  SigmaL_vel {L_vel0:.3e}->{L_vel_final:.3e}  "
          f"SigmaL_Q {L_Q0:.3e}->{L_Q_final:.3e}  kappa_final={kappa_final:g}  "
          f"{_scalars_str(fit)}")
    print(f"[opt] grads={n_grads}  {spg:.1f}s/grad  nan_stopped={nan_stop}  "
          f"converged={converged_run}")

    tau_y_fit = float(_tau_y_of(fit)) if yield_scalar else float('nan')
    if yield_scalar:
        np.savez_compressed(os.path.join(run_dir, 'tau_y_history.npz'),
                            tau_y_hist=np.array(tau_y_hist, dtype=np.float64))

    # --- HARD checkpoint + reload self-check (loss at kappa_final reproduces) ---
    ckpt_ok, ckpt_path = _save_and_verify_ckpt(
        run_dir, theta, fit, args, Gp_fit, lam_fit, nu_fit, kappa_final,
        L_final, make_loss(kappa_final), agnostic, tau_y_t,
        tau_y_fit=tau_y_fit, yield_scalar=yield_scalar, gauge_fixed=gauge_fixed)
    print(f"[ckpt] reload self-check: {'PASS' if ckpt_ok else 'FAIL'} -> {ckpt_path}")

    # --- batch metrics (read by aggregate_channel_batch.py). Written early so
    # even a --no-eval or crashed-eval run leaves the convergence record. ---
    batch_metrics = dict(
        run_name=args.run_name, geometry=geometry, init=args.init,
        theta_init=theta_init_mode,
        theta_init_scale=float(args.theta_init_scale),
        freeze_theta=bool(args.freeze_theta),
        yield_mode=yield_mode, model_name=model_name,
        theta_seed=int(args.theta_seed), schedule=args.kappa_schedule,
        kappa_schedule=kappa_schedule, kappa_final=float(kappa_final),
        Ny=int(cfg['Ny']), outer_steps=int(cfg['outer_steps']),
        fit_solver_tol=float(cfg['solver_tol']),
        adam_steps_per_kappa=int(args.adam_steps_per_kappa),
        tail_step_mult=int(args.tail_step_mult),
        loss_init=float(L0), loss_final=float(L_final),
        L_vel_init=float(L_vel0), L_Q_init=float(L_Q0),
        L_vel_final=float(L_vel_final), L_Q_final=float(L_Q_final),
        lambda_q=float(lambda_q), lambda_q0=float(lambda_q0),
        g_x_list=g_x_list, multi_forcing=bool(multi_forcing),
        Q_truth={f"{td['g_x']:g}": float(td['Q_truth']) for td in truth_forcings},
        vel_weights=vel_weights, vel_W=vel_W,      # ratio weighting, per drive
        yield_pref_floor=float(tb._YIELD_PREF_FLOOR),
        loss_reduction=float(loss_red0), nan_stopped=bool(nan_stop),
        converged=bool(converged_run), n_grads=int(n_grads),
        s_per_grad=float(spg), wall_opt_s=float(wall_opt),
        nu_s_fit=float(nu_fit), nu_s_truth=float(nu_t),
        tau_y_fit=(float(tau_y_fit) if yield_scalar else None),
        Gp_fit=float(Gp_fit), lam_fit=float(lam_fit),
        tau_y_truth=float(tau_y_t), g_x=float(args.g_x),
        y_p_analytic=float(tau_y_t / args.g_x), checkpoint_reload=bool(ckpt_ok))
    if alt_mode:
        batch_metrics['alt_mode'] = True
        batch_metrics['br_init'] = br_init          # BR regression provenance
        batch_metrics['scalar_init'] = dict(Gp=Gp_init, lam=lam_init,
                                            nu_s=nus_init, tau_y=tau_y_init)
        batch_metrics['scalar_recovered'] = dict(
            Gp=float(Gp_fit), lam=float(lam_fit), nu_s=float(nu_fit),
            tau_y=float(tau_y_fit))
        batch_metrics['scalar_truth'] = dict(Gp=float(Gp_t), lam=float(lam_t),
                                             nu_s=float(nu_t), tau_y=float(tau_y_t))
    with open(os.path.join(run_dir, 'batch_metrics.json'), 'w') as f:
        json.dump(batch_metrics, f, indent=2, default=float)

    # Save kappa history unconditionally (trap checklist: kappa history saved).
    np.savez_compressed(os.path.join(run_dir, 'kappa_history.npz'),
                        kappa_hist=np.array(kappa_hist, dtype=np.float64),
                        kappa_schedule=np.array(kappa_schedule, dtype=np.float64),
                        kappa_final=np.float64(kappa_final))

    if args.no_eval:
        print("[no-eval] checkpoint + kappa history saved; skipping eval/figure.")
        return 0

    # --- invariant cloud at the fitted Tier-3 model (correct switches) ---
    g_x_ref = 4.0 if any(abs(g - 4.0) < 1e-9 for g in g_x_list) else g_x_list[-1]
    cfg_ref = dict(cfg, g_x=g_x_ref)
    out_truth_ref = next(td['out_truth'] for td in truth_forcings
                         if abs(td['g_x'] - g_x_ref) < 1e-9)
    cloud = vt.tbnn_invariant_cloud(
        cfg_ref, theta, Gp=Gp_fit, lam=lam_fit, nu_s=nu_fit, bound_c=bound_c,
        model_name=model_name,
        anchored=True if yield_scalar else False,
        mobility='softplus',
        kappa=kappa_final, geometry=geometry,
        tau_y=tau_y_fit if yield_scalar else None)
    print(f"[cloud] active_fraction={cloud['active_fraction']:.2e}  "
          f"x1 q1={cloud['x1']['q1']:.3f} q99={cloud['x1']['q99']:.3f}")

    # --- learned 0D flow-curve intercept vs tau_y (record-don't-fail) ---
    fc = _learned_flow_curve(theta, Gp_fit, lam_fit, kappa_final, bound_c,
                             n_shear=args.n_shear, yield_scalar=yield_scalar,
                             tau_y_fit=tau_y_fit)
    print(f"[recovery] learned flow-curve |tau_d| intercept = {fc['intercept']} "
          f"(gd_min={fc['gd_min']}); tau_y_truth={tau_y_t}")
    intercept_rel = (abs(fc['intercept'] - tau_y_t) / tau_y_t
                     if np.isfinite(fc['intercept']) else float('nan'))

    # --- deliverable yield-surface figure (geometry-specific) ---
    if geometry == 'channel':
        fig_path, yfig = _yield_surface_figure_channel(
            run_dir, out_truth_ref, theta, Gp_t, tau_y_t, bound_c, kappa_final,
            cfg_ref, g_x_ref, yield_scalar=yield_scalar,
            Gp_fit=Gp_fit, tau_y_fit=tau_y_fit)
        per_forcing = _per_forcing_plug_metrics(
            truth_forcings, theta, fit, Gp_t, tau_y_t, bound_c, kappa_final,
            cfg, grid, tbnn_model, tbnn_state, tbnn_perm, _gp_of, _lam_of,
            _nu_of, _tau_y_of=_tau_y_of if yield_scalar else None)
        batch_metrics['per_forcing_plugs'] = per_forcing
        batch_metrics['g_x_ref'] = float(g_x_ref)
    else:
        fig_path, yfig = _yield_surface_figure(
            run_dir, out_truth, theta, Gp_t, tau_y_t, bound_c, kappa_final,
            yield_scalar=yield_scalar, Gp_fit=Gp_fit, tau_y_fit=tau_y_fit)
    print(f"[figure] yield surface -> {fig_path}  "
          f"(learned yielded frac={yfig['learned_yield_frac']:.1%}, "
          f"true yielded frac={yfig['true_yield_frac']:.1%}, "
          f"overlap IoU={yfig['iou']:.2f})")
    if 'plug_half_width_learned' in yfig:
        print(f"[figure] PLUG half-width (headline): learned="
              f"{yfig['plug_half_width_learned']:.4f}  true="
              f"{yfig['plug_half_width_true']:.4f}  analytic y_p=tau_y/g_x="
              f"{yfig['y_p_analytic']:.4f}  rel(learned vs true)={yfig['plug_rel']:.1%}")

    # Augment batch_metrics with the eval-derived headline numbers + re-dump.
    batch_metrics.update(
        flow_intercept=(float(fc['intercept']) if np.isfinite(fc['intercept'])
                        else None),
        intercept_rel=(float(intercept_rel) if np.isfinite(intercept_rel)
                       else None),
        yield_iou=float(yfig['iou']),
        plug_half_width_learned=yfig.get('plug_half_width_learned'),
        plug_half_width_true=yfig.get('plug_half_width_true'),
        plug_rel=yfig.get('plug_rel'), active_fraction=float(cloud['active_fraction']))
    with open(os.path.join(run_dir, 'batch_metrics.json'), 'w') as f:
        json.dump(batch_metrics, f, indent=2, default=float)

    loss_red = L0 / max(L_final, 1e-300)
    gate = {
        'loss_decreased': bool(loss_red > 10.0),
        'checkpoint_reload': bool(ckpt_ok),
        'intercept_within_15pct': bool(np.isfinite(intercept_rel)
                                       and intercept_rel <= 0.15),
    }
    print(f"\n==== P3-G4 (EVP fit) verdict [RECORD-DON'T-FAIL] ====")
    for k, v in gate.items():
        print(f"  {k}: {v}")
    print(f"  [record] loss reduction={loss_red:.2e}x  intercept rel={intercept_rel}")
    print(f"  [record] yielded-fraction(truth)={yf['yielded_fraction']:.1%}  "
          f"kappa_final={kappa_final:g}")
    gate_pass = bool(gate['loss_decreased'] and gate['checkpoint_reload'])

    _save_evp_outputs(run_dir, args, loss_hist, kappa_hist, kappa_schedule,
                      kappa_final, L0, L_final, loss_red, cloud, fc, yf, yfig,
                      gate, gate_pass, Gp_fit, lam_fit, nu_fit, Gp_t, lam_t,
                      nu_t, tau_y_t, intercept_rel, agnostic, ckpt_ok,
                      tau_y_fit=(tau_y_fit if yield_scalar else None),
                      br_init=(br_init if alt_mode else None),
                      model_name=model_name)
    # --- Save fitted-model final trajectories + Q(t) vs truth (offline plots) ---
    print("[save] fitted model trajectories per drive ...", flush=True)
    params_fit = {'Gp': jnp.asarray(Gp_fit), 'lam': jnp.asarray(lam_fit),
                  'theta': theta, 'tbnn_bound_c': bound_c,
                  'tbnn_kappa': float(kappa_final)}
    if yield_scalar:
        params_fit['tau_y'] = jnp.asarray(tau_y_fit)
    for td in truth_forcings:
        gx = td['g_x']
        gx_key = f'{gx:g}'.replace('.', 'p')
        out_m = jax.jit(lambda g=gx: _forward(
            tbnn_state, tbnn_model, tbnn_perm, params_fit, nu_fit, g))()
        out_m['u_traj'].block_until_ready()
        Qm = np.asarray(_flow_rate_Q_traj(out_m['u_traj'], cfg))
        Qt = np.asarray(td['Q_traj'])
        np.savez_compressed(
            os.path.join(run_dir, f'model_traj_gx{gx_key}.npz'),
            u=np.asarray(out_m['u_traj']), v=np.asarray(out_m['v_traj']),
            A_xx=np.asarray(out_m['A_xx_traj']),
            A_xy=np.asarray(out_m['A_xy_traj']),
            A_yy=np.asarray(out_m['A_yy_traj']),
            A_zz=np.asarray(out_m['A_zz_traj']),
            Q_model=Qm, Q_truth=Qt, Q_scale=float(td['Q_scale']), g_x=gx)
        print(f"  [save] model_traj_gx{gx_key}.npz  "
              f"Q_final model={Qm[-1]:.6e} truth={Qt[-1]:.6e}", flush=True)

    print("[done]")
    # Mark complete for chained afterany jobs ONLY if alt_mode finished all
    # cycles (or non-alt path). Incomplete time-budget stops leave train_ckpt
    # for --resume and do not write DONE.
    alt_complete = getattr(args, '_alt_complete', True)
    if alt_complete and (not nan_stop):
        with open(os.path.join(run_dir, 'DONE'), 'w') as f:
            f.write(f"completed {datetime.now().isoformat()}  "
                    f"loss={L_final:.6e} grads={n_grads}\n")
        print(f"[done] wrote {run_dir}/DONE", flush=True)
    else:
        print(f"[done] incomplete -- no DONE marker "
              f"(alt_complete={alt_complete} nan_stop={nan_stop})", flush=True)
    return 0 if gate_pass else 2


def _archive_stage_ckpt(run_dir, fit, tag):
    """Copy current scalars+theta to ckpt_stage{tag}.pkl at stage boundaries
    (stage-1 + each re-solve) for the response-overlay ridge diagnostic.
    Additive; does not replace the final theta_checkpoint.npz."""
    import pickle
    path = os.path.join(run_dir, f'ckpt_stage{tag}.pkl')
    obj = {k: (np.asarray(v) if hasattr(v, 'shape') else v)
           for k, v in fit.items()}
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    print(f"[ckpt] archived {path}", flush=True)


def _save_and_verify_ckpt(run_dir, theta, fit, args, Gp_fit, lam_fit, nu_fit,
                          kappa_final, L_final, loss_fn, agnostic, tau_y_t,
                          *, tau_y_fit=float('nan'), yield_scalar=False,
                          gauge_fixed=False):
    """Save theta + fitted scalars + kappa_final as raw arrays, reload from
    the .npz (cross-env path), and assert loss(reloaded) reproduces the
    logged best loss AT kappa_final (HARD self-check)."""
    arrs, nlayers = theta_to_named_arrays(theta)
    heads = list(theta.keys())
    meta = dict(
        ckpt_heads=np.array(heads),
        ckpt_nlayers=np.array([nlayers[h] for h in heads], dtype=np.int64),
        ckpt_width=np.int64(args.width), ckpt_depth=np.int64(args.depth),
        ckpt_bound_c=np.float64(args.__dict__.get('bound_c', tb.TBNN_DEFAULT_BOUND_C)),
        ckpt_anchored=np.bool_(True if yield_scalar else False),
        ckpt_mobility=np.array('softplus' if yield_scalar else 'relu_annealed'),
        ckpt_yield_mode=np.array('scalar' if yield_scalar else 'off'),
        ckpt_gauge_fixed=np.bool_(gauge_fixed),
        ckpt_kappa_final=np.float64(kappa_final),
        ckpt_Gp_fit=np.float64(Gp_fit), ckpt_lam_fit=np.float64(lam_fit),
        ckpt_nu_s=np.float64(nu_fit), ckpt_loss=np.float64(L_final),
        ckpt_tau_y_truth=np.float64(tau_y_t),
        ckpt_tau_y_fit=np.float64(tau_y_fit if yield_scalar else float('nan')),
        # Lower clamp on the yield prefactor the fit ran under. Recorded so a
        # checkpoint can never be replayed under a different arrest floor than
        # the one it was trained with.
        ckpt_yield_pref_floor=np.float64(tb._YIELD_PREF_FLOOR),
        ckpt_agnostic=np.bool_(agnostic))
    ckpt_path = os.path.join(run_dir, 'theta_checkpoint.npz')
    np.savez_compressed(ckpt_path, **arrs, **meta)
    z = np.load(ckpt_path, allow_pickle=False)
    heads_r = [str(h) for h in z['ckpt_heads']]
    nlayers_r = {h: int(n) for h, n in zip(heads_r, z['ckpt_nlayers'])}
    theta_r = theta_from_named_arrays(z, heads_r, nlayers_r)
    fit_r = {'theta': theta_r}
    gf = bool(z['ckpt_gauge_fixed']) if 'ckpt_gauge_fixed' in z.files else gauge_fixed
    ym = str(z['ckpt_yield_mode'].item()) if 'ckpt_yield_mode' in z.files else (
        'scalar' if yield_scalar else 'off')
    ys = (ym == 'scalar')
    if gf:
        fit_r['nu_s'] = jnp.asarray(float(z['ckpt_nu_s']), dtype=jnp.float64)
        if ys:
            fit_r['tau_y'] = jnp.asarray(float(z['ckpt_tau_y_fit']), dtype=jnp.float64)
    elif agnostic:
        fit_r['Gp'] = jnp.asarray(float(z['ckpt_Gp_fit']), dtype=jnp.float64)
        fit_r['lam'] = jnp.asarray(float(z['ckpt_lam_fit']), dtype=jnp.float64)
        fit_r['nu_s'] = jnp.asarray(float(z['ckpt_nu_s']), dtype=jnp.float64)
        # Un-gauged V2 (yield_scalar and NOT gauge_fixed): tau_y is a fitted
        # top-level scalar and MUST be carried on reload -- else make_loss's
        # _tau_y_of(fit) KeyErrors (the params-reconstruction omission class
        # that bit ckpt+plug before). Extends the round-trip to all 4 scalars.
        if ys:
            fit_r['tau_y'] = jnp.asarray(float(z['ckpt_tau_y_fit']),
                                         dtype=jnp.float64)
    L_reload = float(loss_fn(fit_r))
    rel = abs(L_reload - L_final) / max(abs(L_final), 1e-300)
    print(f"[ckpt] logged={L_final:.10e} reloaded={L_reload:.10e} rel={rel:.2e}")
    return (rel <= 1e-10), ckpt_path


def _learned_flow_curve(theta, Gp_fit, lam_fit, kappa_final, bound_c, *,
                        n_shear=12, gd_lo=0.2, gd_hi=8.0,
                        yield_scalar=False, tau_y_fit=None):
    """For V2 (scalar yield) the model's 0D yield stress is, BY THE ONE-RULER
    CONTRACT, the fitted tau_y measured on the SAME Hookean map ``Gp(A-I)``
    the closure prefactor uses -- so the flow-curve intercept IS tau_y_fit
    (no separate mis-specified relu/unanchored steady solve, which was the
    v2_prod two-ruler intercept). Reported directly; intercept_rel then
    coincides with the tau_y recovery, which is the honest statement for a
    scalar-tau_y criterion."""
    if yield_scalar:
        ty = float(tau_y_fit) if tau_y_fit is not None else float('nan')
        return dict(rows=[], intercept=ty, gd_min=float('nan'))
    return _learned_flow_curve_v1(theta, Gp_fit, lam_fit, kappa_final, bound_c,
                                  n_shear=n_shear, gd_lo=gd_lo, gd_hi=gd_hi)


def _learned_flow_curve_v1(theta, Gp_fit, lam_fit, kappa_final, bound_c, *,
                           n_shear=12, gd_lo=0.2, gd_hi=8.0):
    """Learned 0D steady-shear flow curve for the Tier-3 fit. Returns the
    per-gammadot |tau_d| (from the learned stress N1, N2, tau_xy) and the
    low-gammadot intercept estimate (the learned yield stress). Robust: the
    unanchored/EVP Newton root can fail near yield -- non-converged points
    are dropped, and the intercept is the smallest converged gammadot's
    |tau_d| (record-don't-fail)."""
    params_fit = {'Gp': Gp_fit, 'lam': lam_fit, 'theta': theta,
                  'tbnn_bound_c': bound_c}
    rows = []
    A_warm = None
    for gd in np.geomspace(gd_lo, gd_hi, n_shear):
        try:
            r = vt.tbnn_steady_reference(
                theta, params_fit, float(gd), Gp=Gp_fit, lam=lam_fit,
                bound_c=bound_c, anchored=False, mobility='relu_annealed',
                kappa=kappa_final, A_init=A_warm)
        except Exception:  # noqa: BLE001
            continue
        if not r.get('converged', False):
            continue
        A_warm = r['a']
        N1, N2, txy = r['N1'], r['N2'], r['tau_xy']
        # deviator norm from (N1, N2, tau_xy): a-b=N1, b-c=N2, a+b+c=0.
        a = (2 * N1 + N2) / 3.0
        b = (-N1 + N2) / 3.0
        c = (-N1 - 2 * N2) / 3.0
        td = float(np.sqrt(0.5 * (a * a + b * b + c * c + 2 * txy * txy)))
        rows.append(dict(gammadot=float(gd), tau_d=td, N1=N1, tau_xy=txy))
    if rows:
        intercept = rows[0]['tau_d']
        gd_min = rows[0]['gammadot']
    else:
        intercept = float('nan'); gd_min = float('nan')
    return dict(rows=rows, intercept=intercept, gd_min=gd_min)


def _yield_surface_figure(run_dir, out_truth, theta, Gp_t, tau_y_t, bound_c,
                          kappa_final, eps_mob=0.05, *, yield_scalar=False,
                          Gp_fit=None, tau_y_fit=None):
    """Deliverable figure: learned yield surface over the visited (x1, x3)
    plane overlaid with the true Saramito locus |tau_d|(x) = tau_y. Evaluated
    on the TRUTH's final visited conformation field. ONE RULER: V2 uses the
    model's own Hookean criterion |tau_d|(A;Gp_fit) > tau_y_fit; V1/V3 use the
    mobility min-eig zero-set. Returns figure path + overlap diagnostics."""
    Axx = np.asarray(out_truth['A_xx_traj'][-1]).reshape(-1)
    Axy = np.asarray(out_truth['A_xy_traj'][-1]).reshape(-1)
    Ayy = np.asarray(out_truth['A_yy_traj'][-1]).reshape(-1)
    Azz = np.asarray(out_truth['A_zz_traj'][-1]).reshape(-1)
    x1, x2, x3 = tb.tbnn_invariant_features(
        jnp.asarray(Axx), jnp.asarray(Axy), jnp.asarray(Ayy), jnp.asarray(Azz))
    x1 = np.asarray(x1); x2 = np.asarray(x2); x3 = np.asarray(x3)
    X = jnp.stack([jnp.asarray(x1), jnp.asarray(x2), jnp.asarray(x3)], axis=-1)
    anch = bool(yield_scalar)
    mob_mode = 'softplus' if yield_scalar else 'relu_annealed'
    _, _, _, m0, m1, _ = tb.tbnn_heads(theta, X, bound_c, anchored=anch,
                                       mobility=mob_mode, kappa=kappa_final)
    Aj = (jnp.asarray(Axx), jnp.asarray(Axy), jnp.asarray(Ayy), jnp.asarray(Azz))
    mob_min = np.asarray(tb._coaxial_min_eig(
        m0 + m1 * Aj[0], m1 * Aj[1], m0 + m1 * Aj[2], m0 + m1 * Aj[3]))
    # True yield locus: |tau_d| in closed form; yielded = |tau_d| > tau_y.
    td_true = np.asarray(tb.saramito_tau_d_norm(*Aj, Gp_t))
    true_yielded = td_true > tau_y_t
    if yield_scalar:
        td_model = np.asarray(tb.saramito_tau_d_norm(*Aj, float(Gp_fit)))
        learned_yielded = td_model > float(tau_y_fit)
    else:
        learned_yielded = mob_min > eps_mob     # positive mobility => yielded
    inter = np.logical_and(true_yielded, learned_yielded).sum()
    union = np.logical_or(true_yielded, learned_yielded).sum()
    iou = float(inter / max(union, 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Panel A: (x1, x3) colored by learned mobility min-eig + true locus.
    sc = axes[0].scatter(x1, x3, c=mob_min, s=8, cmap='viridis', vmin=0.0)
    fig.colorbar(sc, ax=axes[0], label='learned min-eig(m0 I + m1 A)')
    near = np.abs(td_true - tau_y_t) < 0.1 * tau_y_t
    axes[0].scatter(x1[near], x3[near], s=18, facecolors='none',
                    edgecolors='red', linewidths=0.8,
                    label='true yield locus |tau_d|=tau_y')
    zero_set = (~learned_yielded) if yield_scalar else (mob_min < eps_mob)
    axes[0].scatter(x1[zero_set], x3[zero_set], s=6, c='k', marker='x',
                    label=f'learned zero-set (mob<{eps_mob})')
    axes[0].set_xlabel('x1 = tau - 3'); axes[0].set_ylabel('x3 = ln det A')
    axes[0].set_title('Learned mobility over visited (x1, x3)\n'
                      'k x = learned yield surface; red o = true locus')
    axes[0].legend(fontsize=7, loc='best'); axes[0].grid(alpha=0.3)
    # Panel B: (x1, x2) natural plane for |tau_d|, colored by true |tau_d|.
    sc2 = axes[1].scatter(x1, x2, c=td_true, s=8, cmap='magma')
    fig.colorbar(sc2, ax=axes[1], label='true |tau_d|')
    axes[1].scatter(x1[near], x2[near], s=18, facecolors='none',
                    edgecolors='cyan', linewidths=0.8, label='|tau_d|=tau_y')
    axes[1].scatter(x1[zero_set], x2[zero_set], s=6, c='lime', marker='x',
                    label='learned zero-set')
    axes[1].set_xlabel('x1 = tau - 3'); axes[1].set_ylabel('x2 = p2 - 3')
    axes[1].set_title(f'True |tau_d| over (x1, x2)  (tau_y={tau_y_t}, IoU={iou:.2f})')
    axes[1].legend(fontsize=7, loc='best'); axes[1].grid(alpha=0.3)
    fig.suptitle('Learned yield surface vs the true Saramito yield locus '
                 '(qualitative)')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(run_dir, 'yield_surface.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    return path, dict(iou=iou, learned_yield_frac=float(learned_yielded.mean()),
                      true_yield_frac=float(true_yielded.mean()),
                      x1=x1, x3=x3, mob_min=mob_min, td_true=td_true)


def _per_forcing_plug_metrics(truth_forcings, theta, fit, Gp_t, tau_y_t,
                              bound_c, kappa_final, cfg, grid, tbnn_model,
                              tbnn_state, tbnn_perm, _gp_of, _lam_of, _nu_of,
                              *, _tau_y_of=None, eps_mob=0.05):
    """Plug half-width per forcing for multi-forcing ablation (Sec.4)."""
    params = {'Gp': _gp_of(fit), 'lam': _lam_of(fit), 'theta': theta,
              'tbnn_bound_c': bound_c, 'tbnn_kappa': float(kappa_final)}
    yield_scalar = _tau_y_of is not None
    if yield_scalar:
        params['tau_y'] = _tau_y_of(fit)
    Gp_fit_v = float(_gp_of(fit))
    tau_y_fit_v = float(_tau_y_of(fit)) if yield_scalar else float('nan')
    nu = float(_nu_of(fit))
    rows = {}
    for td in truth_forcings:
        gx = td['g_x']
        out_t = td['out_truth']
        out_l = p3b._evolve_wall_bounded_with_diagnostics(
            initial_state=tbnn_state, model=tbnn_model, polymer_params=params,
            grid=grid, density=cfg['density'], base_viscosity=nu,
            dt=cfg['dt'], inner_steps=cfg['inner_steps'],
            outer_steps=cfg['outer_steps'], solver_type=cfg['solver_type'],
            use_preconditioner=cfg['use_preconditioner'],
            preconditioner_type=cfg['preconditioner_type'],
            pressure_gradient=(gx, 0.0), permeability=tbnn_perm,
            U_f=cfg['U_f'], solver_tol=cfg['solver_tol'],
            solver_maxiter=cfg['solver_maxiter'])
        Ny, Ly = cfg['Ny'], cfg['Ly']
        dy = Ly / Ny
        Axx = np.asarray(out_l['A_xx_traj'][-1]).mean(axis=0)
        Axy = np.asarray(out_l['A_xy_traj'][-1]).mean(axis=0)
        Ayy = np.asarray(out_l['A_yy_traj'][-1]).mean(axis=0)
        Azz = np.asarray(out_l['A_zz_traj'][-1]).mean(axis=0)
        Aj = (jnp.asarray(Axx), jnp.asarray(Axy), jnp.asarray(Ayy), jnp.asarray(Azz))
        x1, x2, x3 = tb.tbnn_invariant_features(*Aj)
        X = jnp.stack([x1, x2, x3], axis=-1)
        anch = bool(yield_scalar)
        mob_mode = 'softplus' if yield_scalar else 'relu_annealed'
        _, _, _, m0, m1, _ = tb.tbnn_heads(theta, X, bound_c, anchored=anch,
                                           mobility=mob_mode,
                                           kappa=kappa_final)
        mob_min = np.asarray(tb._coaxial_min_eig(
            m0 + m1 * Aj[0], m1 * Aj[1], m0 + m1 * Aj[2], m0 + m1 * Aj[3]))
        td_true = np.asarray(tb.saramito_tau_d_norm(*Aj, Gp_t))
        true_yielded = td_true > tau_y_t
        if yield_scalar:
            # ONE RULER: model's own Hookean-map criterion (Gp_fit, tau_y_fit).
            td_model = np.asarray(tb.saramito_tau_d_norm(*Aj, Gp_fit_v))
            learned_unyielded = td_model <= tau_y_fit_v
        else:
            learned_unyielded = mob_min < eps_mob

        def _plug_hw(uny):
            jc = Ny // 2
            if not uny[jc]:
                return 0.0
            lo = jc
            while lo - 1 >= 0 and uny[lo - 1]:
                lo -= 1
            hi = jc
            while hi + 1 < Ny and uny[hi + 1]:
                hi += 1
            return 0.5 * (hi - lo + 1) * dy

        plug_true = _plug_hw(~true_yielded)
        plug_learned = _plug_hw(learned_unyielded)
        y_p = tau_y_t / gx if gx > tau_y_t else None
        rows[f"{gx:g}"] = dict(
            plug_half_width_learned=float(plug_learned),
            plug_half_width_true=float(plug_true),
            y_p_analytic=float(y_p) if y_p is not None else None,
            plug_rel=(abs(plug_learned - plug_true) / max(plug_true, 1e-30)
                      if plug_true > 0 else None),
            zero_set_fraction=float(learned_unyielded.mean()))
    return rows


def _yield_surface_figure_channel(run_dir, out_truth, theta, Gp_t, tau_y_t,
                                  bound_c, kappa_final, cfg, g_x, eps_mob=0.05,
                                  *, yield_scalar=False, Gp_fit=None,
                                  tau_y_fit=None):
    """Channel result: learned yield surface
    across the channel half-height, overlaid on the true plug boundary
    ``|y| = y_p``. Uses the x-averaged final Saramito truth conformation
    profile. Also measures the RECOVERED plug half-width vs the analytic
    ``y_p = tau_y/g_x`` (the geometry-specific headline).

    ONE RULER: for the V2 scalar-yield model the learned yielded region is the
    model's OWN Saramito criterion ``|tau_d|(A; Gp_fit) > tau_y_fit`` -- the
    SAME closed-form Hookean map used for the truth locus and inside the
    closure prefactor -- NOT the raw mobility min-eig of a mis-specified
    unanchored/relu model (the v2_prod two-ruler bug). Legacy V1/V3 runs keep
    the mobility-min-eig surface."""
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    H = 0.5 * Ly
    y = (np.arange(Ny) + 0.5) * dy
    yc = y - H                                   # from centreline
    Axx = np.asarray(out_truth['A_xx_traj'][-1]).mean(axis=0)
    Axy = np.asarray(out_truth['A_xy_traj'][-1]).mean(axis=0)
    Ayy = np.asarray(out_truth['A_yy_traj'][-1]).mean(axis=0)
    Azz = np.asarray(out_truth['A_zz_traj'][-1]).mean(axis=0)
    u = np.asarray(out_truth['u_traj'][-1]).mean(axis=0)
    Aj = (jnp.asarray(Axx), jnp.asarray(Axy), jnp.asarray(Ayy), jnp.asarray(Azz))
    x1, x2, x3 = tb.tbnn_invariant_features(*Aj)
    X = jnp.stack([x1, x2, x3], axis=-1)
    anch = bool(yield_scalar)
    mob_mode = 'softplus' if yield_scalar else 'relu_annealed'
    _, _, _, m0, m1, _ = tb.tbnn_heads(theta, X, bound_c, anchored=anch,
                                       mobility=mob_mode, kappa=kappa_final)
    mob_min = np.asarray(tb._coaxial_min_eig(
        m0 + m1 * Aj[0], m1 * Aj[1], m0 + m1 * Aj[2], m0 + m1 * Aj[3]))
    td_true = np.asarray(tb.saramito_tau_d_norm(*Aj, Gp_t))
    x1 = np.asarray(x1); x3 = np.asarray(x3)

    true_yielded = td_true > tau_y_t
    if yield_scalar:
        # Learned criterion = the model's own Hookean-ruler yield test.
        td_model = np.asarray(tb.saramito_tau_d_norm(*Aj, float(Gp_fit)))
        learned_yielded = td_model > float(tau_y_fit)
        learned_curve = td_model / max(float(tau_y_fit), 1e-30)
    else:
        td_model = None
        learned_yielded = mob_min > eps_mob
        learned_curve = mob_min
    inter = np.logical_and(true_yielded, learned_yielded).sum()
    union = np.logical_or(true_yielded, learned_yielded).sum()
    iou = float(inter / max(union, 1))

    def _plug_half_width(unyielded):
        jc = Ny // 2
        if not unyielded[jc]:
            return 0.0
        lo = jc
        while lo - 1 >= 0 and unyielded[lo - 1]:
            lo -= 1
        hi = jc
        while hi + 1 < Ny and unyielded[hi + 1]:
            hi += 1
        return 0.5 * (hi - lo + 1) * dy

    y_p_analytic = tau_y_t / g_x
    plug_true = _plug_half_width(~true_yielded)
    plug_learned = _plug_half_width(~learned_yielded)
    plug_rel = (abs(plug_learned - plug_true) / max(plug_true, 1e-30)
                if plug_true > 0 else float('nan'))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Panel A: velocity profile with the plug band shaded.
    axes[0].plot(u, yc, '-', color='C0', label='truth u(y)')
    for yb, c, lab in ((plug_true, 'C3', 'true plug edge |y|=y_p'),
                       (y_p_analytic, 'k', 'analytic y_p=tau_y/g_x')):
        axes[0].axhline(yb, color=c, ls='--', lw=1, label=lab)
        axes[0].axhline(-yb, color=c, ls='--', lw=1)
    axes[0].set_xlabel('u (x-averaged)'); axes[0].set_ylabel('y - centre')
    axes[0].set_title('Channel velocity profile + plug'); axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    # Panel B: learned yield criterion vs y, with true |tau_d|/tau_y overlaid.
    if yield_scalar:
        lbl = 'learned |tau_d|(A;Gp_fit)/tau_y_fit'
        axes[1].axvline(1.0, color='C0', ls=':', lw=0.8)
    else:
        lbl = 'learned min-eig(m0 I+m1 A)'
        axes[1].axvline(eps_mob, color='C0', ls=':', lw=0.8)
    axes[1].plot(learned_curve, yc, '-', color='C0', label=lbl)
    axes[1].plot(td_true / tau_y_t, yc, '--', color='C3',
                 label='true |tau_d|/tau_y')
    axes[1].axvline(1.0, color='C3', ls=':', lw=0.8)
    for yb in (plug_learned, -plug_learned):
        axes[1].axhline(yb, color='C0', ls='--', lw=0.8)
    axes[1].set_xlabel('learned criterion  /  |tau_d|/tau_y')
    axes[1].set_ylabel('y - centre')
    axes[1].set_title(f'Learned yield surface vs true plug\n'
                      f'plug half-width: learned={plug_learned:.3f} '
                      f'true={plug_true:.3f} y_p(analytic)={y_p_analytic:.3f}')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.suptitle('Learned yield surface vs the true Buckingham-Reiner plug')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(run_dir, 'yield_surface.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    return path, dict(iou=iou, learned_yield_frac=float(learned_yielded.mean()),
                      true_yield_frac=float(true_yielded.mean()),
                      x1=x1, x3=x3, mob_min=mob_min, td_true=td_true,
                      plug_half_width_learned=float(plug_learned),
                      plug_half_width_true=float(plug_true),
                      y_p_analytic=float(y_p_analytic), plug_rel=float(plug_rel))


def _save_evp_outputs(run_dir, args, loss_hist, kappa_hist, kappa_schedule,
                      kappa_final, L0, L_final, loss_red, cloud, fc, yf, yfig,
                      gate, gate_pass, Gp_fit, lam_fit, nu_fit, Gp_t, lam_t,
                      nu_t, tau_y_t, intercept_rel, agnostic, ckpt_ok,
                      *, tau_y_fit=None, br_init=None, model_name=None):
    hist = np.array(loss_hist, dtype=np.float64)
    np.savez_compressed(
        os.path.join(run_dir, 'arrays.npz'),
        loss_history=hist, loss_init=np.float64(L0),
        loss_final=np.float64(L_final),
        kappa_hist=np.array(kappa_hist, dtype=np.float64),
        cloud_x1=cloud['cloud']['x1'], cloud_x2=cloud['cloud']['x2'],
        cloud_x3=cloud['cloud']['x3'],
        fig_x1=yfig['x1'], fig_x3=yfig['x3'], fig_mob_min=yfig['mob_min'],
        fig_td_true=yfig['td_true'],
        flow_curve=np.array([[r['gammadot'], r['tau_d'], r['N1'], r['tau_xy']]
                             for r in fc['rows']], dtype=np.float64))
    config_out = dict(
        model_name=(model_name or MODEL_NAME), truth_model=TRUTH_NAME,
        agnostic=agnostic, geometry=args.geometry,
        tau_y_fit=tau_y_fit, br_init=br_init,   # BR-init provenance (alt_mode)
        gate=gate, gate_pass=gate_pass, note='P3-G4 is RECORD-DONT-FAIL',
        loss_init=L0, loss_final=L_final, loss_reduction=loss_red,
        kappa_schedule=kappa_schedule, kappa_final=kappa_final,
        yielded_fraction_truth=yf['yielded_fraction'],
        learned_flow_intercept=fc['intercept'], tau_y_truth=tau_y_t,
        intercept_rel=intercept_rel,
        yield_iou=yfig['iou'], learned_yield_frac=yfig['learned_yield_frac'],
        true_yield_frac=yfig['true_yield_frac'],
        plug_half_width_learned=yfig.get('plug_half_width_learned'),
        plug_half_width_true=yfig.get('plug_half_width_true'),
        y_p_analytic=yfig.get('y_p_analytic'),
        plug_rel=yfig.get('plug_rel'),
        active_fraction=cloud['active_fraction'], checkpoint_reload=bool(ckpt_ok),
        Gp_fit=Gp_fit, Gp_truth=Gp_t, lam_fit=lam_fit, lam_truth=lam_t,
        nu_s_fit=nu_fit, nu_s_truth=nu_t, args=vars(args))
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_out, f, indent=2, default=float)
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write(f"run {args.run_name}  truth=Saramito tau_y={tau_y_t}\n")
        f.write(f"P3-G4 RECORD-DONT-FAIL  gate_pass={gate_pass}\n")
        f.write(f"loss {L0:.3e} -> {L_final:.3e} ({loss_red:.2e}x)  "
                f"kappa_final={kappa_final:g}\n")
        f.write(f"yielded fraction (truth) = {yf['yielded_fraction']:.1%}\n")
        f.write(f"learned flow-curve intercept = {fc['intercept']} vs "
                f"tau_y={tau_y_t} (rel {intercept_rel})\n")
        f.write(f"yield-surface IoU = {yfig['iou']:.2f}  "
                f"(learned yield frac {yfig['learned_yield_frac']:.1%}, "
                f"true {yfig['true_yield_frac']:.1%})\n")
        f.write(f"Gp {Gp_fit:.3f}/{Gp_t} lam {lam_fit:.3f}/{lam_t} "
                f"nu_s {nu_fit:.3f}/{nu_t}\n")
        for k, v in gate.items():
            f.write(f"  {k}: {v}\n")

    # Loss curve with kappa-block boundaries annotated.
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(hist[:, 0], hist[:, 1], '-', color='C0', lw=0.8)
    for gs, kp in kappa_hist:
        ax.axvline(gs, color='C3', ls=':', lw=0.7)
        ax.text(gs, ax.get_ylim()[1], f'k={kp:g}', fontsize=6, rotation=90,
                va='top', ha='right', color='C3')
    ax.set_xlabel('global step'); ax.set_ylabel('velocity-RMSE loss')
    ax.set_title('Tier-3 EVP fit loss (kappa-annealed)')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, 'loss_curve.png'), dpi=150)
    plt.close(fig)
    print(f"[save] outputs -> {run_dir}")


if __name__ == '__main__':
    sys.exit(main())
