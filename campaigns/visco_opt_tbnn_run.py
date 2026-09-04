#!/usr/bin/env python
"""Fit the memory TBNN to a synthetic viscoelastic truth on the constriction.

Fits the TBNN heads ``theta`` (potential + two-term mobility) to a
synthetic family truth on the constriction. Truth = ``giesekus_logconf_bk_v2``
(G3) or ``fene_p_logconf_bk_v2`` (G4) with its true physical scalars.

Scalar protocol:
  * AGNOSTIC (default): the fitted model is run at the declared gauge
    ``Gp = lam = 1`` (Gp is absorbable into the phi-scale, lam into the
    mobility scale -- neither is held at truth), and the genuine solvent
    channel ``nu_s`` is **fitted jointly with theta** (init at a neutral
    1.0, NOT truth). Gate on recovered ``nu_s`` vs truth. The init is then a
    deliberately wrong-scaled Maxwell unit (init loss >> the truth-scaled OB
    loss -- that is the point).
  * ``--unit-test``: hold ``Gp, lam, nu_s`` at truth, fit theta only -- the
    closure-expressivity check (old behavior).

Gates (plan, current revision): loss reduction >= 2 orders (G3); in-cloud
**in-plane** observables (N1, tau_xy, eta_p) within a few %; recovered nu_s
(agnostic); FENE-P steady ``A_zz < 1``; floor inactivity; and the HARD
checkpoint reload self-check. N2-derived quantities are RECORD-only, never
gates (planar-velocity unobservability).

No value bound on the heads; safety comes from the mobility floors.

Kernel-restart note: importing ``tbnn_closure`` registers
``tbnn_potential_logconf_bk_v2`` once; rerun in a fresh process after edits.
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
from jax.flatten_util import ravel_pytree  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import analytic_limits_validation as p3b  # noqa: E402
import visco_families as vf  # noqa: E402
import visco_tbnn as vt  # noqa: E402
from jax_rheology.models import registry as cr  #  noqa: E402,F401
from jax_rheology import log_conformation as lc  #  noqa: E402
from jax_rheology.models import tbnn_memory as tb  #  noqa: E402


# ===========================================================================
# Cross-env checkpoint helpers (raw arrays + metadata, NO pickled treedef).
# ===========================================================================

def theta_to_named_arrays(theta):
    """Flatten the theta pytree to named raw arrays + a layer-count map.

    Layout ``theta = {head: [(W, b), ...]}``; keys are
    ``theta::<head>::<i>::W`` / ``::b``. Plain arrays only, so the diff_rheo
    adapter can rebuild the heads from the ``.npz`` even without importing
    ``jax_rheology`` (plan addendum cross-env handoff)."""
    arrs, nlayers = {}, {}
    for head, layers in theta.items():
        nlayers[head] = len(layers)
        for i, (W, b) in enumerate(layers):
            arrs[f'theta::{head}::{i}::W'] = np.asarray(W, dtype=np.float64)
            arrs[f'theta::{head}::{i}::b'] = np.asarray(b, dtype=np.float64)
    return arrs, nlayers


def theta_from_named_arrays(npz, heads, nlayers):
    theta = {}
    for head in heads:
        layers = []
        for i in range(int(nlayers[head])):
            W = jnp.asarray(npz[f'theta::{head}::{i}::W'], dtype=jnp.float64)
            b = jnp.asarray(npz[f'theta::{head}::{i}::b'], dtype=jnp.float64)
            layers.append((W, b))
        theta[head] = layers
    return theta


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--truth-model', choices=['giesekus', 'fene_p'], default='giesekus')
    p.add_argument('--truth-gp', type=float, default=3.2)
    p.add_argument('--truth-lam', type=float, default=0.7)
    p.add_argument('--truth-nus', type=float, default=0.8)
    p.add_argument('--truth-alpha', type=float, default=0.3, help='Giesekus')
    p.add_argument('--truth-lsq', type=float, default=12.0,
                   help='FENE-P L^2 (plan G4: 10-15, NOT 100).')
    p.add_argument('--unit-test', action='store_true',
                   help='Hold Gp,lam,nu_s at truth; fit theta only '
                        '(closure-expressivity check).')
    p.add_argument('--nus-init', type=float, default=1.0,
                   help='Neutral init for the fitted nu_s (agnostic mode).')
    p.add_argument('--gp-init', type=float, default=1.0,
                   help='Neutral init for the fitted Gp (agnostic mode).')
    p.add_argument('--lam-init', type=float, default=1.0,
                   help='Neutral init for the fitted lam (agnostic mode).')
    # Two-stage hot-start (supersedes the single-stage agnostic fit that stalled,
    # diary Entry 7): the single-stage run pinned nothing wrong but started ALL
    # scalars off (Gp=lam=nu_s=1), so the floored mobility channel was forced to
    # absorb the lam scale-error -> floors load-bearing (active ~100%) and the
    # optimizer walled against the floor. Fix: Stage 1 freezes theta at OB and
    # fits ONLY the 3 dimensional scalars (well-conditioned, no shape d.o.f.),
    # then Stage 2 releases theta from that in-basin start while the scalars
    # stay alive but slow (anchored near the Stage-1 values so the Gp<->phi /
    # lam<->mobility gauge fight does not restart).
    p.add_argument('--stage1-maxiter', type=int, default=120,
                   help='Stage-1 L-BFGS-B max iterations over (Gp, lam, nu_s) '
                        'with theta frozen at OB. This is the SAME smooth 3-D '
                        'subproblem the giesekus/oldroyd direct-fit protocols '
                        'solve with L-BFGS-B (curvature + Wolfe line search) -- '
                        'converges in ~tens of grad evals, vs Adam +-lr '
                        'sign-following taking hundreds. Far better tool here.')
    p.add_argument('--stage1-ftol', type=float, default=1e-12,
                   help='Stage-1 L-BFGS-B relative-loss tolerance. NOTE: the '
                        'OB-best scalar fit is gauge-DEGENERATE (a flat Gp<->lam '
                        'valley with no unique minimum), so a tight 1e-12 just '
                        'makes L-BFGS chase the flat valley floor for many wasted '
                        'evals. Loosen to ~1e-9 to stop once the loss is '
                        'genuinely flat -- nu_s (the only scalar theta cannot '
                        'fake) is locked early, so the gauge-arbitrary Gp/lam '
                        'endpoint need not be polished (Stage 2 re-anchors them).')
    p.add_argument('--scalar-bound-lo', type=float, default=0.02,
                   help='Stage-1 L-BFGS-B lower bound for each scalar (>0).')
    p.add_argument('--scalar-bound-hi', type=float, default=20.0,
                   help='Stage-1 L-BFGS-B upper bound for each scalar.')
    p.add_argument('--scalar-lr', type=float, default=2e-2,
                   help='(Legacy / unused for Stage 1 now that it is L-BFGS-B.)')
    p.add_argument('--scalar-lr2', type=float, default=2e-3,
                   help='Stage-2 peak lr for the scalars (order below Stage 1): '
                        'they are already in-basin and should only nudge while '
                        'theta does the shape work; a small lr keeps lam from '
                        'fighting the mobility heads along the flat gauge dir.')
    p.add_argument('--stage1-steps', type=int, default=150,
                   help='Max Stage-1 (scalars-only) steps; stops early on slope.')
    p.add_argument('--stage1-min-steps', type=int, default=30)
    p.add_argument('--slope-tol', type=float, default=3e-3,
                   help='Converged if the relative loss drop over --slope-window '
                        'steps falls below this (Stage-1 early stop AND the '
                        'gate-the-gates convergence assert).')
    p.add_argument('--slope-window', type=int, default=30)
    p.add_argument('--floor-tol', type=float, default=1e-2,
                   help='Floor active_fraction <= this counts as inactive '
                        '(gate-the-gates: floor-pinned fits are NOT valid).')
    p.add_argument('--azz-tol', type=float, default=0.10,
                   help='FENE-P A_zz two-sided band: worst in-cloud '
                        '|A_zz_learned - A_zz_truth|/A_zz_truth <= this '
                        '(replaces the mis-designed one-sided A_zz<1, which '
                        'passed on collapse).')
    p.add_argument('--g-x', type=float, default=8.0)
    p.add_argument('--outer-steps', type=int, default=300)
    p.add_argument('--inner-steps', type=int, default=None,
                   help='Override inner_steps (FENE-P stiffness rule, Sec. 0.8).')
    p.add_argument('--adam-steps', type=int, default=600)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--bound-c', type=float, default=tb.TBNN_DEFAULT_BOUND_C,
                   help='INERT (no value bound); recorded only.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-shear', type=int, default=12)
    p.add_argument('--obs-rtol', type=float, default=0.05)
    p.add_argument('--time-budget-s', type=float, default=11000.0,
                   help='Stop Adam early if wall time exceeds this.')
    p.add_argument('--out-dir', type=str, default='./work/constriction_memory_fit')
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--resume', type=str, default=None,
                   help='checkpoint .npz to resume theta + (Gp,lam,nu_s) from '
                        '(skips the OB init / Stage-1 OB scalar solve). For a '
                        'resumed chain keep --inner-steps identical to the run '
                        'that produced the checkpoint (the truth trajectory '
                        'depends on it).')
    p.add_argument('--joint-polish', action='store_true',
                   help='after the scheme, run a LATE joint L-BFGS over the full '
                        '[theta, scalars] vector (scalars handled by L-BFGS, not '
                        'free Adam) to unstick the gauge scalars (Gp<->phi, '
                        'lam<->mobility) once theta has matured on the shape.')
    p.add_argument('--joint-maxiter', type=int, default=400)
    p.add_argument('--polish-only', action='store_true',
                   help='skip scheme training; use the resumed fit as-is, then '
                        'run --joint-polish + eval (fast final chain round).')
    p.add_argument('--no-eval', action='store_true',
                   help='train + save checkpoint only; skip the recovery eval / '
                        'cloud / plots (fast intermediate rounds of a resume chain).')
    p.add_argument('--scheme', type=str, default='twostage',
                   choices=['twostage', 's1', 's1b', 's4', 's2', 's3', 's5'],
                   help='Optimization schedule (agnostic mode only):\n'
                        ' twostage = L-BFGS(scalars) then Adam(theta), scalars slow (current).\n'
                        ' s1  = alternating 6x[L-BFGS(scalars,40) + Adam(theta,100)].\n'
                        ' s1b = alternating 12x[L-BFGS(scalars,25) + Adam(theta,50)].\n'
                        ' s4  = theta-heavy alternating 24x[L-BFGS(scalars,20)+Adam(theta,25)].\n'
                        ' s2  = co-step: per step Adam(theta,1)+L-BFGS(scalars,2).\n'
                        ' s3  = inverted hierarchy, single-optimizer Adam: scalars fast'
                        ' (lr 3e-2) lead, theta slow (lr 2e-4) follows.\n'
                        ' s5  = L-BFGS(scalars) then JOINT L-BFGS over [theta,scalars].\n'
                        'All alternating schemes persist the theta Adam state (cosine'
                        ' does NOT reset per block) and re-solve scalars with L-BFGS'
                        ' each block (no curvature dragged across, full re-equilibration).')
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(p)
    return args


def main():
    args = parse_args()
    if args.run_name is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode = 'unit' if args.unit_test else 'agn'
        args.run_name = f'tbnn_{args.truth_model}_{mode}_T{args.outer_steps}_{stamp}'
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[setup] device = {jax.devices()}")
    print(f"[setup] run dir = {run_dir}")

    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    cfg['outer_steps'] = args.outer_steps
    cfg['g_x'] = args.g_x
    if args.inner_steps is not None:
        cfg['inner_steps'] = args.inner_steps
    Gp_t, lam_t, nu_t = args.truth_gp, args.truth_lam, args.truth_nus
    bound_c = float(args.bound_c)

    # Agnostic: ALL dimensional scalars (Gp, lam, nu_s) are FIT via the
    # two-stage hot-start, NOT pinned at a wrong constant. Unit-test: scalars
    # held at truth, fit theta only.
    agnostic = not args.unit_test
    Gp_init = args.gp_init if agnostic else Gp_t
    lam_init = args.lam_init if agnostic else lam_t
    nus_init = args.nus_init if agnostic else nu_t
    T_final = cfg['outer_steps'] * cfg['inner_steps'] * cfg['dt']
    print(f"[setup] truth={args.truth_model} Gp={Gp_t} lam={lam_t} nu_s={nu_t} "
          f"g_x={args.g_x} T={T_final:.4f}={T_final/lam_t:.3f}lam_t "
          f"outer={cfg['outer_steps']} inner={cfg['inner_steps']}")
    if agnostic:
        print(f"[setup] mode=AGNOSTIC two-stage: fit (Gp,lam,nu_s) [init "
              f"{Gp_init},{lam_init},{nus_init}] then release theta")
    else:
        print("[setup] mode=UNIT-TEST (scalars at truth, fit theta only)")

    # --- truth family model + its source/viscometric for the reference ---
    if args.truth_model == 'giesekus':
        truth_name = 'giesekus_logconf_bk_v2'
        truth_pp = {'Gp': Gp_t, 'lam': lam_t, 'alpha': args.truth_alpha}
        truth_R, truth_visc = vf.giesekus_source_R, vf.hookean_viscometric
        truth_ref_pp = {'alpha': args.truth_alpha}
    else:
        truth_name = 'fene_p_logconf_bk_v2'
        truth_pp = {'Gp': Gp_t, 'lam': lam_t, 'Lsq': args.truth_lsq}
        truth_R, truth_visc = vf.fene_p_source_R, vf.fene_p_viscometric
        truth_ref_pp = {'Lsq': args.truth_lsq}

    grid, truth_model, truth_state, truth_perm = vf._build_constriction(
        cfg, truth_name)
    _, tbnn_model, tbnn_state, tbnn_perm = vf._build_constriction(
        cfg, 'tbnn_potential_logconf_bk_v2')

    def _forward(state, model, perm, params, nu):
        return p3b._evolve_wall_bounded_with_diagnostics(
            initial_state=state, model=model, polymer_params=params, grid=grid,
            density=cfg['density'], base_viscosity=nu, dt=cfg['dt'],
            inner_steps=cfg['inner_steps'], outer_steps=cfg['outer_steps'],
            solver_type=cfg['solver_type'],
            use_preconditioner=cfg['use_preconditioner'],
            preconditioner_type=cfg['preconditioner_type'],
            pressure_gradient=(args.g_x, 0.0), permeability=perm, U_f=cfg['U_f'],
            solver_tol=cfg['solver_tol'], solver_maxiter=cfg['solver_maxiter'])

    print("[truth] generating truth trajectory ...")
    t0 = time.time()
    truth_pp_j = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in truth_pp.items()}
    out_truth = jax.jit(lambda: _forward(truth_state, truth_model, truth_perm,
                                          truth_pp_j, nu_t))()
    u_truth = out_truth['u_traj']
    v_truth = out_truth['v_traj']
    u_truth.block_until_ready()
    trA_t = (np.asarray(out_truth['A_xx_traj']) + np.asarray(out_truth['A_yy_traj'])
             + np.asarray(out_truth['A_zz_traj']))
    print(f"[truth] forward warm = {time.time()-t0:.1f}s  "
          f"max|u|={float(jnp.max(jnp.abs(u_truth))):.3f}  trA max={trA_t.max():.3f}")

    # --- fitted model loss over the pytree `fit` (theta [+ Gp,lam,nu_s]) ---
    # All scalars are LINEAR with a soft positivity floor (inactive at the
    # operating point). In unit-test mode they collapse to truth constants.
    def _gp_of(fit):
        return jnp.maximum(fit['Gp'], 1e-4) if agnostic else jnp.asarray(Gp_t, dtype=jnp.float64)

    def _lam_of(fit):
        return jnp.maximum(fit['lam'], 1e-4) if agnostic else jnp.asarray(lam_t, dtype=jnp.float64)

    def _nu_of(fit):
        return jnp.maximum(fit['nu_s'], 1e-4) if agnostic else jnp.asarray(nu_t, dtype=jnp.float64)

    def loss_fn(fit):
        params = {'Gp': _gp_of(fit), 'lam': _lam_of(fit),
                  'theta': fit['theta'], 'tbnn_bound_c': bound_c}
        out = _forward(tbnn_state, tbnn_model, tbnn_perm, params, _nu_of(fit))
        return (jnp.sum((out['u_traj'] - u_truth) ** 2)
                + jnp.sum((out['v_traj'] - v_truth) ** 2))

    theta0, _ = tb.init_tbnn_theta(jax.random.PRNGKey(args.seed),
                                   width=args.width, depth=args.depth,
                                   bound_c=bound_c)
    fit = {'theta': theta0}
    if agnostic:
        fit['Gp'] = jnp.asarray(Gp_init, dtype=jnp.float64)
        fit['lam'] = jnp.asarray(lam_init, dtype=jnp.float64)
        fit['nu_s'] = jnp.asarray(nus_init, dtype=jnp.float64)

    resumed = False
    if args.resume:
        z = np.load(args.resume, allow_pickle=False)
        heads_r = [str(h) for h in z['ckpt_heads']]
        nlayers_r = {h: int(n) for h, n in zip(heads_r, z['ckpt_nlayers'])}
        fit['theta'] = theta_from_named_arrays(z, heads_r, nlayers_r)
        if agnostic:
            fit['Gp'] = jnp.asarray(float(z['ckpt_Gp_fit']), dtype=jnp.float64)
            fit['lam'] = jnp.asarray(float(z['ckpt_lam_fit']), dtype=jnp.float64)
            fit['nu_s'] = jnp.asarray(float(z['ckpt_nu_s']), dtype=jnp.float64)
        resumed = True
        print(f"[resume] loaded {args.resume}: Gp={float(z['ckpt_Gp_fit']):.4f} "
              f"lam={float(z['ckpt_lam_fit']):.4f} nu_s={float(z['ckpt_nu_s']):.4f} "
              f"(prev loss {float(z['ckpt_loss']):.4e})")

    vg = jax.jit(jax.value_and_grad(loss_fn))
    print("[opt] warm-compiling value_and_grad ...")
    t0 = time.time()
    L0, _g0 = vg(fit)
    L0 = float(L0)
    print(f"[opt] vag warm = {time.time()-t0:.1f}s  loss(init)={L0:.6e}  "
          f"(init Gp={float(_gp_of(fit)):.3f} lam={float(_lam_of(fit)):.3f} "
          f"nu_s={float(_nu_of(fit)):.3f}; wrong-scaled in agnostic mode)")

    def _scalars_str(fit):
        if not agnostic:
            return "scalars@truth"
        return (f"Gp={float(_gp_of(fit)):.4f} lam={float(_lam_of(fit)):.4f} "
                f"nu_s={float(_nu_of(fit)):.4f}")

    # Label fn for multi_transform: theta -> 'theta', the 3 scalars -> 'scalars'
    # (each scalar still gets its own Adam m/v; the group just shares lr/schedule).
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

    def _scalar_opt(peak, n_steps, lo_frac):
        sched = optax.warmup_cosine_decay_schedule(
            init_value=peak * lo_frac, peak_value=peak,
            warmup_steps=max(1, args.warmup // 2),
            decay_steps=max(args.warmup + 1, n_steps), end_value=peak * 0.02)
        return optax.adam(sched)  # no clip: clip is a theta-only concern

    progress_path = os.path.join(run_dir, 'progress.csv')
    with open(progress_path, 'w') as pf:
        pf.write("stage,step,loss,Gp,lam,nu_s\n")
        pf.write("0,0,%.6e,%.6f,%.6f,%.6f\n" % (
            L0, float(_gp_of(fit)), float(_lam_of(fit)), float(_nu_of(fit))))

    loss_hist = [(0, L0, float(_nu_of(fit)))]
    t_opt = time.time()

    def _rel_slope(window):
        if len(loss_hist) <= window:
            return float('nan')
        L_old = loss_hist[-1 - window][1]
        L_new = loss_hist[-1][1]
        return (L_old - L_new) / max(abs(L_old), 1e-300)

    def run_phase(fit, opt, n_steps, *, stage, slope_stop, track_best, step0):
        """Run one optimizer phase. Returns (best_loss, best_fit, last_gstep).
        `step` evaluates the loss at the PRE-update params and returns the
        POST-update params, so we pair Lf with prev_fit (diary off-by-one fix)."""
        opt_state = opt.init(fit)

        @jax.jit
        def step(fit, opt_state):
            L, g = vg(fit)
            updates, opt_state = opt.update(g, opt_state, fit)
            return optax.apply_updates(fit, updates), opt_state, L

        best = (float('inf'), fit)
        last_g = step0
        for it in range(1, n_steps + 1):
            prev_fit = fit
            fit, opt_state, L = step(fit, opt_state)
            Lf = float(L)
            gstep = step0 + it
            last_g = gstep
            loss_hist.append((gstep, Lf, float(_nu_of(prev_fit))))
            if track_best and Lf < best[0]:
                best = (Lf, prev_fit)
            if it % 5 == 0 or it == 1:
                with open(progress_path, 'a') as pf:
                    pf.write(f"{stage},{gstep},{Lf:.6e},{float(_gp_of(prev_fit)):.6f},"
                             f"{float(_lam_of(prev_fit)):.6f},{float(_nu_of(prev_fit)):.6f}\n")
                print(f"  [{stage}] step {gstep:>4}  loss={Lf:.6e}  "
                      f"best={best[0]:.4e}  {_scalars_str(prev_fit)}  "
                      f"[{time.time()-t_opt:.0f}s]", flush=True)
            if (slope_stop and it >= max(args.stage1_min_steps, args.slope_window)):
                rel = _rel_slope(args.slope_window)
                if np.isfinite(rel) and rel < args.slope_tol:
                    print(f"  [{stage}] converged: rel drop over "
                          f"{args.slope_window} steps = {rel:.2e} < {args.slope_tol}")
                    break
            if time.time() - t_opt > args.time_budget_s:
                print(f"  [{stage}] time budget hit at step {gstep}; stopping.")
                break
        if not track_best:  # phase end-state is the result (Stage 1)
            best = (loss_hist[-1][1], fit)
        return best[0], best[1], last_g

    # ---- reusable L-BFGS-B scalar solve at the CURRENT theta (jit shared over
    # (svec, theta) so blocks do NOT recompile; theta passed as a traced arg).
    def _loss_vec_th(svec, th):
        return loss_fn({'theta': th, 'Gp': svec[0], 'lam': svec[1], 'nu_s': svec[2]})
    vag_scalars = jax.jit(jax.value_and_grad(_loss_vec_th, argnums=0))

    def _lbfgs_scalars(cur_fit, maxiter, ftol, label):
        th = cur_fit['theta']
        x0 = np.array([float(_gp_of(cur_fit)), float(_lam_of(cur_fit)),
                       float(_nu_of(cur_fit))], dtype=np.float64)

        def _obj(x):
            val, g = vag_scalars(jnp.asarray(x, dtype=jnp.float64), th)
            fv, fg = float(val), np.asarray(g, dtype=np.float64)
            loss_hist.append((len(loss_hist), fv, float(x[2])))
            with open(progress_path, 'a') as pf:
                pf.write(f"{label},{len(loss_hist)-1},{fv:.6e},"
                         f"{x[0]:.6f},{x[1]:.6f},{x[2]:.6f}\n")
            return fv, fg

        res = minimize(_obj, x0, jac=True, method='L-BFGS-B',
                       bounds=[(args.scalar_bound_lo, args.scalar_bound_hi)] * 3,
                       options=dict(maxiter=maxiter, ftol=ftol, gtol=1e-10))
        nf = {'theta': th,
              'Gp': jnp.asarray(res.x[0], dtype=jnp.float64),
              'lam': jnp.asarray(res.x[1], dtype=jnp.float64),
              'nu_s': jnp.asarray(res.x[2], dtype=jnp.float64)}
        return nf, float(res.fun), int(res.nfev), int(res.nit)

    # ---- alternating driver (schemes s1/s1b/s4/s2): persistent theta Adam
    # state (cosine spans the WHOLE run, never reset per block) + a fresh L-BFGS
    # scalar re-solve each block (full re-equilibration at the current theta).
    def _alternating(n_theta_block, n_scalar_iters, n_blocks, scalar_ftol):
        nonlocal_fit = fit
        total_theta = n_theta_block * n_blocks
        topt = optax.multi_transform(
            {'theta': _theta_opt(total_theta), 'scalars': optax.set_to_zero()}, _label)
        tstate = topt.init(nonlocal_fit)

        @jax.jit
        def tstep(f, st):
            L, g = vg(f)
            upd, st = topt.update(g, st, f)
            return optax.apply_updates(f, upd), st, L

        cur = nonlocal_fit
        best = (float('inf'), cur)
        gstep = 0
        for b in range(n_blocks):
            cur, lsc, nf, ni = _lbfgs_scalars(
                cur, n_scalar_iters, scalar_ftol, f'sc{b}')
            if lsc < best[0]:
                best = (lsc, cur)
            print(f"  [{args.scheme} blk {b}] L-BFGS scalars nfev={nf} nit={ni} "
                  f"loss={lsc:.6e}  {_scalars_str(cur)}  [{time.time()-t_opt:.0f}s]",
                  flush=True)
            for _i in range(n_theta_block):
                prev = cur
                cur, tstate, L = tstep(cur, tstate)
                Lf = float(L)
                gstep += 1
                loss_hist.append((len(loss_hist), Lf, float(_nu_of(prev))))
                if Lf < best[0]:
                    best = (Lf, prev)
                if gstep % 25 == 0:
                    with open(progress_path, 'a') as pf:
                        pf.write(f"th{b},{len(loss_hist)-1},{Lf:.6e},"
                                 f"{float(_gp_of(prev)):.6f},{float(_lam_of(prev)):.6f},"
                                 f"{float(_nu_of(prev)):.6f}\n")
                    print(f"  [{args.scheme} blk {b}] theta step {gstep}/{total_theta} "
                          f"loss={Lf:.6e} best={best[0]:.4e} {_scalars_str(prev)} "
                          f"[{time.time()-t_opt:.0f}s]", flush=True)
            if time.time() - t_opt > args.time_budget_s:
                print(f"  [{args.scheme}] time budget hit at block {b}; stopping.")
                break
        return best

    def _joint_lbfgs(fit, maxiter, tag):
        """Joint L-BFGS over the full raveled [theta, scalars] vector. Used by
        s5 and the late --joint-polish phase: the scalars are handled by L-BFGS
        (not free Adam), so they can migrate off the OB-best along the gauge
        direction without the Adam sloshing that wrecked the s3 run."""
        x0, unravel = ravel_pytree(fit)
        x0 = np.asarray(x0, dtype=np.float64)
        jc = [0]

        def _obj(x):
            fitc = unravel(jnp.asarray(x, dtype=jnp.float64))
            val, g = vg(fitc)
            gflat, _ = ravel_pytree(g)
            jc[0] += 1
            fv = float(val)
            loss_hist.append((len(loss_hist), fv, float(_nu_of(fitc))))
            if jc[0] % 5 == 0 or jc[0] == 1:
                with open(progress_path, 'a') as pf:
                    pf.write(f"{tag},{len(loss_hist)-1},{fv:.6e},"
                             f"{float(_gp_of(fitc)):.6f},{float(_lam_of(fitc)):.6f},"
                             f"{float(_nu_of(fitc)):.6f}\n")
                print(f"  [{tag}] eval {jc[0]:>3} loss={fv:.6e} {_scalars_str(fitc)} "
                      f"[{time.time()-t_opt:.0f}s]", flush=True)
            return fv, np.asarray(gflat, dtype=np.float64)

        res = minimize(_obj, x0, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=maxiter, ftol=1e-11, gtol=1e-9))
        return unravel(jnp.asarray(res.x, dtype=jnp.float64)), float(res.fun), res

    stage1_end = 0
    if args.polish_only:
        if not resumed:
            raise ValueError("--polish-only requires --resume")
        L_final = L0
        print(f"[polish-only] skipping scheme; resumed fit loss {L0:.4e}; {_scalars_str(fit)}")
    elif not agnostic:
        # Unit-test: single theta-only phase (scalars fixed at truth).
        print(f"[opt] UNIT-TEST single-stage theta fit; {args.adam_steps} steps")
        L_final, fit, _ = run_phase(
            fit, _theta_opt(args.adam_steps), args.adam_steps, stage='theta',
            slope_stop=False, track_best=True, step0=0)
    elif args.scheme == 'twostage':
        # L-BFGS(scalars, theta frozen at OB) -> Adam(theta) + slow scalars.
        print(f"[stage1] scalars-only L-BFGS-B (theta frozen at OB); maxiter={args.stage1_maxiter}")
        fit, _Ls1, nf, ni = _lbfgs_scalars(fit, args.stage1_maxiter, args.stage1_ftol, 'stage1')
        stage1_end = len(loss_hist) - 1
        print(f"[stage1] L-BFGS-B done nfev={nf} nit={ni} loss={_Ls1:.6e} "
              f"{_scalars_str(fit)} (truth Gp={Gp_t} lam={lam_t} nu_s={nu_t})")
        s1_ok, s1_path = _save_and_verify_checkpoint(
            run_dir, fit['theta'], fit, args, float(_gp_of(fit)), float(_lam_of(fit)),
            float(_nu_of(fit)), _Ls1, loss_fn, agnostic, tag='stage1')
        print(f"[stage1] checkpoint reload self-check: {'PASS' if s1_ok else 'FAIL'} -> {s1_path}")
        print(f"[stage2] release theta; scalars slow (peak {args.scalar_lr2}); {args.adam_steps} steps")
        opt2 = optax.multi_transform(
            {'theta': _theta_opt(args.adam_steps),
             'scalars': _scalar_opt(args.scalar_lr2, args.adam_steps, 0.5)}, _label)
        L_final, fit, _ = run_phase(
            fit, opt2, args.adam_steps, stage='stage2', slope_stop=False,
            track_best=True, step0=stage1_end)
    elif args.scheme in ('s1', 's1b', 's4', 's2'):
        recipe = {'s1': (100, 40, 6), 's1b': (50, 25, 12),
                  's4': (25, 20, 24), 's2': (1, 2, args.adam_steps)}[args.scheme]
        n_theta_block, n_scalar_iters, n_blocks = recipe
        print(f"[{args.scheme}] alternating: {n_blocks}x[L-BFGS(scalars,{n_scalar_iters}) "
              f"+ Adam(theta,{n_theta_block})]  (theta cosine spans {n_theta_block*n_blocks})")
        L_final, fit = _alternating(n_theta_block, n_scalar_iters, n_blocks, args.stage1_ftol)
    elif args.scheme == 's3':
        # Inverted hierarchy, single-optimizer Adam: scalars fast lead, theta slow follow.
        print("[s3] Stage A: scalars-only Adam (lr 3e-2), theta frozen, 80 steps")
        optA = optax.multi_transform(
            {'theta': optax.set_to_zero(), 'scalars': _scalar_opt(3e-2, 80, 0.3)}, _label)
        _LA, fit, _ = run_phase(fit, optA, 80, stage='s3A', slope_stop=False,
                                track_best=True, step0=0)
        stage1_end = len(loss_hist) - 1
        n_b = max(1, args.adam_steps - 80)
        print(f"[s3] Stage B: theta slow (lr 2e-4, warmup 40) + scalars fast (lr 1e-2); {n_b} steps")
        optB = optax.multi_transform(
            {'theta': _theta_opt_p(2e-4, 40, n_b),
             'scalars': _scalar_opt(1e-2, n_b, 0.5)}, _label)
        L_final, fit, _ = run_phase(fit, optB, n_b, stage='s3B', slope_stop=False,
                                    track_best=True, step0=stage1_end)
    elif args.scheme == 's5':
        # L-BFGS(scalars) then JOINT L-BFGS over the full [theta, scalars] vector.
        print(f"[s5] Stage 1: L-BFGS(scalars), maxiter={args.stage1_maxiter}")
        fit, _Ls1, nf, ni = _lbfgs_scalars(fit, args.stage1_maxiter, args.stage1_ftol, 'stage1')
        stage1_end = len(loss_hist) - 1
        print(f"[s5] stage1 nfev={nf} nit={ni} loss={_Ls1:.6e} {_scalars_str(fit)}")
        print("[s5] Stage 2: JOINT L-BFGS over [theta,scalars], maxiter=300")
        fit, L_final, resj = _joint_lbfgs(fit, 300, 's5joint')
        print(f"[s5] joint done: {resj.message}  nfev={resj.nfev} nit={resj.nit} loss={L_final:.6e}")
    else:
        raise ValueError(f"unknown scheme {args.scheme}")

    # --- LATE joint polish (optional): unstick the gauge scalars once theta has
    # matured on the shape (scalars via L-BFGS, never free Adam). ---
    if agnostic and args.joint_polish:
        print(f"[joint-polish] late joint L-BFGS over [theta,scalars], "
              f"maxiter={args.joint_maxiter}  (pre: {_scalars_str(fit)})")
        fit, L_jp, rjp = _joint_lbfgs(fit, args.joint_maxiter, 'jointpolish')
        L_final = min(L_final, L_jp)
        print(f"[joint-polish] done: {rjp.message} nfev={rjp.nfev} nit={rjp.nit} "
              f"loss={L_jp:.6e}  (post: {_scalars_str(fit)}; truth Gp={Gp_t} lam={lam_t})")

    theta = fit['theta']
    Gp_fit, lam_fit, nu_fit = (float(_gp_of(fit)), float(_lam_of(fit)),
                               float(_nu_of(fit)))
    print(f"[opt] done {time.time()-t_opt:.0f}s  loss {L0:.3e} -> {L_final:.3e}"
          f"  ({L0/max(L_final,1e-300):.2e}x)  recovered {_scalars_str(fit)} "
          f"(truth Gp={Gp_t} lam={lam_t} nu_s={nu_t})")

    # --- HARD checkpoint + reload self-check (plan Task 0b / addendum Prereq 0) ---
    ckpt_ok, ckpt_path = _save_and_verify_checkpoint(
        run_dir, theta, fit, args, Gp_fit, lam_fit, nu_fit, L_final, loss_fn,
        agnostic)
    print(f"[ckpt] reload self-check: {'PASS' if ckpt_ok else 'FAIL'}  -> {ckpt_path}")

    if args.no_eval:
        print("[no-eval] checkpoint saved; skipping recovery/cloud/plots "
              "(intermediate resume-chain round). [done]")
        return 0

    # --- invariant cloud at the fitted model (recovered scalars) ---
    cloud = vt.tbnn_invariant_cloud(cfg, theta, Gp=Gp_fit, lam=lam_fit,
                                    nu_s=nu_fit, bound_c=bound_c)
    print(f"[cloud] active_fraction={cloud['active_fraction']:.2e}  "
          f"x1(tau-3) q1={cloud['x1']['q1']:.3f} q99={cloud['x1']['q99']:.3f}  "
          f"x3(ldet) q1={cloud['x3']['q1']:.3f} q99={cloud['x3']['q99']:.3f}")

    # --- recovery eval: learned 0D in-plane observables vs truth root ---
    params_fit = {'Gp': Gp_fit, 'lam': lam_fit, 'theta': theta,
                  'tbnn_bound_c': bound_c}
    rec = vt.tbnn_recovery_eval(
        theta, params_fit, truth_R, truth_visc, truth_ref_pp,
        lam_truth=lam_t, lam_fit=lam_fit, Gp_truth=Gp_t, Gp_fit=Gp_fit,
        bound_c=bound_c, cloud=cloud, n_shear=args.n_shear,
        obs_rtol=args.obs_rtol)
    print(f"[recovery] in-cloud points: {rec['n_in_cloud']}/{args.n_shear} "
          f"(need >= {rec['min_in_cloud']}); gd_edge={rec['gd_edge']:.3f}")
    print(f"[recovery] worst in-cloud rel err: N1={rec['worst']['N1']:.2%}  "
          f"tau_xy={rec['worst']['tau']:.2%}  eta_p={rec['worst']['eta']:.2%}")

    # --- GATE THE GATES (run-order): a fit that is still descending or whose
    # floors are active is NOT a valid recovery verdict, just "not converged /
    # floor-pinned". Assert convergence (loss slope) AND floor inactivity first.
    loss_red = L0 / max(L_final, 1e-300)
    final_slope = _rel_slope(args.slope_window)
    converged = bool(np.isfinite(final_slope) and abs(final_slope) < args.slope_tol)
    floor_inactive = bool(cloud['active_fraction'] <= args.floor_tol)
    valid = converged and floor_inactive and bool(ckpt_ok)
    print(f"[gate-the-gates] final slope({args.slope_window})={final_slope:.2e} "
          f"converged={converged}  active_fraction={cloud['active_fraction']:.2e} "
          f"floor_inactive={floor_inactive}")

    incloud = [r for r in rec['rows'] if r['in_cloud']]
    obs_ok = (rec['enough_in_cloud'] and np.isfinite(rec['worst']['N1'])
              and rec['worst']['N1'] <= args.obs_rtol
              and rec['worst']['tau'] <= args.obs_rtol
              and rec['worst']['eta'] <= args.obs_rtol)
    nus_relerr = abs(nu_fit - nu_t) / nu_t
    gp_relerr = abs(Gp_fit - Gp_t) / Gp_t
    lam_relerr = abs(lam_fit - lam_t) / lam_t

    gate = {
        'converged': converged,
        'floor_inactive': floor_inactive,
        'checkpoint_reload': bool(ckpt_ok),
        'observables_in_plane': bool(obs_ok),
    }
    if agnostic:
        gate['nu_s_recovered'] = bool(nus_relerr <= 0.05)
    if args.truth_model == 'giesekus':
        gate['loss_2orders'] = bool(loss_red >= 100.0)
    else:
        # Two-sided A_zz band around the truth steady curve (replaces the
        # mis-designed one-sided A_zz<1, which passed on collapse). In-cloud only.
        azz_errs = [abs(r['l_Azz'] - r['t_Azz']) / max(abs(r['t_Azz']), 1e-12)
                    for r in incloud]
        worst_azz = max(azz_errs) if azz_errs else float('nan')
        gate['Azz_band'] = bool(np.isfinite(worst_azz) and worst_azz <= args.azz_tol)
        print(f"[recovery] FENE-P worst in-cloud |A_zz-truth|/truth = {worst_azz:.2%} "
              f"(band {args.azz_tol:.0%})")

    # RECORD-only (never gated): N2, -N2/N1.
    r_lo = min(incloud, key=lambda r: r['Wi']) if incloud else None
    n2n1 = (-(r_lo['l_N2']) / r_lo['l_N1']) if r_lo and abs(r_lo['l_N1']) > 1e-12 else float('nan')
    print(f"[record] low-shear -N2/N1(learned)={n2n1:.4f}  "
          f"(Giesekus alpha/2={args.truth_alpha/2:.3f}; unobservable in planar data)")
    print(f"[record] recovered scalars vs truth: Gp {Gp_fit:.4f}/{Gp_t} "
          f"({gp_relerr:.2%}; gauge), lam {lam_fit:.4f}/{lam_t} "
          f"({lam_relerr:.2%}; gauge), nu_s {nu_fit:.4f}/{nu_t} ({nus_relerr:.2%})")

    gname = 'G3 (Giesekus)' if args.truth_model == 'giesekus' else 'G4 (FENE-P)'
    print(f"\n==== {gname} recovery verdict ({'agnostic' if agnostic else 'unit-test'}) ====")
    for k, v in gate.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  [record] loss reduction = {loss_red:.2e}x; nu_s relerr = {nus_relerr:.2%}")
    if not valid:
        gate_pass = False
        print(f"  OVERALL {gname}: INVALID -- not converged / floor-pinned / ckpt "
              f"fail (the observable gates above are diagnostic, NOT a recovery "
              f"pass/fail; extend steps or investigate the gauge fight).")
    else:
        gate_pass = all(gate.values())
        print(f"  OVERALL {gname}: {'PASS' if gate_pass else 'FAIL'}")

    _save_outputs(run_dir, args, theta, loss_hist, L0, L_final, rec, cloud,
                  gate, gate_pass, valid, truth_ref_pp, Gp_fit, lam_fit, nu_fit,
                  Gp_t, lam_t, nu_t, gp_relerr, lam_relerr, nus_relerr, n2n1,
                  loss_red, stage1_end, final_slope, bound_c, agnostic)
    print("[done]")
    return 0 if gate_pass else 2


def _save_and_verify_checkpoint(run_dir, theta, fit, args, Gp_fit, lam_fit,
                                nu_fit, L_final, loss_fn, agnostic, tag='final'):
    """Save theta + fitted scalars as raw arrays + metadata, then reload from
    the ``.npz`` (NOT an in-memory treedef) and assert ``loss`` reproduces the
    logged value to float64. HARD gate (plan Task 0b / addendum Prereq 0).
    ``tag`` selects the filename (``stage1`` vs ``final``)."""
    arrs, nlayers = theta_to_named_arrays(theta)
    heads = list(theta.keys())
    sizes = (3,) + (int(args.width),) * int(args.depth) + (1,)
    meta = dict(
        ckpt_heads=np.array(heads),
        ckpt_nlayers=np.array([nlayers[h] for h in heads], dtype=np.int64),
        ckpt_sizes=np.array(sizes, dtype=np.int64),
        ckpt_width=np.int64(args.width), ckpt_depth=np.int64(args.depth),
        ckpt_bound_c=np.float64(args.bound_c),
        ckpt_Gp_fit=np.float64(Gp_fit), ckpt_lam_fit=np.float64(lam_fit),
        ckpt_nu_s=np.float64(nu_fit), ckpt_log_nu_s=np.float64(np.log(max(nu_fit, 1e-12))),
        ckpt_loss=np.float64(L_final), ckpt_agnostic=np.bool_(agnostic),
        ckpt_truth_model=np.array(args.truth_model))
    fname = 'theta_checkpoint.npz' if tag == 'final' else f'theta_checkpoint_{tag}.npz'
    ckpt_path = os.path.join(run_dir, fname)
    np.savez_compressed(ckpt_path, **arrs, **meta)

    # Reload purely from the file (cross-env path), rebuild fit, recompute loss.
    z = np.load(ckpt_path, allow_pickle=False)
    heads_r = [str(h) for h in z['ckpt_heads']]
    nlayers_r = {h: int(n) for h, n in zip(heads_r, z['ckpt_nlayers'])}
    theta_r = theta_from_named_arrays(z, heads_r, nlayers_r)
    fit_r = {'theta': theta_r}
    if agnostic:
        fit_r['Gp'] = jnp.asarray(float(z['ckpt_Gp_fit']), dtype=jnp.float64)
        fit_r['lam'] = jnp.asarray(float(z['ckpt_lam_fit']), dtype=jnp.float64)
        fit_r['nu_s'] = jnp.asarray(float(z['ckpt_nu_s']), dtype=jnp.float64)
    L_reload = float(loss_fn(fit_r))
    rel = abs(L_reload - L_final) / max(abs(L_final), 1e-300)
    print(f"[ckpt:{tag}] logged={L_final:.10e}  reloaded={L_reload:.10e}  rel={rel:.2e}")
    return (rel <= 1e-12), ckpt_path


def _save_outputs(run_dir, args, theta, loss_hist, L0, L_final, rec, cloud,
                  gate, gate_pass, valid, truth_ref_pp, Gp_fit, lam_fit, nu_fit,
                  Gp_t, lam_t, nu_t, gp_relerr, lam_relerr, nus_relerr, n2n1,
                  loss_red, stage1_end, final_slope, bound_c, agnostic):
    hist = np.array(loss_hist, dtype=np.float64)
    np.savez_compressed(
        os.path.join(run_dir, 'arrays.npz'),
        loss_history=hist, loss_init=np.float64(L0),
        loss_final=np.float64(L_final), stage1_end=np.int64(stage1_end),
        rows=np.array([[r['Wi'], r['gammadot'], r['in_cloud'],
                        r['t_N1'], r['l_N1'], r['t_N2'], r['l_N2'],
                        r['t_tau'], r['l_tau'], r['t_eta'], r['l_eta'],
                        r['t_Azz'], r['l_Azz']] for r in rec['rows']],
                      dtype=np.float64),
        cloud_x1=cloud['cloud']['x1'], cloud_x2=cloud['cloud']['x2'],
        cloud_x3=cloud['cloud']['x3'])

    config_out = dict(
        truth_model=args.truth_model, agnostic=agnostic, gate=gate,
        gate_pass=gate_pass, valid=valid, loss_init=L0, loss_final=L_final,
        loss_reduction=loss_red, final_slope=final_slope, stage1_end=stage1_end,
        worst_obs_rel=rec['worst'],
        n_in_cloud=rec['n_in_cloud'], min_in_cloud=rec['min_in_cloud'],
        enough_in_cloud=rec['enough_in_cloud'], gd_edge=rec['gd_edge'],
        bands=rec['bands'], active_fraction=cloud['active_fraction'],
        Gp_fit=Gp_fit, Gp_truth=Gp_t, Gp_relerr=gp_relerr,
        lam_fit=lam_fit, lam_truth=lam_t, lam_relerr=lam_relerr,
        nu_s_fit=nu_fit, nu_s_truth=nu_t, nu_s_relerr=nus_relerr,
        low_shear_N2N1=n2n1,
        schedule=dict(theta_lr=args.lr, warmup=args.warmup, clip=args.clip,
                      scalar_lr_stage1=args.scalar_lr, scalar_lr_stage2=args.scalar_lr2,
                      stage1_steps=args.stage1_steps, adam_steps=args.adam_steps,
                      slope_tol=args.slope_tol, slope_window=args.slope_window,
                      floor_tol=args.floor_tol, azz_tol=args.azz_tol),
        args=vars(args))
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_out, f, indent=2, default=float)
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write(f"run {args.run_name}  truth={args.truth_model}  "
                f"mode={'agnostic' if agnostic else 'unit-test'}\n")
        f.write(f"VALID={valid} (converged + floor-inactive + ckpt)  "
                f"gate_pass={gate_pass}\n")
        f.write(f"loss {L0:.3e} -> {L_final:.3e} ({loss_red:.2e}x)  "
                f"final_slope={final_slope:.2e}\n")
        f.write(f"stage1_end_step={stage1_end}\n")
        f.write(f"Gp:   fit={Gp_fit:.4f} truth={Gp_t:.4f} relerr={gp_relerr:.2%} (gauge)\n")
        f.write(f"lam:  fit={lam_fit:.4f} truth={lam_t:.4f} relerr={lam_relerr:.2%} (gauge)\n")
        f.write(f"nu_s: fit={nu_fit:.4f} truth={nu_t:.4f} relerr={nus_relerr:.2%}\n")
        f.write(f"in-cloud {rec['n_in_cloud']}/{args.n_shear} "
                f"(need >= {rec['min_in_cloud']})\n")
        f.write(f"worst in-plane obs rel: {rec['worst']}\n")
        f.write(f"active_fraction={cloud['active_fraction']:.2e}\n")
        f.write(f"[record] low-shear -N2/N1(learned)={n2n1:.4f}\n")
        for k, v in gate.items():
            f.write(f"  {k}: {v}\n")

    # loss curve with the Stage-1 -> Stage-2 boundary annotated
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(hist[:, 0], hist[:, 1], '-', color='C0')
    if stage1_end > 0:
        ax.axvline(stage1_end, color='C3', ls='--', lw=1.0,
                   label=f'stage1->stage2 (step {stage1_end})')
        ax.legend()
    ax.set_xlabel('Adam step'); ax.set_ylabel('velocity-RMSE loss')
    ax.set_title(f'TBNN fit loss ({args.truth_model}, '
                 f"{'agnostic two-stage' if agnostic else 'unit-test'})")
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, 'loss_curve.png'), dpi=150)
    plt.close(fig)

    # flow curves: learned vs truth (N1, tau, eta gated; N2 RECORD-only)
    rows = rec['rows']
    Wi = [r['Wi'] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [('N1', 'N1 (gated)'), ('tau', 'tau_xy (gated)'),
              ('eta', 'eta_p (gated)'), ('N2', 'N2 (RECORD only)')]
    for ax, (key, ttl) in zip(axes.ravel(), panels):
        ax.plot(Wi, [r[f't_{key}'] for r in rows], 'o-', color='C3', label='truth')
        ax.plot(Wi, [r[f'l_{key}'] for r in rows], 's--', color='C0', label='learned')
        for r in rows:
            if not r['in_cloud']:
                ax.axvspan(r['Wi'] * 0.97, r['Wi'] * 1.03, color='gray', alpha=0.10)
        ax.set_xlabel('Wi = lam_truth * gammadot'); ax.set_ylabel(ttl)
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle(f'0D steady-shear: learned vs truth ({args.truth_model})  '
                 '(gray = out-of-cloud, not gated)')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(run_dir, 'flow_curves.png'), dpi=150)
    plt.close(fig)

    # A_zz vs Wi (FENE-P zz-channel diagnostic) + head curves
    x1 = cloud['cloud']['x1']; x2 = cloud['cloud']['x2']; x3 = cloud['cloud']['x3']
    X = jnp.stack([jnp.asarray(x1), jnp.asarray(x2), jnp.asarray(x3)], axis=-1)
    phi_tau, _phi_p2, _phi_l, _m0, m1, _ = tb.tbnn_heads(theta, X, bound_c)
    tau = np.asarray(x1) + 3.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(Wi, [r['t_Azz'] for r in rows], 'o-', color='C3', label='truth A_zz')
    axes[0].plot(Wi, [r['l_Azz'] for r in rows], 's--', color='C0', label='learned A_zz')
    axes[0].axhline(1.0, color='k', ls=':', lw=0.8)
    axes[0].set_xlabel('Wi'); axes[0].set_ylabel('steady A_zz'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].scatter(tau, 2 * np.asarray(phi_tau), s=4, color='C0', label='learned 2*phi_tau')
    if args.truth_model == 'fene_p':
        f_true = np.asarray(lc._fene_p_peterlin_f(jnp.asarray(tau), truth_ref_pp['Lsq']))
        axes[1].scatter(tau, f_true, s=4, color='C3', label='true f(tau)')
    else:
        axes[1].axhline(1.0, color='C3', ls='--', label='OB/Giesekus 2*phi_tau=1')
    axes[1].set_xlabel('tau = tr A'); axes[1].set_ylabel('2*phi_tau (gauge-annot.)')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].scatter(tau, np.asarray(m1), s=4, color='C0', label='learned m1')
    axes[2].axhline(args.truth_alpha if args.truth_model == 'giesekus' else 0.0,
                    color='C3', ls='--',
                    label=('Giesekus m1=alpha' if args.truth_model == 'giesekus'
                           else 'FENE-P m1=0'))
    axes[2].set_xlabel('tau = tr A'); axes[2].set_ylabel('m1 (gauge-annot.)')
    axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle(f'Diagnostics ({args.truth_model}) -- A_zz gated (FENE-P); '
                 'heads RECORD-only, gauge-annotated')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(run_dir, 'head_curves.png'), dpi=150)
    plt.close(fig)
    print(f"[save] outputs -> {run_dir}")


if __name__ == '__main__':
    sys.exit(main())
