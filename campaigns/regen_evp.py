#!/usr/bin/env python
"""EVP channel field regeneration for the publication figure pass.

Reuses the production forward machinery (visco_tbnn._build_geometry,
analytic_limits_validation._evolve_wall_bounded_with_diagnostics,
tbnn_memory.init_tbnn_theta) to regenerate, per run and per drive:
  - initial-guess trajectory at T_fit (OB-init theta + BR/ones init scalars)
  - 60lam extension of the FITTED model (for the steady flow-curve fig l2)
and, once (shared across runs), truth 60lam at Nx=16.

The learned + truth trajectories at T_fit already exist (model_traj_*.npz /
truth_traj_*.npz in the run dirs) and are NOT recomputed here.

Saves full u, v, A_* fields at every outer step + derived scalars
(Q(t), u_cl(t), plug_hw(t) via the fixed |tau_d|<=tau_y yield ruler) to
<run>/regen/ (+ shared campaign dir for truth). regen_manifest.json per dir.

Env: cfd_md_optimization (jax_rheology).
"""
import argparse
import hashlib
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
from visco_opt_tbnn_run import theta_from_named_arrays

V2_MODEL_NAME = 'tbnn_potential_yield_logconf_bk_v2'
TRUTH_NAME = 'saramito_logconf_bk_v2'
LAM_TRUTH = 0.7
OUTER_TFIT = 200          # T = 5.0 = 7.14 lam
OUTER_60LAM = 1680        # T = 60 lam (dt=2.5e-3, inner=10)


def _cfg(outer):
    c = dict(vf.DEFAULT_CHANNEL_CONFIG)
    c['Nx'] = 16
    c['Ny'] = 128
    c['outer_steps'] = int(outer)
    c['solver_tol'] = 1e-8
    return c


def _plug_hw_unyielded(uny, dy):
    Ny = len(uny); jc = Ny // 2
    if not bool(uny[jc]):
        return 0.0
    lo = jc
    while lo - 1 >= 0 and uny[lo - 1]:
        lo -= 1
    hi = jc
    while hi + 1 < Ny and uny[hi + 1]:
        hi += 1
    return 0.5 * (hi - lo + 1) * dy


def _derived(out, cfg, Gp, tau_y):
    """Q(t), u_cl(t), plug_hw(t) [yield ruler] from a forward trajectory."""
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    y = (np.arange(Ny) + 0.5) * dy
    u = np.asarray(out['u_traj'])           # (T, Nx, Ny)
    Axx = np.asarray(out['A_xx_traj']); Axy = np.asarray(out['A_xy_traj'])
    Ayy = np.asarray(out['A_yy_traj']); Azz = np.asarray(out['A_zz_traj'])
    T = u.shape[0]; jc = Ny // 2
    Q = np.empty(T); ucl = np.empty(T); phw = np.empty(T); yf = np.empty(T)
    for k in range(T):
        uk = u[k].mean(axis=0)
        ucl[k] = float(uk[jc])
        Q[k] = float(np.trapz(uk, y))
        td = np.asarray(tb.saramito_tau_d_norm(
            jnp.asarray(Axx[k].mean(axis=0)), jnp.asarray(Axy[k].mean(axis=0)),
            jnp.asarray(Ayy[k].mean(axis=0)), jnp.asarray(Azz[k].mean(axis=0)), Gp))
        uny = td <= tau_y
        phw[k] = _plug_hw_unyielded(uny, dy)
        yf[k] = float((~uny).mean())
    return dict(Q_traj=Q, ucl_traj=ucl, plug_hw_traj=phw, yielded_frac_traj=yf, y=y)


def _pack(out, cfg, Gp, tau_y, g_x):
    d = dict(
        u=np.asarray(out['u_traj']), v=np.asarray(out['v_traj']),
        A_xx=np.asarray(out['A_xx_traj']), A_xy=np.asarray(out['A_xy_traj']),
        A_yy=np.asarray(out['A_yy_traj']), A_zz=np.asarray(out['A_zz_traj']),
        g_x=g_x)
    d.update(_derived(out, cfg, Gp, tau_y))
    d['Q_final'] = float(d['Q_traj'][-1])
    return d


def _make_forward(cfg, grid, state, model, perm):
    """EAGER forward f(params, nu, g_x) -> out. cfg/model/state/perm static.

    Runs on gpu/seas_gpu (A100); the channel grid (16x128) is tiny so eager
    dispatch is fast even at 60 lambda.
    """
    dens = cfg['density']; dt = cfg['dt']; inner = cfg['inner_steps']
    outer = cfg['outer_steps']; stype = cfg['solver_type']
    usepc = cfg['use_preconditioner']; pctype = cfg['preconditioner_type']
    Uf = cfg['U_f']; stol = cfg['solver_tol']; smax = cfg['solver_maxiter']

    def f(params, nu, g_x):
        return p3b._evolve_wall_bounded_with_diagnostics(
            initial_state=state, model=model, polymer_params=params, grid=grid,
            density=dens, base_viscosity=nu, dt=dt, inner_steps=inner,
            outer_steps=outer, solver_type=stype, use_preconditioner=usepc,
            preconditioner_type=pctype, pressure_gradient=(g_x, 0.0),
            permeability=perm, U_f=Uf, solver_tol=stol, solver_maxiter=smax)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evp-root', default='work/evp_channel')
    ap.add_argument('--runs', nargs='+', default=['v2_final_Nx16_br', 'v2_final_Nx16_ones'])
    ap.add_argument('--drives', nargs='+', type=float, default=[1.8, 2.5, 4.0])
    ap.add_argument('--out-root', default='work/regen_evp')
    ap.add_argument('--jobid', default=os.environ.get('SLURM_JOB_ID', 'local'))
    ap.add_argument('--truth-60', action='store_true',
                    help='also regen truth 60lam (shared); default on')
    ap.add_argument('--smoke', action='store_true',
                    help='login-node plumbing check: 1 run, 1 drive, tiny outer')
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(ap)

    global OUTER_TFIT, OUTER_60LAM
    if args.smoke:
        OUTER_TFIT, OUTER_60LAM = 3, 4
        args.runs = args.runs[:1]
        args.drives = args.drives[:1]
        args.out_root = args.out_root.rstrip('/') + '_smoke'

    Gp_t, lam_t, nu_t, tau_y_t = 3.2, 0.7, 0.8, 1.45
    tag = lambda g: f"{g:g}".replace('.', 'p')

    cfg_fit = _cfg(OUTER_TFIT)
    cfg_60 = _cfg(OUTER_60LAM)

    # geometries (built once; states are re-usable initial conditions)
    grid, truth_model, truth_state, truth_perm = vt._build_geometry(cfg_60, TRUTH_NAME, 'channel')
    _, tbnn_model, tbnn_state, tbnn_perm = vt._build_geometry(cfg_60, V2_MODEL_NAME, 'channel')
    truth_fwd_60 = _make_forward(cfg_60, grid, truth_state, truth_model, truth_perm)
    tbnn_fwd_fit = _make_forward(cfg_fit, grid, tbnn_state, tbnn_model, tbnn_perm)
    tbnn_fwd_60 = _make_forward(cfg_60, grid, tbnn_state, tbnn_model, tbnn_perm)

    # ---- shared truth 60lam (once) -----------------------------------------
    shared = os.path.join(args.out_root, '_campaign_evp', 'regen')
    os.makedirs(shared, exist_ok=True)
    truth_pp = {'Gp': jnp.asarray(Gp_t), 'lam': jnp.asarray(lam_t),
                'tau_y': jnp.asarray(tau_y_t)}
    for g in args.drives:
        t0 = time.time()
        out = truth_fwd_60(truth_pp, nu_t, float(g))
        out['u_traj'].block_until_ready()
        d = _pack(out, cfg_60, Gp_t, tau_y_t, g)
        np.savez_compressed(os.path.join(shared, f'ext60_truth_gx{tag(g)}.npz'), **d)
        print(f"[truth60 g={g}] {time.time()-t0:.1f}s Q_final={d['Q_final']:.6e} "
              f"plug_hw={d['plug_hw_traj'][-1]:.4f}", flush=True)
    with open(os.path.join(shared, 'regen_manifest.json'), 'w') as f:
        json.dump(dict(kind='evp_truth_60lam', drives=args.drives,
                       outer=OUTER_60LAM, Nx=16, Ny=128,
                       truth=dict(Gp=Gp_t, lam=lam_t, nu_s=nu_t, tau_y=tau_y_t),
                       jobid=args.jobid), f, indent=2)

    # ---- per-run: initial-guess (T_fit) + fitted 60lam extension ------------
    for run in args.runs:
        rd = os.path.join(args.evp_root, run)
        cfgj = json.load(open(os.path.join(rd, 'config.json')))
        a = cfgj['args']
        z = np.load(os.path.join(rd, 'theta_checkpoint.npz'), allow_pickle=False)
        heads = [str(h) for h in z['ckpt_heads']]
        nlayers = {h: int(n) for h, n in zip(heads, z['ckpt_nlayers'])}
        theta = theta_from_named_arrays(z, heads, nlayers)
        Gp_f = float(z['ckpt_Gp_fit']); lam_f = float(z['ckpt_lam_fit'])
        nu_f = float(z['ckpt_nu_s']); tau_y_f = float(z['ckpt_tau_y_fit'])
        bound_c = float(z['ckpt_bound_c'])
        width = int(z['ckpt_width']); depth = int(z['ckpt_depth'])
        br = cfgj['br_init']
        Gp0 = float(br['Gp_init_clipped']); lam0 = float(br['lam_init_clipped'])
        nu0 = float(br['nu_s_init_clipped']); tau_y0 = float(br['tau_y_init_clipped'])
        seed = int(a.get('seed', 0))
        outdir = os.path.join(args.out_root, run, 'regen')
        os.makedirs(outdir, exist_ok=True)

        # OB-init theta for the initial guess (V2 anchored + yield scalar)
        theta0, _ = tb.init_tbnn_theta(jax.random.PRNGKey(seed), width=width,
                                       depth=depth, bound_c=bound_c, anchored=True,
                                       mobility='softplus', yield_mode='scalar')

        for g in args.drives:
            # initial guess at T_fit
            t0 = time.time()
            p0 = {'Gp': jnp.asarray(Gp0), 'lam': jnp.asarray(lam0), 'theta': theta0,
                  'tbnn_bound_c': bound_c, 'tbnn_kappa': 1.0, 'tau_y': jnp.asarray(tau_y0)}
            out = tbnn_fwd_fit(p0, nu0, float(g))
            out['u_traj'].block_until_ready()
            d = _pack(out, cfg_fit, Gp0, tau_y0, g)
            np.savez_compressed(os.path.join(outdir, f'initguess_gx{tag(g)}.npz'), **d)
            print(f"[init {run} g={g}] {time.time()-t0:.1f}s Q_final={d['Q_final']:.6e}",
                  flush=True)

            # fitted-model 60lam extension
            t0 = time.time()
            pL = {'Gp': jnp.asarray(Gp_f), 'lam': jnp.asarray(lam_f), 'theta': theta,
                  'tbnn_bound_c': bound_c, 'tbnn_kappa': 1.0, 'tau_y': jnp.asarray(tau_y_f)}
            out = tbnn_fwd_60(pL, nu_f, float(g))
            out['u_traj'].block_until_ready()
            d = _pack(out, cfg_60, Gp_f, tau_y_f, g)
            np.savez_compressed(os.path.join(outdir, f'ext60_model_gx{tag(g)}.npz'), **d)
            print(f"[ext60 {run} g={g}] {time.time()-t0:.1f}s Q_final={d['Q_final']:.6e} "
                  f"plug_hw={d['plug_hw_traj'][-1]:.4f}", flush=True)

        with open(os.path.join(outdir, 'regen_manifest.json'), 'w') as f:
            json.dump(dict(kind='evp_run_regen', run=run, drives=args.drives,
                           regenerated=[f'initguess_gx{tag(g)}.npz' for g in args.drives]
                                       + [f'ext60_model_gx{tag(g)}.npz' for g in args.drives],
                           checkpoint=os.path.relpath(os.path.join(rd, 'theta_checkpoint.npz')),
                           init_scalars=dict(Gp0=Gp0, lam0=lam0, nu_s0=nu0, tau_y0=tau_y0,
                                             method=br.get('method')),
                           fit_scalars=dict(Gp_fit=Gp_f, lam_fit=lam_f, nu_s_fit=nu_f,
                                            tau_y_fit=tau_y_f),
                           tfit_outer=OUTER_TFIT, ext60_outer=OUTER_60LAM,
                           existing_tfit='model_traj_*.npz / truth_traj_*.npz (run dir)',
                           jobid=args.jobid), f, indent=2)
        print(f"[regen-evp] run={run} DONE", flush=True)


if __name__ == '__main__':
    main()
