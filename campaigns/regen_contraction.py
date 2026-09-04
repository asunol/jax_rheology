#!/usr/bin/env python
"""Contraction field regeneration for the publication figure pass.

Reproduces each run's EXACT config (from summary.json args) and regenerates the
three forward trajectories needed for figs a/c/f: ground-truth, initial-guess
(OB-init, gauge 1), and learned (fitted theta + scalars). Truth and init are
identical across schemes within a campaign, so they are computed ONCE per
family (the initial-guess field is generated once per campaign) and
saved to a shared campaign dir; the learned trajectory is saved per run.

Every trajectory is saved as npz with full u, v, A_* fields at every outer step
+ final fields + derived scalars (centerline u_x(x), max A_xx(t); FENE also the
p1-referenced dp2/dp3/dp4 tap-difference trajectories). A regen_manifest.json is
written per output dir for provenance.

Env: cfd_md_optimization (jax_rheology). One family per invocation.
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

from jax_rheology.models import registry as cr
from jax_rheology.geometries import planar_contraction as cg
from jax_rheology.forward import contraction as cf
from jax_rheology.models import tbnn_memory as tb
from visco_opt_tbnn_run import theta_from_named_arrays
import visco_opt_tbnn_contraction_run as C   # TRUTH_NAME, TBNN_NAME, dp helpers


def _cfg_hash(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _build(a, model_name):
    """Explicit 128x256 contraction grid + state for a given model name."""
    from jax_ib.base import grids as _grids
    H, R = float(a['H']), float(a['ratio'])
    L_up, L_down = float(a['L_up']) * H, float(a['L_down']) * H
    domain = cg.contraction_domain(H, L_up, L_down, R)
    grid = _grids.Grid((int(a['nx']), int(a['ny'])), domain=domain)
    _dx, dy = grid.step
    wlog = 0.5 * dy
    model = cr.get_model(model_name)
    state, perm, bc = cg.build_contraction_viscoelastic_state(
        grid, H=H, L_down=L_down, U_inlet=float(a['U']), logistic_width=wlog,
        model=model, contraction_ratio=R)
    return grid, model, state, perm, bc


def _make_evolver(a, model, grid, perm, bc):
    """Return an EAGER forward f(state, params, nu) -> (final, out).

    Runs on gpu/seas_gpu where this env binds the A100; the eager forward is
    ~23s at 128x256 (exactly as the production truth generation ran), with no
    jit-compile overhead. model/grid/perm/bc are captured as closure constants.
    """
    dens = float(a['density']); dt = float(a['dt']); inner = int(a['inner'])
    outer = int(a['outer']); U = float(a['U']); ramp = float(a['ramp_time'])
    stol = float(a['solver_tol']); smax = int(a['solver_maxiter'])

    def f(state, params, nu):
        return cf.evolve_contraction(
            state, model, params, grid, density=dens, base_viscosity=nu, dt=dt,
            inner_steps=inner, outer_steps=outer, U_inlet=U, ramp_time=ramp,
            perm_f=perm, bc_spec=bc, solver_type='bicgstab', solver_tol=stol,
            solver_maxiter=smax)
    return f


def _grid_axes(grid):
    return dict(
        xc=np.asarray(grid.axes(grid.cell_center)[0]),
        yc=np.asarray(grid.axes(grid.cell_center)[1]),
        xfu=np.asarray(grid.axes(grid.cell_faces[0])[0]),
        yfu=np.asarray(grid.axes(grid.cell_faces[0])[1]))


def _pack(final, out, grid, a, pressure_on, tap_idx, factor):
    xf = np.asarray(grid.axes(grid.cell_faces[0])[0])
    yf = np.asarray(grid.axes(grid.cell_faces[0])[1])
    jc = int(np.argmin(np.abs(yf)))
    u_final = np.asarray(final.velocity[0].array.data)
    v_final = np.asarray(final.velocity[1].array.data)
    pk = dict(
        u=u_final, v=v_final,
        A_xx=np.asarray(final.memory_fields[0].array.data),
        A_yy=np.asarray(final.memory_fields[2].array.data),
        A_zz=np.asarray(final.memory_fields[3].array.data),
        u_traj=np.asarray(out['u_traj']), v_traj=np.asarray(out['v_traj']),
        A_xx_traj=np.asarray(out['A_xx_traj']),
        A_xy_traj=np.asarray(out['A_xy_traj']),
        A_yy_traj=np.asarray(out['A_yy_traj']),
        A_zz_traj=np.asarray(out['A_zz_traj']),
        # derived scalars
        ucl_x=u_final[:, jc],
        max_Axx_traj=np.asarray(out['A_xx_traj']).max(axis=(1, 2)),
    )
    if pressure_on:
        dp = np.asarray(C.dp_from_ptraj(jnp.asarray(out['p_traj']), tap_idx, factor))
        pk['dp_traj'] = dp            # (outer, 3) = dp2, dp3, dp4
        pk['p_traj'] = np.asarray(out['p_traj'])
    return pk


def _save(path, arrays, axes, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays, **axes,
                        H=meta['H'], R=meta['R'], U=meta['U'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, choices=['giesekus', 'fene'])
    ap.add_argument('--campaign-root', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out-root', default='work/regen_contraction')
    ap.add_argument('--jobid', default=os.environ.get('SLURM_JOB_ID', 'local'))
    ap.add_argument('--smoke', action='store_true',
                    help='login-node plumbing check: 1 run, tiny outer')
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(ap)

    if args.smoke:
        args.runs = args.runs[:1]
        args.out_root = args.out_root.rstrip('/') + '_smoke'

    rep = json.load(open(os.path.join(args.campaign_root, args.runs[0],
                                      'summary.json')))
    a = rep['args']
    if args.smoke:
        a = dict(a); a['outer'] = 3
    H, R, U = float(a['H']), float(a['ratio']), float(a['U'])
    pressure_on = bool(rep.get('pressure_on', False)) or (args.family == 'fene')
    print(f"[regen-ctr] family={args.family} runs={args.runs} "
          f"grid={a['nx']}x{a['ny']} outer={a['outer']} pressure={pressure_on}",
          flush=True)

    # grids/states (truth + tbnn share geometry)
    grid, truth_model, truth_state, perm_f, bc_f = _build(a, C.TRUTH_NAME[a['truth_model']])
    _, tbnn_model, tbnn_state, perm_t, bc_t = _build(a, C.TBNN_NAME)
    axes = _grid_axes(grid)
    meta = dict(H=H, R=R, U=U)
    truth_evolve = _make_evolver(a, truth_model, grid, perm_f, bc_f)
    tbnn_evolve = _make_evolver(a, tbnn_model, grid, perm_t, bc_t)

    factor = float(a['density']) / (int(a['inner']) * float(a['dt']))
    tap_idx = None
    if pressure_on:
        Xc, Yc = grid.mesh(grid.cell_center)
        xc = np.asarray(Xc)[:, 0]; yc = np.asarray(Yc)[0, :]
        tap_idx = [C.bilinear_idx(xc, yc, x, y) for (x, y) in C.TAPS_PHYS]

    shared = os.path.join(args.out_root, f'_campaign_{args.family}', 'regen')
    os.makedirs(shared, exist_ok=True)
    cfg_hash = _cfg_hash({k: a[k] for k in ('H', 'ratio', 'L_up', 'L_down', 'nx',
                                            'ny', 'U', 'dt', 'inner', 'outer',
                                            'ramp_time', 'truth_model')})

    # ---- truth (once per family) -------------------------------------------
    t0 = time.time()
    Gp_t, lam_t, nu_t = float(a['truth_gp']), float(a['truth_lam']), float(a['truth_nus'])
    if a['truth_model'] == 'giesekus':
        pp = dict(Gp=jnp.asarray(Gp_t), lam=jnp.asarray(lam_t),
                  alpha=jnp.asarray(float(a['truth_alpha'])))
    else:
        pp = dict(Gp=jnp.asarray(Gp_t), lam=jnp.asarray(lam_t),
                  Lsq=jnp.asarray(float(a['truth_lsq'])))
    final, out = truth_evolve(truth_state, pp, nu_t)
    out['u_traj'].block_until_ready()
    tr = _pack(final, out, grid, a, pressure_on, tap_idx, factor)
    _save(os.path.join(shared, 'truth_traj.npz'), tr, axes, meta)
    print(f"[truth] {time.time()-t0:.1f}s max|u|={np.abs(tr['u']).max():.3f} "
          f"maxA_xx={tr['A_xx'].max():.3f}", flush=True)

    # ---- initial guess (once per family): OB-init theta, gauge 1 ------------
    t0 = time.time()
    theta0, _ = tb.init_tbnn_theta(jax.random.PRNGKey(int(a['seed'])),
                                   width=int(a['width']), depth=int(a['depth']),
                                   bound_c=float(a['bound_c']))
    pp0 = dict(Gp=jnp.asarray(1.0), lam=jnp.asarray(1.0), theta=theta0,
               tbnn_bound_c=float(a['bound_c']))
    final, out = tbnn_evolve(tbnn_state, pp0, 1.0)
    out['u_traj'].block_until_ready()
    ig = _pack(final, out, grid, a, pressure_on, tap_idx, factor)
    _save(os.path.join(shared, 'init_traj.npz'), ig, axes, meta)
    print(f"[init]  {time.time()-t0:.1f}s max|u|={np.abs(ig['u']).max():.3f} "
          f"maxA_xx={ig['A_xx'].max():.3f}", flush=True)

    with open(os.path.join(shared, 'regen_manifest.json'), 'w') as f:
        json.dump(dict(kind='contraction_shared', family=args.family,
                       regenerated=['truth_traj.npz', 'init_traj.npz'],
                       cfg_hash=cfg_hash, truth_ckpt=None,
                       init='OB-init theta seed=%s gauge Gp=lam=nu_s=1' % a['seed'],
                       jobid=args.jobid, source_run=args.runs[0]), f, indent=2)

    # ---- learned (per run) --------------------------------------------------
    for run in args.runs:
        rd = os.path.join(args.campaign_root, run)
        ck = os.path.join(rd, 'theta_checkpoint.npz')
        z = np.load(ck, allow_pickle=False)
        heads = [str(h) for h in z['ckpt_heads']]
        nlayers = {h: int(n) for h, n in zip(heads, z['ckpt_nlayers'])}
        theta = theta_from_named_arrays(z, heads, nlayers)
        Gp_f, lam_f, nu_f = (float(z['ckpt_Gp_fit']), float(z['ckpt_lam_fit']),
                             float(z['ckpt_nu_s']))
        ppL = dict(Gp=jnp.asarray(Gp_f), lam=jnp.asarray(lam_f), theta=theta,
                   tbnn_bound_c=float(a['bound_c']))
        t0 = time.time()
        final, out = tbnn_evolve(tbnn_state, ppL, nu_f)
        out['u_traj'].block_until_ready()
        lr = _pack(final, out, grid, a, pressure_on, tap_idx, factor)
        outdir = os.path.join(args.out_root, run, 'regen')
        _save(os.path.join(outdir, 'learned_traj.npz'), lr, axes, meta)
        with open(os.path.join(outdir, 'regen_manifest.json'), 'w') as f:
            json.dump(dict(kind='contraction_learned', run=run,
                           regenerated=['learned_traj.npz'],
                           shared_truth_init=os.path.relpath(shared, outdir),
                           checkpoint=os.path.relpath(ck),
                           scalars=dict(Gp_fit=Gp_f, lam_fit=lam_f, nu_s_fit=nu_f),
                           cfg_hash=cfg_hash, jobid=args.jobid), f, indent=2)
        print(f"[learned {run}] {time.time()-t0:.1f}s "
              f"max|u|={np.abs(lr['u']).max():.3f} maxA_xx={lr['A_xx'].max():.3f} "
              f"(Gp={Gp_f:.3f} lam={lam_f:.3f} nu={nu_f:.3f})", flush=True)

    print(f"[regen-ctr] family={args.family} DONE", flush=True)


if __name__ == '__main__':
    main()
