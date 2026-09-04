#!/usr/bin/env python
"""Direct differentiable FENE-P calibration on the 4:1 contraction (no TBNN).

Fits log-parameters (Gp, lam, nu_s, Lsq) with L-BFGS against the solver's own
FENE-P (fene_p_logconf_bk_v2). Lsq is a traced polymer_params key.

Loss matches the paired TBNN ROI + per-rate alpha_v machinery, including
--rate-balance {legacy|equal}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
from scipy.optimize import minimize

from jax_rheology.models import registry as cr
from jax_rheology.geometries import planar_contraction as cg
from jax_rheology.forward import contraction as cf

# Reuse ROI helper + campaign hash from the TBNN contraction driver (read-only).
from visco_opt_tbnn_contraction_run import (  # noqa: E402
    TAPS_PHYS, bilinear_idx, dp_from_ptraj, roi_weight,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate the four FENE-P parameters directly against "
                    "contraction-flow velocity data (no neural closure).",
        epilog="Usual invocation, which sets every option below from a config "
               "file:\n"
               "  python experiments/fenep_direct.py "
               "--config experiments/configs/fenep_direct_fit_u05.yaml\n"
               "The individual flags exist so a config can be overridden for "
               "one run; see REPRODUCE.md for the published settings.")
    p.add_argument('--truth-gp', type=float, default=3.2)
    p.add_argument('--truth-lam', type=float, default=0.7)
    p.add_argument('--truth-nus', type=float, default=0.8)
    p.add_argument('--truth-lsq', type=float, default=12.0)
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--ny', type=int, default=256)
    p.add_argument('--H', type=float, default=1.0)
    p.add_argument('--ratio', type=float, default=4.0)
    p.add_argument('--L-up', type=float, default=6.0)
    p.add_argument('--L-down', type=float, default=12.0)
    p.add_argument('--density', type=float, default=1.0)
    p.add_argument('--dt', type=float, default=1e-4)
    p.add_argument('--inner', type=int, default=50)
    p.add_argument('--outer', type=int, default=400)
    p.add_argument('--ramp-time', type=float, default=1.0)
    p.add_argument('--solver-tol', type=float, default=1e-10)
    p.add_argument('--solver-maxiter', type=int, default=300)
    p.add_argument('--loss-weight', default='roi', choices=('roi', 'stretch', 'uniform'))
    p.add_argument('--roi-a', type=float, default=0.5)
    p.add_argument('--roi-c', type=float, default=1.75)
    p.add_argument('--roi-ell', type=float, default=0.4)
    p.add_argument('--roi-sigma-y', type=float, default=0.4)
    p.add_argument('--roi-kappa', type=float, default=4.0)
    p.add_argument('--roi-xc', type=float, default=0.0)
    p.add_argument('--U', type=float, default=None,
                   help='single-rate mode (e.g. 0.5). Mutually exclusive with --U-list.')
    p.add_argument('--U-list', type=str, default=None,
                   help='comma rates for multi-flow, e.g. "0.5,4".')
    p.add_argument('--w-p', type=float, default=0.0)
    p.add_argument('--w-p-scale', type=float, default=None)
    p.add_argument('--n-sub', type=int, default=8)
    p.add_argument('--norms-json', type=str, default=None)
    p.add_argument('--rate-balance', choices=('legacy', 'equal'), default='legacy')
    p.add_argument('--maxiter', type=int, default=60)
    p.add_argument('--init-gp', type=float, default=2.0)
    p.add_argument('--init-lam', type=float, default=1.5)
    p.add_argument('--init-nus', type=float, default=0.5)
    p.add_argument('--init-lsq', type=float, default=50.0)
    p.add_argument('--truth-init', action='store_true',
                   help='start at the ground-truth parameters (gradient check).')
    p.add_argument('--gate-iters', type=int, default=10,
                   help='with --truth-init: require drift <0.1%% over this many iters.')
    p.add_argument('--out-dir', type=str, default='./work/fenep_direct')
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--seed', type=int, default=0,
                   help='recorded RNG seed (I5 randomized starts).')
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(p)
    return args


def main():
    args = parse_args()
    if args.run_name is None:
        args.run_name = f'fenep_direct_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, 'DONE')):
        print(f"[resume] DONE present in {run_dir}; exit.", flush=True)
        return 0

    if args.U is not None and args.U_list is not None:
        raise SystemExit("[HALT] pass only one of --U / --U-list")
    if args.U_list is not None:
        U_list = [float(x) for x in args.U_list.split(',')]
    elif args.U is not None:
        U_list = [float(args.U)]
    else:
        U_list = [0.5]
    multi_flow = len(U_list) > 1

    H, R = args.H, args.ratio
    L_up, L_down = args.L_up * H, args.L_down * H
    from jax_ib.base import grids as _grids
    grid = _grids.Grid((args.nx, args.ny),
                       domain=(( -L_up, L_down), (-R * H, R * H)))
    model = cr.get_model('fene_p_logconf_bk_v2')
    print(f"[setup] device={jax.devices()} run_dir={run_dir}", flush=True)
    print(f"[setup] U_list={U_list} multi_flow={multi_flow} "
          f"rate_balance={args.rate_balance}", flush=True)

    # Build a reference state at U0 (geometry); per-rate evolve uses U_inlet.
    U0 = U_list[0]
    truth_state, perm_f, bc_spec = cg.build_contraction_viscoelastic_state(
        grid, H=H, L_down=L_down, U_inlet=U0, logistic_width=0.15,
        model=model, contraction_ratio=R)

    def _evolve(params, nu, U):
        _final, out = cf.evolve_contraction(
            truth_state, model, params, grid, density=args.density,
            base_viscosity=nu, dt=args.dt, inner_steps=args.inner,
            outer_steps=args.outer, U_inlet=U, ramp_time=args.ramp_time,
            perm_f=perm_f, bc_spec=bc_spec, solver_type='bicgstab',
            solver_tol=args.solver_tol, solver_maxiter=args.solver_maxiter)
        return out

    Gp_t, lam_t, nu_t, lsq_t = (
        args.truth_gp, args.truth_lam, args.truth_nus, args.truth_lsq)
    truth_pp = dict(
        Gp=jnp.asarray(Gp_t, jnp.float64),
        lam=jnp.asarray(lam_t, jnp.float64),
        Lsq=jnp.asarray(lsq_t, jnp.float64),
    )

    rate_data = {}
    for U in U_list:
        print(f"[truth] U={U} ...", flush=True)
        t0 = time.time()
        out_t = jax.jit(lambda U=U: _evolve(truth_pp, nu_t, U))()
        u_t, v_t = out_t['u_traj'], out_t['v_traj']
        u_t.block_until_ready()
        if args.loss_weight == 'roi':
            ubar = R * U
            w2d, w_raw = roi_weight(
                grid, x_c=args.roi_xc, y_c=0.0, ubar=ubar, lam=lam_t,
                a=args.roi_a, c=args.roi_c, ell=args.roi_ell,
                sigma_y=args.roi_sigma_y, kappa=args.roi_kappa)
            w_raw_np = np.asarray(w_raw)
            finite = bool(np.all(np.isfinite(np.asarray(w2d))))
            floor_ok = abs(float(w_raw_np.min()) - 1.0) < 1e-6
            peak = float(w_raw_np.max())
            peak_ok = peak > 1.0 + 0.5 * args.roi_kappa
            X, Y = grid.mesh(grid.cell_center)
            X = np.asarray(X); Y = np.asarray(Y)
            ic = np.unravel_index(
                np.argmin((X - args.roi_xc) ** 2 + (Y - args.H) ** 2), X.shape)
            corner = float(w_raw_np[ic])
            corner_ok = corner < 1.2
            if not (finite and floor_ok and peak_ok and corner_ok):
                raise SystemExit(
                    f"[HALT] ROI weight sanity failed @U={U}: "
                    f"finite={finite} floor_ok={floor_ok} peak_ok={peak_ok} "
                    f"corner_ok={corner_ok} corner={corner}")
            w_U = jnp.asarray(w2d, jnp.float64)
        elif args.loss_weight == 'uniform':
            w_U = jnp.ones_like(u_t)
        else:
            trA = (np.asarray(out_t['A_xx_traj']) + np.asarray(out_t['A_yy_traj'])
                   + np.asarray(out_t['A_zz_traj']))
            stretch = np.clip(trA - 3.0, 0.0, None)
            mw = stretch.mean()
            w_U = jnp.asarray(
                stretch / mw if mw > 0 else np.ones_like(stretch), jnp.float64)
        su2 = float(jnp.sum(w_U * (u_t ** 2 + v_t ** 2)) / (2.0 * jnp.sum(w_U)))
        alpha_v = (1.0 / max(su2, 1e-300)) if multi_flow else 1.0
        roi_mass = float(jnp.sum(w_U))
        vel_mass = float(jnp.sum(w_U * (u_t ** 2 + v_t ** 2)))
        rate_data[U] = dict(
            u_truth=u_t, v_truth=v_t, w=w_U, su2=su2, alpha_v=alpha_v,
            alpha_v_legacy=float(alpha_v), roi_mass=roi_mass, vel_mass=vel_mass,
            balance_scale=1.0, p_traj=out_t['p_traj'],
        )
        print(f"[truth] U={U} {time.time()-t0:.1f}s su2={su2:.4g} "
              f"alpha_v={alpha_v:.4g}", flush=True)

    rate_balance_info = dict(mode=args.rate_balance, scales={}, masses={})
    if args.rate_balance == 'equal':
        if not multi_flow:
            raise SystemExit("[HALT] --rate-balance equal needs >=2 rates")
        masses = {U: float(rate_data[U]['alpha_v_legacy'] * rate_data[U]['roi_mass'])
                  for U in U_list}
        log_sum = sum(math.log(max(m, 1e-300)) for m in masses.values())
        target = math.exp(log_sum / len(masses))
        for U in U_list:
            scale = target / max(masses[U], 1e-300)
            rate_data[U]['balance_scale'] = float(scale)
            rate_data[U]['alpha_v'] = float(rate_data[U]['alpha_v_legacy'] * scale)
            rate_balance_info['scales'][f'{U:g}'] = float(scale)
        m0 = masses[U_list[0]]; m1 = masses[U_list[1]]
        rate_balance_info['mass_ratio_legacy'] = float(m0 / max(m1, 1e-300))
        mb0 = rate_data[U_list[0]]['alpha_v'] * rate_data[U_list[0]]['roi_mass']
        mb1 = rate_data[U_list[1]]['alpha_v'] * rate_data[U_list[1]]['roi_mass']
        rate_balance_info['mass_ratio_balanced'] = float(mb0 / max(mb1, 1e-300))
        print(f"[rate-balance] equal scales={rate_balance_info['scales']} "
              f"ratio_legacy={rate_balance_info['mass_ratio_legacy']:.6g} "
              f"ratio_bal={rate_balance_info['mass_ratio_balanced']:.6g}",
              flush=True)
    else:
        print("[rate-balance] legacy", flush=True)

    if args.norms_json is not None:
        print(f"[norms-json] referenced {args.norms_json} "
              f"(direct driver records path; TBNN campaign-hash check N/A)",
              flush=True)

    pressure_on = (args.w_p_scale is not None) or (float(args.w_p) > 0.0)
    w_p_by_U = {U: float(args.w_p) for U in U_list}
    tap_idx = None
    factor = args.density / (args.inner * args.dt)
    if pressure_on:
        Xc, Yc = grid.mesh(grid.cell_center)
        xc = np.asarray(Xc)[:, 0]; yc = np.asarray(Yc)[0, :]
        tap_idx = [bilinear_idx(xc, yc, x, y) for (x, y) in TAPS_PHYS]
        for U in U_list:
            dp_truth = jax.lax.stop_gradient(
                dp_from_ptraj(rate_data[U]['p_traj'], tap_idx, factor))
            tframe = (np.arange(args.outer) + 1) * args.inner * args.dt
            post = np.where(tframe > args.ramp_time)[0]
            if len(post) < args.n_sub:
                post = np.arange(args.outer)
            sub = np.unique(np.clip(
                np.linspace(post[0], post[-1], args.n_sub).round().astype(int),
                0, args.outer - 1))
            dp_np = np.asarray(dp_truth)
            tap_norm = np.max(np.abs(dp_np[post]), axis=0)
            tap_w_np = np.zeros(3, dtype=np.float64)
            for k in range(3):
                if tap_norm[k] >= 1e-8:
                    tap_w_np[k] = 1.0 / (tap_norm[k] ** 2)
            rate_data[U]['dp_truth'] = dp_truth
            rate_data[U]['sub_j'] = jnp.asarray(sub)
            rate_data[U]['tap_w'] = jnp.asarray(tap_w_np, jnp.float64)
            rate_data[U]['tap_norm'] = tap_norm
        if args.w_p_scale is not None:
            if args.truth_init:
                init_phys = (Gp_t, lam_t, nu_t, lsq_t)
            else:
                init_phys = (args.init_gp, args.init_lam, args.init_nus,
                             args.init_lsq)
            pp_i = {
                'Gp': jnp.asarray(init_phys[0], jnp.float64),
                'lam': jnp.asarray(init_phys[1], jnp.float64),
                'Lsq': jnp.asarray(max(init_phys[3], 1.0 + 1e-6), jnp.float64),
            }
            nu_i = float(init_phys[2])
            for U in U_list:
                rd = rate_data[U]
                out0 = jax.jit(lambda U=U: _evolve(pp_i, nu_i, U))()
                du0 = out0['u_traj'] - rd['u_truth']
                dv0 = out0['v_traj'] - rd['v_truth']
                Lv0 = float((jnp.sum(rd['w'] * du0 * du0)
                             + jnp.sum(rd['w'] * dv0 * dv0)) * rd['alpha_v'])
                dp0 = dp_from_ptraj(out0['p_traj'], tap_idx, factor)
                r0 = (dp0 - rd['dp_truth'])[rd['sub_j']]
                R0 = float(jnp.sum(r0 * r0 * rd['tap_w']))
                wb = Lv0 / max(R0, 1e-300)
                w_p_by_U[U] = float(args.w_p_scale) * wb
                print(f"[pressure U={U}] w_bal={wb:.6g} "
                      f"w_p={w_p_by_U[U]:.6g}", flush=True)
        print(f"[pressure] ACTIVE w_p_by_U={w_p_by_U}", flush=True)
    else:
        print("[pressure] OFF (velocity-only)", flush=True)

    def unpack_log(z):
        Gp, lam, nu_s, Lsq = jnp.exp(z)
        return (jnp.maximum(Gp, 1e-8), jnp.maximum(lam, 1e-8),
                jnp.maximum(nu_s, 1e-8), jnp.maximum(Lsq, 1.0 + 1e-6))

    def loss_from_phys(Gp, lam, nu_s, Lsq):
        pp = {'Gp': Gp, 'lam': lam, 'Lsq': Lsq}
        total = 0.0
        for U in U_list:
            rd = rate_data[U]
            out = _evolve(pp, nu_s, U)
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
        return total / float(len(U_list))

    def loss_fn(z):
        Gp, lam, nu_s, Lsq = unpack_log(z)
        return loss_from_phys(Gp, lam, nu_s, Lsq)

    vag = jax.jit(jax.value_and_grad(loss_fn))

    if args.truth_init:
        z0 = np.log(np.array([Gp_t, lam_t, nu_t, lsq_t], dtype=np.float64))
    else:
        z0 = np.log(np.array(
            [args.init_gp, args.init_lam, args.init_nus, args.init_lsq],
            dtype=np.float64))

    print("[opt] warm-compiling value_and_grad ...", flush=True)
    t0 = time.time()
    L0, g0 = vag(jnp.asarray(z0))
    L0.block_until_ready()
    g0 = np.asarray(g0)
    print(f"[opt] compile {time.time()-t0:.1f}s loss={float(L0):.6e} "
          f"grad_finite={bool(np.all(np.isfinite(g0)))} "
          f"|g|={float(np.linalg.norm(g0)):.4g}", flush=True)
    if not (np.isfinite(float(L0)) and np.all(np.isfinite(g0))):
        raise SystemExit("[FATAL] non-finite loss/grad at init")

    progress_path = os.path.join(run_dir, 'progress.csv')
    with open(progress_path, 'w') as pf:
        pf.write('nfev,loss,Gp,lam,nu_s,Lsq,log_Gp,log_lam,log_nus,log_Lsq\n')
        pf.flush()
    hist = []
    nfev = [0]

    def obj_and_grad(z):
        val, g = vag(jnp.asarray(z))
        nfev[0] += 1
        fv = float(val)
        Gp, lam, nu_s, Lsq = [float(v) for v in unpack_log(jnp.asarray(z))]
        hist.append((nfev[0], fv, Gp, lam, nu_s, Lsq))
        with open(progress_path, 'a') as pf:
            pf.write(
                f'{nfev[0]},{fv:.8e},{Gp:.8g},{lam:.8g},{nu_s:.8g},{Lsq:.8g},'
                f'{z[0]:.8g},{z[1]:.8g},{z[2]:.8g},{z[3]:.8g}\n')
            pf.flush()
        print(f'  nfev {nfev[0]:>3}  loss={fv:.4e}  '
              f'Gp={Gp:.4f} lam={lam:.4f} nu_s={nu_s:.4f} Lsq={Lsq:.4f}',
              flush=True)
        return fv, np.asarray(g, dtype=np.float64)

    maxiter = args.gate_iters if args.truth_init else args.maxiter
    print(f"[opt] L-BFGS-B maxiter={maxiter} truth_init={args.truth_init}",
          flush=True)
    t0 = time.time()
    res = minimize(
        obj_and_grad, z0, jac=True, method='L-BFGS-B',
        options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-10),
    )
    t_opt = time.time() - t0
    Gp, lam, nu_s, Lsq = [float(v) for v in unpack_log(jnp.asarray(res.x))]
    drift = {
        'Gp': abs(Gp - Gp_t) / Gp_t * 100,
        'lam': abs(lam - lam_t) / lam_t * 100,
        'nu_s': abs(nu_s - nu_t) / nu_t * 100,
        'Lsq': abs(Lsq - lsq_t) / lsq_t * 100,
    }
    gate = None
    if args.truth_init:
        max_drift = max(drift.values())
        gate = {
            'gradients_finite': bool(np.all(np.isfinite(g0))),
            'loss_init': float(L0),
            'loss_final': float(res.fun),
            'max_param_drift_pct': float(max_drift),
            'drift_pct': drift,
            'pass': bool(
                np.all(np.isfinite(g0))
                and np.isfinite(float(L0))
                and max_drift < 0.1
            ),
        }
        print(f"[gate] truth-init pass={gate['pass']} "
              f"max_drift%={max_drift:.4g}", flush=True)

    summary = dict(
        run_name=args.run_name,
        truth=dict(Gp=Gp_t, lam=lam_t, nu_s=nu_t, Lsq=lsq_t),
        init=dict(Gp=args.init_gp, lam=args.init_lam, nu_s=args.init_nus,
                  Lsq=args.init_lsq, truth_init=bool(args.truth_init)),
        final=dict(Gp=Gp, lam=lam, nu_s=nu_s, Lsq=Lsq, eta_p=Gp * lam),
        loss_init=float(L0), loss_final=float(res.fun),
        nfev=int(res.nfev), nit=int(res.nit),
        success=bool(res.success), message=str(res.message),
        converged=bool(res.success),
        U_list=U_list, multi_flow=multi_flow,
        rate_balance=rate_balance_info,
        pressure_on=bool(pressure_on),
        w_p=float(args.w_p),
        w_p_scale=(float(args.w_p_scale) if args.w_p_scale is not None else None),
        norms_json=args.norms_json,
        seed=int(args.seed),
        opt_seconds=t_opt,
        truth_init_gate=gate,
        drift_pct_vs_truth=drift,
        args=vars(args),
    )
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(run_dir, 'DONE'), 'w') as f:
        f.write(f'completed {datetime.now().isoformat()} loss={float(res.fun):.6e}\n')
    print(f"[done] Gp={Gp:.4f} lam={lam:.4f} nu_s={nu_s:.4f} Lsq={Lsq:.4f} "
          f"eta_p={Gp*lam:.4f} loss {float(L0):.4e}->{float(res.fun):.4e}",
          flush=True)
    if gate is not None and not gate['pass']:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
