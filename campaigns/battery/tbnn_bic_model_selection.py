"""BIC model selection on the viscoelastic TBNN (runs in the rheometry environment).

Mirrors the paper's TBNN-interrogation (arXiv:2510.24673): probe the learned
closure under prescribed AOS forcings, then fit a LIBRARY of classical models to
the TBNN's stress response and select via BIC (diff_rheo's own
``fit_model_to_experimental_data`` + ``calculate_bic_from_l2``). The paper showed
the (inelastic) TBNN is best captured by Carreau-Yasuda, not Newtonian; here the
VISCOELASTIC TBNN should be best captured by Giesekus (the truth family) over
Newtonian / Oldroyd-B / FENE-P / Linear-PTT -- and the recovered Giesekus
parameters should match the truth.

Done for ALL THREE converged fits (s1b, s1, s4): which model wins on BIC, and
how well it trains to the correct parameters.

Protocol (paper Methods .4, Eq.36): gammadot(t)=f sin(omega t),
f in {0.01,0.1,1,10}, omega in {0.33,1,2}; sigma_12(t) is the reported quantity
(loss Eq.34 = MSE on sigma_12), Gaussian noise added. Fit with Adam lr 0.1 (paper).

The TBNN stress data is generated through the SAME diff_rheo DiffraxSolver as the
candidate-model fits (conformation A(t) for the TBNN, native stress ODEs for the
candidates) -- apples-to-apples machinery.
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import diff_rheo as dr
from diff_rheo.models import Newtonian, OldroydB, Giesekus, FENEP, LinearPTT
from diff_rheo.parameters import LogParameter
from diff_rheo._forcing import VelocityGradient
import tbnn_diff_rheo_adapter as ad

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tbnn_phase1_data')
OUT = os.path.join(DATA, 'bic_model_selection')
os.makedirs(OUT, exist_ok=True)

F_GRID = [0.01, 0.1, 1.0, 10.0]
W_GRID = [0.33, 1.0, 2.0]


def _candidate_models():
    """Initial guesses ~1 (paper). FENE-P extension_length init > sqrt(3);
    Giesekus alpha / PTT eps,zeta init small for a stable start."""
    return {
        'Newtonian': lambda: Newtonian(viscosity=LogParameter(1.0)),
        'OldroydB': lambda: OldroydB(polymer_viscosity=LogParameter(1.0),
                                     relaxation_time=LogParameter(1.0),
                                     solvent_viscosity=LogParameter(1.0)),
        'Giesekus': lambda: Giesekus(polymer_viscosity=LogParameter(1.0),
                                     relaxation_time=LogParameter(1.0),
                                     solvent_viscosity=LogParameter(1.0),
                                     alpha=LogParameter(0.1)),
        'FENEP': lambda: FENEP(polymer_viscosity=LogParameter(1.0),
                               relaxation_time=LogParameter(1.0),
                               solvent_viscosity=LogParameter(1.0),
                               extension_length=LogParameter(5.0)),
        'LinearPTT': lambda: LinearPTT(polymer_viscosity=LogParameter(1.0),
                                       relaxation_time=LogParameter(1.0),
                                       solvent_viscosity=LogParameter(1.0),
                                       epsilon=LogParameter(0.1),
                                       zeta=LogParameter(0.1)),
    }


def _gen_tbnn_data(ck, data_solver, args, key):
    """Generate the TBNN's AOS sigma_12(t) battery as dr.BatchedData."""
    rhs = ad.make_tbnn_rhs(ck['theta'], ck['lam'])
    Gp, nu_s = ck['Gp'], ck['nu_s']
    exps = []
    meta = []
    for omega in W_GRID:
        period = 2.0 * np.pi / omega
        n = args.n_cycles * args.pts_per_cycle
        ts = jnp.linspace(0.0, args.n_cycles * period, n + 1)
        for f in F_GRID:
            vg = VelocityGradient.from_components(
                grad_u_12=lambda t, f=f, w=omega: f * jnp.sin(w * t))
            sol = data_solver.integrate(rhs, ad.A_REST, ts, vg)
            A = np.asarray(sol.ys)
            gd = np.asarray(f * jnp.sin(omega * ts))
            K, _, _ = ad.tbnn_K_and_frozen(A[:, 0], A[:, 1], A[:, 2], A[:, 3], ck['theta'])
            sig = np.asarray(Gp * K[1]) + nu_s * gd
            key, sub = jax.random.split(key)
            noise = float(args.noise) * np.asarray(jax.random.normal(sub, sig.shape))
            sig_noisy = sig + noise
            exps.append(dr.ShearStrainRateData(
                time=jnp.asarray(ts), data=jnp.asarray(sig_noisy),
                forcing_data=jnp.asarray(gd), initial_condition=jnp.zeros((3, 3))))
            meta.append(dict(f=f, omega=omega, max_sig=float(np.max(np.abs(sig))),
                             finite=bool(np.all(np.isfinite(sig)))))
    return dr.BatchedData.from_data(*exps), meta, key


def _fit_one(name, make_model, data, fit_solver, args, key):
    model = make_model()
    rheo = dr.VirtualRheometer.setup(model, "strain_rate_response", fit_solver)
    cfg = dr.FittingConfig(num_epochs=args.epochs, learning_rate=args.lr,
                           ensemble_size=1, key=None, verbose=False)
    try:
        fit = dr.fit_model_to_experimental_data(model, rheo, data, cfg)
        bic = float(dr.calculate_bic_from_l2(fit, rheo, data))
        pv = {k: float(v) for k, v in fit.parameter_values.items()}
        # pooled MSE for record
        sse, n = 0.0, 0
        for ref in data.data:
            sim = rheo.run_experiment(fit, ref.get_forcing_function(),
                                      ref.time, ref.initial_condition)
            pred = np.asarray(ref.extract_from_simulation(sim))
            sse += float(np.sum((pred - np.asarray(ref.data)) ** 2)); n += pred.size
        mse = sse / max(n, 1)
        ok = bool(np.isfinite(bic))
    except Exception as e:  # noqa: BLE001
        print(f"      [{name}] FIT FAILED: {repr(e)[:160]}")
        fit, bic, pv, mse, ok = None, float('inf'), {}, float('inf'), False
    return dict(name=name, bic=bic, params=pv, mse=mse, ok=ok, fit=fit, rheo=rheo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=['g3_s1b', 'g3_s1', 'g3_s4'])
    ap.add_argument('--models', nargs='+', default=None,
                    help='subset of candidate models (default all 5)')
    ap.add_argument('--epochs', type=int, default=800)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--noise', type=float, default=0.03)
    ap.add_argument('--n-cycles', type=int, default=6)
    ap.add_argument('--pts-per-cycle', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--port-rtol', type=float, default=1e-4,
                    help='cross-env adapter port tolerance (default 1e-4). '
                         'Higher-stretch curriculum closures port at ~1e-4 '
                         '(vs ~1e-14 low-stretch) due to jax/XLA transcendental '
                         'differences across envs -- still negligible vs the '
                         '3%% protocol noise. Loosen (e.g. 1e-3) for those.')
    # Optional truth override (used when a run's config.json is absent, e.g. a
    # training that timed out during the post-fit eval before config was saved).
    ap.add_argument('--truth-model', default=None, choices=[None, 'giesekus', 'fene_p'])
    ap.add_argument('--truth-gp', type=float, default=None)
    ap.add_argument('--truth-lam', type=float, default=None)
    ap.add_argument('--truth-nus', type=float, default=None)
    ap.add_argument('--truth-alpha', type=float, default=0.3)
    ap.add_argument('--truth-lsq', type=float, default=12.0)
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(ap)
    global _TRUTH_OVERRIDE
    _TRUTH_OVERRIDE = (dict(truth_model=args.truth_model, truth_gp=args.truth_gp,
                            truth_lam=args.truth_lam, truth_nus=args.truth_nus,
                            truth_alpha=args.truth_alpha, truth_lsq=args.truth_lsq)
                       if args.truth_model else None)

    data_solver = dr.DiffraxSolver(solver='tsit5', rtol=1e-8, atol=1e-8,
                                   dt0=1e-3, max_steps=4_000_000, throw=False)
    fit_solver = dr.DiffraxSolver(solver='tsit5', rtol=1e-6, atol=1e-6,
                                  dt0=1e-3, max_steps=1_000_000, throw=False)

    key = jax.random.PRNGKey(args.seed)
    all_results = {}
    for run in args.runs:
        print(f"\n===== {run} =====")
        truth, fam = _truth_for_run(run)
        print(f"  truth family = {fam} (diff_rheo params): {truth}")
        ck = ad.load_tbnn_checkpoint(os.path.join(DATA, run, 'theta_checkpoint.npz'))
        v = ad.verify_against_reference(ck, os.path.join(DATA, run, f'{run}_refio.npz'),
                                        rtol=args.port_rtol)
        if not v['ok']:
            raise RuntimeError(f"{run} cross-env port FAILED: {v}")
        print(f"  port OK (K_relerr={v['K_relerr']:.1e}); Gp={ck['Gp']:.4f} "
              f"lam={ck['lam']:.4f} nu_s={ck['nu_s']:.4f}")
        data, meta, key = _gen_tbnn_data(ck, data_solver, args, key)
        print(f"  generated {len(data)} AOS conditions (noise={args.noise})")

        cand = _candidate_models()
        if args.models:
            cand = {k: v for k, v in cand.items() if k in args.models}
        results = []
        for name, mk in cand.items():
            r = _fit_one(name, mk, data, fit_solver, args, key)
            results.append(r)
            print(f"    {name:<10} BIC={r['bic']:>12.2f}  MSE={r['mse']:.4e}  "
                  f"params={ {k: round(x,4) for k,x in r['params'].items()} }", flush=True)
            _dump_run_json(run, results, truth, fam, args, OUT)  # incremental
        finite = [r for r in results if r['ok']]
        winner = min(finite, key=lambda r: r['bic']) if finite else None
        print(f"  WINNER: {winner['name']} (lowest BIC = {winner['bic']:.2f})"
              if winner else "  WINNER: NONE")
        print(f"  (truth family is {fam}; hope it wins)")
        all_results[run] = dict(results=results, winner=winner['name'] if winner else None,
                                meta=meta, data=data, ck=ck)
        _plot_run(run, results, data, all_results[run], OUT)
        _write_run_report(run, results, winner, truth, fam, args, OUT)

    print(f"\n[done] outputs -> {OUT}")


_TRUTH_OVERRIDE = None


def _truth_for_run(run):
    """Per-run truth model + diff_rheo-parameter dict + candidate family name
    we hope BIC selects (Giesekus for G3, FENEP for G4). Uses the CLI truth
    override if given (config.json may be absent for a timed-out training),
    else reads the run's config.json."""
    if _TRUTH_OVERRIDE is not None:
        a = _TRUTH_OVERRIDE
    else:
        a = json.load(open(os.path.join(DATA, run, 'config.json')))['args']
    Gp = float(a['truth_gp']); lam = float(a['truth_lam']); nu = float(a['truth_nus'])
    tm = a.get('truth_model', 'giesekus')
    if tm == 'fene_p':
        L = float(a['truth_lsq']) ** 0.5  # diff_rheo extension_length = sqrt(L^2)
        return (dict(polymer_viscosity=Gp * lam, relaxation_time=lam,
                     solvent_viscosity=nu, extension_length=L), 'FENEP')
    return (dict(polymer_viscosity=Gp * lam, relaxation_time=lam,
                 solvent_viscosity=nu, alpha=float(a['truth_alpha'])), 'Giesekus')


def _dump_run_json(run, results, truth, fam, args, out):
    js = dict(run=run, truth_family=fam, truth_params=truth,
              protocol=dict(F_GRID=F_GRID, W_GRID=W_GRID, noise=args.noise,
                            epochs=args.epochs, lr=args.lr,
                            n_cycles=args.n_cycles, pts_per_cycle=args.pts_per_cycle),
              winner=(min((r for r in results if r['ok']), key=lambda r: r['bic'])['name']
                      if any(r['ok'] for r in results) else None),
              results=[dict(name=r['name'], bic=r['bic'], mse=r['mse'],
                            params=r['params'], ok=r['ok']) for r in results])
    with open(os.path.join(out, f'bic_results_{run}.json'), 'w') as fp:
        json.dump(js, fp, indent=2, default=float)


def _write_run_report(run, results, winner, truth, fam, args, out):
    wname = winner['name'] if winner else 'NONE'
    lines = [f"==== BIC model selection: {run} (winner: {wname}; truth family: {fam}) ====",
             f"protocol: gammadot=f sin(wt), f in {F_GRID}, omega in {W_GRID}; "
             f"noise={args.noise}; Adam lr={args.lr}, epochs={args.epochs}; "
             f"{args.n_cycles} cycles x {args.pts_per_cycle} pts/cycle.",
             f"truth {fam} (diff_rheo params): {truth}",
             f"BIC selected the truth family: {'YES' if wname == fam else 'NO (' + wname + ')'}"]
    finite = [r for r in results if r['ok']]
    bmin = min((r['bic'] for r in finite), default=float('nan'))
    lines.append(f"   {'model':<10}{'k':>3}{'BIC':>13}{'dBIC':>11}{'MSE':>12}")
    for r in results:
        k = len([kk for kk in r['params'] if kk != 'observation_noise'])
        db = (r['bic'] - bmin) if r['ok'] else float('nan')
        lines.append(f"   {r['name']:<10}{k:>3}{r['bic']:>13.2f}{db:>11.2f}{r['mse']:>12.3e}")
    tf = next((r for r in results if r['name'] == fam and r['ok']), None)
    if tf:
        lines.append(f"   {fam} (truth-family) recovered vs truth:")
        for kk, tv in truth.items():
            fv = tf['params'].get(kk, float('nan'))
            lines.append(f"      {kk:<20} fit={fv:.4f}  truth={tv:.4f}  ({abs(fv-tv)/abs(tv)*100:.1f}%)")
    with open(os.path.join(out, f'bic_summary_{run}.txt'), 'w') as fp:
        fp.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _plot_run(run, results, data, bundle, out):
    # BIC bar chart (relative to min)
    finite = [r for r in results if r['ok']]
    if finite:
        bmin = min(r['bic'] for r in finite)
        fig, ax = plt.subplots(figsize=(6, 4))
        names = [r['name'] for r in results]
        dbic = [r['bic'] - bmin if r['ok'] else np.nan for r in results]
        colors = ['C2' if r['name'] == bundle['winner'] else 'C0' for r in results]
        ax.bar(names, dbic, color=colors)
        ax.set_ylabel(r'$\Delta$BIC (vs best)'); ax.set_title(f'{run}: model selection (lower=better)')
        ax.tick_params(axis='x', rotation=20); ax.grid(alpha=0.3, axis='y')
        fig.tight_layout(); fig.savefig(os.path.join(out, f'{run}_bic_bar.png'), dpi=150)
        plt.close(fig)

    # waveform overlays at representative conditions: TBNN data vs each fitted model
    sel = [(1.0, 1.0), (10.0, 1.0), (1.0, 0.33)]
    fig, axes = plt.subplots(1, len(sel), figsize=(5 * len(sel), 3.8))
    for ax, (f, omega) in zip(axes, sel):
        idx = next((i for i, m in enumerate(bundle['meta']) if m['f'] == f and m['omega'] == omega), None)
        if idx is None:
            continue
        ref = data.data[idx]
        t = np.asarray(ref.time)
        ax.plot(t, np.asarray(ref.data), 'k.', ms=2, label='TBNN data', zorder=5)
        for r in results:
            if not r['ok']:
                continue
            sim = r['rheo'].run_experiment(r['fit'], ref.get_forcing_function(),
                                           ref.time, ref.initial_condition)
            pred = np.asarray(ref.extract_from_simulation(sim))
            lw = 2.2 if r['name'] == bundle['winner'] else 1.0
            ax.plot(t, pred, '-', lw=lw, label=r['name'])
        ax.set_title(f'{run}: f={f}, omega={omega}')
        ax.set_xlabel('t'); ax.set_ylabel(r'$\sigma_{12}$'); ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle(f'{run}: classical-model fits to TBNN stress (winner bold)')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out, f'{run}_fit_overlays.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
