"""Withheld-N1 transferability on the synthetic sweep, for the per-dataset BIC winner.

Each synthetic dataset: random-parameter ground-truth model -> sigma12(t) (+noise)
under gammadot(t)=sin(t); five candidates fit to sigma12; BIC selects one. N1 is
NEVER used in fitting. Here we reconstruct, per dataset, the true model and the
BIC-SELECTED model (stored params), simulate both through the strain-rate protocol
(N1 = sigma11 - sigma22, correct sign after the convection fix), and show how well
the selected model recovers the withheld true N1 -- including when selection is
"wrong" (e.g. LinearPTT truth -> Giesekus selected).
"""
import json, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import apply as apply_style, SAVE
apply_style()

import diff_rheo as dr
from diff_rheo.models import Newtonian, CarreauYasuda, OldroydB, Giesekus, LinearPTT

jax.config.update("jax_enable_x64", True)
RES = Path(__file__).resolve().parent / "results_synthetic"
OUT = RES / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
LOOKUP = {"Newtonian": Newtonian, "CarreauYasuda": CarreauYasuda, "OldroydB": OldroydB,
          "Giesekus": Giesekus, "LinearPTT": LinearPTT}
VE = {"OldroydB", "Giesekus", "LinearPTT"}                 # models with nonzero N1
solver = dr.DiffraxSolver(rtol=1e-6, atol=1e-8, max_steps=100000, throw=False)
T = jnp.linspace(0.0, 12.0, 100)
FORCING = dr.VelocityGradient.from_components(grad_u_12=lambda t: jnp.sin(t))


def build(name, params):
    p = {k: float(v) for k, v in params.items()}
    return LOOKUP[name](**p)


def simulate(name, params):
    m = build(name, params)
    rheo = dr.VirtualRheometer.setup(m, "strain_rate_response", solver)
    ys = np.asarray(rheo.run_experiment(m, FORCING, T, jnp.zeros((3, 3))).data)
    return ys[:, 0, 1], ys[:, 0, 0] - ys[:, 1, 1]          # sigma12, N1


def r2(p, d):
    sse = np.sum((p - d) ** 2); sst = np.sum((d - d.mean()) ** 2)
    return float(1 - sse / sst) if np.all(np.isfinite(p)) and sst > 1e-12 else float("nan")


def load_runs():
    runs = []
    for f in RES.glob("results_l2_*.json"):
        lines = [json.loads(l) for l in open(f) if l.strip()]
        for rec in lines[1:]:
            if rec.get("best_params") is None:
                continue
            runs.append(rec)
    return runs


if __name__ == "__main__":
    runs = load_runs()
    print(f"{len(runs)} runs with a selected model")
    rows = []
    for rec in runs:
        tname, sname = rec["model_name"], rec["best_model_name"]
        if tname not in VE:                                # N1 ~ 0 for GN truth; skip
            continue
        try:
            s_t, n1_t = simulate(tname, rec["model_parameters"])
            s_s, n1_s = simulate(sname, rec["best_params"])
        except Exception:
            continue
        if not (np.all(np.isfinite(n1_t)) and np.all(np.isfinite(n1_s))):
            continue
        rows.append({"true": tname, "sel": sname, "correct": tname == sname,
                     "n1_r2": r2(n1_s, n1_t), "sig_r2": r2(s_s, s_t),
                     "t": T, "n1_t": n1_t, "n1_s": n1_s})
    print(f"{len(rows)} VE-truth runs evaluated")

    # ---- aggregate: N1 recovery for the selected model ----
    corr = [r["n1_r2"] for r in rows if r["correct"] and np.isfinite(r["n1_r2"])]
    inc = [r["n1_r2"] for r in rows if not r["correct"] and np.isfinite(r["n1_r2"])]
    print(f"selected model recovers withheld N1:  correct-selection median R2 = {np.median(corr):.3f} (n={len(corr)})"
          f"   incorrect-selection median R2 = {np.median(inc) if inc else float('nan'):.3f} (n={len(inc)})")
    import pickle
    cache = Path(__file__).resolve().parent / "gel" / "results" / "paper_figures" / "synthetic_n1.pkl"
    pickle.dump({"corr": corr, "inc": inc}, open(cache, "wb"))
    print("wrote", cache)

    # ---- figure: example transferability panels (selected model's N1 vs true N1) ----
    def pick(pred):  # first run matching predicate with finite, well-scaled N1
        cand = [r for r in rows if pred(r) and np.isfinite(r["n1_r2"]) and np.std(r["n1_t"]) > 1e-3]
        return sorted(cand, key=lambda r: -r["n1_r2"])
    panels = []
    for tn in ["OldroydB", "Giesekus", "LinearPTT"]:
        c = pick(lambda r, tn=tn: r["true"] == tn and r["correct"])
        if c: panels.append(c[0])
    for tn in ["Giesekus", "LinearPTT"]:                   # misspecified-selection cases
        c = pick(lambda r, tn=tn: r["true"] == tn and not r["correct"])
        if c: panels.append(c[0])
    panels = panels[:6]

    ncol = 3; nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax, r in zip(axes.ravel(), panels):
        ax.plot(r["t"], r["n1_t"], color="black", lw=2, label=f"true ({r['true']})")
        tag = "selected" if r["correct"] else "selected, mis-ID"
        ax.plot(r["t"], r["n1_s"], color="C1", ls="--", lw=1.8, label=f"{r['sel']} ({tag})")
        ax.set_title(rf"{r['true']} $\to$ {r['sel']}: $N_1$ $R^2={r['n1_r2']:.2f}$, $\sigma_{{12}}$ $R^2={r['sig_r2']:.2f}$")
        ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$N_1$"); ax.legend()
    fig.suptitle(r"Synthetic sweep: withheld $N_1$ recovered by the BIC-selected model (fit on $\sigma_{12}$ only)")
    fig.tight_layout()
    out = OUT / "synthetic_n1_recovery.png"
    fig.savefig(out, **SAVE); plt.close(fig)
    print("wrote", out)

    # ---- summary scatter: selected-model N1 R2, correct vs incorrect ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lab, vals, c in [("correct selection", corr, "C2"), ("incorrect selection", inc, "C3")]:
        if vals:
            ax.hist(np.clip(vals, -1, 1), bins=np.linspace(-1, 1, 21), alpha=0.6,
                    label=rf"{lab} (med {np.median(vals):.2f})", color=c)
    ax.set_xlabel(r"withheld $N_1$ $R^2$ (selected model vs.\ true)"); ax.set_ylabel("count")
    ax.set_title(r"$N_1$ recovery by the BIC-selected model across the synthetic sweep"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "synthetic_n1_recovery_hist.png", **SAVE); plt.close(fig)
    print("wrote", OUT / "synthetic_n1_recovery_hist.png")
