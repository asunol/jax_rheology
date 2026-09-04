"""Revised, more compact paper figures — rebuilt from the cached curves
(curves.pkl / identifiability.pkl / forward BIC json); NO experiments re-run.

Changes vs gel_composites.py:
  Rev Fig 1  rev_fig1_fits.png
     (a) single in-sample overlay: data + Oldroyd-B + White-Metzner + RUDE
         (sigma12 / N1 / Lissajous, all 4 amplitudes, 3 kPa held out). This
         merges old Fig 1(a) and Fig 2(a) — RUDE is no longer pulled out into
         its own figure.
     (b) old Fig 2(b): WM vs RUDE trained on {1,2} kPa, extrapolated to 4 kPa.
  Rev Fig 2  rev_fig2_bic.png     the BIC ranking on its own, compact.
  Rev Fig 3  rev_fig3_saos.png    old Fig 3(b) alone (SAOS moduli resolve the
         backbones). The old Fig 3(a) LAOS-overlap panel is dropped (stated in
         text). The SAOS-fixed backbone is relabelled "SAOS-fit" (not "WM").

Everything uses layout="constrained" so the Lissajous x-labels no longer
collide with the row above.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paper_style import apply as apply_style, SAVE, C as STY
apply_style()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import gel_data as gd

HERE = Path(__file__).resolve().parent
SRC = HERE / "results" / "paper_figures"
OUT = HERE / "results" / "paper_figures_revised"
OUT.mkdir(parents=True, exist_ok=True)
CURVES = SRC / "curves.pkl"
IDENT = SRC / "identifiability.pkl"
BIC = HERE / "results" / "forward" / "results.json"
G_M, TAU_M = gd.G_M, gd.TAU_M
ALL_KPA = [1.0, 2.0, 3.0, 4.0]
FIT_KPA = [1.0, 2.0, 4.0]
HELD = "#fbf3f3"


def panel_label(fig, s):
    fig.text(0.005, 0.99, s, fontsize=19, fontweight="bold", va="top", ha="left")


# ==================================================== Rev Fig 1: fits + extrap
def fig1(C):
    fig = plt.figure(figsize=(15, 13), layout="constrained")
    sa, sb = fig.subfigures(2, 1, height_ratios=[9.2, 3.6])

    # ---- (a) data + Oldroyd-B + White-Metzner + RUDE (Lennon), 3x4 ----
    ax = sa.subplots(3, 4)
    for j, kpa in enumerate(ALL_KPA):
        held = kpa not in FIT_KPA
        od, wl, le = C["oldroyd"][kpa], C["wm_learn"][kpa], C["lennon"][kpa]
        for r, (q, qd) in enumerate([("sig", "sig_d"), ("n1", "n1_d")]):
            a = ax[r, j]
            if held: a.set_facecolor(HELD)
            a.plot(od["t"], od[qd], ".", color=STY.DATA, ms=2.5)
            a.plot(od["t"], od[q], color=STY.OLDROYD, ls="--", lw=1.3)
            a.plot(le["t"], le[q], color=STY.RUDE, lw=1.2)
            a.plot(wl["t"], wl[q], color=STY.WM, lw=1.3)
        ax[0, j].set_title(rf"$\sigma_0 = {kpa:.0f}$ kPa" + ("  (held out)" if held else ""))
        ax[1, j].set_xlabel(r"$t/\tau_m$")
        # Lissajous (last ~cycle)
        a = ax[2, j]
        if held: a.set_facecolor(HELD)
        ss = od["t"] >= (od["t"].max() - 22.6)
        a.plot(od["gamma"][ss], od["sig_d"][ss], ".", color=STY.DATA, ms=2.5)
        a.plot(od["gamma"][ss], od["sig"][ss], color=STY.OLDROYD, ls="--", lw=1.1)
        a.plot(le["gamma"][ss], le["sig"][ss], color=STY.RUDE, lw=1.0)
        a.plot(wl["gamma"][ss], wl["sig"][ss], color=STY.WM, lw=1.1)
        a.set_xlabel(r"$\gamma$")
    ax[0, 0].set_ylabel(r"$\sigma_{12}/G_m$")
    ax[1, 0].set_ylabel(r"$N_1/G_m$")
    ax[2, 0].set_ylabel(r"$\sigma_{12}/G_m$")
    handles = [Line2D([0], [0], marker=".", ls="", color=STY.DATA, label="data"),
               Line2D([0], [0], color=STY.OLDROYD, ls="--", label="Oldroyd-B"),
               Line2D([0], [0], color=STY.RUDE, label="RUDE (Lennon)"),
               Line2D([0], [0], color=STY.WM, label="White-Metzner")]
    sa.legend(handles=handles, loc="outside upper center", ncol=4)
    panel_label(sa, "(a)")

    # ---- (b) extrapolation break: WM vs RUDE trained on {1,2}, predict 4 ----
    ax = sb.subplots(2, 2)
    for j, kpa in enumerate([2.0, 4.0]):
        wmv, rdv = C["wm_12"][kpa], C["rude_12"][kpa]
        extrap = kpa not in (1.0, 2.0)
        for r, (q, qd, pad) in enumerate([("sig", "sig_d", 0.5), ("n1", "n1_d", 0.8)]):
            a = ax[r, j]
            if extrap: a.set_facecolor(HELD)
            a.plot(wmv["t"], wmv[qd], ".", color=STY.DATA, ms=2.5)
            a.plot(rdv["t"], rdv[q], color=STY.RUDE, lw=1.4)
            a.plot(wmv["t"], wmv[q], color=STY.WM, lw=1.4)
            lo, hi = np.nanmin(wmv[qd]), np.nanmax(wmv[qd]); d = pad * (hi - lo)
            a.set_ylim(lo - d, hi + d)
        ax[0, j].set_title(rf"$\sigma_0 = {kpa:.0f}$ kPa" + ("  (unseen)" if extrap else ""))
        ax[1, j].set_xlabel(r"$t/\tau_m$")
    ax[0, 0].set_ylabel(r"$\sigma_{12}/G_m$")
    ax[1, 0].set_ylabel(r"$N_1/G_m$")
    handles = [Line2D([0], [0], marker=".", ls="", color=STY.DATA, label="data"),
               Line2D([0], [0], color=STY.RUDE, label="RUDE (retrained)"),
               Line2D([0], [0], color=STY.WM, label="White-Metzner"),
               Line2D([0], [0], color="none", label="trained on 1, 2 kPa")]
    sb.legend(handles=handles, loc="outside upper center", ncol=4)
    panel_label(sb, "(b)")

    fig.savefig(OUT / "rev_fig1_fits.png", **SAVE); plt.close(fig)
    print("wrote", OUT / "rev_fig1_fits.png")


# ==================================================== Rev Fig 2: BIC ranking
def fig2_bic():
    fig, bx = plt.subplots(figsize=(6.8, 3.8), layout="constrained")
    data = json.load(open(BIC))
    EXCLUDE = {"MultiMode2", "MultiMode3"}
    LAB = {"WhiteMetzner": "White-Metzner", "ExponentialPTT": "Exp.\\ PTT", "LinearPTT": "Linear PTT",
           "Giesekus": "Giesekus", "XPomPom": "XPom-Pom", "OldroydB": "Oldroyd-B",
           "FENECR": "FENE-CR", "FENEP": "FENE-P"}
    rows = [(r["name"], r["bic"]) for r in data
            if r.get("name") not in EXCLUDE and r.get("bic") is not None
            and np.isfinite(r.get("bic", np.nan))]
    rows.sort(key=lambda x: x[1]); best = rows[0][1]
    y = np.arange(len(rows))[::-1]
    bx.barh(y, [b - best for _, b in rows],
            color=[STY.WM if n == "WhiteMetzner" else "0.55" for n, _ in rows], height=0.72)
    bx.set_yticks(y); bx.set_yticklabels([LAB.get(n, n) for n, _ in rows])
    bx.set_xlabel(r"$\Delta\mathrm{BIC} = \mathrm{BIC} - \mathrm{BIC}_{\min}$")
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    fig.savefig(OUT / "rev_fig2_bic.png", **SAVE); plt.close(fig)
    print("wrote", OUT / "rev_fig2_bic.png")


# ==================================================== Rev Fig 3: SAOS moduli
def fig3_saos(R):
    _d = np.loadtxt(gd.DATA_DIR / "gel_saos_1.csv", delimiter=",")
    W, GP, GPP = _d[2:, 3], _d[2:, 0] / G_M, _d[2:, 1] / G_M
    # relabelled: SAOS-fixed backbone -> "SAOS-fit" (not "WM")
    sty = {"saos": (STY.WM, "-", "SAOS-fit"),
           "laos": ("C0", "--", "WM (LAOS-only)"),
           "joint": ("C2", "-.", "WM (SAOS+LAOS)"),
           "rude": (STY.RUDE, ":", "RUDE (LAOS-only)")}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    wv = np.logspace(np.log10(W.min()), np.log10(W.max()), 200)
    for a, (dat, idx, ylab) in zip(ax, [(GP, 0, r"$G'$ (Pa)"), (GPP, 1, r"$G''$ (Pa)")]):
        a.loglog(W, dat * G_M, "o", color=STY.DATA, mfc="none", ms=5)
        for k in ["saos", "laos", "joint", "rude"]:
            col, ls, _ = sty[k]
            wl = wv * (R[k]["lam"] * TAU_M); G = R[k]["G"]
            gp = G * wl ** 2 / (1 + wl ** 2); gpp = G * wl / (1 + wl ** 2)
            a.loglog(wv, (gp if idx == 0 else gpp) * G_M, color=col, ls=ls, lw=1.6)
        a.set_xlabel(r"$\omega$ (rad/s)"); a.set_ylabel(ylab)
    handles = [Line2D([0], [0], marker="o", ls="", mfc="none", color=STY.DATA, label="SAOS data")] + \
              [Line2D([0], [0], color=sty[k][0], ls=sty[k][1], label=sty[k][2])
               for k in ["saos", "laos", "joint", "rude"]]
    fig.legend(handles=handles, loc="outside upper center", ncol=5)
    fig.savefig(OUT / "rev_fig3_saos.png", **SAVE); plt.close(fig)
    print("wrote", OUT / "rev_fig3_saos.png")


def main():
    C = pickle.load(open(CURVES, "rb"))
    R = pickle.load(open(IDENT, "rb"))
    fig1(C)
    fig2_bic()
    fig3_saos(R)
    print("[done] ->", OUT)


if __name__ == "__main__":
    main()
