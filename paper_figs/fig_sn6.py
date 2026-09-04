"""Figure SN6 (supplement S13) -- FENE-P direct calibration.

Same single-rate contraction data as SN4 -- U = 0.5, velocity only -- but the
four FENE-P parameters are fitted directly by L-BFGS-B instead of being read
back out of a trained network, which makes the network-vs-direct comparison
like-for-like.  Four arbitrary starts land on the truth; the fifth (I4) is
excluded, with the evidence in :func:`i4_exclusion` and in SN6_notes.txt.

Two panels, deliberately not merged into one lettered grid: the parameter
group (2x2, one cell per fitted scalar) is the claim, and the loss panel is
corroboration of how deep the minimum is.  Parameters come first.

The optimiser works in log space -- ``z = log[Gp, lam, nu_s, Lsq]``, physical
values recovered by ``exp`` (``visco_opt_fenep_direct_contraction_run.py``
:290-324, :364-367) -- so the trajectories drawn here are the physical
parameters, not the variables L-BFGS-B moved.
"""
from __future__ import annotations

import numpy as np

from . import loaders as ld
from . import panels as pn
from . import style as st

FIG = "SN6"

#: I4 is excluded from every panel and every aggregate; see ``i4_exclusion``.
FITS = ("I1", "I2", "I3", "I5")
EXCLUDED = "I4"
FIT_COLOR = {"I1": st.C_DRIVE[0], "I2": st.C_DRIVE[1], "I3": st.C_WINNER,
             "I4": st.C_LEARN, "I5": st.C_DRIVE[2]}
#: Legend names.  The archives number the starts I1..I5 with I4 excluded; the
#: figure numbers the four drawn starts 1..4 in that order, so the reader is
#: not left looking for a missing fourth.  The mapping is in SN6_notes.txt.
FIT_LABEL = {f: f"Initialization {i}" for i, f in enumerate(FITS, start=1)}
FIT_LABEL[EXCLUDED] = "Initialization I4 (excluded)"
#: The four fitted scalars, in the order the 2x2 grid reads.  ``nu_s`` is the
#: archive's name for the solvent viscosity; the figure calls it eta_s.
PARAMS = (("Gp", r"$G_p$", 3.2),
          ("lam", r"$\lambda$", 0.7),
          ("nu_s", r"$\eta_s$", 0.8),
          ("Lsq", r"$L^2$", 12.0))
#: Per-cell y limits.  Set individually: the four scalars span different
#: ranges and a shared scale would flatten three of them.
PARAM_YLIM = {"Gp": (1.5, 8.4), "lam": (0.28, 1.62), "nu_s": (0.42, 1.86),
              "Lsq": (6.0, 53.0)}
#: L^2 spans 8 to 50 once I4 is out, well under one decade, so the cell is
#: linear like the other three; ``plot_SN6a(lsq_log=True)`` renders the log
#: alternative that was compared against it.
LSQ_LOG_DEFAULT = False


def _lw(scale, k=1.0):
    return k * st.BASE_LINEWIDTH * scale


def trace(fit: str) -> dict:
    """Physical parameter trajectory of one start, plus the loss."""
    d = ld.direct_progress(fit)
    return {"nfev": d["nfev"], "loss": d["loss"], "Gp": d["Gp"],
            "lam": d["lam"], "nu_s": d["nu_s"], "Lsq": d["Lsq"],
            "eta_p": d["Gp"] * d["lam"]}


def summary_table(fits=FITS) -> list[dict]:
    """One row per start.  Finals come from summary.json, not from the last
    progress row: the CSV stores the parameters to 8 significant digits, which
    rounds the residual 1e-9 offsets away."""
    rows = []
    for f in fits:
        t = trace(f)
        s = ld.load_json(f"direct_summary_{f}")
        final = {k: float(s["final"][k]) for k, _, _ in PARAMS}
        rows.append({"fit": f, "nfev": int(s["nfev"]), "nit": int(s["nit"]),
                     "loss_init": float(s["loss_init"]),
                     "loss_final": float(s["loss_final"]),
                     "start": {k: float(t[k][0]) for k, _, _ in PARAMS},
                     "final": final,
                     "rel_err": {k: abs(final[k] - truth) / truth
                                 for k, _, truth in PARAMS}})
    return rows


def peterlin_f(Lsq: float, trA: float) -> float:
    """FENE-P Peterlin factor ``f = L^2/(L^2 - tr A)``.

    Transcribed from ``jax_rheology/log_conformation.py:568-593`` (READ-ONLY);
    the smooth floor there only acts within 0.1% of the wall, so far from it
    this is the exact factor.
    """
    return float(Lsq) / (float(Lsq) - float(trA))


def i4_exclusion() -> dict:
    """Everything on disk about the excluded I4 start.

    Three independent failures at the same start, and one physical reason why
    the start is outside the regime this figure is about.
    """
    s = ld.load_json("direct_summary_I4")
    probe = ld.load_json("direct_i4_u4_probe")
    trA_peak = ld.fene_summary("R3")["max_trA_truth"]
    lsq0 = float(s["init"]["Lsq"])
    return {
        "start": {k: float(v) for k, v in s["init"].items()
                  if k in ("Gp", "lam", "nu_s", "Lsq")},
        "single_rate": {"loss_init": float(s["loss_init"]),
                        "loss_final": float(s["loss_final"]),
                        "nfev": int(s["nfev"]),
                        "final": {k: float(v) for k, v in s["final"].items()},
                        "message": s["message"]},
        # Both dual-rate I4 jobs died; only the equal-weighted one gives a
        # clean diagnosis (finite primal, non-finite adjoint).  The legacy job
        # printed a corrupted truth norm before it hit CUDA_ERROR_ILLEGAL_-
        # ADDRESS, so its host log is not evidence about the I4 adjoint.
        "dual_rate": {
            "equal": {"jobid": "36139679", "loss": 3453.431,
                      "grad_finite": False, "grad_norm": "nan",
                      "exit": "[FATAL] non-finite loss/grad at init"},
            "legacy": {"jobid": "36139678",
                       "exit": "CUDA_ERROR_ILLEGAL_ADDRESS",
                       "host_log": "contaminated (truth su2 1.164e+57 before "
                                   "the I4 adjoint was reached)"},
        },
        "u4_probe": {"jobid": probe["job_id"], "U": probe["U"],
                     "loss": probe["loss"],
                     "grad_all_finite": probe["grad_all_finite"],
                     "grad": probe["grad_abs"]},
        "stiffness": {"Lsq": lsq0, "trA_peak_truth": float(trA_peak),
                      "f_at_peak": peterlin_f(lsq0, trA_peak),
                      "f_at_rest": peterlin_f(lsq0, 3.0),
                      "f_at_truth_Lsq": peterlin_f(12.0, trA_peak)},
    }


def loss_floor_spread() -> dict:
    """Final losses of the drawn starts, and whether they justify a floor line."""
    vals = {f: float(trace(f)["loss"][-1]) for f in FITS}
    lo, hi = min(vals.values()), max(vals.values())
    decades = float(np.log10(hi / lo))
    return {"final_loss": vals, "min": lo, "max": hi, "decades": decades,
            "draw_line": bool(decades <= 1.0)}


def _para(text: str, indent: str = "", hang: str = None) -> str:
    import textwrap

    return textwrap.fill(" ".join(text.split()), width=78,
                         initial_indent=indent,
                         subsequent_indent=hang if hang is not None
                         else " " * len(indent))


def notes() -> str:
    """Caption material for SN6: the fit, the floor, and the excluded start."""
    from .fig_sn5 import ladder_table

    fl = loss_floor_spread()
    i4 = i4_exclusion()
    stf = i4["stiffness"]
    sr = i4["single_rate"]
    rows = summary_table()
    drop = [np.log10(r["loss_init"] / r["loss_final"]) for r in rows]
    readback = [r["Lsq"] for r in ladder_table()]
    out = [_para(
        "Figure SN6 -- FENE-P direct calibration.  Two panels: (a) the four "
        "fitted scalars against iteration, (b) the loss.  They stay separate "
        "figures on purpose -- convergence in parameter space is the claim, "
        "the depth of the minimum is corroboration -- and (a) comes first."),
        ""]

    out += [_para(
        "THE X AXIS is labelled 'Objective evaluation'.  The plotted x is "
        "nfev, one row per objective evaluation, not the accepted-step count "
        "used on the TBNN training axes.  L-BFGS-B reports fewer accepted "
        "iterations than evaluations because its line "
        "search can call the objective more than once per step: nit is "
        + ", ".join(str(r["nit"]) for r in rows) + " against nfev "
        + ", ".join(str(r["nfev"]) for r in rows)
        + ".  Worth knowing if the caption quotes an iteration count."), ""]

    out += [_para(
        "WHAT IS FITTED.  Exactly four scalars: Gp, lam, eta_s (archive "
        "column nu_s) and L^2, driver visco_opt_fenep_direct_contraction_run.py "
        "lines 295-315.  There is no network and no fifth parameter.  eta_p "
        "is not fitted -- it is the product Gp*lam -- so the panel plots Gp "
        "itself."), ""]

    out += [_para(
        "LOG SPACE.  L-BFGS-B moves z = log[Gp, lam, eta_s, L^2]; the physical "
        "values come back through exp with floors at 1e-8 (and L^2 >= "
        "1 + 1e-6): driver lines 290-293, 313-324, 364-367.  Each progress.csv "
        "therefore carries both the physical columns and log_Gp, log_lam, "
        "log_nus, log_Lsq.  The panels plot the physical parameters; the log "
        "columns are the optimiser's own variables and are not drawn.  Worth "
        "one clause in the caption, because equal steps on the figure are not "
        "equal steps for the optimiser."), ""]

    out += [_para(
        "THE DATA.  Single-rate, U = 0.5, velocity only (w_p = 0 and "
        "pressure_on false in every summary.json), 128 x 256 grid with the "
        "same ROI weighting as the network runs.  That is the same data as "
        "SN4's representative network, so the network-vs-direct comparison is "
        "like-for-like.  Dual-rate direct fits do exist "
        "(fene8_direct/direct_dual_{legacy,equal}_I1..I5, eight of ten "
        "completed) but they belong to the SN5 condition ladder, not here."),
        ""]

    out += [_para(
        "STARTS DRAWN, and how they map to the archives.  The legend numbers "
        "them 1 to 4 in the order below; the run directories keep the campaign "
        "identifiers, and I4 is excluded (see the end of this file), so the "
        "figure's 4 is the archive's I5: "
        + ", ".join(f"{FIT_LABEL[f]} = fene8_direct/direct_u05_{f}"
                    for f in FITS)
        + ".  Truth is Gp = 3.2, lam = 0.7, eta_s = 0.8, L^2 = 12."), ""]
    for r in rows:
        s, fin = r["start"], r["final"]
        out.append(
            f"  {FIT_LABEL[r['fit']]} (archive {r['fit']}), {r['nfev']} "
            f"evaluations / {r['nit']} iterations")
        out.append(
            f"      start  Gp {s['Gp']:g}, lam {s['lam']:g}, "
            f"eta_s {s['nu_s']:g}, L^2 {s['Lsq']:g}")
        out.append(
            f"      final  Gp {fin['Gp']:.10f}, lam {fin['lam']:.10f}, "
            f"eta_s {fin['nu_s']:.10f},")
        out.append(f"             L^2 {fin['Lsq']:.8f}")
    worst = max(max(r["rel_err"].values()) for r in rows)
    out += ["", _para(
        "Every start recovers all four parameters to better than "
        f"{worst:.0e} relative (worst single parameter over the four runs), "
        "from starting points spread over Gp 2-7.9, lam 0.4-1.5, eta_s 0.5-1.0 "
        "and L^2 8-50."), ""]

    out += ["THE FLOOR.  Final loss of each start:"]
    for f in FITS:
        out.append(f"  {FIT_LABEL[f]}  {fl['final_loss'][f]:.6e}")
    out += ["", _para(
        f"These span {fl['min']:.2e} to {fl['max']:.2e}, "
        f"{fl['decades']:.1f} decades, so NO horizontal 'double-precision "
        "floor' line is drawn and the annotation that used to claim one has "
        "been removed.  Say it in words in the caption instead: each start "
        f"drops {min(drop):.0f} to {max(drop):.0f} orders of magnitude below "
        "its own initial loss and then stops where L-BFGS-B's "
        "relative-reduction test is met (ftol = 1e-12, gtol = 1e-10, driver "
        "line 366), which is a stopping rule, not a numerical floor.  As a "
        "residual rather than a squared loss that is sqrt(loss) = "
        f"{np.sqrt(fl['max']):.0e} down to {np.sqrt(fl['min']):.0e} on the "
        "ROI-weighted velocity mismatch."), ""]

    out += [_para(
        "AXIS SCALES.  Per cell, not shared: Gp 1.5-8.4, lam 0.28-1.62, "
        "eta_s 0.42-1.86, L^2 6-53, all four linear.  The L^2 cell was "
        "rendered both ways and the log version is kept as SN6a_lsq_log.jpg "
        "(plot_SN6a_lsq_log()).  Linear is the one used: with I4 gone the L^2 "
        "trajectories only span 8 to 50, well under one decade, so log "
        "separates the converged curves no better, squeezes the approach to "
        "12 into the middle of the cell, and labels its ticks 2x10^1, 3x10^1 "
        "where the reader wants a tick near the truth."), ""]

    out += [_para(
        "NAMING.  The solvent viscosity is eta_s (paper Eq. 6).  The "
        "archives still store the column as nu_s.  N1d, SN4d and SN4d_table "
        "now print eta_s as well; Table S2 still uses the archive name."), ""]

    out += [_para(
        f"EXCLUDED START I4 = (Gp {i4['start']['Gp']:g}, lam "
        f"{i4['start']['lam']:g}, eta_s {i4['start']['nu_s']:g}, L^2 "
        f"{i4['start']['Lsq']:g}), archive fene8_direct/direct_u05_I4.  "
        "Removed from both panels and from every aggregate, which is why the "
        "legend runs 1 to 4 over four archive identifiers that end at I5.  "
        "Three failures at that one start:"), ""]
    out += [_para(
        f"1. Single-rate: stalled at loss {sr['loss_final']:,.0f} after "
        f"{sr['nfev']} evaluations (from {sr['loss_init']:,.0f}), ending at Gp "
        f"{sr['final']['Gp']:.2f}, lam {sr['final']['lam']:.3f}, eta_s "
        f"{sr['final']['nu_s']:.3f}, L^2 {sr['final']['Lsq']:.1f} -- nowhere "
        "near the truth, with L-BFGS-B reporting convergence there.",
        indent="  ", hang="     ")]
    out += [_para(
        "2. Both dual-rate I4 fits failed.  The equal-weighted one (job "
        f"{i4['dual_rate']['equal']['jobid']}) printed a finite loss "
        f"{i4['dual_rate']['equal']['loss']:,.3f} with grad_finite = False and "
        "|g| = nan at the first value_and_grad, then exited '[FATAL] "
        "non-finite loss/grad at init'.  The legacy-weighted one (job "
        f"{i4['dual_rate']['legacy']['jobid']}) died on "
        "CUDA_ERROR_ILLEGAL_ADDRESS after printing a corrupted truth norm "
        "(su2 1.164e+57 against 313.7 on every healthy job), so that log is "
        "node contamination and is not evidence about the I4 adjoint: the "
        "clean non-finite-gradient diagnosis rests on the equal arm alone.",
        indent="  ", hang="     ")]
    out += [_para(
        f"3. A U = {i4['u4_probe']['U']:g} gradient probe at the I4 start (job "
        f"{i4['u4_probe']['jobid']}) returned a finite loss "
        f"{i4['u4_probe']['loss']:.3f} with all four parameter gradients NaN, "
        "which localises the non-finite adjoint to the high-rate component.",
        indent="  ", hang="     "), ""]
    out += [_para(
        "The physical reason, and why no rerun is planned.  The Peterlin "
        "factor is f = L^2/(L^2 - tr A) (jax_rheology/log_conformation.py:"
        f"568-593).  At the I4 start L^2 = {stf['Lsq']:g}, so even at the "
        f"truth's peak tr A = {stf['trA_peak_truth']:.2f} the fluid sits at "
        f"f = {stf['f_at_peak']:.3f}, and at rest f = {stf['f_at_rest']:.3f} "
        f"-- numerically Oldroyd-B, against f = {stf['f_at_truth_Lsq']:.2f} at "
        "the truth L^2 = 12.  On that plateau d(solution)/d(L^2) is ~0: the "
        "observable barely responds to the parameter being fitted.  It is also "
        "far outside anything a TBNN readback produces -- across the whole "
        f"FENE-P campaign the readbacks land L^2 between {min(readback):.1f} "
        f"and {max(readback):.1f} -- so the start does not test the claim this "
        "figure makes.  It stays in the archives and in i4_exclusion(); it is "
        "absent from the panels."), ""]

    out += [_para(
        "RETIRED PANEL.  The old third panel (recovered/truth per parameter) "
        "moved to retired/: with I4 out, all sixteen markers land on 1.0 and "
        "it shows nothing the parameter grid does not already show."), ""]
    return "\n".join(out)


def write_notes():
    out = ld.dp.out_dir(FIG) / "SN6_notes.txt"
    out.write_text(notes())
    print(f"[SN6_notes] {out}", flush=True)
    return out


# --------------------------------------------------------------------------
# a -- parameter trajectories, one cell per fitted scalar
# --------------------------------------------------------------------------

def plot_SN6a(ax=None, save=True, dpi=None, lsq_log=LSQ_LOG_DEFAULT,
              name="SN6a"):
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(
            2, 15.0, 10.5, ncols=2, sharex=True, axes_width=6.0,
            gridspec_kw=dict(hspace=0.14, wspace=0.26))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 2, 2, sharex=True, hspace=0.14, wspace=0.26)
        scale = pn.adopt(axs[0])
        for a in axs[1:]:
            pn.adopt(a)

    for a, (key, label, truth) in zip(axs, PARAMS):
        first = key == PARAMS[0][0]
        a.axhline(truth, color=st.C_TRUTH, ls=":", lw=_lw(scale, 1.0),
                  zorder=1, label="Truth" if first else None)
        for f in FITS:
            t = trace(f)
            a.plot(t["nfev"], t[key], "-", color=FIT_COLOR[f],
                   lw=_lw(scale, 1.1), label=FIT_LABEL[f] if first else None)
        a.set_ylabel(label)
        if key == "Lsq" and lsq_log:
            a.set_yscale("log")
            a.set_ylim(7.0, 60.0)
        else:
            a.set_ylim(*PARAM_YLIM[key])
    for a in axs[2:]:
        a.set_xlabel("Objective evaluation")
    st.legend(axs[0], scale, loc="upper right", ncol=2, columnspacing=0.8,
              handlelength=1.2,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.8)

    if own:
        pn.tidy(fig)
        if save:
            print(f"[{name}] {pn.save_panel(fig, FIG, name, dpi)}", flush=True)
            write_notes()
        return fig
    return axs


def plot_SN6a_lsq_log(ax=None, save=True, dpi=None):
    """The log-``L^2`` alternative that was compared against the linear cell."""
    return plot_SN6a(ax=ax, save=save, dpi=dpi, lsq_log=True,
                     name="SN6a_lsq_log")


# --------------------------------------------------------------------------
# b -- loss
# --------------------------------------------------------------------------

def plot_SN6b(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 8.0)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    for f in FITS:
        t = trace(f)
        ax.plot(t["nfev"], t["loss"], "-", color=FIT_COLOR[f],
                lw=_lw(scale, 1.1), label=FIT_LABEL[f])
    ax.set_yscale("log")
    ax.set_ylim(1e-19, 1e6)
    ax.set_yticks([1e-18, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e0, 1e3, 1e6])
    ax.set_xlabel("Objective evaluation")
    ax.set_ylabel("Loss")
    # No floor line: the four final losses span 8.9e-19 to 9.6e-14, five
    # decades, because L-BFGS-B stops on a relative-reduction test (ftol
    # 1e-12) wherever its line search happens to be, not at a fixed floor.
    # The numbers are in SN6_notes.txt for the caption.
    # Upper right: the curves all descend from the top left, so it is the only
    # empty corner once the labels are this long.
    st.legend(ax, scale, loc="upper right", ncol=1,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.8)
    return pn.finish(fig, ax, FIG, "SN6b", save, dpi, own)


# --------------------------------------------------------------------------
# c -- final parameters against truth.  RETIRED: with I4 out, all four starts
# agree with the truth to within 4e-9 relative, so every marker lands on the
# 1.0 line and the panel carries no information the parameter grid does not
# already show.  Kept callable (it still draws I4 if FITS is widened); no
# longer in the assembled figure or the index.
# --------------------------------------------------------------------------

def plot_SN6c(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 8.0)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    x = np.arange(len(PARAMS), dtype=float)
    ax.axhline(1.0, color=st.C_TRUTH, ls="--", lw=_lw(scale, 0.9),
               label="Truth", zorder=2)
    for j, f in enumerate(FITS):
        t = trace(f)
        vals = [t[k][-1] / truth for k, _, truth in PARAMS]
        ax.plot(x + (j - 1.5) * 0.12, vals, "o", color=FIT_COLOR[f],
                ms=11 * scale, mfc="none", mew=1.8 * scale, ls="none",
                zorder=4, label=FIT_LABEL[f])
    ax.set_yscale("log")
    ax.set_ylim(0.1, 20.0)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in PARAMS])
    ax.set_xlim(-0.5, len(PARAMS) - 0.5)
    ax.set_ylabel("recovered / truth")
    st.legend(ax, scale, loc="upper left", ncol=3, columnspacing=1.0,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.8)
    return pn.finish(fig, ax, FIG, "SN6c", save, dpi, own)


# --------------------------------------------------------------------------
# assembled figure
# --------------------------------------------------------------------------

def plot_SN6(save=True, dpi=None):
    """Preview of the two panels side by side, parameters first.

    The panels stay separate figures in the manuscript -- parameter-space
    convergence and depth of the minimum are different claims -- so this is a
    layout preview, not a merged five-cell grid.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(26.0, 10.0))
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.05, right=0.98, top=0.93,
                          bottom=0.12)
    slots = ((gs[0, 0:2], plot_SN6a, "(a)"), (gs[0, 2], plot_SN6b, "(b)"))
    with pn.uniform_scale(0.60) as scale:
        for spec, fn, tag in slots:
            ax = fig.add_subplot(spec)
            res = fn(ax=ax)
            first = res[0] if isinstance(res, list) else res
            pn.panel_tag(first, tag, scale, loc="outside")
    if save:
        print(f"[SN6_full] {pn.save_panel(fig, FIG, 'SN6_full', dpi)}",
              flush=True)
    return fig
