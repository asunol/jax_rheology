"""Figure SN2 (supplement S8-S9) -- training convergence and model extraction.

Panels a-c come from the three Giesekus schedules' own ``progress.csv``; d and
e come from the publication BIC battery.  Panel d needs each candidate's
predicted stress trace, which the battery does not store, so it is replayed
forward-only by ``paper_figs/derive_aos.py`` and cached; the replay reproduces
the battery's recorded MSE and BIC (see that script).
"""
from __future__ import annotations

import numpy as np

from . import data_paths as dp
from . import loaders as ld
from . import panels as pn
from . import style as st

FIG = "SN2"

SCHEDULES = ld.GIE_SCHEDULES
SCHEDULE_LABEL = ld.GIE_SCHEDULE_LABEL
SCHEDULE_COLOR = dict(zip(SCHEDULES, st.C_SCHEDULE))
#: Schedule curves carry the paper line weight with a little extra on top.
SCHEDULE_LW = st.PAPER_LINE["linewidth"] * 1.2

AOS_TARGET = "gie_A_s4"
#: Legs of the AOS protocol drawn in SN2d: strain-rate amplitude 1 and 10 at
#: omega = 1, the pair where the rejected families depart most visibly.
AOS_LEGS = (6, 7)
AOS_CYCLES = 2


def _lw(scale, k=1.0):
    return k * st.BASE_LINEWIDTH * scale


def _accepted(run: str) -> dict:
    return ld.accepted_progress(ld.gie_progress(run))


def notes() -> str:
    """Caption material for panels a and b, kept out of the panels."""
    return "\n".join([
        "Figure SN2, panels (a) and (b) -- Giesekus training curves.",
        "",
        "Curriculum training alternates blocks of scalar optimisation with",
        "blocks of network optimisation.  During a scalar block the optimiser",
        "records trial evaluations that are not adopted, so the raw",
        "progress.csv oscillates by orders of magnitude.  Both panels plot the",
        "accepted state at the end of each scalar / network block, not the raw",
        "per-evaluation trace.",
        "",
        "In (b) the dashed black line in each cell is the ground-truth value:",
        f"G_p = {ld.GIE_TRUTH['Gp']:g}, lambda = {ld.GIE_TRUTH['lam']:g}, "
        f"eta_s = {ld.GIE_TRUTH['nu_s']:g}, alpha = {ld.GIE_TRUTH['alpha']:g}.",
        "",
        "Circles mark the BIC battery readback: the Giesekus family refitted",
        "to the finished network's stress response.  That refit is independent",
        "of the trained scalars, so it need not land where the curve ends.",
        "The battery reports eta_p rather than G_p; G_p = eta_p / lambda.",
        "",
        "Only G_p, lambda and eta_s are trained; progress.csv holds no alpha",
        "column, because the network carries the Giesekus nonlinearity",
        "implicitly.  Alpha therefore appears as a readback only.",
        "",
        f"  {'schedule':<12}{'scalar':<8}{'trained':>12}{'readback':>12}"
        f"{'truth':>10}",
        *(f"  {SCHEDULE_LABEL[r]:<12}{name:<8}"
          f"{ld.gie_checkpoint_scalars()[r][key]:>12.6f}"
          f"{ld.gie_battery_scalars(r)[key]:>12.6f}"
          f"{ld.GIE_TRUTH[key]:>10.2f}"
          if key != "alpha" else
          f"  {SCHEDULE_LABEL[r]:<12}{name:<8}{'--':>12}"
          f"{ld.gie_battery_scalars(r)[key]:>12.6f}"
          f"{ld.GIE_TRUTH[key]:>10.2f}"
          for r in SCHEDULES
          for key, name in (("Gp", "G_p"), ("lam", "lambda"),
                            ("nu_s", "eta_s"), ("alpha", "alpha"))),
        "",
        "Sources:",
        *(f"  {SCHEDULE_LABEL[r]}: {dp.path(f'gie_progress_{r}')}"
          for r in SCHEDULES),
        "",
        "AUTHOR REFERENCE -- schedule mapping (internal names must not appear",
        "in any figure or caption):",
        *(f"  {SCHEDULE_LABEL[r]}  =  {r}  "
          f"final accepted loss {ld.accepted_progress(ld.gie_progress(r))['loss'][-1]:.6g}"
          for r in SCHEDULES),
        "  Training 1 = gie_A_s1, Training 2 = gie_A_s1b, Training 3 = gie_A_s4.",
        "",
        "SPREAD DENOMINATOR.  The three curves on SN2a/SN2b are across",
        "TRAINING SCHEDULES at one fixed network initialisation.  That is a",
        "different quantity from the SN4d +- (five seeds at one fixed",
        "schedule).  Do not compare them as like.",
        "",
    ])


def write_notes():
    out = dp.out_dir(FIG) / "SN2_notes.txt"
    out.write_text(notes())
    print(f"[SN2_notes] {out}", flush=True)
    return out


# --------------------------------------------------------------------------
# a -- loss vs iteration
# --------------------------------------------------------------------------

def plot_SN2a(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 8.0)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    for run in SCHEDULES:
        p = _accepted(run)
        ax.plot(p["step"], p["loss"], "-", color=SCHEDULE_COLOR[run],
                lw=SCHEDULE_LW * scale, label=SCHEDULE_LABEL[run])
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    st.legend(ax, scale, loc="upper right", ncol=1)
    out = pn.finish(fig, ax, FIG, "SN2a", save, dpi, own)
    if save:
        write_notes()
    return out


# --------------------------------------------------------------------------
# b -- Gp, lam, nu_s vs iteration
# --------------------------------------------------------------------------

#: The archives call the solvent viscosity nu_s; the paper writes eta_s.
SCALARS = (("Gp", r"$G_p$"), ("lam", r"$\lambda$"), ("nu_s", r"$\eta_s$"))


def _readback_points(a, scale, key):
    """Mark the BIC battery's readback of one scalar, per schedule.

    The battery refits the Giesekus family to the finished network's stress
    response, so its scalars need not agree with the trained ones the curves
    end on; plotting both at the same iteration makes the gap visible.
    """
    for run in SCHEDULES:
        last = float(_accepted(run)["step"][-1])
        a.plot([last], [ld.gie_battery_scalars(run)[key]], "o",
               color=SCHEDULE_COLOR[run], ms=14 * scale, mec="black",
               mew=st.NOTEBOOK_EFFECTIVE["marker_edge_width"] * scale,
               zorder=4)


def _alpha_axes(a, scale, thick):
    """Fourth cell: alpha, which has no training trajectory.

    progress.csv carries only Gp, lam and nu_s.  The network represents the
    Giesekus nonlinearity implicitly, so alpha exists only as a readback.
    """
    a.axhline(ld.GIE_TRUTH["alpha"], color=st.C_TRUTH, ls="--",
              lw=thick * 0.7, zorder=1)
    _readback_points(a, scale, "alpha")
    a.set_ylabel(r"$\alpha$")
    a.set_ylim(0.2948, 0.3026)
    pn.annotate(a, 0.03, 0.95, "not a trained scalar:\nreadback only", scale,
                va="top", color="0.35")


def _scalar_legend(a, scale, readback: bool):
    handles, labels = a.get_legend_handles_labels()
    if readback:
        from matplotlib.lines import Line2D

        handles.append(Line2D(
            [], [], ls="none", marker="o", ms=11 * scale, mfc="0.75",
            mec="black",
            mew=st.NOTEBOOK_EFFECTIVE["marker_edge_width"] * scale))
        labels.append("BIC readback")
    st.legend(a, scale, handles=handles, labels=labels, loc="upper right",
              ncol=1)


def plot_SN2b(ax=None, save=True, dpi=None, layout="2x2"):
    """Scalar trajectories.

    ``layout='2x2'`` adds the alpha readback as a fourth cell; ``'3x1'`` is the
    three trained scalars stacked, saved beside it as ``SN2b_3x1`` so the two
    can be compared.
    """
    grid = {"2x2": (2, 2), "3x1": (3, 1)}[layout]
    name = "SN2b" if layout == "2x2" else f"SN2b_{layout}"
    own = ax is None
    if own:
        size = (15.0, 10.5) if layout == "2x2" else (10.0, 13.0)
        fig, axs, scale = pn.new_stack(
            grid[0], *size, ncols=grid[1], sharex=True,
            axes_width=6.0 if layout == "2x2" else None,
            gridspec_kw=dict(hspace=0.14, wspace=0.26))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, *grid, sharex=True, hspace=0.14, wspace=0.26)
        scale = pn.adopt(axs[0])
        for a in axs[1:]:
            pn.adopt(a)

    thick = SCHEDULE_LW * scale
    for a, (key, label) in zip(axs, SCALARS):
        a.axhline(ld.GIE_TRUTH[key], color=st.C_TRUTH, ls="--",
                  lw=thick * 0.7, zorder=1)
        for run in SCHEDULES:
            p = _accepted(run)
            a.plot(p["step"], p[key], "-", color=SCHEDULE_COLOR[run],
                   lw=thick, label=SCHEDULE_LABEL[run])
        a.set_ylabel(label)
    axs[0].set_ylim(0.9, 6.6)
    axs[1].set_ylim(0.25, 1.05)
    axs[2].set_ylim(0.6, 1.05)
    if len(axs) > 3:
        for a, (key, _) in zip(axs, SCALARS):
            _readback_points(a, scale, key)
        _alpha_axes(axs[3], scale, thick)
        for a in axs[2:]:
            a.set_xlabel("Iteration")
    else:
        axs[-1].set_xlabel("Iteration")
    _scalar_legend(axs[0], scale, readback=len(axs) > 3)

    if own:
        pn.tidy(fig)
        if save:
            print(f"[{name}] {pn.save_panel(fig, FIG, name, dpi)}", flush=True)
            write_notes()
        return fig
    return axs


def plot_SN2b_3x1(ax=None, save=True, dpi=None):
    return plot_SN2b(ax=ax, save=save, dpi=dpi, layout="3x1")


# --------------------------------------------------------------------------
# c -- eta_p = Gp lam vs iteration.  RETIRED: eta_p is the product of the two
# scalars panel b already tracks, so it carries nothing new.  Kept callable;
# no longer part of the assembled figure or the index.
# --------------------------------------------------------------------------

def plot_SN2c(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 8.0)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    ax.axhline(ld.GIE_TRUTH["eta_p"], color=st.C_TRUTH, ls="--",
               lw=_lw(scale, 0.9),
               label=rf"truth $\eta_p={ld.GIE_TRUTH['eta_p']:.2f}$")
    for run in SCHEDULES:
        p = _accepted(run)
        ax.plot(p["step"], p["eta_p"], "-", color=SCHEDULE_COLOR[run],
                lw=_lw(scale), label=SCHEDULE_LABEL[run])
    ax.set_xlabel("optimiser step")
    ax.set_ylabel(r"$\eta_p = G_p\lambda$")
    ax.set_ylim(0.6, 3.4)
    st.legend(ax, scale, loc="lower right", ncol=2, columnspacing=1.2)
    return pn.finish(fig, ax, FIG, "SN2c", save, dpi, own)


# --------------------------------------------------------------------------
# d -- AOS stress response of the trained TBNN, candidate fits overlaid
# --------------------------------------------------------------------------

AOS_STYLE = {
    "Giesekus": (st.C_LEARN, "-", 1.15),
    "LinearPTT": (st.C_DRIVE[2], "--", 1.0),
    "FENEPConformation": ("#56B4E9", "-.", 1.0),
    "OldroydB": (st.C_DRIVE[1], ":", 1.0),
    "Newtonian": ("0.45", (0, (1, 1)), 1.0),
}


def plot_SN2d(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(2, 12.0, 11.0,
                                       gridspec_kw=dict(hspace=0.32))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 2, 1, hspace=0.32)
        scale = pn.adopt(axs[0])
        pn.adopt(axs[1])

    aos = ld.battery_aos(AOS_TARGET)
    for a, leg in zip(axs, AOS_LEGS):
        t = aos["time"][leg]
        omega = float(aos["legs"][leg]["omega"])
        amp = float(aos["legs"][leg]["f"])
        window = t <= AOS_CYCLES * 2 * np.pi / omega
        a.plot(t[window], aos["sigma_noisy"][leg][window], "o",
               color="0.15", ms=6 * scale, mfc="none",
               mew=1.1 * scale, label="TBNN" if leg == AOS_LEGS[0] else None,
               zorder=2)
        for fam in ld.FAMILY_ORDER:
            key = f"pred_{fam}"
            if key not in aos:
                continue
            color, ls, k = AOS_STYLE[fam]
            a.plot(t[window], aos[key][leg][window], ls=ls, color=color,
                   lw=_lw(scale, k), zorder=3,
                   label=ld.FAMILY_LABEL[fam] if leg == AOS_LEGS[0] else None)
        a.set_ylabel(r"$\sigma_{12}$")
        pn.annotate(a, 0.015, 0.04,
                    rf"$\dot\gamma_0={amp:g}$, $\omega={omega:g}$", scale,
                    ha="left", va="bottom", color="0.05")
    axs[-1].set_xlabel(r"$t$")
    st.legend(axs[0], scale, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.30), columnspacing=1.0,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.8)

    if own:
        pn.tidy(fig)
        if save:
            print(f"[SN2d] {pn.save_panel(fig, FIG, 'SN2d', dpi)}", flush=True)
        return fig
    return axs


# --------------------------------------------------------------------------
# e -- per-restart spread of the readback.  RETIRED alongside c; kept callable.
# --------------------------------------------------------------------------

def restart_alphas() -> dict:
    """alpha from each of the three restarts of the Giesekus candidate fit."""
    return {run: ld.battery_restart_spread(run, "Giesekus")
            for run in SCHEDULES}


def plot_SN2e(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(
            2, 10.0, 11.0, sharex=True,
            gridspec_kw=dict(height_ratios=[1.4, 1.0], hspace=0.10))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 2, 1, sharex=True, hspace=0.10,
                         height_ratios=[1.4, 1.0])
        scale = pn.adopt(axs[0])
        pn.adopt(axs[1])

    spread = restart_alphas()
    x = np.arange(len(SCHEDULES), dtype=float)
    seed_marker = {101: "o", 202: "s", 303: "^"}
    offsets = {101: -0.13, 202: 0.0, 303: 0.13}

    axs[0].axhline(ld.GIE_TRUTH["alpha"], color=st.C_TRUTH, ls="--",
                   lw=_lw(scale, 0.9), label=r"truth $\alpha=0.30$")
    seen = set()
    per_scheme = []
    for i, run in enumerate(SCHEDULES):
        s = spread[run]
        alphas = [float(p["alpha"]) for p in s["params"]]
        per_scheme.append(alphas)
        for seed, a_ in zip(s["seeds"], alphas):
            axs[0].plot(x[i] + offsets[seed], a_, seed_marker[seed],
                        color=st.C_LEARN, ms=13 * scale, mec="black",
                        mew=st.NOTEBOOK_EFFECTIVE["marker_edge_width"] * scale,
                        label=None if seed in seen else f"seed {seed}")
            seen.add(seed)
    axs[0].set_ylabel(r"$\alpha$ (readback)")
    axs[0].set_ylim(0.2948, 0.3024)
    st.legend(axs[0], scale, loc="upper left", ncol=1, columnspacing=1.0)

    within = np.array([max(a) - min(a) for a in per_scheme])
    across = float(np.std([np.mean(a) for a in per_scheme], ddof=1))
    axs[1].bar(x, within, width=0.5, color=st.C_LEARN, edgecolor="black",
               linewidth=1.2 * scale, zorder=3)
    axs[1].axhline(across, color=st.C_TRUTH, ls="--", lw=_lw(scale, 0.9),
                   zorder=4)
    axs[1].set_yscale("log")
    axs[1].set_ylim(1e-8, 1e-1)
    axs[1].set_yticks([1e-7, 1e-5, 1e-3])
    axs[1].set_ylabel(r"$\alpha$ spread")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([SCHEDULE_LABEL[r] for r in SCHEDULES])
    axs[1].set_xlabel("training schedule")
    axs[1].set_xlim(-0.45, len(SCHEDULES) - 0.55)
    pn.annotate(axs[1], 0.02, 0.86,
                f"across schemes, s.d. $={across:.1e}$", scale, color="0.2")
    pn.annotate(axs[1], 0.02, 0.50, "within scheme, 3 restarts", scale,
                color=st.C_LEARN)

    if own:
        pn.tidy(fig)
        if save:
            print(f"[SN2e] {pn.save_panel(fig, FIG, 'SN2e', dpi)}", flush=True)
        return fig
    return axs


# --------------------------------------------------------------------------
# assembled figure
# --------------------------------------------------------------------------

def plot_SN2(save=True, dpi=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20.0, 20.0))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28, left=0.06,
                          right=0.98, top=0.95, bottom=0.06)
    slots = ((gs[0, 0], plot_SN2a, "(a)"), (gs[0, 1], plot_SN2b, "(b)"),
             (gs[1, 0:2], plot_SN2d, "(c)"))
    with pn.uniform_scale(0.62) as scale:
        for spec, fn, tag in slots:
            ax = fig.add_subplot(spec)
            res = fn(ax=ax)
            first = res[0] if isinstance(res, list) else res
            pn.panel_tag(first, tag, scale, loc="outside")
    if save:
        print(f"[SN2_full] {pn.save_panel(fig, FIG, 'SN2_full', dpi)}",
              flush=True)
    return fig
