"""Figure SN5 (supplement S12) -- FENE-P condition ladder.

Four training conditions, one representative network each, chosen inside the
condition by the SN4 criterion (lowest mean absolute relative error over
eta_p, lam, nu_s, L^2).  The grouping is: the
headline single-rate runs, the original dual-rate legacy-weighted arm, the
equal-weight ("balanced") experimental arm -- which is dual-rate *and*
pressure-observing -- and the single-rate pressure ablation.

Panel (b) groups the runs by drive, so that "+pressure" reads as the single
rate U = 0.5 plus pressure that it is; panel (c) asks why L^2 is the parameter
that will not come back, and answers it with the state distribution the truth
fields actually cover relative to the Peterlin pole.

Panel d of the brief (legacy vs balanced loss curves) is deliberately absent;
see :func:`optional_panel_d_evidence` for the numbers behind that decision.
"""
from __future__ import annotations

import json

import numpy as np

from . import geometry as gm
from . import loaders as ld
from . import panels as pn
from . import style as st

FIG = "SN5"

CONDITIONS = (
    ("single-U", ("R3", "fene8_u05_s1", "fene8_u05_s2", "fene8_u05_s3",
                  "fene8_u05_s4")),
    ("dual-U legacy", ("R1", "R2", "R5", "R6", "R7", "fene8_leg_s3",
                       "fene8_leg_s4")),
    ("dual-U balanced", ("fene8_bal_s0", "fene8_bal_s1", "fene8_bal_s2",
                         "fene8_bal_s3", "fene8_bal_s4")),
    ("+pressure", ("R4",)),
)
COND_LABEL = {"single-U": "single-$U$", "dual-U legacy": "dual-$U$\nlegacy",
              "dual-U balanced": "dual-$U$\nbalanced",
              "+pressure": "$+$pressure"}   # retired: superseded by DRIVE_LABEL
#: Tick labels that name the *drive* each condition trains on, so that
#: "+pressure" cannot be read as dual-rate plus pressure: it is the single rate
#: U = 0.5 with the wall-tap pressure observable added.
DRIVE_LABEL = {"single-U": r"$U = 0.5$",
               "dual-U legacy": r"$U = 0.5,\,4$ (legacy)",
               "dual-U balanced": r"$U = 0.5,\,4$",
               "+pressure": r"$U = 0.5$ $+$ pressure"}
#: The same labels as plain text, for the notes file.
DRIVE_TEXT = {"single-U": "U = 0.5", "dual-U legacy": "U = 0.5, 4 (legacy)",
              "dual-U balanced": "U = 0.5, 4",
              "+pressure": "U = 0.5 + pressure"}
#: Conditions drawn in the panels.  The legacy dual-rate arm stays in the
#: inventory (Table S2, and ``optional_panel_d_evidence``) but is out of the
#: figure: its weighting is superseded by the balanced arm, which is the
#: comparison panel c is about.
PLOT_CONDITIONS = tuple(c for c in CONDITIONS if c[0] != "dual-U legacy")
#: One line each: with the legacy arm gone there are three groups, and a
#: wrapped tick label put "balanced" under the wrong group.
PLOT_LABEL = [DRIVE_LABEL[c] for c, _ in PLOT_CONDITIONS]
#: Same three groups, with the number of runs behind each.  Panel (b) draws
#: every run, so n belongs on its axis; (a) and (c) draw one representative.
PLOT_DRIVE_LABEL = [f"{DRIVE_LABEL[c]}\n$n = {len(m)}$"
                    for c, m in PLOT_CONDITIONS]
#: Five families, five hues that survive printing; the campaign palette had a
#: red, a pink and an orange in it.
FAMILY_COLOR = {"FENEPConformation": st.C_TRUTHFAM, "LinearPTT": "#1f77b4",
                "Giesekus": "#e08214", "OldroydB": "#9467bd",
                "Newtonian": "#d1495b"}
#: The one run in the whole program whose battery verdict is not FENE-P.
PTT_EXCEPTION = "fene8_bal_s3"


def _lw(scale, k=1.0):
    return k * st.BASE_LINEWIDTH * scale


def _drive_ticks(ax, scale):
    """Shrink the drive tick labels when the panel is inside the full figure.

    "U = 0.5 + pressure" is wide for a third of a three-group axis; standing
    alone it fits, and in the assembled figure it needs 15% off.
    """
    if pn.assembling():
        ax.tick_params(
            axis="x",
            labelsize=st.NOTEBOOK_EFFECTIVE["tick_labelsize_major"] * scale * 0.85)


def representatives() -> dict[str, str]:
    """Lowest-MARE run inside each condition."""
    return {name: min(members, key=ld.fene_mare)
            for name, members in CONDITIONS}


REPS = representatives()


def ladder_table() -> list[dict]:
    rows = []
    for cond, members in CONDITIONS:
        for t in members:
            rec, err = ld.fene_recovery(t), ld.fene_recovery_errors(t)
            winner, margin = ld.battery_winner(t)
            rows.append({"condition": cond, "target": t,
                         "representative": t == REPS[cond],
                         "Lsq": rec["Lsq"], "Lsq_err": err["Lsq"],
                         "mare": ld.fene_mare(t), "winner": winner,
                         "margin": margin,
                         "resid_x_floor": rec["mse"] / ld.NOISE_FLOOR})
    return rows


def _mean_abs_lsq_err(members) -> float:
    return float(np.abs([100 * ld.fene_recovery_errors(t)["Lsq"]
                         for t in members]).mean())


def notes() -> str:
    """Caption material: what the panels show, and the numbers behind them."""
    leg = dict(CONDITIONS)["dual-U legacy"]
    leg_pressure = [t for t in leg if ld.fene_summary(t)["pressure_on"]]
    leg_velocity = [t for t in leg if not ld.fene_summary(t)["pressure_on"]]
    ptt_resid = (ld.fene_recovery(PTT_EXCEPTION)["mse"] / ld.NOISE_FLOOR)
    f = state_fields()
    lines = [
        "Figure SN5 -- what the panels show, and what they leave out.",
        "",
        "THE THREE CONDITIONS.  Each is a drive plus an observable at one rate",
        "weighting; the tick labels in (b) and (c) name the drive.",
        f"  {DRIVE_TEXT['single-U']:22s} "
        f"{'velocity only':42s} n = {len(dict(PLOT_CONDITIONS)['single-U'])}",
        f"  {DRIVE_TEXT['dual-U balanced']:22s} "
        f"{'velocity + pressure, equal rate weighting':42s} "
        f"n = {len(dict(PLOT_CONDITIONS)['dual-U balanced'])}",
        f"  {DRIVE_TEXT['+pressure']:22s} "
        f"{'velocity + pressure':42s} "
        f"n = {len(dict(PLOT_CONDITIONS)['+pressure'])}",
        "",
        '"+pressure" is SINGLE-rate (U = 0.5) plus the wall-tap pressure',
        "observable, NOT dual-rate plus pressure.  It is one run (R4), and its",
        "6% L^2 error sits inside the single-rate spread (five runs, +4% to",
        "+18%), so it is not evidence that pressure helps.",
        "",
        "n per condition in panel (b): 5 (U = 0.5) / 5 (U = 0.5, 4) / 1",
        "(U = 0.5 + pressure); printed under each group's tick label.",
        "",
        "CORRECTION to an earlier reading of this figure.  The balanced arm is",
        "itself dual-rate PLUS pressure: all five runs carry w_p_scale = 1",
        f"(w_p = {ld.fene_summary('fene8_bal_s1')['w_p']:.3f}), so the dual + "
        "pressure cell at balanced weighting is",
        "exactly what is drawn as the middle group.  The cell that was never",
        "run is dual-rate VELOCITY-ONLY at balanced weighting, which is why the",
        "pressure observable cannot be separated from the dual drive inside the",
        "balanced weighting.  The legacy weighting does contain both cells:",
        f"  dual + pressure, legacy: {', '.join(leg_pressure)} "
        f"({len(leg_pressure)} runs)",
        f"  dual, velocity only, legacy: {', '.join(leg_velocity)} "
        f"({len(leg_velocity)} run)",
        "All of them select FENE-P.  They are excluded from the panels so the",
        "comparison does not confound rate weighting with observable; they",
        "remain in Table S2 and in optional_panel_d_evidence().  For the record,",
        "the legacy representative is "
        f"{ld.FENE_LABEL.get(REPS['dual-U legacy'], REPS['dual-U legacy'])} at "
        f"{100 * ld.fene_recovery_errors(REPS['dual-U legacy'])['Lsq']:+.0f}% "
        f"L^2 error, mean absolute L^2 error",
        f"over the arm {_mean_abs_lsq_err(leg):.0f}%.",
        "",
        "PANEL (a).  dBIC is measured against the selected model, so the FENE-P",
        "bar is zero in every condition and shows as the gap in each group.",
        "Selected family and margin over the runner-up, per condition:",
    ]
    for cond, _ in PLOT_CONDITIONS:
        win, margin = ld.battery_winner(REPS[cond])
        lines.append(f"  {cond}: {ld.FAMILY_LABEL[win]}, margin "
                     f"{margin:,.0f} "
                     f"(representative "
                     f"{ld.FENE_LABEL.get(REPS[cond], REPS[cond])})")
    lines += [
        "",
        "PANEL (b).  The sole non-FENE-P verdict in the whole program is",
        f"{PTT_EXCEPTION}, at "
        f"{100 * ld.fene_recovery_errors(PTT_EXCEPTION)['Lsq']:+.0f}% L^2 error",
        f"with residual {ptt_resid:.2f} x the 3% noise floor -- the worst fit of",
        "any run in the campaign, which is why its verdict carries no weight",
        f"against the other {sum(len(m) for _, m in CONDITIONS) - 1} runs.  It "
        "is encoded as an OPEN SQUARE in the",
        f"Linear PTT blue of panel (a) ({FAMILY_COLOR['LinearPTT']}); every "
        "other run is a circle (filled",
        "red = the condition's representative, open grey = the rest).  The",
        "legend carries the verdict, so no callout crosses the cloud.  Group",
        "means are printed above the axes:",
        "  mean absolute L^2 error: "
        + " / ".join(f"{_mean_abs_lsq_err(m):.0f}% ({DRIVE_TEXT[c]})"
                     for c, m in PLOT_CONDITIONS),
        "",
        "PANEL (c).  Distribution of truth tr A / L^2 over all fluid cells of",
        "the final solver frame, one histogram per condition, from that",
        "condition's representative run.  Every condition is normalised by the",
        f"TRUTH extensibility L^2 = {POLE:g}, not by its own recovered L^2, so the",
        "Peterlin pole is at 1.0 on one common axis; counts are on a log scale.",
        "State this in the caption.",
    ]
    for cond, _ in PLOT_CONDITIONS:
        d = f[cond]
        c = d["cov"]
        lines.append(
            f"  {DRIVE_TEXT[cond]:22s} rep {d['run']:13s} "
            f"{d['n_fluid']} fluid cells of {d['n_fluid'] + d['n_solid']}, "
            f"max {c.max():.3f}, "
            f"{int((c > 0.8).sum())} cells above 0.8, "
            f"{int((c > 0.9).sum())} above 0.9")
    lines += [
        "",
        "The single-rate and the +pressure histograms are the SAME curve: both",
        "conditions drive at U = 0.5, so they see an identical truth state",
        "distribution and differ only in what is observed.  The +pressure curve",
        "is therefore drawn dashed and unfilled on top of the single-rate one.",
        "The dual-rate archive holds the U = 4 frame (the driver's per-rate loop",
        "leaves the last rate in place); the U = 0.5 half of that condition's",
        "training set is the single-rate curve already drawn, so the two curves",
        "together are the dual condition's full state coverage.",
        "",
        "The caption argument here is STIFFNESS, not weighting.  What would pin",
        "L^2 is probability mass near the Peterlin pole, where the FENE-P",
        "spring stiffens and the model becomes distinguishable from its",
        "competitors; no condition has that mass.  The dual-rate drive reaches",
        f"{f['dual-U balanced']['cov'].max():.3f} of the pole against "
        f"{f['single-U']['cov'].max():.3f} for the single rate, yet recovers L^2",
        f"WORSE ({_mean_abs_lsq_err(dict(PLOT_CONDITIONS)['dual-U balanced']):.0f}% "
        f"mean absolute error against "
        f"{_mean_abs_lsq_err(dict(PLOT_CONDITIONS)['single-U']):.0f}%), which is why",
        "the maximum coverage alone was a misleading statistic: the extra reach",
        f"is {int((f['dual-U balanced']['cov'] > 0.9).sum())} cells out of "
        f"{f['dual-U balanced']['n_fluid']}, while the bulk of both distributions",
        "sits near the rest state tr A = 3 (0.25 on this axis).  The maxima",
        "quoted over the whole trajectory rather than the final frame are",
        f"{coverage()['dual-U balanced']:.3f} (dual) and {coverage()['single-U']:.3f} "
        "(single); the panel plots final-frame fields, so",
        "its maxima are the slightly lower numbers tabulated above.",
        "CAPTION: the histogram is over final-frame fluid cells,",
        "tr A / L^2_truth with L^2 = 12.  Support maxima 0.774 / 0.959 /",
        "0.774 (U = 0.5 / dual / +pressure).  The previously quoted",
        "0.780 / 0.959 / 0.780 are the whole-trajectory maxima.  Both are",
        "correct; they are different quantities.  The dual tail to 0.959 is",
        "15 of 16448 cells.",
        "",
        "NOT DRAWN.  The effective-loss-mass sub-panel (the 114:1 legacy rate",
        "imbalance) has been removed: legacy weighting appears nowhere else in",
        "this figure.  The numbers survive in rate_balance_record().",
        "",
    ]
    return "\n".join(lines)


def write_notes():
    out = ld.dp.out_dir(FIG) / "SN5_notes.txt"
    out.write_text(notes())
    print(f"[SN5_notes] {out}", flush=True)
    return out


# --------------------------------------------------------------------------
# a -- representative dBIC per condition
# --------------------------------------------------------------------------

def plot_SN5a(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(12.0, 8.5)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    fams = ld.FAMILY_ORDER
    x = np.arange(len(PLOT_CONDITIONS), dtype=float)
    width = 0.16
    for k, fam in enumerate(fams):
        vals = [ld.battery_delta_bic(REPS[c])[fam] for c, _ in PLOT_CONDITIONS]
        ax.bar(x + (k - 2) * width, vals, width=width,
               color=FAMILY_COLOR[fam], edgecolor="black",
               linewidth=1.0 * scale, zorder=3, label=ld.FAMILY_LABEL[fam])
    # Linear, as in N1c and SN4c: every rejected family is within a factor of
    # four of the others, so a symlog warp only emptied the lower half.
    top = max(ld.battery_delta_bic(REPS[c])[f]
              for c, _ in PLOT_CONDITIONS for f in fams)
    ax.set_ylim(0, 1.08 * top)
    ax.axhline(0.0, color="black", lw=st.BASE_AXES_LINEWIDTH * scale, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(PLOT_LABEL)
    _drive_ticks(ax, scale)
    ax.set_xlim(-0.5, len(PLOT_CONDITIONS) - 0.5)
    ax.set_ylabel(r"$\Delta$BIC")
    # Above the axes: the bars run from zero, so there is no clear space inside.
    st.legend(ax, scale, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=3, columnspacing=0.9, handlelength=1.2,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.8)
    out = pn.finish(fig, ax, FIG, "SN5a", save, dpi, own)
    if save:
        write_notes()
    return out


# --------------------------------------------------------------------------
# b -- L^2 readback error per condition
# --------------------------------------------------------------------------

def plot_SN5b(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(12.0, 8.5)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    ax.axhline(0.0, color=st.C_TRUTH, ls="--", lw=_lw(scale, 0.9),
               label=r"truth $L^2 = 12$")
    fs = st.NOTEBOOK_EFFECTIVE["tick_labelsize_minor"] * scale
    ptt_color = FAMILY_COLOR["LinearPTT"]
    for i, (cond, members) in enumerate(PLOT_CONDITIONS):
        errs = np.array([100 * ld.fene_recovery_errors(t)["Lsq"]
                         for t in members])
        xs = i + np.linspace(-0.18, 0.18, len(members))
        seen = set()
        for x, t, e in zip(xs, members, errs):
            rep = t == REPS[cond]
            # The one non-FENE-P verdict in the program carries the LinearPTT
            # colour of panel (a) and a square marker, so the exception is read
            # off the marker instead of a callout leader crossing the cloud.
            if t == PTT_EXCEPTION:
                ax.plot(x, e, "s", ms=13 * scale, color=ptt_color, mfc="none",
                        mew=2.4 * scale, zorder=5,
                        label=f"sole {ld.FAMILY_LABEL['LinearPTT']} verdict")
                continue
            tag = "representative" if rep else "other runs"
            label = tag if (i == 0 and tag not in seen) else None
            seen.add(tag)
            ax.plot(x, e, "o", ms=(15 if rep else 10) * scale,
                    color=st.C_LEARN if rep else "0.45",
                    mfc=st.C_LEARN if rep else "none", mew=1.6 * scale,
                    zorder=4 if rep else 3, label=label)
        # Above the axes: the mean is a summary of the group, not a datum in it.
        ax.annotate(rf"$\langle|{{\rm err}}|\rangle = {np.abs(errs).mean():.0f}\%$",
                    xy=(i, 1.015), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", color="0.25", fontsize=fs)
    ax.set_xticks(np.arange(len(PLOT_CONDITIONS)))
    ax.set_xticklabels(PLOT_DRIVE_LABEL)
    _drive_ticks(ax, scale)
    ax.set_xlim(-0.5, len(PLOT_CONDITIONS) - 0.5)
    ax.set_ylim(-72, 62)
    ax.set_ylabel(r"$L^2$ readback error (%)")
    # Inside now that the group means have vacated the top of the axes; boxed
    # so the marker key does not read as three more data points.
    handles, labels = ax.get_legend_handles_labels()
    order = ["representative", "other runs",
             f"sole {ld.FAMILY_LABEL['LinearPTT']} verdict", r"truth $L^2 = 12$"]
    by_label = dict(zip(labels, handles))
    st.legend(ax, scale, handles=[by_label[k] for k in order if k in by_label],
              labels=[k for k in order if k in by_label],
              loc="upper left", ncol=2, columnspacing=1.2, handletextpad=0.4,
              frameon=True, fancybox=True, framealpha=1.0, edgecolor="0.6",
              borderpad=0.35,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.7)
    return pn.finish(fig, ax, FIG, "SN5b", save, dpi, own)


# --------------------------------------------------------------------------
# c -- mechanism: where the training states sit relative to the Peterlin pole
# --------------------------------------------------------------------------

#: Every condition is normalised by the TRUTH extensibility, not by its own
#: recovered L^2, so the Peterlin pole is at 1.0 on one common axis.
POLE = ld.FENE_TRUTH["Lsq"]
#: One hue per drawn condition; the same green / blue / red triple the training
#: -schedule panels use.
COND_COLOR = {"single-U": st.C_SCHEDULE[0], "dual-U balanced": st.C_SCHEDULE[1],
              "+pressure": st.C_SCHEDULE[2]}


def state_fields() -> dict:
    """Final-frame truth ``tr A / L^2_truth`` over fluid cells, per condition.

    One entry per drawn condition, from that condition's representative run.
    The dual-rate archive holds the U = 4 frame (the driver's per-rate loop
    leaves the last rate in place), which is the higher-stretch half of that
    condition's training set; its U = 0.5 half is the same truth field the
    single-rate conditions are drawn from.
    """
    out = {}
    ref = ld.load_npz("fene_truth_traj", "xc", "yc")
    for cond, _ in PLOT_CONDITIONS:
        t = REPS[cond]
        d = ld.fene_truth_state_cloud(t)
        grid = gm.contraction_grid_from_config(ld.fene_config(t))
        if not (np.allclose(grid.xc, ref["xc"]) and
                np.allclose(grid.yc, ref["yc"])):
            raise ValueError(f"{t}: reconstructed grid differs from the archive")
        fluid = grid.fluid_mask()
        out[cond] = {"run": t, "U_archived": d["U"], "U_list": d["U_list"],
                     "cov": d["trA"][fluid] / POLE,
                     "n_fluid": int(fluid.sum()), "n_solid": int((~fluid).sum())}
    return out


def audit_state_fields() -> str:
    """What is on disk for panel (c), printed before the panel is built."""
    lines = ["Panel (c) audit -- final-frame truth tr A fields, per condition:"]
    f = state_fields()
    for cond, _ in PLOT_CONDITIONS:
        d = f[cond]
        c = d["cov"]
        lines.append(
            f"  {cond:16s} rep {d['run']:13s} "
            f"U_list={d['U_list']} archived frame U={d['U_archived']:g}  "
            f"{d['n_fluid']} fluid cells (of "
            f"{d['n_fluid'] + d['n_solid']}), "
            f"tr A / L^2: min {c.min():.3f} median {np.median(c):.3f} "
            f"max {c.max():.3f}")
    a, b = f["single-U"]["cov"], f["+pressure"]["cov"]
    lines.append(f"  single-U and +pressure clouds identical: "
                 f"{np.array_equal(a, b)} (same U = 0.5 truth field; the "
                 f"conditions differ only in the observable)")
    return "\n".join(lines)


def rate_balance_record(run: str = "fene8_bal_s0") -> dict:
    """``rate_balance`` block written by the balanced-arm trainer.

    It reports the effective ROI-weighted loss mass of each rate under the
    legacy weighting and under equal weighting, so the 114:1 imbalance is read
    from the run itself rather than recomputed here.
    """
    return ld.fene_summary(run)["rate_balance"]


def coverage() -> dict[str, float]:
    """Peak truth ``tr A`` as a fraction of the FENE-P pole ``L^2``."""
    out = {}
    for cond, members in CONDITIONS:
        vals = {ld.fene_summary(t)["max_trA_truth"] for t in members}
        if len(vals) != 1:
            raise ValueError(f"{cond}: inconsistent truth fields {vals}")
        out[cond] = vals.pop() / ld.FENE_TRUTH["Lsq"]
    return out


def plot_SN5c(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(12.0, 8.5)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    f = state_fields()
    edges = np.arange(0.22, 1.0001, 0.02)
    # Widest distribution first, so the narrower single-rate outline sits on
    # top of it rather than under its fill.
    draw_order = ("dual-U balanced", "single-U", "+pressure")
    for cond in draw_order:
        cov, color = f[cond]["cov"], COND_COLOR[cond]
        # The +pressure condition trains on the same U = 0.5 truth field as the
        # single-rate condition, so its histogram lies exactly on top: drawn
        # last, unfilled and dashed, both curves stay readable.
        same = cond == "+pressure"
        if not same:
            ax.hist(cov, bins=edges, histtype="stepfilled", color=color,
                    alpha=0.30, zorder=2)
        ax.hist(cov, bins=edges, histtype="step", color=color,
                lw=_lw(scale, 1.2 if same else 1.0),
                ls="--" if same else "-", zorder=4 if same else 3,
                label=DRIVE_LABEL[cond])
    ax.set_yscale("log")
    ax.set_xlim(0.20, 1.06)
    ax.set_ylim(0.7, 4e4)
    ax.axvline(1.0, color=st.C_TRUTH, ls="-.", lw=_lw(scale, 1.0), zorder=5)
    ax.annotate("FENE-P pole", xy=(0.988, 0.34), xycoords=("data",
                                                           "axes fraction"),
                rotation=90, ha="right", va="center", color=st.C_TRUTH,
                fontsize=st.NOTEBOOK_EFFECTIVE["tick_labelsize_minor"] * scale)
    ax.annotate(r"rest state, $\mathrm{tr}\,\mathbf{A} = 3$", xy=(0.252, 1.1e4),
                xytext=(0.325, 2.2e4), ha="left", va="center", color="0.25",
                arrowprops=dict(arrowstyle="-", color="0.45", lw=1.1 * scale),
                fontsize=st.NOTEBOOK_EFFECTIVE["tick_labelsize_minor"] * scale)
    ax.set_xlabel(r"$\mathrm{tr}\,\mathbf{A}\,/\,L^2_{\rm truth}$")
    ax.set_ylabel("fluid cells")
    st.legend(ax, scale, loc="upper right", ncol=1, handlelength=1.6,
              frameon=True, fancybox=True, framealpha=1.0, edgecolor="0.6",
              borderpad=0.35,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.7)
    return pn.finish(fig, ax, FIG, "SN5c", save, dpi, own)


# --------------------------------------------------------------------------
# the optional panel d, and why it is not here
# --------------------------------------------------------------------------

def optional_panel_d_evidence() -> dict:
    """Numbers behind the decision to omit the legacy-vs-balanced loss panel.

    Two findings, both against plotting it.  First, the balanced arm optimises
    a *rescaled* objective (``--rate-balance equal`` divides the U=0.5
    velocity weight by ~10.7 and multiplies the U=4 weight by the same), so
    raw losses are not on a common scale; normalised by each run's own initial
    loss the balanced arm converges *further*, not worse, which is the
    opposite of the mechanism the panel was meant to show.  Second, the
    balanced ``progress.csv`` files are not clean single-stream histories.
    """
    out = {"arms": {}}
    for arm, members in (("legacy", dict(CONDITIONS)["dual-U legacy"]),
                         ("balanced", dict(CONDITIONS)["dual-U balanced"])):
        rows = []
        for t in members:
            s = ld.fene_summary(t)
            p = ld.load_csv(ld.FENE_PROGRESS_KEY.get(t, f"fene_progress_{t}"))
            step = p["step"]
            rows.append({"run": t, "loss_init": s["loss_init"],
                         "loss_final": s["loss_final"],
                         "final_over_init": s["loss_final"] / s["loss_init"],
                         "progress_rows": int(step.size),
                         "step_monotone": bool(np.all(np.diff(step) >= 0)),
                         "duplicate_rows": int(step.size
                                               - np.unique(step).size)})
        out["arms"][arm] = rows
        out[f"{arm}_median_final_over_init"] = float(np.median(
            [r["final_over_init"] for r in rows]))
    out["rate_balance_scales"] = rate_balance_record()["scales"]
    out["decision"] = "omit"
    return out


# --------------------------------------------------------------------------
# assembled figure
# --------------------------------------------------------------------------

def plot_SN5(save=True, dpi=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(24.0, 9.5))
    gs = fig.add_gridspec(1, 3, wspace=0.28, left=0.05, right=0.98, top=0.92,
                          bottom=0.14)
    slots = ((gs[0, 0], plot_SN5a, "(a)"), (gs[0, 1], plot_SN5b, "(b)"),
             (gs[0, 2], plot_SN5c, "(c)"))
    with pn.uniform_scale(0.62) as scale:
        for spec, fn, tag in slots:
            ax = fig.add_subplot(spec)
            res = fn(ax=ax)
            first = res[0] if isinstance(res, list) else res
            pn.panel_tag(first, tag, scale, loc="outside")
    if save:
        print(f"[SN5_full] {pn.save_panel(fig, FIG, 'SN5_full', dpi)}",
              flush=True)
    return fig
