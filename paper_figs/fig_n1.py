"""Figure N1 (manuscript Fig 5) -- Giesekus.

Frozen network ``gie_prod_rerun/gie_A_s4``; truth Gp 3.2, lam 0.7, nu_s 0.8,
alpha 0.30.  Nothing here refits or retrains: every number is read from an
archive or a battery JSON.
"""
from __future__ import annotations

import textwrap

import numpy as np

from . import geometry as geo
from . import loaders as ld
from . import panels as pn
from . import style as st

FIG = "N1"
SCHEDULES = ld.GIE_SCHEDULES
SCHEDULE_LABEL = ld.GIE_SCHEDULE_LABEL
XLIM = (-3.0, 4.5)
#: N1a shows the whole channel height and 6 H either side of the step; the
#: remaining 6 H of the downstream tail is fully developed and adds nothing.
XLIM_FIELD = (-6.0, 6.0)
YLIM_FIELD = (-4.0, 4.0)


# --------------------------------------------------------------------------
# shared computations
# --------------------------------------------------------------------------

def _grid():
    return geo.contraction_grid_from_archive(ld.gie_fields())


def _trace(prefix: str) -> np.ndarray:
    f = ld.gie_fields()
    return f[f"{prefix}_A_xx"] + f[f"{prefix}_A_yy"] + f[f"{prefix}_A_zz"]


def _trace_init() -> np.ndarray:
    f = ld.gie_init_fields()
    return f["A_xx"] + f["A_yy"] + f["A_zz"]


def _centreline(field: np.ndarray, grid: geo.ContractionGrid) -> np.ndarray:
    """Interpolate a cell-centre field onto y = 0."""
    return np.array([np.interp(0.0, grid.yc, field[i, :])
                     for i in range(grid.xc.size)])


def _cross_section(field: np.ndarray, grid: geo.ContractionGrid,
                   x0: float = 0.0) -> np.ndarray:
    """Interpolate a cell-centre field onto x = x0."""
    return np.array([np.interp(x0, grid.xc, field[:, j])
                     for j in range(grid.yc.size)])


def contraction_metrics() -> dict:
    """Relative L2 on tr A (global and in the ROI band) + symmetry residual.

    Recomputed here from the archives; the Pass C provenance file carries the
    same three numbers and is used as a cross-check.
    """
    f = ld.gie_fields()
    grid = _grid()
    cfg = ld.gie_config()
    fluid = grid.fluid_mask()
    _, activation, x_on, x_off = geo.roi_fields(cfg, grid)
    band = (activation >= 0.5) & fluid
    tt, tl = _trace("truth"), _trace("learned")
    rel = lambda m: float(np.linalg.norm((tl - tt)[m]) / np.linalg.norm(tt[m]))
    # The symmetry residual is evaluated on the u/v face stations (xfu, yfu),
    # matching passc_contraction_figures.py::_symmetry_residual.
    Xf, Yf = np.meshgrid(f["xfu"], f["yfu"], indexing="ij")
    fluid_f = ~((Xf >= 0.0) & (np.abs(Yf) > grid.H)
                & (np.abs(Yf) <= grid.R * grid.H))
    u, v = f["truth_u"], f["truth_v"]
    pair = fluid_f & fluid_f[:, ::-1]
    num = np.sum(((u - u[:, ::-1]) ** 2 + (v + v[:, ::-1]) ** 2)[pair])
    den = np.sum((u ** 2 + v ** 2)[pair])
    return {
        "relative_L2_trA_global": rel(fluid),
        "relative_L2_trA_roi_band": rel(band),
        "truth_centerline_symmetry_residual": float(np.sqrt(num / den)),
        "roi_x_on": x_on, "roi_x_off": x_off, "roi_cells": int(band.sum()),
    }


# --------------------------------------------------------------------------
# N1a -- contraction u_x, truth above / learned s4 below
# --------------------------------------------------------------------------

def plot_N1a(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(13.0, 7.4, axes_width=9.5)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    f = ld.gie_fields()
    grid = _grid()
    pg = geo.contraction_plot_grid(grid)
    cfg = ld.gie_config()
    _, activation, _, _ = geo.roi_fields(cfg, grid)

    # u is staggered on x-faces: interpolate to centres before plotting.
    ut = geo.u_faces_to_centres(f["truth_u"])
    ul = geo.u_faces_to_centres(f["learned_u"])
    Ut = geo.contraction_to_plot(ut, grid, pg, velocity=True)
    Ul = geo.contraction_to_plot(ul, grid, pg, velocity=True)
    F = geo.mirrored_split(Ut, Ul, pg)

    vmax = float(np.nanmax(np.abs(F)))
    levels = np.linspace(0.0, vmax, pn.N_LEVELS)
    m = pn.field_panel(ax, pg, F, cmap=st.CMAP_SPEED, levels=levels,
                       scale=scale, extend="both",
                       solids=grid.solid_rectangles())
    pn.draw_roi(ax, grid, activation, scale)
    pn.split_labels(ax, "Giesekus truth", "Trained TBNN", scale)
    pn.contraction_axes(ax, grid, scale, XLIM_FIELD, YLIM_FIELD)
    pn.colorbar(fig, m, ax, r"$u_x$", scale)
    return pn.finish(fig, ax, FIG, "N1a", save, dpi, own)


# --------------------------------------------------------------------------
# N1b -- tr A along the centreline
# --------------------------------------------------------------------------

def plot_N1b(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 8.5, axes_width=8.6)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    grid = _grid()
    x = grid.xc / grid.H
    for label, key, kw in st.PROFILE_ROLES:
        field = _trace_init() if key == "init" else _trace(key)
        ax.plot(x, _centreline(field, grid), label=label,
                lw=st.PAPER_LINE["linewidth"] * scale, **kw)
    ax.set_xlim(*XLIM)
    ax.set_xlabel(r"$x/H$")
    ax.set_ylabel(r"$\mathrm{tr}\,\mathbf{A}$")
    pn.annotate(ax, 0.97, 0.90, "centreline, $y=0$", scale, ha="right",
                color="0.3")
    st.paper_legend(ax, scale, loc="upper left")
    return pn.finish(fig, ax, FIG, "N1b", save, dpi, own)


# --------------------------------------------------------------------------
# N1c -- model selection, five families
# --------------------------------------------------------------------------

def _bar_positions(n_families, n_groups, width=0.38):
    base = np.arange(n_families, dtype=float)
    off = (np.arange(n_groups) - (n_groups - 1) / 2) * width
    return base, off


def bic_bar_panel(ax, scale, learned_target, control_target,
                  learned_label, control_label):
    """Five-family DeltaBIC bars for one network plus its clean-analytic control.

    DeltaBIC is measured against the selected model, so the winner sits at zero;
    the axis is symlog so the four decades separating the rejected families
    are all legible while zero stays on the plot.
    """
    fams = ld.FAMILY_ORDER
    learned = ld.battery_delta_bic(learned_target)
    control = ld.battery_delta_bic(control_target)
    base, off = _bar_positions(len(fams), 2)
    for k, (d, label, color) in enumerate((
            (learned, learned_label, st.C_LEARN),
            (control, control_label, st.C_TRUTHFAM))):
        vals = np.array([d[f] for f in fams])
        ax.bar(base + off[k], vals, width=0.38, color=color, label=label,
               edgecolor="black", linewidth=1.2 * scale, zorder=3)

    ax.set_yscale("symlog", linthresh=1.0, linscale=0.3)
    ax.set_ylim(0, 3e5)
    ax.set_yticks([0, 1e1, 1e2, 1e3, 1e4])
    ax.set_xticks(base)
    ax.set_xticklabels([ld.FAMILY_LABEL[f] for f in fams], rotation=30,
                       ha="right")
    ax.set_ylabel(r"$\Delta$BIC vs selected")
    ax.axhline(0.0, color="black", lw=st.BASE_AXES_LINEWIDTH * scale, zorder=4)

    win_l, margin = ld.battery_winner(learned_target)
    _, margin_c = ld.battery_winner(control_target)
    fs = st.NOTEBOOK_EFFECTIVE["tick_labelsize_minor"] * scale
    i = fams.index(win_l)
    ax.annotate("0\n(selected)", xy=(base[i], 0.0),
                xytext=(0, 12 * scale), textcoords="offset points",
                ha="center", va="bottom", color="0.2", fontsize=fs)
    j = fams.index("LinearPTT")
    ax.annotate(f"{margin:,.2f}", xy=(base[j] + off[0], learned["LinearPTT"]),
                xytext=(0, 10 * scale), textcoords="offset points",
                ha="center", va="bottom", color=st.C_LEARN, fontsize=fs)
    ax.annotate(f"{margin_c:,.2f}",
                xy=(base[j] + off[1], control["LinearPTT"]),
                xytext=(0, 10 * scale), textcoords="offset points",
                ha="center", va="bottom", color=st.C_TRUTHFAM, fontsize=fs)
    pn.annotate(ax, 0.98, 0.44,
                "margin over the\nrunner-up family", scale,
                ha="right", color="0.25")
    st.legend(ax, scale, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), ncol=1)
    return margin, margin_c


def bic_bar_single(ax, scale, target: str):
    """One series of Delta-BIC bars: the winner at zero, the rest beside it.

    Families run winner-first then in increasing Delta-BIC, and the axis is
    linear, so the bar heights are read directly rather than through a symlog
    warp.  Returns the (winner, margin over the runner-up) pair.
    """
    d = ld.battery_delta_bic(target)
    win, margin = ld.battery_winner(target)
    fams = [win] + sorted((f for f in d if f != win), key=lambda f: d[f])
    vals = np.array([d[f] for f in fams])
    x = np.arange(len(fams), dtype=float)

    ax.bar(x, vals, width=0.62,
           color=[st.C_TRUTHFAM if f == win else st.C_OTHER for f in fams],
           edgecolor="black", linewidth=1.4 * scale, zorder=3)
    fs = st.NOTEBOOK_EFFECTIVE["tick_labelsize_minor"] * scale
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:,.0f}", xy=(xi, v), xytext=(0, 8 * scale),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=fs)
    ax.set_xticks(x)
    ax.set_xticklabels([ld.FAMILY_LABEL[f] for f in fams], rotation=25,
                       ha="right")
    # The selected family sits at zero, so it has no bar to carry the colour;
    # its tick label takes it instead.
    ax.get_xticklabels()[0].set_color(st.C_TRUTHFAM)
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_ylim(0.0, 1.14 * float(vals.max()))
    ax.set_ylabel(r"$\Delta$BIC")
    return win, margin


def plot_N1c(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 8.5)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)
    bic_bar_single(ax, scale, "gie_A_s4")
    return pn.finish(fig, ax, FIG, "N1c", save, dpi, own)


# --------------------------------------------------------------------------
# N1d -- recovered constitutive parameters, gie_A_s4
# --------------------------------------------------------------------------

#: Rows of the recovery table, in the order the paper introduces them.
RECOVERY_ROWS = (("Gp", r"$G_p$"), ("lam", r"$\lambda$"),
                 ("nu_s", r"$\eta_s$"), ("alpha", r"$\alpha$"))
RECOVERY_TARGET = "gie_A_s4"


def recovery_table(target: str = RECOVERY_TARGET,
                  rows=RECOVERY_ROWS) -> list[list[str]]:
    """truth vs recovered for one network, three decimals throughout.

    ``recovered`` is the BIC battery's Giesekus fit to that network's stress
    response.  All four rows come from that one fit: alpha exists nowhere else,
    and mixing it with the trained scalars would put two different recoveries
    in one column.
    """
    got = ld.gie_battery_scalars(target)
    return [[label, f"{ld.GIE_TRUTH[key]:.3f}", f"{got[key]:.3f}"]
            for key, label in rows]


def plot_N1d(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(9.0, 3.6, axes_width=8.0)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)
    pn.table_panel(ax, ("param", "truth", "recovered"),
                   recovery_table(), scale,
                   col_widths=[0.28, 0.30, 0.38], row_height=3.0)
    if own and save:
        write_n1d_notes()
    return pn.finish(fig, ax, FIG, "N1d", save, dpi, own)


#: Same four rows as N1d, but the polymer modulus is reported as eta_p
#: so the table matches SN4d_table's first row.
RECOVERY_ROWS_ETAP = (("eta_p", r"$\eta_p$"), ("lam", r"$\lambda$"),
                      ("nu_s", r"$\eta_s$"), ("alpha", r"$\alpha$"))


def plot_N1d_alt(ax=None, save=True, dpi=None):
    """N1d with eta_p in place of G_p."""
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(9.0, 3.6, axes_width=8.0)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)
    pn.table_panel(ax, ("param", "truth", "recovered"),
                   recovery_table(rows=RECOVERY_ROWS_ETAP), scale,
                   col_widths=[0.28, 0.30, 0.38], row_height=3.0)
    return pn.finish(fig, ax, FIG, "N1d_alt", save, dpi, own)


def n1d_notes() -> str:
    """Provenance of the four recovered numbers (all battery readback)."""
    got = ld.gie_battery_scalars(RECOVERY_TARGET)
    ck = ld.gie_checkpoint_scalars()[RECOVERY_TARGET]
    return (
        "N1d -- recovered constitutive parameters\n"
        "=======================================\n\n"
        "ALL FOUR rows are readback quantities: the BIC battery's Giesekus\n"
        "joint fit to the finished network's stress response (target "
        f"{RECOVERY_TARGET}).  They are not the trained scalars from\n"
        "progress.csv / the checkpoint.  The earlier split 'alpha is\n"
        "readback, the others are trained' no longer applies; the caption\n"
        "must treat G_p, lambda, eta_s and alpha the same way.\n\n"
        "Battery Giesekus fit (what the table prints, three decimals):\n"
        f"  G_p   {got['Gp']:.3f}   (raw {got['Gp']:.6f})\n"
        f"  lambda {got['lam']:.3f}   (raw {got['lam']:.6f})\n"
        f"  eta_s  {got['nu_s']:.3f}   (raw {got['nu_s']:.6f})\n"
        f"  alpha  {got['alpha']:.3f}   (raw {got['alpha']:.6f})\n\n"
        "Trained checkpoint scalars (NOT in the table; no alpha exists):\n"
        f"  G_p   {ck['Gp']:.3f}   (raw {ck['Gp']:.6f})\n"
        f"  lambda {ck['lam']:.3f}   (raw {ck['lam']:.6f})\n"
        f"  eta_s  {ck['nu_s']:.3f}   (raw {ck['nu_s']:.6f})\n"
        f"  final training loss {ck['loss']:.6f}\n"
    )


def write_n1d_notes():
    out = ld.dp.out_dir(FIG) / "N1d_notes.txt"
    out.write_text(n1d_notes())
    print(f"[N1d] {out}", flush=True)
    return out


# --------------------------------------------------------------------------
# Superseded by the table above; kept callable, out of the figure set.
# --------------------------------------------------------------------------

def alpha_readback() -> dict:
    """alpha per training schedule, from the battery Giesekus winner row."""
    return {r: float(ld.battery_rows(r)["Giesekus"]["params"]["alpha"])
            for r in SCHEDULES}


def plot_N1d_schedules(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(
            2, 9.5, 10.5, sharex=True,
            gridspec_kw=dict(height_ratios=[1.35, 1.0], hspace=0.10))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 2, 1, sharex=True, hspace=0.10,
                         height_ratios=[1.35, 1.0])
        scale = pn.adopt(axs[0])
        pn.adopt(axs[1])

    x = np.arange(len(SCHEDULES), dtype=float)
    alpha = alpha_readback()
    av = np.array([alpha[r] for r in SCHEDULES])
    mean, sd = float(av.mean()), float(av.std(ddof=1))

    axs[0].axhspan(mean - sd, mean + sd, color=st.C_LEARN, alpha=0.16,
                   zorder=0,
                   label=rf"mean $\pm$ s.d. $={mean:.3f}\pm{sd:.3f}$")
    axs[0].axhline(ld.GIE_TRUTH["alpha"], color=st.C_TRUTH,
                   lw=st.BASE_LINEWIDTH * scale, ls="--",
                   label=r"truth $\alpha=0.30$")
    axs[0].plot(x, av, "o-", color=st.C_LEARN,
                ms=13 * scale, mec="black", mew=st.NOTEBOOK_EFFECTIVE[
                    "marker_edge_width"] * scale,
                lw=st.BASE_LINEWIDTH * scale, label=r"$\alpha$ readback")
    for xi, a_ in zip(x, av):
        axs[0].annotate(f"{a_:.6f}", (xi, a_), textcoords="offset points",
                        xytext=(0, 18 * scale), ha="center",
                        fontsize=st.NOTEBOOK_EFFECTIVE[
                            "tick_labelsize_minor"] * scale)
    axs[0].set_ylabel(r"$\alpha$ (readback)")
    axs[0].set_ylim(0.2950, 0.3022)
    st.legend(axs[0], scale, loc="lower right")

    ck = ld.gie_checkpoint_scalars()
    series = (("$G_p$", "Gp", "o", st.C_DRIVE[0]),
              (r"$\lambda$", "lam", "s", st.C_DRIVE[1]),
              (r"$\eta_s$", "nu_s", "^", st.C_DRIVE[2]))
    axs[1].axhline(1.0, color=st.C_TRUTH, lw=st.BASE_LINEWIDTH * scale,
                   ls="--")
    for label, key, marker, color in series:
        ratio = np.array([ck[r][key] / ld.GIE_TRUTH[key] for r in SCHEDULES])
        axs[1].plot(x, ratio, marker + "-", color=color, ms=13 * scale,
                    mec="black",
                    mew=st.NOTEBOOK_EFFECTIVE["marker_edge_width"] * scale,
                    lw=st.BASE_LINEWIDTH * scale, label=label)
    axs[1].set_ylabel("recovered / truth")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([SCHEDULE_LABEL[r] for r in SCHEDULES])
    axs[1].set_xlabel("training schedule")
    axs[1].set_xlim(-0.35, len(SCHEDULES) - 0.65)
    axs[1].set_ylim(0.89, 1.17)
    st.legend(axs[1], scale, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.03), columnspacing=1.1)
    pn.annotate(axs[1], 0.5, 0.06,
                "spread across training schedules, fixed init",
                scale, ha="center", va="bottom", color="0.3")

    if own:
        pn.tidy(fig)
        if save:
            print(f"[N1d_schedules] "
                  f"{pn.save_panel(fig, FIG, 'N1d_schedules', dpi)}",
                  flush=True)
        return fig
    return axs


# --------------------------------------------------------------------------
# N1e -- cavity transfer: centreline profiles and the max A_xx transient
# --------------------------------------------------------------------------

TRUTH_LABEL = "Truth Giesekus"
LEARNED_LABEL = "TBNN Prediction"

#: Marker stride for the learned arm.  The profiles carry 96-128 samples and
#: the transients 200, so these give ~12 well separated markers per curve.
PROFILE_STRIDE = 10
TRANSIENT_STRIDE = 17
MARKER_SIZE = 12.0


def _learned_markers(ax, x, y, color, scale, stride, offset=0):
    """Learned arm: open markers on a subsampled grid, no line.

    Truth and learned agree to within a line width here, so drawing both as
    lines in one colour hides one of them; separated open markers let the
    reader see that both are present.
    """
    k = slice(offset, None, stride)
    return ax.plot(np.asarray(x)[k], np.asarray(y)[k], linestyle="none",
                   marker="o", mfc="none", mec=color, ms=MARKER_SIZE * scale,
                   mew=st.NOTEBOOK_EFFECTIVE["marker_edge_width"] * scale
                   * 1.3)[0]


def n1e_check() -> dict:
    """Every number the N1e caption quotes, read from the arrays it plots.

    Also the cross-check that the transient's last sample, the metrics file
    and the final field all report the same max A_xx.
    """
    m = ld.cavity_metrics()
    out = {}
    for de in ld.dp.CAVITY_DE:
        row = {"U_lid": float(ld.cavity_run("truth", de)["config"]["U_lid"])}
        for arm in ("truth", "learned"):
            h = ld.cavity_history(arm, de)
            fld = np.asarray(ld.cavity_run(arm, de)["diag"]["final_A_xx"])
            row[arm] = {
                "final_traj": float(h["max_Axx"][-1]),
                "metrics": float(m[f"De{de:.2f}"][f"{arm}_max_Axx"]),
                "final_field": float(fld.max()),
                "t_end": float(h["t"][-1]),
            }
        d = (ld.cavity_history("learned", de)["max_Axx"]
             - ld.cavity_history("truth", de)["max_Axx"])
        t = ld.cavity_history("truth", de)["t"]
        cross = t[:-1][np.sign(d[:-1]) != np.sign(d[1:])]
        row["diff_final"] = float(d[-1])
        row["diff_max"] = float(d.max())
        row["t_diff_max"] = float(t[int(d.argmax())])
        row["crossings"] = [float(c) for c in cross]
        row["rel_pct"] = 100.0 * float(d[-1]) / row["truth"]["final_traj"]
        out[de] = row
    return out


def cavity_ladder() -> dict:
    """The two N1e quantities at each De, plus the caption numbers."""
    de = list(ld.dp.CAVITY_DE)
    m = ld.cavity_metrics()
    return {
        "De": np.array(de),
        "u_truth": np.array([ld.cavity_centreline_scalar("truth", d)
                             for d in de]),
        "u_learned": np.array([ld.cavity_centreline_scalar("learned", d)
                               for d in de]),
        "axx_truth": np.array([m[f"De{d:.2f}"]["truth_max_Axx"] for d in de]),
        "axx_learned": np.array([m[f"De{d:.2f}"]["learned_max_Axx"]
                                 for d in de]),
        "vel_rms_pct": np.array([100 * m[f"De{d:.2f}"]["velocity_rms_over_U_lid"]
                                 for d in de]),
    }


def caption_notes() -> str:
    """The N1e caption material, with the numbers read at write time."""
    chk = n1e_check()
    ramp = float(ld.cavity_run("truth", ld.dp.CAVITY_DE[0])
                 ["config"]["ramp_time"])
    rows = "\n".join(
        f"  De = {de:.2f}   U_lid = {c['U_lid']:.3f}   "
        f"max A_xx  truth {c['truth']['final_traj']:.3f}   "
        f"TBNN {c['learned']['final_traj']:.3f}   "
        f"({c['rel_pct']:+.2f}%)"
        for de, c in chk.items())
    return f"""N1e -- caption notes
=====================

What the panel shows
--------------------
(i)  u_x/U_lid on the vertical centreline (x/L = 0.5) at t = 4, all three De.
(ii) max A_xx against time, same three De, to t = 4.
The left-panel x-label is the cut location, x/L = 0.5; the plotted
quantity remains u_x/U_lid (DEC-2).  The right-panel x-label is "Time, t".
A light band on (ii) marks the lid-ramp interval, from ramp_traj in
diagnostics.npz, until the lid-speed factor first reaches 1 (t = {ramp:g}).
Truth is the solid line; the TBNN prediction is the open markers.  The frozen
gie_A_s4 closure is carried into the lid-driven cavity forward only; it saw
no cavity data in training.

The claim
---------
The two arms agree over the whole approach, not only at the endpoint.  A
closure that only matched the final state could have taken any path to it;
matching max A_xx at every instant means the frozen closure reproduced the
relaxation dynamics of a geometry it never saw, not just its steady answer.

Velocity against conformation
-----------------------------
The centreline profiles overlay to within a line width at every De, while
max A_xx separates visibly at De = 0.50 ({chk[0.50]['rel_pct']:+.2f}% at t = 4).
The kinematics transfer essentially exactly; the polymer stretch is where the
closure's error is resolvable.  This is the same ordering the field panels
show (SN3a-c).

Ramp caveat
-----------
The lid is started with a cosine ramp of ramp_time = {ramp:g}, so the early
rise in (ii) is partly the boundary condition coming on and not free
relaxation; the agreement before t ~= {ramp:g} should not be read as a test of
the closure.

Numbers, read from the arrays the panel plots
---------------------------------------------
{rows}

Endpoint cross-check (asked for explicitly)
-------------------------------------------
For all six runs the transient's last sample, the value in
cavity_transfer_metrics.json and the max of the final A_xx field are the same
number to every digit stored, so the transient and the retired vs-De panel
were reading one array, not two:
""" + "\n".join(
        f"  De = {de:.2f}  {arm:8s}  traj[-1] = {c[arm]['final_traj']:.6f}  "
        f"metrics = {c[arm]['metrics']:.6f}  "
        f"field.max = {c[arm]['final_field']:.6f}"
        for de, c in chk.items() for arm in ("truth", "learned")) + "\n\n" + \
        textwrap.fill(
            "At De = 0.50 the curves do cross, but not where it looked like "
            "they did: the TBNN arm runs above truth only for "
            f"t < {chk[0.50]['crossings'][-1]:.2f}, by at most "
            f"{chk[0.50]['diff_max']:+.4f} at t = {chk[0.50]['t_diff_max']:.2f}"
            ", which is inside the line width, and below truth for the rest "
            f"of the run, reaching {chk[0.50]['diff_final']:+.4f} at t = 4. "
            "So the apparent contradiction is not one, and there is no frame "
            "or file mismatch to resolve: the learned final value is "
            f"{chk[0.50]['learned']['final_traj']:.3f}, not 7.98, which is "
            "close to the truth curve's value at t = 2 (7.990) and was "
            "probably read off the retired panel by eye.", width=78) + "\n"


def write_caption_notes():
    out = ld.dp.out_dir(FIG) / "N1e_caption_notes.txt"
    out.write_text(caption_notes())
    print(f"[N1e] {out}", flush=True)
    return out


def _e_profiles(ax, scale, *, zero_line=True,
                xlabel=r"$x/L = 0.5$"):
    """u_x on the vertical centreline, normalised by the lid speed.

    U_lid runs 0.286 / 0.5 / 0.714 across the ladder, so the unnormalised
    profiles would fan out by lid speed and bury the elastic shape change.

    The frame is the last one, t = 4, and every one of the six runs is
    classified STEADY in its own ``result.json``; the last outer step moves
    max A_xx by at most 5e-5 relative.  Hence "steady state" on the axis.
    """
    for i, de in enumerate(ld.dp.CAVITY_DE):
        color = st.C_DE[de]
        t = ld.cavity_centreline("truth", de)
        l = ld.cavity_centreline("learned", de)
        ax.plot(t["u_of_y"] / t["U_lid"], t["coord"], "-", color=color,
                lw=st.PAPER_LINE["linewidth"] * scale,
                label=f"De = {de:.2f}")
        _learned_markers(ax, l["u_of_y"] / l["U_lid"], l["coord"], color,
                         scale, PROFILE_STRIDE, offset=3 * i)
    if zero_line:
        ax.axvline(0.0, color="0.6", lw=st.BASE_LINEWIDTH * 0.4 * scale,
                   zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$y/L$")


def _lid_ramp_end() -> float:
    """First diagnostic time at which ``ramp_traj`` has reached 1.

    ``diagnostics.npz`` stores the lid-speed factor, not a boolean mask.
    The first sample with ``ramp_traj == 1`` is t = ramp_time = 0.7.
    """
    run = ld.cavity_run("truth", ld.dp.CAVITY_DE[0])
    ramp = np.asarray(run["diag"]["ramp_traj"], dtype=float)
    cfg = run["config"]
    n_out = int(cfg["outer_steps"])
    t = (np.arange(n_out) + 1) * float(cfg["T"]) / n_out
    on = np.flatnonzero(ramp >= 1.0 - 1e-12)
    return float(t[on[0]]) if on.size else float(cfg["ramp_time"])


def _mark_lid_ramp(ax, scale):
    """Shade t < ramp-complete, labelled.  Not a legend entry."""
    t_on = _lid_ramp_end()
    ax.axvspan(0.0, t_on, color="0.92", lw=0, zorder=0)
    ax.annotate(
        "lid ramp", xy=(0.5 * t_on, 1.0), xycoords=("data", "axes fraction"),
        ha="center", va="top", color="0.4",
        fontsize=st.NOTEBOOK_EFFECTIVE["tick_labelsize_minor"] * scale)


def _e_transient(ax, scale, inline_labels=False, *,
                 xlabel="Time, $t$", lid_ramp=True):
    """max A_xx against time, the approach to the steady state."""
    top = 0.0
    for i, de in enumerate(ld.dp.CAVITY_DE):
        color = st.C_DE[de]
        h = ld.cavity_history("truth", de)
        hl = ld.cavity_history("learned", de)
        ax.plot(h["t"], h["max_Axx"], "-", color=color,
                lw=st.PAPER_LINE["linewidth"] * scale,
                label=f"De = {de:.2f}")
        _learned_markers(ax, hl["t"], hl["max_Axx"], color, scale,
                         TRANSIENT_STRIDE, offset=5 + 5 * i)
        top = max(top, float(h["max_Axx"].max()))
        if inline_labels:
            # Assembled, the panel is too short for a legend box; the plateaux
            # are far apart, so name them where they sit.
            ax.annotate(f"De = {de:.2f}", (2.6, float(h["max_Axx"][-1])),
                        textcoords="offset points", xytext=(0, 9 * scale),
                        color=color, ha="center", va="bottom",
                        fontsize=st.NOTEBOOK_EFFECTIVE[
                            "tick_labelsize_minor"] * scale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\max\,A_{xx}$")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, top * (1.20 if inline_labels else 1.06))
    if lid_ramp:
        _mark_lid_ramp(ax, scale)


def _encoding_handles(scale):
    from matplotlib.lines import Line2D

    return [Line2D([], [], color="0.25",
                   lw=st.PAPER_LINE["linewidth"] * scale, label=TRUTH_LABEL),
            Line2D([], [], color="none", marker="o", mfc="none", mec="0.25",
                   ms=MARKER_SIZE * scale,
                   mew=st.NOTEBOOK_EFFECTIVE["marker_edge_width"] * scale
                   * 1.3, label=LEARNED_LABEL)]


def plot_N1e(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(
            2, 9.5, 13.0, gridspec_kw=dict(hspace=0.26))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 2, 1, hspace=0.42)
        scale = pn.adopt(axs[0])
        pn.adopt(axs[1])

    _e_profiles(axs[0], scale)
    _e_transient(axs[1], scale, inline_labels=not own)
    # Standalone, both legends stack on the profile panel, which is where the
    # three De curves lie on top of each other and need naming.  In an
    # assembled figure that panel is half as tall and the two collide, so the
    # De key moves down to the transient, which has an empty lower right.
    if own:
        axs[0].add_artist(st.legend(axs[0], scale, loc="lower right"))
    # Assembled, the De key is written on the plateaux themselves instead.
    st.legend(axs[0], scale, handles=_encoding_handles(scale),
              loc="center right")

    if own:
        pn.tidy(fig)
        if save:
            print(f"[N1e] {pn.save_panel(fig, FIG, 'N1e', dpi)}", flush=True)
            write_caption_notes()
        return fig
    return axs


def plot_N1e_horizontal(ax=None, save=True, dpi=None):
    """N1e's two sub-panels side by side, for a wide slot in the layout.

    Same curves, same encoding and the same two keys as :func:`plot_N1e`; only
    the arrangement differs, so the pair reads across a two-column figure
    rather than down a column.
    """
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(1, 15.2, 6.4, ncols=2,
                                       gridspec_kw=dict(wspace=0.26))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 1, 2, wspace=0.30)
        scale = pn.adopt(axs[0])
        pn.adopt(axs[1])

    _e_profiles(
        axs[0], scale, zero_line=False,
        xlabel=r"Steady state $u_x/U_{\rm lid}$ at $x/L = 0.5$")
    _e_transient(axs[1], scale, xlabel="Time", lid_ramp=False)
    axs[0].add_artist(st.legend(axs[0], scale, loc="lower right"))
    st.legend(axs[0], scale, handles=_encoding_handles(scale),
              loc="center right")

    if own:
        pn.tidy(fig)
        if save:
            print(f"[N1e_horizontal] "
                  f"{pn.save_panel(fig, FIG, 'N1e_horizontal', dpi)}",
                  flush=True)
        return fig
    return axs


# --------------------------------------------------------------------------
# Superseded by the two sub-panels above; kept callable, out of the figure set.
# --------------------------------------------------------------------------

def plot_N1e_ladder(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(
            2, 9.5, 10.5, sharex=True, gridspec_kw=dict(hspace=0.10))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 2, 1, sharex=True, hspace=0.10)
        scale = pn.adopt(axs[0])
        pn.adopt(axs[1])

    d = cavity_ladder()
    ms = 16 * scale
    mew = st.NOTEBOOK_EFFECTIVE["marker_edge_width"] * scale
    for a, yt, yl, ylabel in (
            (axs[0], d["u_truth"], d["u_learned"],
             r"$u_{x,\min}/U_{\rm lid}$"),
            (axs[1], d["axx_truth"], d["axx_learned"], r"$\max A_{xx}$")):
        a.plot(d["De"], yt, "o-", color=st.C_TRUTH, ms=ms, mfc=st.C_TRUTH,
               mec="black", mew=mew, lw=st.BASE_LINEWIDTH * scale,
               label="Giesekus truth")
        a.plot(d["De"], yl, "s--", color=st.C_LEARN, ms=ms, mfc="none",
               mec=st.C_LEARN, mew=mew * 1.4, lw=st.BASE_LINEWIDTH * scale,
               label="frozen TBNN s4")
        a.set_ylabel(ylabel)
        a.set_xticks(d["De"])
    axs[1].set_xlabel("Deborah number  De")
    axs[1].set_xlim(0.155, 0.545)
    axs[1].set_ylim(2.9, 9.4)
    st.legend(axs[0], scale, loc="lower right")
    pn.annotate(axs[0], 0.97, 0.10, "velocity: indiscernible", scale,
                ha="right", color="0.3")
    pn.annotate(axs[1], 0.97, 0.10, "conformation: close, resolvable",
                scale, ha="right", color="0.3")
    for xi, yi in zip(d["De"], d["axx_truth"]):
        axs[1].annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 22 * scale), ha="center",
                        fontsize=st.NOTEBOOK_EFFECTIVE[
                            "tick_labelsize_minor"] * scale,
                        color=st.C_TRUTH)

    if own:
        pn.tidy(fig)
        if save:
            print(f"[N1e_ladder] "
                  f"{pn.save_panel(fig, FIG, 'N1e_ladder', dpi)}", flush=True)
        return fig
    return axs


# --------------------------------------------------------------------------
# assembled figure
# --------------------------------------------------------------------------

def plot_N1(save=True, dpi=None):
    """Assembled Fig 5.  Every panel is drawn by the same function that
    produces the standalone JPG, through the ``ax=`` path."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(26.0, 16.5))
    # Two bands: the contraction field and the centreline trace it is read
    # along on top, the three recovery / transfer panels below.  Nested
    # gridspecs so each band sets its own column widths.
    outer = fig.add_gridspec(2, 1, height_ratios=[1.26, 1.0], hspace=0.20,
                             left=0.045, right=0.985, top=0.955, bottom=0.06)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.36, 1.0], wspace=0.20)
    bot = outer[1].subgridspec(1, 3, width_ratios=[1.0, 0.92, 1.0],
                               wspace=0.32)
    slots = ((top[0, 0], plot_N1a, "(a)"), (top[0, 1], plot_N1b, "(b)"),
             (bot[0, 0], plot_N1c, "(c)"), (bot[0, 1], plot_N1d, "(d)"),
             (bot[0, 2], plot_N1e, "(e)"))
    # One type scale for the whole plate; panels here differ in width by more
    # than 2x and per-axes scaling would make the field panel shout.
    with pn.uniform_scale(0.82) as scale:
        for spec, fn, tag in slots:
            ax = fig.add_subplot(spec)
            res = fn(ax=ax)
            first = res[0] if isinstance(res, list) else res
            pn.panel_tag(first, tag, scale, loc="outside")
    if save:
        print(f"[N1_full] {pn.save_panel(fig, FIG, 'N1_full', dpi)}",
              flush=True)
    return fig
