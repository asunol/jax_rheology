"""Figure SN4 (supplement S11) -- FENE-P single-rate recovery.

Direct mirror of N1 for the FENE-P fluid.  The representative network is chosen
by best parameter recovery (mean absolute relative error over the four
parameters), not by BIC margin; :func:`representative_table` prints the
comparison and :data:`REPRESENTATIVE` records the winner.

Truth and OB-init contraction fields come from the archived campaign
regeneration, which is at the same config hash as the representative run; the
learned field is the forward-only run written by
``paper_figs/run_learned_forward.py``.
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo
from . import loaders as ld
from . import panels as pn
from . import style as st
from .fig_n1 import (XLIM, XLIM_FIELD, YLIM_FIELD, _centreline, _cross_section,
                     bic_bar_single)

FIG = "SN4"

CANDIDATES = ld.FENE_SINGLE_RATE_CANDIDATES
PARAMS = (("eta_p", r"$\eta_p$"), ("nu_s", r"$\eta_s$"),
          ("Lsq", r"$L^2$"), ("lam", r"$\lambda$"))
#: Five hues that stay apart in print and for the common colour deficiencies;
#: the campaign palette put two reds next to each other.
SEED_COLOR = dict(zip(CANDIDATES, ("#1f77b4", "#e08214", "#56B4E9", "#9467bd",
                                   "#d1495b")))
#: The five runs are one training recipe repeated, so they are seeds, not
#: variants; the archive names (R3, s1..s4) mean nothing to a reader.
SEED_LABEL = {t: f"Seed {i + 1}" for i, t in enumerate(CANDIDATES)}


def representative_table() -> list[dict]:
    """All five single-rate headline runs, ranked by parameter recovery.

    Criterion, fixed in advance: the mean absolute relative error over
    ``eta_p``, ``lam``, ``nu_s`` and ``L^2``, equally weighted.  ``L^2`` is
    ``extension_length`` squared, since the battery stores L.
    """
    rows = []
    for t in CANDIDATES:
        rec, err = ld.fene_recovery(t), ld.fene_recovery_errors(t)
        winner, margin = ld.battery_winner(t)
        rows.append({"target": t, "label": ld.FENE_LABEL[t], **rec,
                     **{f"err_{k}": v for k, v in err.items()},
                     "mare": ld.fene_mare(t), "winner": winner,
                     "margin": margin,
                     "resid_x_floor": rec["mse"] / ld.NOISE_FLOOR})
    return sorted(rows, key=lambda r: r["mare"])


REPRESENTATIVE = representative_table()[0]["target"]


def _lw(scale, k=1.0):
    return k * st.BASE_LINEWIDTH * scale


def _grid():
    return geo.contraction_grid_from_archive(ld.fene_repr_fields())


def _trace(arm: str) -> np.ndarray:
    f = ld.fene_repr_fields()
    return f[f"{arm}_A_xx"] + f[f"{arm}_A_yy"] + f[f"{arm}_A_zz"]


# --------------------------------------------------------------------------
# a -- contraction u_x, truth above / learned below
# --------------------------------------------------------------------------

def plot_SN4a(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(13.0, 7.4, axes_width=9.5)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    f = ld.fene_repr_fields()
    grid = _grid()
    pg = geo.contraction_plot_grid(grid)
    cfg = ld.fene_config(REPRESENTATIVE)
    _, activation, _, _ = geo.roi_fields(cfg, grid)

    Ut = geo.contraction_to_plot(geo.u_faces_to_centres(f["truth_u"]), grid,
                                 pg, velocity=True)
    Ul = geo.contraction_to_plot(geo.u_faces_to_centres(f["learned_u"]), grid,
                                 pg, velocity=True)
    F = geo.mirrored_split(Ut, Ul, pg)
    vmax = float(np.nanmax(np.abs(F)))
    m = pn.field_panel(ax, pg, F, cmap=st.CMAP_SPEED,
                       levels=np.linspace(0.0, vmax, pn.N_LEVELS),
                       scale=scale, extend="both",
                       solids=grid.solid_rectangles())
    pn.draw_roi(ax, grid, activation, scale)
    pn.split_labels(ax, "FENE-P truth", "Trained TBNN", scale)
    pn.contraction_axes(ax, grid, scale, XLIM_FIELD, YLIM_FIELD)
    pn.colorbar(fig, m, ax, r"$u_x$", scale)
    return pn.finish(fig, ax, FIG, "SN4a", save, dpi, own)


# --------------------------------------------------------------------------
# b -- tr A along the centreline, as N1b
# --------------------------------------------------------------------------

def plot_SN4b(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 8.5, axes_width=8.6)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    grid = _grid()
    x = grid.xc / grid.H
    for label, key, kw in st.PROFILE_ROLES:
        ax.plot(x, _centreline(_trace(key), grid), label=label,
                lw=st.PAPER_LINE["linewidth"] * scale, **kw)
    ax.set_xlim(*XLIM)
    ax.set_xlabel("$x/H$")
    ax.set_ylabel(r"$\mathrm{tr}\,\mathbf{A}$")
    pn.annotate(ax, 0.97, 0.90, "centreline, $y=0$", scale, ha="right",
                color="0.3")
    st.paper_legend(ax, scale, loc="upper left")
    return pn.finish(fig, ax, FIG, "SN4b", save, dpi, own)


def plot_SN4b_throat(ax=None, save=True, dpi=None):
    """Retired: the throat cross-section N1b dropped, kept for reference."""
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 8.5, axes_width=8.6)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    grid = _grid()
    core = np.abs(grid.yc) <= grid.H
    for label, key, kw in st.PROFILE_ROLES:
        ax.plot(grid.yc[core] / grid.H, _cross_section(_trace(key), grid)[core],
                label=label, lw=st.PAPER_LINE["linewidth"] * scale, **kw)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("$y/H$")
    ax.set_ylabel(r"$\mathrm{tr}\,\mathbf{A}$")
    pn.annotate(ax, 0.5, 0.06, "throat cross-section, $x=0$", scale,
                ha="center", color="0.3")
    st.paper_legend(ax, scale, loc="upper left")
    return pn.finish(fig, ax, FIG, "SN4b_throat", save, dpi, own)


# --------------------------------------------------------------------------
# c -- model selection, as N1c
# --------------------------------------------------------------------------

def plot_SN4c(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 8.5)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)
    bic_bar_single(ax, scale, REPRESENTATIVE)
    return pn.finish(fig, ax, FIG, "SN4c", save, dpi, own)


# --------------------------------------------------------------------------
# d -- four-parameter recovery across the five initialisations
# --------------------------------------------------------------------------

def recovery_ensemble() -> dict:
    """Ratio-to-truth of each parameter for all five runs, plus mean and s.d."""
    out = {"targets": list(CANDIDATES)}
    for key, _ in PARAMS:
        vals = np.array([ld.fene_recovery(t)[key] / ld.FENE_TRUTH[key]
                         for t in CANDIDATES])
        out[key] = vals
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_sd"] = float(vals.std(ddof=1))
    return out


#: param key, printed name, decimals -- the table's rows.
TABLE_ROWS = (("eta_p", r"$\eta_p$", 3), ("lam", r"$\lambda$", 3),
              ("nu_s", r"$\eta_s$", 3), ("Lsq", r"$L^2$", 2))


def _row_values(key: str) -> tuple[float, np.ndarray]:
    """Truth and the five-seed recovered values for one table row."""
    if key == "Gp":
        truth = ld.FENE_TRUTH["eta_p"] / ld.FENE_TRUTH["lam"]
        vals = np.array([ld.fene_recovery(t)["eta_p"] / ld.fene_recovery(t)["lam"]
                         for t in CANDIDATES])
        return truth, vals
    return ld.FENE_TRUTH[key], np.array([ld.fene_recovery(t)[key]
                                         for t in CANDIDATES])


def recovery_table(rows=TABLE_ROWS) -> list[list[str]]:
    """truth against the five-seed mean, with the seed spread as the +-.

    The uncertainty is the sample standard deviation over the five runs, which
    is a spread across initialisations at one fixed schedule, not a posterior
    width; the seeds are the only repeat measurement here.
    """
    out = []
    for key, label, nd in rows:
        truth, vals = _row_values(key)
        out.append([label, f"{truth:.{nd}f}",
                    f"{vals.mean():.{nd}f} $\\pm$ {vals.std(ddof=1):.{nd}f}"])
    return out


def plot_SN4d_table(ax=None, save=True, dpi=None):
    """Alternate to SN4d: the same recovery as a table, styled as N1d."""
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(9.6, 3.6, axes_width=8.0)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)
    pn.table_panel(ax, ("param", "truth", "recovered"), recovery_table(),
                   scale, col_widths=[0.24, 0.26, 0.46], row_height=3.0)
    return pn.finish(fig, ax, FIG, "SN4d_table", save, dpi, own)


#: Same four rows as SN4d_table, but the polymer viscosity is reported as
#: G_p = eta_p / lambda so the table matches N1d's first row.
TABLE_ROWS_GP = (("Gp", r"$G_p$", 3), ("lam", r"$\lambda$", 3),
                 ("nu_s", r"$\eta_s$", 3), ("Lsq", r"$L^2$", 2))


def plot_SN4d_table_alt(ax=None, save=True, dpi=None):
    """SN4d_table with G_p in place of eta_p."""
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(9.6, 3.6, axes_width=8.0)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)
    pn.table_panel(ax, ("param", "truth", "recovered"),
                   recovery_table(TABLE_ROWS_GP),
                   scale, col_widths=[0.24, 0.26, 0.46], row_height=3.0)
    return pn.finish(fig, ax, FIG, "SN4d_table_alt", save, dpi, own)


def plot_SN4d(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 8.0)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    e = recovery_ensemble()
    x = np.arange(len(PARAMS), dtype=float)
    ax.axhline(1.0, color=st.C_TRUTH, ls="-.", lw=_lw(scale, 0.9),
               label="Ground truth")
    for i, (key, _) in enumerate(PARAMS):
        for j, t in enumerate(CANDIDATES):
            ax.plot(x[i] + (j - 2) * 0.11, e[key][j], "o",
                    color=SEED_COLOR[t], ms=10 * scale, mfc="none",
                    mew=1.6 * scale,
                    label=SEED_LABEL[t] if i == 0 else None)
        ax.errorbar(x[i], e[f"{key}_mean"], yerr=e[f"{key}_sd"], fmt="s",
                    color=st.C_TRUTH, ms=13 * scale, mfc=st.C_LEARN,
                    mec="black", mew=1.6 * scale, capsize=8 * scale,
                    elinewidth=_lw(scale, 0.6), zorder=5,
                    label="Seed mean $\\pm$ s.d." if i == 0 else None)
        ax.annotate(f"{100*(e[f'{key}_mean']-1):+.0f}%",
                    (x[i], e[f"{key}_mean"] + e[f"{key}_sd"]),
                    textcoords="offset points",
                    xytext=(13 * scale, 3 * scale), ha="left",
                    fontsize=st.NOTEBOOK_EFFECTIVE[
                        "tick_labelsize_minor"] * scale)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in PARAMS])
    ax.set_xlim(-0.5, len(PARAMS) - 0.5)
    ax.set_ylim(0.76, 1.32)
    ax.set_ylabel("recovered / truth")
    st.legend(ax, scale, loc="upper left", ncol=2, columnspacing=1.0,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.8)
    pn.annotate(ax, 0.02, 0.04, "spread across five seeds, fixed schedule",
                scale, ha="left", va="bottom", color="0.3")
    if own and save:
        write_notes()
    return pn.finish(fig, ax, FIG, "SN4d", save, dpi, own)


def notes() -> str:
    return (
        "Figure SN4 -- FENE-P single-rate recovery.\n\n"
        "SPREAD DENOMINATOR.  On SN4d and SN4d_table the +- is the sample\n"
        "standard deviation across five SEEDS at one fixed schedule.  That is\n"
        "a different quantity from the three SN2a/SN2b curves, which are\n"
        "across TRAINING SCHEDULES at a fixed initialisation.  Do not compare\n"
        "them as like.\n"
    )


def write_notes():
    out = ld.dp.out_dir(FIG) / "SN4_notes.txt"
    out.write_text(notes())
    print(f"[SN4_notes] {out}", flush=True)
    return out


# --------------------------------------------------------------------------
# e -- training curves, five runs
# --------------------------------------------------------------------------

def plot_SN4e(ax=None, save=True, dpi=None):
    """Loss and eta_p against iteration, side by side, as in N1e_horizontal."""
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(1, 15.2, 6.4, ncols=2,
                                       gridspec_kw=dict(wspace=0.26))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 1, 2, wspace=0.30)
        scale = pn.adopt(axs[0])
        pn.adopt(axs[1])

    lw = st.PAPER_LINE["linewidth"] * scale
    for t in CANDIDATES:
        p = ld.accepted_progress(ld.fene_progress(t))
        axs[0].plot(p["step"], p["loss"], "-", color=SEED_COLOR[t], lw=lw,
                    label=SEED_LABEL[t])
        axs[1].plot(p["step"], p["eta_p"], "-", color=SEED_COLOR[t], lw=lw)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Iteration")
    # Anchored on the left axes but sized to span both: one seed key for the
    # pair, above the panels and clear of every curve.
    st.legend(axs[0], scale, loc="lower left", bbox_to_anchor=(0.0, 1.005),
              ncol=5, columnspacing=1.1, handlelength=1.3, frameon=False,
              borderpad=0.1,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.8)

    axs[1].axhline(ld.FENE_TRUTH["eta_p"], color=st.C_TRUTH, ls="-.",
                   lw=lw * 0.7, label="Ground truth")
    axs[1].set_ylabel(r"$\eta_p = G_p\lambda$")
    axs[1].set_ylim(0.85, 2.85)
    axs[1].set_xlabel("Iteration")
    st.legend(axs[1], scale, loc="lower right", ncol=1,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.8)

    if own:
        pn.tidy(fig)
        if save:
            print(f"[SN4e] {pn.save_panel(fig, FIG, 'SN4e', dpi)}", flush=True)
        return fig
    return axs


def plot_SN4e_alt(ax=None, save=True, dpi=None):
    """SN4e with G_p on the right panel instead of eta_p."""
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(1, 15.2, 6.4, ncols=2,
                                       gridspec_kw=dict(wspace=0.26))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 1, 2, wspace=0.30)
        scale = pn.adopt(axs[0])
        pn.adopt(axs[1])

    lw = st.PAPER_LINE["linewidth"] * scale
    gp_truth = ld.FENE_TRUTH["eta_p"] / ld.FENE_TRUTH["lam"]
    for t in CANDIDATES:
        p = ld.accepted_progress(ld.fene_progress(t))
        axs[0].plot(p["step"], p["loss"], "-", color=SEED_COLOR[t], lw=lw,
                    label=SEED_LABEL[t])
        axs[1].plot(p["step"], p["Gp"], "-", color=SEED_COLOR[t], lw=lw)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Iteration")
    st.legend(axs[0], scale, loc="lower left", bbox_to_anchor=(0.0, 1.005),
              ncol=5, columnspacing=1.1, handlelength=1.3, frameon=False,
              borderpad=0.1,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.8)

    axs[1].axhline(gp_truth, color=st.C_TRUTH, ls="-.",
                   lw=lw * 0.7, label="Ground truth")
    axs[1].set_ylabel(r"$G_p$")
    axs[1].set_ylim(0.85, 5.15)
    axs[1].set_xlabel("Iteration")
    st.legend(axs[1], scale, loc="lower right", ncol=1,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.8)

    if own:
        pn.tidy(fig)
        if save:
            print(f"[SN4e_alt] {pn.save_panel(fig, FIG, 'SN4e_alt', dpi)}",
                  flush=True)
        return fig
    return axs


# --------------------------------------------------------------------------
# assembled figure
# --------------------------------------------------------------------------

def plot_SN4(save=True, dpi=None):
    import matplotlib.pyplot as plt

    # Four columns: the field panel and the recovery scatter take two each,
    # and so does (e), which is itself two axes side by side.
    fig = plt.figure(figsize=(30.0, 16.0))
    gs = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.34, left=0.05,
                          right=0.98, top=0.94, bottom=0.07)
    slots = ((gs[0, 0:2], plot_SN4a, "(a)"), (gs[0, 2], plot_SN4b, "(b)"),
             (gs[0, 3], plot_SN4c, "(c)"), (gs[1, 0:2], plot_SN4d, "(d)"),
             (gs[1, 2:4], plot_SN4e, "(e)"))
    with pn.uniform_scale(0.62) as scale:
        for spec, fn, tag in slots:
            ax = fig.add_subplot(spec)
            res = fn(ax=ax)
            first = res[0] if isinstance(res, list) else res
            pn.panel_tag(first, tag, scale, loc="outside")
    if save:
        print(f"[SN4_full] {pn.save_panel(fig, FIG, 'SN4_full', dpi)}",
              flush=True)
    return fig
