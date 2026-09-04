"""Figure SN3 (supplement S10) -- cavity transfer detail.

The frozen gie_A_s4 closure is carried, forward only, into a lid-driven cavity
it never saw in training.  Every panel here compares that run against the
Giesekus truth solver at the same configuration; panel d is the source
computation that N1e(i) reduces to a single scalar per De.
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo
from . import loaders as ld
from . import panels as pn
from . import style as st

FIG = "SN3"

DE_LADDER = ld.dp.CAVITY_DE
DE_FIELD = 0.50                     # De shown in the field panels a-c
DE_COLOR = st.C_DE                  # shared with N1e, which shows these runs
#: N1's name for the frozen closure, carried through this figure.
LEARNED = "TBNN prediction"
#: One math group, so mathtext does not open a gap around the "=".
DE_TAG = rf"$\mathrm{{De}} = {DE_FIELD:.2f}$"
#: Titles of the two difference panels, named after the quantity rather than
#: the arithmetic, as in SN1.
DIFF_TITLE = {"A_xx": r"Absolute difference in $A_{xx}$",
              "speed": "Absolute difference in velocity magnitude"}


def _diff_tag_shrink() -> float:
    """Difference titles at the same size as every other SN3 title.

    They are the two long ones, and inside the assembled figure -- where type
    is a larger fraction of the axes width -- at full size they overrun the
    axes onto the panel letter and the colourbar label, so there only they drop
    a step.
    """
    return 0.80 if pn.assembling() else 1.0


def _lw(scale, k=1.0):
    return k * st.BASE_LINEWIDTH * scale


def _grid(de: float = DE_FIELD) -> geo.CavityGrid:
    return geo.cavity_grid_from_config(ld.cavity_run("truth", de)["config"])


def _field(arm: str, name: str, de: float = DE_FIELD) -> np.ndarray:
    return np.asarray(ld.cavity_run(arm, de)["diag"][f"final_{name}"])


def _axx_limits():
    return (min(float(_field(a, "A_xx").min()) for a in ("truth", "learned")),
            max(float(_field(a, "A_xx").max()) for a in ("truth", "learned")))


def _U_lid(de: float = DE_FIELD) -> float:
    return float(ld.cavity_run("truth", de)["config"]["U_lid"])


def _speed(arm: str, de: float = DE_FIELD) -> np.ndarray:
    """|u| at cell centres, in units of the lid speed."""
    return np.hypot(_field(arm, "u", de), _field(arm, "v", de)) / _U_lid(de)


def _speed_error(de: float = DE_FIELD) -> np.ndarray:
    """|u_learned - u_truth|, same normalisation, so the two panels compare."""
    du = _field("learned", "u", de) - _field("truth", "u", de)
    dv = _field("learned", "v", de) - _field("truth", "v", de)
    return np.hypot(du, dv) / _U_lid(de)


def _cavity_panel(ax, field, *, cmap, label, scale, vmin=None, vmax=None,
                  tag=None, streamlines=None, extend="neither",
                  bc=None, cb_rotation=0.0, tag_shrink=1.0,
                  label_align="center"):
    grid = _grid()
    pg = geo.cavity_plot_grid(grid)
    if bc is None:
        F = geo.cavity_to_plot(field, grid, pg, velocity=False)
    else:
        # Speed panels: no-slip on three walls, the lid speed on the fourth,
        # so the boundary is drawn as the solver imposed it rather than
        # extrapolated from the first cell.
        F = geo.cavity_to_plot(field, grid, pg, velocity=True,
                               lid_component=bc[0], U_lid=bc[1])
    art = pn.field_panel(ax, pg, F, cmap=cmap, vmin=vmin, vmax=vmax,
                         scale=scale, extend=extend, contour_color="0.55",
                         contour_lw=0.7)
    if streamlines is not None:
        u, v = streamlines
        c = grid.centres
        # White: A_xx is dark over most of the cavity, so a dark streamline
        # would be invisible where the vortex actually is.
        ax.streamplot(c, c, u.T, v.T, color="white", linewidth=0.9 * scale,
                      density=1.0, arrowsize=1.0 * scale, zorder=6)
    ax.set_xlim(0.0, grid.L)
    ax.set_ylim(0.0, grid.L)
    ax.set_aspect("equal")
    ax.set_xlabel("$x/L$")
    ax.set_ylabel("$y/L$")
    st.style_axes(ax, scale)
    pn.colorbar(ax.get_figure(), art, ax, label, scale, rotation=cb_rotation,
                label_align=label_align)
    if tag:
        # Above the axes, not printed on the field: white-on-field text cannot
        # be cropped away if the caption ends up carrying the same words.
        ax.set_title(tag, pad=8.0 * scale,
                     fontsize=st.NOTEBOOK_EFFECTIVE["axes_label_fontsize"]
                     * scale * tag_shrink)
    return art


def _make_field(name, builder, width=11.0, height=8.6, axes_width=6.6):
    def plot(ax=None, save=True, dpi=None):
        own = ax is None
        if own:
            fig, ax, scale = pn.new_panel(width, height, axes_width=axes_width)
        else:
            fig, scale = ax.get_figure(), pn.adopt(ax)
        builder(ax, scale)
        return pn.finish(fig, ax, FIG, name, save, dpi, own)

    plot.__name__ = f"plot_{name}"
    plot.__qualname__ = plot.__name__
    return plot


# --------------------------------------------------------------------------
# a, b -- A_xx with streamlines; c -- its error
# --------------------------------------------------------------------------

def _sn3a(ax, scale):
    lo, hi = _axx_limits()
    _cavity_panel(ax, _field("truth", "A_xx"), cmap=st.CMAP_STRETCH,
                  label=r"$A_{xx}$", scale=scale, vmin=lo, vmax=hi,
                  tag=f"Ground truth, {DE_TAG}",
                  streamlines=(_field("truth", "u"), _field("truth", "v")))


def _sn3b(ax, scale):
    lo, hi = _axx_limits()
    _cavity_panel(ax, _field("learned", "A_xx"), cmap=st.CMAP_STRETCH,
                  label=r"$A_{xx}$", scale=scale, vmin=lo, vmax=hi,
                  tag=f"{LEARNED}, {DE_TAG}",
                  streamlines=(_field("learned", "u"), _field("learned", "v")))


def _sn3c(ax, scale):
    d = np.abs(_field("learned", "A_xx") - _field("truth", "A_xx"))
    _cavity_panel(ax, d, cmap=st.CMAP_ERROR, label=r"$|\Delta A_{xx}|$",
                  scale=scale, vmin=0.0, tag=DIFF_TITLE["A_xx"],
                  tag_shrink=_diff_tag_shrink(), label_align="left")
    rel = ld.cavity_metrics()[f"De{DE_FIELD:.2f}"]["relative_L2_A_xx"]
    pn.annotate(ax, 0.015, 0.03, rf"$\mathrm{{rel.}}\ L_2 = {rel:.4f}$", scale,
                va="bottom", color="white",
                bbox=dict(facecolor="0.12", alpha=0.65, edgecolor="none",
                          pad=2.5))


plot_SN3a = _make_field("SN3a", _sn3a)
plot_SN3b = _make_field("SN3b", _sn3b)
plot_SN3c = _make_field("SN3c", _sn3c)


# --------------------------------------------------------------------------
# d -- centreline profiles, the source computation for N1e(i)
# --------------------------------------------------------------------------

def plot_SN3d(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(2, 10.0, 12.0,
                                       gridspec_kw=dict(hspace=0.30))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 2, 1, hspace=0.30)
        scale = pn.adopt(axs[0])
        pn.adopt(axs[1])

    for de in DE_LADDER:
        for arm, ls, lw in (("truth", "-", 1.0), ("learned", "--", 0.9)):
            c = ld.cavity_centreline(arm, de)
            axs[0].plot(c["u_of_y"] / c["U_lid"], c["coord"], ls,
                        color=DE_COLOR[de], lw=_lw(scale, lw),
                        label=f"De {de:.2f}" if arm == "truth" else None)
            axs[1].plot(c["coord"], c["v_of_x"] / c["U_lid"], ls,
                        color=DE_COLOR[de], lw=_lw(scale, lw))
    axs[0].axvline(0.0, color="0.6", lw=_lw(scale, 0.4), zorder=0)
    axs[0].set_xlabel(r"$u_x/U_{\rm lid}$ at $x = 0.5$")
    axs[0].set_ylabel("$y/L$")
    st.legend(axs[0], scale, loc="upper left")
    pn.annotate(axs[0], 0.98, 0.06,
                f"solid ground truth, dashed {LEARNED}", scale,
                ha="right", color="0.3")

    axs[1].axhline(0.0, color="0.6", lw=_lw(scale, 0.4), zorder=0)
    axs[1].set_xlabel("$x/L$")
    axs[1].set_ylabel(r"$u_y/U_{\rm lid}$ at $y = 0.5$")

    if own:
        pn.tidy(fig)
        if save:
            print(f"[SN3d] {pn.save_panel(fig, FIG, 'SN3d', dpi)}", flush=True)
        return fig
    return axs


# --------------------------------------------------------------------------
# e -- steadiness histories
# --------------------------------------------------------------------------

HISTORIES = (("ke", "kinetic energy"),
             ("max_Axx", r"$\max A_{xx}$"),
             ("psi_min_over_UL", r"$\psi_{\min}/(U L)$"))


def plot_SN3e(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(3, 10.0, 13.0, sharex=True,
                                       gridspec_kw=dict(hspace=0.12))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 3, 1, sharex=True, hspace=0.12)
        scale = pn.adopt(axs[0])
        for a in axs[1:]:
            pn.adopt(a)

    for a, (key, label) in zip(axs, HISTORIES):
        for de in DE_LADDER:
            for arm, ls, lw in (("truth", "-", 1.0), ("learned", "--", 0.9)):
                h = ld.cavity_history(arm, de)
                a.plot(h["t"], h[key], ls, color=DE_COLOR[de],
                       lw=_lw(scale, lw),
                       label=(f"De {de:.2f}" if arm == "truth"
                              and key == "ke" else None))
        a.set_ylabel(label)
    axs[0].set_yscale("log")
    axs[-1].set_xlabel("$t$")
    st.legend(axs[0], scale, loc="lower right", ncol=3, columnspacing=1.0)
    pn.annotate(axs[1], 0.98, 0.10,
                f"solid ground truth, dashed {LEARNED}", scale,
                ha="right", color="0.3")

    if own:
        pn.tidy(fig)
        if save:
            print(f"[SN3e] {pn.save_panel(fig, FIG, 'SN3e', dpi)}", flush=True)
        return fig
    return axs


# --------------------------------------------------------------------------
# f -- vortex strength, eye position and SPD margin vs De
# --------------------------------------------------------------------------

def plot_SN3f(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(3, 10.0, 13.0, sharex=True,
                                       gridspec_kw=dict(hspace=0.12))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 3, 1, sharex=True, hspace=0.12)
        scale = pn.adopt(axs[0])
        for a in axs[1:]:
            pn.adopt(a)

    m = ld.cavity_ladder_metrics()
    de = m["De"]
    ms = 14 * scale
    mew = st.NOTEBOOK_EFFECTIVE["marker_edge_width"] * scale

    def _pair(a, truth, learned, marker="o"):
        a.plot(de, truth, marker + "-", color=st.C_TRUTH, ms=ms, mfc=st.C_TRUTH,
               mew=mew, lw=_lw(scale, 0.8), label="Ground truth")
        a.plot(de, learned, marker + "--", color=st.C_LEARN, ms=ms,
               mfc="none", mew=mew, lw=_lw(scale, 0.8), label=LEARNED)

    _pair(axs[0], m["truth_psi_min_over_UL"], m["learned_psi_min_over_UL"])
    axs[0].set_ylabel(r"$\psi_{\min}/(UL)$")
    st.legend(axs[0], scale, loc="lower right", ncol=2, columnspacing=1.0)

    axs[1].plot(de, m["truth_eye"][:, 0], "o-", color=st.C_TRUTH, ms=ms,
                mew=mew, lw=_lw(scale, 0.8), label="$x$ truth")
    axs[1].plot(de, m["learned_eye"][:, 0], "o--", color=st.C_LEARN, ms=ms,
                mfc="none", mew=mew, lw=_lw(scale, 0.8), label="$x$ TBNN")
    axs[1].plot(de, m["truth_eye"][:, 1], "^-", color=st.C_TRUTHFAM, ms=ms,
                mew=mew, lw=_lw(scale, 0.8), label="$y$ truth")
    axs[1].plot(de, m["learned_eye"][:, 1], "^--", color=st.C_DRIVE[3], ms=ms,
                mfc="none", mew=mew, lw=_lw(scale, 0.8), label="$y$ TBNN")
    axs[1].set_ylabel("vortex eye")
    axs[1].set_ylim(0.40, 1.30)
    st.legend(axs[1], scale, loc="upper center", ncol=2, columnspacing=0.8,
              handlelength=1.3,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.8)

    _pair(axs[2], m["truth_min_eig"], m["learned_min_eig"], marker="s")
    for x, v in zip(de, m["truth_min_eig"]):
        axs[2].annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                        xytext=(0, 16 * scale), ha="center",
                        fontsize=st.NOTEBOOK_EFFECTIVE[
                            "tick_labelsize_minor"] * scale)
    axs[2].set_ylabel(r"SPD margin $\min\,\mathrm{eig}\,\mathbf{A}$")
    axs[2].set_ylim(0.16, 0.45)
    axs[2].set_xlabel("Deborah number De")
    axs[2].set_xticks(list(de))
    axs[2].set_xlim(0.16, 0.54)

    if own:
        pn.tidy(fig)
        if save:
            print(f"[SN3f] {pn.save_panel(fig, FIG, 'SN3f', dpi)}", flush=True)
        return fig
    return axs


# --------------------------------------------------------------------------
# g, h -- speed with streamlines; i -- its error
# --------------------------------------------------------------------------

def _speed_limits():
    return (0.0, max(float(_speed(a).max()) for a in ("truth", "learned")))


def _speed_panel(ax, arm, scale, tag):
    lo, hi = _speed_limits()
    _cavity_panel(ax, _speed(arm), cmap=st.CMAP_SPEED,
                  label=r"$|\mathbf{u}|/U_{\rm lid}$", scale=scale,
                  vmin=lo, vmax=hi, tag=tag, bc=("u", 1.0), cb_rotation=90.0,
                  streamlines=(_field(arm, "u"), _field(arm, "v")))


def _sn3g(ax, scale):
    _speed_panel(ax, "truth", scale, f"Ground truth, {DE_TAG}")


def _sn3h(ax, scale):
    _speed_panel(ax, "learned", scale, f"{LEARNED}, {DE_TAG}")


def _sn3i(ax, scale):
    _cavity_panel(ax, _speed_error(), cmap=st.CMAP_ERROR,
                  label=r"$|\Delta\mathbf{u}|/U_{\rm lid}$", scale=scale,
                  vmin=0.0, tag=DIFF_TITLE["speed"], bc=(None, 0.0),
                  cb_rotation=90.0, tag_shrink=_diff_tag_shrink())
    rms = ld.cavity_metrics()[f"De{DE_FIELD:.2f}"]["velocity_rms_over_U_lid"]
    man, exp = f"{rms:.2e}".split("e")
    pn.annotate(ax, 0.015, 0.03,
                rf"$\mathrm{{rms}} = {man}\times10^{{{int(exp)}}}$", scale,
                va="bottom",
                color="white",
                bbox=dict(facecolor="0.12", alpha=0.65, edgecolor="none",
                          pad=2.5))


plot_SN3g = _make_field("SN3g", _sn3g)
plot_SN3h = _make_field("SN3h", _sn3h)
plot_SN3i = _make_field("SN3i", _sn3i)


# --------------------------------------------------------------------------
# assembled figure
# --------------------------------------------------------------------------

def plot_SN3(save=True, dpi=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(24.0, 28.0))
    gs = fig.add_gridspec(3, 3, hspace=0.28, wspace=0.32, left=0.05,
                          right=0.97, top=0.965, bottom=0.045)
    slots = ((gs[0, 0], plot_SN3a, "(a)"), (gs[0, 1], plot_SN3b, "(b)"),
             (gs[0, 2], plot_SN3c, "(c)"), (gs[1, 0], plot_SN3d, "(d)"),
             (gs[1, 1], plot_SN3e, "(e)"), (gs[1, 2], plot_SN3f, "(f)"),
             (gs[2, 0], plot_SN3g, "(g)"), (gs[2, 1], plot_SN3h, "(h)"),
             (gs[2, 2], plot_SN3i, "(i)"))
    with pn.uniform_scale(0.60) as scale:
        for spec, fn, tag in slots:
            ax = fig.add_subplot(spec)
            res = fn(ax=ax)
            first = res[0] if isinstance(res, list) else res
            pn.panel_tag(first, tag, scale, loc="outside")
    if save:
        print(f"[SN3_full] {pn.save_panel(fig, FIG, 'SN3_full', dpi)}",
              flush=True)
    return fig
