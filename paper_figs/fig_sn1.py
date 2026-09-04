"""Figure SN1 (supplement S7) -- contraction fields and errors.

Same July-12 archives as N1: the Giesekus truth forward and the frozen
gie_A_s4 learned forward at the July-10 production configuration.  Every
panel obeys the field-plot rules: staggered velocity interpolated to cell
centres, zero enforced on every no-slip wall and inside the blocks, blocks in
flat grey cut on grid lines.

Five panels.  The two field comparisons are split the way N1a is -- truth in
the upper half, trained TBNN in the lower half of the same axes -- and the
three difference maps carry their name as a black title above the axes, so the
words can be cropped if the caption repeats them.  The previous set, one axes
per arm, is in ``final_figures/SN1/original2/`` and still callable through the
``plot_SN1_*`` functions at the bottom of this module.
"""
from __future__ import annotations

import numpy as np

from . import data_paths as dp
from . import geometry as geo
from . import loaders as ld
from . import panels as pn
from . import style as st
from .fig_n1 import XLIM_FIELD, YLIM_FIELD, _grid, _trace, contraction_metrics

FIG = "SN1"
TRUTH_TAG = "Giesekus truth"
LEARNED_TAG = "Trained TBNN"
DIFF_TAG = r"Trained TBNN $-$ truth"
DIFF_TITLE = {
    "ux": r"Absolute difference in $x$-velocity",
    "uy": r"Absolute difference in $y$-velocity",
    "trA": r"Absolute difference in $\mathrm{tr}\,\mathbf{A}$",
}


def _velocity_centres(arm: str, component: str) -> np.ndarray:
    f = ld.gie_fields()
    if component == "u":
        return geo.u_faces_to_centres(f[f"{arm}_u"])
    return geo.v_faces_to_centres(f[f"{arm}_v"])


def _panel(ax, field, *, cmap, label, velocity, vmin=None, vmax=None,
           scale=1.0, roi=False, tag=None, symmetric=False, mirror=None,
           title=None, split_color="white"):
    grid = _grid()
    pg = geo.contraction_plot_grid(grid)
    F = geo.contraction_to_plot(field, grid, pg, velocity=velocity)
    if mirror is not None:
        Fb = geo.contraction_to_plot(mirror, grid, pg, velocity=velocity)
        F = geo.mirrored_split(F, Fb, pg)
    if symmetric:
        m = float(np.nanmax(np.abs(F)))
        vmin, vmax = -m, m
    fig = ax.get_figure()
    art = pn.field_panel(ax, pg, F, cmap=cmap, vmin=vmin, vmax=vmax,
                         scale=scale, extend="both",
                         contour_color="white" if not symmetric else "0.25",
                         solids=grid.solid_rectangles())
    if mirror is not None:
        pn.split_labels(ax, TRUTH_TAG, LEARNED_TAG, scale, color=split_color)
    if roi:
        _, activation, _, _ = geo.roi_fields(ld.gie_config(), grid)
        pn.draw_roi(ax, grid, activation, scale)
    pn.contraction_axes(ax, grid, scale, XLIM_FIELD, YLIM_FIELD)
    pn.colorbar(fig, art, ax, label, scale)
    if tag:
        # Above the grey blocks, which carry zorder 5.
        pn.annotate(ax, 0.015, 0.94, tag, scale, va="top", color="white",
                    fontweight="bold", zorder=8)
    if title:
        # Black, above the axes: printed on the field in white it could not be
        # cropped away if the caption ends up carrying the same words.  Left
        # aligned and at tick size, because centred at label size it runs into
        # the colourbar's own label.
        ax.set_title(title, pad=8.0 * scale, loc="left",
                     fontsize=st.NOTEBOOK_EFFECTIVE["tick_labelsize_major"]
                     * scale)
    return art


def _make(name, builder, width=13.0, height=7.4, axes_width=9.5):
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
# a -- u_y, truth above the centreline and trained TBNN below
# --------------------------------------------------------------------------

def _uy_limits():
    a = _velocity_centres("truth", "v")
    b = _velocity_centres("learned", "v")
    m = max(float(np.abs(a).max()), float(np.abs(b).max()))
    return -m, m


def _sn1a(ax, scale):
    lo, hi = _uy_limits()
    # Split labels in near-black: RdBu_r is white through zero, and the upper
    # corners of this field are close to zero.
    _panel(ax, _velocity_centres("truth", "v"),
           mirror=_velocity_centres("learned", "v"), cmap=st.CMAP_SIGNED,
           label=r"$u_y$", velocity=True, vmin=lo, vmax=hi, scale=scale,
           split_color="0.15")


# --------------------------------------------------------------------------
# b, c -- absolute velocity differences
# --------------------------------------------------------------------------

def _sn1b(ax, scale):
    d = np.abs(_velocity_centres("learned", "u") - _velocity_centres("truth", "u"))
    _panel(ax, d, cmap=st.CMAP_ERROR, label=r"$|\Delta u_x|$", velocity=True,
           vmin=0.0, vmax=_robust_max([d], 99.9), scale=scale, roi=True,
           title=DIFF_TITLE["ux"])


def _sn1c(ax, scale):
    d = np.abs(_velocity_centres("learned", "v") - _velocity_centres("truth", "v"))
    _panel(ax, d, cmap=st.CMAP_ERROR, label=r"$|\Delta u_y|$", velocity=True,
           vmin=0.0, vmax=_robust_max([d], 99.9), scale=scale, roi=True,
           title=DIFF_TITLE["uy"])


# --------------------------------------------------------------------------
# d -- tr A, truth above the centreline and trained TBNN below
# --------------------------------------------------------------------------

def _robust_max(fields, pct: float) -> float:
    """Upper colour limit from a percentile over the fluid only.

    The re-entrant corner cells carry a stretch spike ~2.5x the bulk maximum;
    scaling to it would crush every other feature, so the map is scaled to the
    given percentile and the colourbar is extended.
    """
    m = _grid().fluid_mask()
    return float(max(np.percentile(f[m], pct) for f in fields))


def _tra_limits():
    return 3.0, _robust_max([_trace("truth"), _trace("learned")], 99.5)


def _sn1d(ax, scale):
    lo, hi = _tra_limits()
    _panel(ax, _trace("truth"), mirror=_trace("learned"), cmap=st.CMAP_STRETCH,
           label=r"$\mathrm{tr}\,\mathbf{A}$", velocity=False, vmin=lo,
           vmax=hi, scale=scale)


# --------------------------------------------------------------------------
# e -- tr A difference, with the ROI band outlined
# --------------------------------------------------------------------------

def _sn1e(ax, scale):
    d = np.abs(_trace("learned") - _trace("truth"))
    _panel(ax, d, cmap=st.CMAP_ERROR,
           label=r"$|\Delta\,\mathrm{tr}\,\mathbf{A}|$", velocity=False,
           vmin=0.0, vmax=_robust_max([d], 99.9), scale=scale, roi=True,
           title=DIFF_TITLE["trA"])


# --------------------------------------------------------------------------
# Retired: one axes per arm, the previous panels (a), (b), (e) and (f).  Their
# content is now the two split panels; the images they produced are in
# final_figures/SN1/original2/.
# --------------------------------------------------------------------------

def _uy_single(ax, scale, arm, tag):
    lo, hi = _uy_limits()
    _panel(ax, _velocity_centres(arm, "v"), cmap=st.CMAP_SIGNED,
           label=r"$u_y$", velocity=True, vmin=lo, vmax=hi, scale=scale,
           tag=tag)


def _tra_single(ax, scale, arm, tag):
    lo, hi = _tra_limits()
    _panel(ax, _trace(arm), cmap=st.CMAP_STRETCH,
           label=r"$\mathrm{tr}\,\mathbf{A}$", velocity=False, vmin=lo,
           vmax=hi, scale=scale, tag=tag)


# --------------------------------------------------------------------------
# Metrics note.  These used to be printed inside SN1g; they now travel beside
# the panels as text so they can go in the caption instead.
# --------------------------------------------------------------------------

def metrics_note() -> str:
    m = contraction_metrics()
    fluid = int(_grid().fluid_mask().sum())
    global_row = f"whole fluid domain ({fluid} cells)"
    roi_row = f"training ROI band ({m['roi_cells']} cells)"
    return "\n".join([
        "Figure SN1 -- Giesekus 4:1 contraction, trained TBNN (gie_A_s4, "
        "frozen) against the Giesekus truth forward.",
        "",
        "Relative L2 error on tr A, ||learned - truth|| / ||truth||:",
        f"  {global_row:<36}{m['relative_L2_trA_global']:.6f}",
        f"  {roi_row:<36}{m['relative_L2_trA_roi_band']:.6f}",
        "",
        "The ROI band is the region the training loss weighted, "
        "psi_x psi_y >= 0.5,",
        f"spanning x/H = {m['roi_x_on']:.3f} to {m['roi_x_off']:.3f}; it is "
        "the dashed cyan outline in the three difference panels, (b), (c) and "
        "(e).",
        "",
        "Panels (a) and (d) are split at y = 0, truth above and trained TBNN "
        "below, as in",
        "Fig N1a.  tr A is symmetric about the centreline, so agreement there "
        "reads as a",
        "field continuous across the split.  u_y is antisymmetric, so the two "
        "halves of (a)",
        "carry opposite signs: agreement reads as the lower half being the "
        "mirror image of",
        "the upper one with the colour reversed, not as a continuous field.",
        "",
        "Truth centreline symmetry residual (a solver check, not a model "
        f"error): {m['truth_centerline_symmetry_residual']:.3e}",
        "",
        f"Fields: {dp.path('gie_passc_fields')}",
        "Recomputed at figure time by paper_figs.fig_n1.contraction_metrics().",
        "",
    ])


def write_metrics_note():
    out = dp.out_dir(FIG) / "SN1_metrics.txt"
    out.write_text(metrics_note())
    print(f"[SN1_metrics] {out}", flush=True)
    return out


plot_SN1a = _make("SN1a", _sn1a)
plot_SN1b = _make("SN1b", _sn1b)
plot_SN1c = _make("SN1c", _sn1c)
plot_SN1d = _make("SN1d", _sn1d)
_plot_SN1e = _make("SN1e", _sn1e)

plot_SN1_uy_truth = _make(
    "SN1_uy_truth", lambda ax, s: _uy_single(ax, s, "truth", TRUTH_TAG))
plot_SN1_uy_tbnn = _make(
    "SN1_uy_tbnn", lambda ax, s: _uy_single(ax, s, "learned", LEARNED_TAG))
plot_SN1_trA_truth = _make(
    "SN1_trA_truth", lambda ax, s: _tra_single(ax, s, "truth", TRUTH_TAG))
plot_SN1_trA_tbnn = _make(
    "SN1_trA_tbnn", lambda ax, s: _tra_single(ax, s, "learned", LEARNED_TAG))


def plot_SN1e(ax=None, save=True, dpi=None):
    out = _plot_SN1e(ax=ax, save=save, dpi=dpi)
    if save:
        write_metrics_note()
    return out


def plot_SN1(save=True, dpi=None):
    import matplotlib.pyplot as plt

    st.apply_style()
    fig = plt.figure(figsize=(20.0, 18.5))
    gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.30, left=0.05,
                          right=0.97, top=0.96, bottom=0.05)
    order = [plot_SN1a, plot_SN1b, plot_SN1c, plot_SN1d, plot_SN1e]
    for i, fn in enumerate(order):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        fn(ax=ax)
        pn.panel_tag(ax, f"({chr(ord('a') + i)})",
                     st.scale_for(pn.axes_width(ax)), loc="outside")
    if save:
        print(f"[SN1_full] {pn.save_panel(fig, FIG, 'SN1_full', dpi)}",
              flush=True)
        write_metrics_note()
    return fig
