"""Shared drawing helpers: field panels, colourbars, annotations, saving."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from . import data_paths as dp
from . import geometry as geo
from . import style as st


# --------------------------------------------------------------------------
# Figure / axes plumbing
# --------------------------------------------------------------------------

def new_panel(width: float = 10.0, height: float = 7.5,
              axes_width: float | None = None, **kw):
    """Create a single-axes figure at the notebook's type scale.

    ``axes_width`` is the width the *axes* will actually occupy, which differs
    from the figure width whenever the axes is aspect-locked (field panels) or
    shares the figure with a colourbar.  Type is scaled to that, not to the
    figure, so the apparent type size matches the notebook.
    """
    scale = st.apply_style(axes_width if axes_width is not None
                           else width * AXES_FRACTION)
    fig, ax = plt.subplots(figsize=(width, height), **kw)
    st.style_axes(ax, scale)
    return fig, ax, scale


#: Fraction of the figure width a single axes typically occupies once the
#: y-label, tick labels and (where present) colourbar have taken their margins.
#: Used to pick the type scale before the axes exists.
AXES_FRACTION = 0.68


def new_stack(nrows: int = 2, width: float = 9.0, height: float = 10.0,
              axes_width: float | None = None, ncols: int = 1, **kw):
    """Stack of axes sharing one figure, at the notebook type scale."""
    scale = st.apply_style(axes_width if axes_width is not None
                           else width * AXES_FRACTION / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, height), **kw)
    axs = list(np.atleast_1d(axs).ravel())
    for a in axs:
        st.style_axes(a, scale)
    return fig, axs, scale


def axes_width(ax) -> float:
    """Drawing width of an axes in inches (used to pick the type scale)."""
    fig = ax.get_figure()
    return float(ax.get_position().width * fig.get_size_inches()[0])


_SCALE_OVERRIDE: float | None = None


class uniform_scale:
    """Force one type scale for every axes drawn inside the block.

    Assembled figures mix panels of very different widths; scaling type to
    each axes separately would make the wide panels shout.  Inside this
    context every ``adopt`` returns the same scale, so the assembled figure
    carries a single type size.
    """

    def __init__(self, scale: float):
        self.scale = float(scale)

    def __enter__(self):
        global _SCALE_OVERRIDE
        self._prev = _SCALE_OVERRIDE
        _SCALE_OVERRIDE = self.scale
        st.apply_style(self.scale * st.REF_AXES_WIDTH_IN)
        return self.scale

    def __exit__(self, *exc):
        global _SCALE_OVERRIDE
        _SCALE_OVERRIDE = self._prev
        return False


def assembling() -> bool:
    """True while an assembled figure is forcing one type scale on its panels.

    Assembled figures put type at a larger fraction of the axes width than the
    standalone panels do, so a long annotation that fits on its own can overrun
    there; a panel can use this to give that one item a smaller size.
    """
    return _SCALE_OVERRIDE is not None


def adopt(ax):
    """Style an externally supplied axes and return its scale factor.

    Also re-installs the rcParams at that axes' scale, so labels and legends
    created after this call match the axes actually being drawn into.  Panels
    draw sequentially, so a global rcParams update is the right granularity.
    """
    if _SCALE_OVERRIDE is not None:
        st.apply_style(_SCALE_OVERRIDE * st.REF_AXES_WIDTH_IN)
        st.style_axes(ax, _SCALE_OVERRIDE)
        return _SCALE_OVERRIDE
    scale = st.apply_style(axes_width(ax))
    st.style_axes(ax, scale)
    return scale


def tidy(fig):
    """tight_layout without the sub-gridspec compatibility warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        fig.tight_layout()


def subaxes(ax, nrows: int = 2, ncols: int = 1, sharex: bool = False,
            sharey: bool = False, **kw):
    """Replace ``ax`` by an nrows x ncols block in the same slot.

    Lets a panel that is intrinsically two stacked sub-axes still honour the
    ``ax=`` contract used by the assembled-figure path.
    """
    fig = ax.get_figure()
    spec = ax.get_subplotspec()
    ax.remove()
    gs = spec.subgridspec(nrows, ncols, **kw)
    out = []
    first = None
    for i in range(nrows):
        for j in range(ncols):
            share = {}
            if first is not None:
                if sharex:
                    share["sharex"] = first
                if sharey:
                    share["sharey"] = first
            a = fig.add_subplot(gs[i, j], **share)
            first = first or a
            out.append(a)
    return out


def archive_original(out: Path) -> Path | None:
    """Move a pre-existing panel into ``original/`` before it is overwritten.

    Only the first version ever written is kept: once ``original/<name>`` is
    present, later regenerations overwrite the working copy and leave the
    archived one alone.
    """
    if not out.exists():
        return None
    keep = out.parent / "original" / out.name
    if keep.exists():
        return keep
    keep.parent.mkdir(parents=True, exist_ok=True)
    out.replace(keep)
    return keep


def save_panel(fig, figure_id: str, panel_id: str,
               dpi: int | None = None) -> Path:
    """Write ``final_figures/<figure_id>/<panel_id>.jpg``."""
    out = dp.out_dir(figure_id) / f"{panel_id}.jpg"
    archive_original(out)
    fig.savefig(out, dpi=dpi or st.DEFAULT_DPI, bbox_inches="tight",
                facecolor="white", format="jpg", pil_kwargs={"quality": 95})
    return out


def finish(fig, ax, figure_id, panel_id, save, dpi, own_fig):
    """Common tail of every ``plot_*`` function."""
    if own_fig:
        fig.tight_layout()
        if save:
            path = save_panel(fig, figure_id, panel_id, dpi)
            print(f"[{panel_id}] {path}", flush=True)
        return fig
    return ax


def panel_tag(ax, text: str, scale: float = 1.0, loc: str = "upper left",
              color: str = "black", pad: float = 0.02):
    """Panel letter, drawn in the axes corner."""
    x, y, ha, va = {
        "upper left": (pad, 1 - pad, "left", "top"),
        "upper right": (1 - pad, 1 - pad, "right", "top"),
        "lower left": (pad, pad, "left", "bottom"),
        "lower right": (1 - pad, pad, "right", "bottom"),
        # Above the axes, clear of the data -- the only placement that is
        # legible on a field panel, where every corner is either data or a
        # grey solid.
        "outside": (-0.02, 1.02, "right", "bottom"),
    }[loc]
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                   color=color, fontweight="bold",
                   fontsize=st.NOTEBOOK_RCPARAMS["axes.titlesize"] * scale)


def annotate(ax, x, y, text, scale=1.0, **kw):
    kw.setdefault("fontsize", st.NOTEBOOK_EFFECTIVE["tick_labelsize_minor"] * scale)
    return ax.text(x, y, text, transform=ax.transAxes, **kw)


def colorbar(fig, mappable, ax, label, scale=1.0, nticks=6, rotation=0.0,
             label_align="center", **kw):
    """Colourbar with the label upright above the bar, or beside it if rotated.

    ``label_align="left"`` starts the label at the bar's left edge instead of
    centring it there.  A label much wider than the bar, centred, reaches back
    across the panel's right spine and prints its first character on the field;
    left aligned it stays above the bar and off the plotted square.
    """
    from matplotlib.ticker import MaxNLocator

    kw.setdefault("fraction", 0.030)
    kw.setdefault("pad", 0.018)
    cb = fig.colorbar(mappable, ax=ax, **kw)
    lo, hi = mappable.get_clim()
    ticks = MaxNLocator(nbins=nticks).tick_values(lo, hi)
    # Drop ticks that fall in the "extend" triangles, where their labels would
    # collide with the arrow.
    cb.set_ticks([t for t in ticks if lo - 1e-12 <= t <= hi + 1e-12])
    fs = st.NOTEBOOK_EFFECTIVE["axes_label_fontsize"] * scale
    if rotation == 0:
        # Upright, and above the bar rather than beside it: beside it the label
        # sits in the next panel's y-label lane once the figure is assembled.
        cb.ax.set_title(label, pad=8.0 * scale, fontsize=fs,
                        loc=label_align)
    else:
        cb.set_label(label, rotation=rotation, labelpad=28.0 * scale,
                     va="center", ha="center", fontsize=fs)
    cb.ax.tick_params(
        labelsize=st.NOTEBOOK_EFFECTIVE["tick_labelsize_major"] * scale,
        width=st.NOTEBOOK_EFFECTIVE["tick_width_major"] * scale,
        length=st.NOTEBOOK_EFFECTIVE["tick_length_major"] * scale,
        direction="in")
    cb.outline.set_linewidth(st.BASE_AXES_LINEWIDTH * scale)
    return cb


# --------------------------------------------------------------------------
# 2D field panels (filled colour + contours, solids cut on cell edges)
# --------------------------------------------------------------------------

N_LEVELS = 18
N_CONTOUR_LINES = 9


def field_panel(ax, pg: geo.PlotGrid, F: np.ndarray, *, cmap: str,
                levels=None, vmin=None, vmax=None, scale: float = 1.0,
                contour_color: str = "white", contour_lw: float = 0.9,
                extend: str = "neither", solids=(), grey=st.C_SOLID):
    """Filled colour map with overlaid contour lines, solids cut on cell edges.

    ``F`` must already have been put on ``pg`` by ``geometry.*_to_plot`` so
    that wall nodes carry the exact boundary value and solid nodes are NaN.
    """
    finite = F[np.isfinite(F)]
    lo = float(vmin if vmin is not None else finite.min())
    hi = float(vmax if vmax is not None else finite.max())
    if levels is None:
        levels = np.linspace(lo, hi, N_LEVELS)
    filled = ax.contourf(pg.X, pg.Y, F, levels=levels, cmap=cmap,
                         extend=extend)
    for c in filled.collections:
        c.set_edgecolor("face")
    line_levels = np.linspace(levels[0], levels[-1], N_CONTOUR_LINES)[1:-1]
    ax.contour(pg.X, pg.Y, F, levels=line_levels, colors=contour_color,
               linewidths=contour_lw * scale, alpha=0.75)
    for (x0, y0, w, h) in solids:
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=grey,
                               edgecolor="black",
                               linewidth=0.8 * scale, zorder=5))
    return filled


#: Plotted window.  x per the archived Pass C figure; y cropped from the full
#: +-4 H domain to +-2.6 H, which keeps the whole step face and a clear band of
#: the solid block while dropping the stagnant upstream corners.
CONTRACTION_XLIM = (-3.0, 4.5)
CONTRACTION_YLIM = (-2.6, 2.6)


def contraction_axes(ax, grid: geo.ContractionGrid, scale: float = 1.0,
                     xlim=CONTRACTION_XLIM, ylim=CONTRACTION_YLIM,
                     ylabel=True, xlabel=True):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    if xlabel:
        ax.set_xlabel(r"$x/H$")
    if ylabel:
        ax.set_ylabel(r"$y/H$")
    st.style_axes(ax, scale)


def draw_roi(ax, grid: geo.ContractionGrid, activation: np.ndarray,
             scale: float = 1.0, color: str = st.C_ROI):
    """Outline the ROI band used by the training loss (psi_x psi_y = 0.5)."""
    X, Y = np.meshgrid(grid.xc, grid.yc, indexing="ij")
    return ax.contour(X, Y, activation, levels=[0.5], colors=[color],
                      linewidths=1.8 * scale, linestyles="--", zorder=7)


def split_labels(ax, top: str, bottom: str, scale: float = 1.0,
                 color: str = "white"):
    ax.axhline(0.0, color=color, linewidth=1.0 * scale, alpha=0.85, zorder=6)
    fs = st.NOTEBOOK_EFFECTIVE["tick_labelsize_minor"] * scale
    ax.text(0.015, 0.96, top, transform=ax.transAxes, va="top", color=color,
            fontweight="bold", fontsize=fs, zorder=8)
    ax.text(0.015, 0.04, bottom, transform=ax.transAxes, va="bottom",
            color=color, fontweight="bold", fontsize=fs, zorder=8)


# --------------------------------------------------------------------------
# Table panels
# --------------------------------------------------------------------------

TABLE_HEADER_BG = "#4a9d8e"
TABLE_ROW_BG = ("#ffffff", "#f4f5f5")
TABLE_EDGE = "#9e9e9e"


def table_panel(ax, columns, rows, scale: float = 1.0, col_widths=None,
                row_height: float = 2.2, font_k: float = 1.0):
    """A rendered table: teal header, alternating row shading, grey rules.

    ``font_k`` trims the type: the default is sized for a four-row, three-column
    table like N1d, and a wider table needs less.
    """
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=list(columns), loc="center",
                   cellLoc="center", colWidths=col_widths)
    tab.auto_set_font_size(False)
    fs = st.NOTEBOOK_EFFECTIVE["axes_label_fontsize"] * scale * font_k
    for (r, _), cell in tab.get_celld().items():
        cell.set_edgecolor(TABLE_EDGE)
        cell.set_linewidth(1.3 * scale)
        text = cell.get_text()
        text.set_fontsize(fs)
        if r == 0:
            cell.set_facecolor(TABLE_HEADER_BG)
            text.set_color("white")
            text.set_fontweight("bold")
        else:
            cell.set_facecolor(TABLE_ROW_BG[(r - 1) % 2])
    tab.scale(1.0, row_height)
    return tab


# --------------------------------------------------------------------------
# Assembled-figure plumbing
# --------------------------------------------------------------------------

def assemble(figure_id: str, panels, ncols: int, panel_w: float = 8.0,
             panel_h: float = 6.0, save: bool = True, dpi: int | None = None,
             letters: bool = True, **gridkw):
    """Build ``<figure_id>_full.jpg`` by calling each panel with ``ax=``.

    ``panels`` is a sequence of callables taking ``ax``; every one of them is
    the same function that produces the standalone panel.
    """
    nrows = int(np.ceil(len(panels) / ncols))
    scale = st.apply_style(panel_w)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             **gridkw)
    flat = np.atleast_1d(axes).ravel()
    for i, fn in enumerate(panels):
        fn(ax=flat[i])
        if letters:
            panel_tag(flat[i], f"({chr(ord('a') + i)})", scale,
                      loc="upper right")
    for ax in flat[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    if save:
        path = save_panel(fig, figure_id, f"{figure_id}_full", dpi)
        print(f"[{figure_id}_full] {path}", flush=True)
    return fig
