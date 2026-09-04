"""Publication style for the paper figures.

Every number here was read out of ``paper_plots.ipynb`` cell 6 (the only cell
in that notebook that touches ``rcParams``).  Where the notebook sets an
rcParam and then overrides it in the call that actually draws, the value that
reaches the page is recorded as the *effective* value and is what we use.

Notebook context: a single-axes parity figure at ``figsize=(10, 10)``.  The
type sizes are therefore tied to a 10-inch-wide axes.  ``apply_style`` keeps
that ratio by scaling with the axes width, so a six-panel supplementary figure
carries the same apparent type size as the notebook's parity plots.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Verbatim from paper_plots.ipynb cell 6.
# --------------------------------------------------------------------------
NOTEBOOK_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 32,
    "axes.labelsize": 48,
    "axes.titlesize": 44,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
    "legend.fontsize": 36,
    "figure.titlesize": 46,
}

# In-call overrides in the same cell (these are what actually render).
NOTEBOOK_EFFECTIVE = {
    "figsize": (10.0, 10.0),
    "figure_dpi": 600,
    "axes_label_fontsize": 36,      # set_xlabel/set_ylabel(..., fontsize=36)
    "tick_labelsize_major": 30,     # tick_params(which='major', labelsize=30)
    "tick_labelsize_minor": 24,
    "tick_width_major": 2.0,
    "tick_length_major": 10.0,
    "tick_width_minor": 1.5,
    "tick_length_minor": 6.0,
    "tick_direction": "in",
    "tick_top": True,
    "tick_right": True,
    "grid": False,
    "reference_line_width": 2.5,    # identity line 'k--', linewidth=2.5
    "marker_size": 150,             # scatter s=150
    "marker_edge_width": 2.0,
    "savefig_bbox": "tight",
    "savefig_facecolor": "white",
}

# The notebook contains no legend() call, so it fixes no frame settings.  The
# repository's own figure scripts (cavity_transfer_comparison.py, and every
# readback figure script) call legend(frameon=False); we adopt
# that as the fallback and record it here explicitly.
LEGEND_FRAME = dict(frameon=False, handlelength=1.8, borderaxespad=0.4)

# Arial is not installed on this machine, so both the notebook and this module
# fall back to the second entry, DejaVu Sans.  Recorded so the report does not
# claim Arial output.
RESOLVED_SANS = "DejaVu Sans"

# Line-plot conventions of the throat velocity-profile comparison the notebook
# draws (paper_plots.ipynb cells 3/7 -> plot_tbnn_training_results.py
# :1289-1307).  Thick unmarked lines told apart by dash pattern and colour,
# with a boxed legend.
PAPER_LINE = {
    "linewidth": 3.0,
    # The reference figure sets legend 22 against ticks 26 (porous_media_flow
    # PLOT 7); carried across as the same ratio of this style's tick size,
    # since 22pt read against 30pt ticks would look like a footnote.
    "legend_fontsize": 25.0,
}
PAPER_LEGEND = dict(frameon=True, fancybox=True, shadow=True,
                    handlelength=1.8, borderaxespad=0.4)

#: Ground truth / initial / trained, in plotting order and verbatim style.
PROFILE_ROLES = (
    ("Ground truth", "truth", dict(color="k", linestyle="-.", zorder=3)),
    ("Initial TBNN", "init",
     dict(color="#ff7f0e", linestyle="--", alpha=0.7, zorder=2)),
    ("Trained TBNN", "learned",
     dict(color="#2ca02c", linestyle="-", zorder=2)),
)

REF_AXES_WIDTH_IN = 10.0
DEFAULT_DPI = 300

# Line widths are not set in the notebook cell (it uses per-call values 2.0 /
# 2.5).  We keep 2.5 for reference/truth lines at reference scale.
BASE_LINEWIDTH = 2.5
BASE_AXES_LINEWIDTH = 2.0

# --------------------------------------------------------------------------
# Colour roles.  Kept identical to pub_style.py, which is already used by the
# archived figures we are regenerating; the notebook fixes no palette.
# --------------------------------------------------------------------------
C_INIT = "#9aa0a6"
C_TRUTH = "#111111"
C_LEARN = "#d1495b"
C_TRUTHFAM = "#2a9d8f"
C_WINNER = "#4c72b0"
C_OTHER = "#b0b6bf"
C_DRIVE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
#: Training 3 / De = 0.50 / +pressure.  Independent of C_LEARN (learned-role
#: red).  Okabe-Ito reddish purple so it is not a red/green pair with
#: C_SCHEDULE[0].
C_SCHED_HI = "#CC79A7"
#: Training-schedule curves.  A true green, a blue and a reddish purple.
C_SCHEDULE = ("#2ca02c", "#1f77b4", C_SCHED_HI)
#: The cavity Deborah ladder.  N1e and SN3 show the same three runs, so they
#: share one colour per De; green / blue / reddish-purple keeps them separable.
C_DE = {0.20: C_SCHEDULE[0], 0.35: C_SCHEDULE[1], 0.50: C_SCHEDULE[2]}
C_ROI = "#00ffff"
C_SOLID = "#9e9e9e"

CMAP_SPEED = "viridis"
CMAP_STRETCH = "magma"
CMAP_ERROR = "inferno"
CMAP_SIGNED = "RdBu_r"


def scale_for(axes_width_in: float) -> float:
    """Type scale for an axes of the given drawing width."""
    return float(axes_width_in) / REF_AXES_WIDTH_IN


def apply_style(axes_width_in: float = REF_AXES_WIDTH_IN) -> float:
    """Install the notebook style, scaled to ``axes_width_in``.

    Returns the scale factor so callers can size markers/line widths to match.
    """
    s = scale_for(axes_width_in)
    plt.rcParams.update({
        "font.family": NOTEBOOK_RCPARAMS["font.family"],
        "font.sans-serif": NOTEBOOK_RCPARAMS["font.sans-serif"],
        "font.size": NOTEBOOK_RCPARAMS["font.size"] * s,
        "axes.labelsize": NOTEBOOK_EFFECTIVE["axes_label_fontsize"] * s,
        "axes.titlesize": NOTEBOOK_RCPARAMS["axes.titlesize"] * s,
        "xtick.labelsize": NOTEBOOK_EFFECTIVE["tick_labelsize_major"] * s,
        "ytick.labelsize": NOTEBOOK_EFFECTIVE["tick_labelsize_major"] * s,
        "legend.fontsize": NOTEBOOK_RCPARAMS["legend.fontsize"] * s,
        "figure.titlesize": NOTEBOOK_RCPARAMS["figure.titlesize"] * s,
        # The notebook never touches mathtext, so its math renders in the
        # matplotlib default sans face, matching the surrounding text.  Setting
        # "cm" here would put serif Computer Modern next to sans tick labels.
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "it",
        "axes.linewidth": BASE_AXES_LINEWIDTH * s,
        "lines.linewidth": BASE_LINEWIDTH * s,
        "axes.grid": False,
        "xtick.direction": NOTEBOOK_EFFECTIVE["tick_direction"],
        "ytick.direction": NOTEBOOK_EFFECTIVE["tick_direction"],
        "xtick.top": NOTEBOOK_EFFECTIVE["tick_top"],
        "ytick.right": NOTEBOOK_EFFECTIVE["tick_right"],
        "xtick.major.width": NOTEBOOK_EFFECTIVE["tick_width_major"] * s,
        "ytick.major.width": NOTEBOOK_EFFECTIVE["tick_width_major"] * s,
        "xtick.major.size": NOTEBOOK_EFFECTIVE["tick_length_major"] * s,
        "ytick.major.size": NOTEBOOK_EFFECTIVE["tick_length_major"] * s,
        "xtick.minor.width": NOTEBOOK_EFFECTIVE["tick_width_minor"] * s,
        "ytick.minor.width": NOTEBOOK_EFFECTIVE["tick_width_minor"] * s,
        "xtick.minor.size": NOTEBOOK_EFFECTIVE["tick_length_minor"] * s,
        "ytick.minor.size": NOTEBOOK_EFFECTIVE["tick_length_minor"] * s,
        "xtick.labelsize": NOTEBOOK_EFFECTIVE["tick_labelsize_major"] * s,
        "legend.frameon": LEGEND_FRAME["frameon"],
        "savefig.facecolor": NOTEBOOK_EFFECTIVE["savefig_facecolor"],
        "figure.facecolor": "white",
        "savefig.dpi": DEFAULT_DPI,
    })
    return s


def style_axes(ax, scale: float = 1.0) -> None:
    """Apply the notebook's explicit ``tick_params`` calls to one axes."""
    e = NOTEBOOK_EFFECTIVE
    ax.grid(False)
    ax.tick_params(
        axis="both", which="major",
        labelsize=e["tick_labelsize_major"] * scale,
        width=e["tick_width_major"] * scale,
        length=e["tick_length_major"] * scale,
        direction=e["tick_direction"], top=e["tick_top"], right=e["tick_right"],
    )
    ax.tick_params(
        axis="both", which="minor",
        labelsize=e["tick_labelsize_minor"] * scale,
        width=e["tick_width_minor"] * scale,
        length=e["tick_length_minor"] * scale,
        direction=e["tick_direction"], top=e["tick_top"], right=e["tick_right"],
    )


def legend(ax, scale: float = 1.0, **kwargs):
    opts = dict(LEGEND_FRAME)
    opts.update(kwargs)
    opts.setdefault("fontsize", NOTEBOOK_RCPARAMS["legend.fontsize"] * scale)
    return ax.legend(**opts)


def paper_legend(ax, scale: float = 1.0, **kwargs):
    """Legend in the style of the Carreau-Yasuda comparison figure.

    ``paper_plots.ipynb`` cell 2 draws the shear-thinning (Carreau-Yasuda vs
    TBNN) comparison through ``porous_media_flow.run_demo_comparison``; its one
    legend-bearing line plot uses a boxed, rounded, shadowed frame.
    """
    opts = dict(PAPER_LEGEND)
    opts.update(kwargs)
    opts.setdefault("fontsize", PAPER_LINE["legend_fontsize"] * scale)
    return ax.legend(**opts)


def describe() -> str:
    """Human-readable dump of the extracted style, for the figure report."""
    lines = ["paper_plots.ipynb cell 6 -- rcParams as written:"]
    for k, v in NOTEBOOK_RCPARAMS.items():
        lines.append(f"  {k} = {v!r}")
    lines.append("in-call overrides (what actually renders):")
    for k, v in NOTEBOOK_EFFECTIVE.items():
        lines.append(f"  {k} = {v!r}")
    lines.append(f"legend frame: not set in notebook; using {LEGEND_FRAME!r}")
    lines.append(f"reference axes width = {REF_AXES_WIDTH_IN} in; "
                 f"default save dpi = {DEFAULT_DPI}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Verbatim from pub_style.py (cavity serial report). Not applied at import --
# notebook apply_style sizes must stay the paper_figs default.
# --------------------------------------------------------------------------
PUB_REPORT_RCPARAMS = {
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.dpi": 120, "savefig.dpi": 300, "axes.linewidth": 1.0,
    "font.family": "DejaVu Sans", "mathtext.fontset": "cm",
}

SHORT = {"init": "initial guess", "truth": "ground truth", "learned": "learned TBNN"}


def save_fig(fig, outdir, name):
    """Write <outdir>/<name>.pdf (vector) and <outdir>/<name>.png (300 dpi)."""
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[fig]", os.path.join(outdir, name + ".{pdf,png}"), flush=True)


def bingham_u_profile(y, g, eta, tau_y, H=1.0, y_center=1.0):
    """Steady planar Bingham/ (steady-)Saramito velocity profile on grid y
    (walls at y_center-H, y_center+H; centerline y_center). Plug where
    |g d| <= tau_y, d = |y - y_center|."""
    d = np.abs(y - y_center)
    y_p = tau_y / g
    u = np.where(
        d >= y_p,
        (g / (2 * eta)) * (H ** 2 - d ** 2) - (tau_y / eta) * (H - d),
        (g / (2 * eta)) * (H - y_p) ** 2)
    u = np.clip(u, 0.0, None)
    return u


def bingham_Q(g, eta, tau_y, H=1.0, Ny=2048):
    """Flow rate Q = trapz(u, y) over the full channel [0, 2H]."""
    y = np.linspace(0.0, 2 * H, Ny)
    u = bingham_u_profile(y, g, eta, tau_y, H=H, y_center=H)
    return float(np.trapz(u, y))

