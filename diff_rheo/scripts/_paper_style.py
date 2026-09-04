"""Shared Matplotlib style for paper figures, matching scripts/visualize_data.py.

Usage:
    import matplotlib; matplotlib.use("Agg")
    from _paper_style import apply, SAVE, C
    apply()
    ...
    fig.savefig(path, **SAVE)

All text labels must be LaTeX-safe (usetex=True): use $...$ for math, \% for
percent, and avoid unicode (use $\sigma$, $\dot\gamma$, R$^2$, etc.).
"""
import matplotlib.pyplot as plt

_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
}


def apply():
    plt.rcParams.update(_RC)


# savefig defaults (match visualize_data.py)
SAVE = dict(dpi=300, bbox_inches="tight")


# consistent colour convention across the RUDE/N1 figures
class C:
    DATA = "black"      # measured / ground-truth data
    WM = "C3"           # White-Metzner (red family)
    RUDE = "C1"         # our RUDE (orange)
    LENNON = "C0"       # Lennon's shipped RUDE (blue)
    OLDROYD = "0.5"     # Oldroyd-B / linear baseline (grey)
    REF = "0.7"         # reference lines
