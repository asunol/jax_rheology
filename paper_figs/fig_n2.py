"""Figure N2 (manuscript Fig 6) -- EVP yield stress.

Five-seed ensemble ``evp_fix_A_3lam_agn{,_s2.._s5}``, truth Gp 3.2, lam 0.7,
nu_s 0.8, tau_y 1.45.  Every curve is read from the common seed-eval store
(15-lambda ladder, 30 lambda below yield, early stopping off); the OB-init
curves come from the forward-only run G3, which used that same protocol.
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo
from . import loaders as ld
from . import panels as pn
from . import style as st

FIG = "N2"

DRIVES = (1.8, 2.5, 4.0, 5.0)
#: The drive N2a shows on its own; the other three are in the supplement.
MAIN_DRIVE = 4.0
TRAINING_DRIVES = (1.8, 2.5, 4.0)
CEILING = 5.0                      # well-posedness ceiling, N2b stops here
TAU_Y = ld.EVP_TRUTH["tau_y"]

#: Ground truth / Initial TBNN / Trained TBNN, named and drawn as in N1b.
ROLE = {key: (label, dict(kw)) for label, key, kw in st.PROFILE_ROLES}
LEARNED_LABEL = ROLE["learned"][0]

#: One dash-dotted rule marks the yield location in both panels: horizontal in
#: N2a (y = +-tau_y/G_x), vertical in N2b (G_x = G_c).
YIELD_RULE = dict(color=st.C_TRUTHFAM, ls="-.")
YIELD_RULE_K = 0.9


def _lw(scale, k=1.0):
    return k * st.BASE_LINEWIDTH * scale


def _role_plot(ax, x, y, key, scale, k=1.0, label=None, **over):
    kw = dict(ROLE[key][1])
    kw.update(over)
    return ax.plot(x, y, lw=st.PAPER_LINE["linewidth"] * k * scale,
                   label=label, **kw)


def _role_handles(scale, keys=("truth", "init", "learned")):
    """Legend proxies in N1b's order, whatever order the curves were drawn."""
    from matplotlib.lines import Line2D

    out = []
    for key in keys:
        label, kw = ROLE[key]
        kw = dict(kw)
        kw.pop("zorder", None)
        kw.pop("alpha", None)
        out.append(Line2D([], [], lw=st.PAPER_LINE["linewidth"] * scale,
                          label=label, **kw))
    return out


def analytic_plug(gx: float) -> float:
    """Yield surface of the Bingham/Saramito plug: |y| < tau_y / G_x."""
    return TAU_Y / gx


def _profile(arm: str, gx: float):
    u = ld.evp_profile(arm, gx)["u"]
    return geo.evp_profile_with_walls(u)


# --------------------------------------------------------------------------
# u(y) at one drive -- the panel body, shared by N2a and its supplement grid
# --------------------------------------------------------------------------

def drive_panel(a, gx, scale, show_init=False, keyed=False, title=None,
                yield_label=None):
    """One drive: truth, the five trained seeds, optionally the initial TBNN.

    ``keyed`` puts this panel's curves in the legend; in a grid only the first
    panel is keyed.  ``yield_label`` spells the yield rule out in words rather
    than giving the bare ratio, and defaults to ``keyed``.
    """
    if yield_label is None:
        yield_label = keyed
    fs = st.NOTEBOOK_EFFECTIVE["tick_labelsize_minor"] * scale
    if show_init:
        # At true amplitude.  The initial closure carries 2.2x the flux of the
        # yielded solution at this drive, and that gap is the point.
        yy, ui = _profile("obinit", gx)
        _role_plot(a, ui, yy, "init", scale,
                   label=ROLE["init"][0] if keyed else None)
    for j, sd in enumerate(ld.EVP_SEEDS):
        yy, us = _profile(sd, gx)
        _role_plot(a, us, yy, "learned", scale, k=0.85, alpha=0.85,
                   label=LEARNED_LABEL if (keyed and j == 0) else None)
    yy, ut = _profile("truth", gx)
    _role_plot(a, ut, yy, "truth", scale, k=0.9,
               label=ROLE["truth"][0] if keyed else None)

    yp = analytic_plug(gx)
    for s in (-1, 1):
        a.axhline(s * yp, lw=_lw(scale, YIELD_RULE_K), zorder=1, **YIELD_RULE)
    eq = rf"$\tau_y/G_x = {yp:.3f}$"
    if yield_label:
        # Inside the plug, clear of both the yielded profile on its left and
        # the initial closure on its right.
        a.annotate(f"Yield surface\n{eq}", xy=(0.46, yp),
                   xycoords=("axes fraction", "data"),
                   xytext=(0, -6 * scale), textcoords="offset points",
                   ha="left", va="top", color=st.C_TRUTHFAM, fontsize=fs)
    else:
        a.annotate(eq, xy=(0.04, yp), xycoords=("axes fraction", "data"),
                   xytext=(0, 5 * scale), textcoords="offset points",
                   ha="left", va="bottom", color=st.C_TRUTHFAM, fontsize=fs)
    # A hair of slack past the walls and past u_x = 0, so the corner tick
    # labels "-1.0" and "0.0" are not written on top of each other.
    a.set_ylim(-1.06, 1.06)
    a.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    a.set_xlim(left=-0.018 * a.get_xlim()[1])
    a.set_xlabel("$u_x$")
    if title is None:
        title = f"$G_x = {gx:g}$"
    if title:
        a.set_title(
            title,
            fontsize=st.NOTEBOOK_EFFECTIVE["axes_label_fontsize"] * scale,
            pad=6 * scale)


def drive_grid(axs, drives, scale, show_init=False):
    """The four-drive version, one axes per drive."""
    for k, (a, gx) in enumerate(zip(axs, drives)):
        drive_panel(a, gx, scale, show_init=show_init, keyed=(k == 0),
                    yield_label=False)
        if gx not in TRAINING_DRIVES:
            pn.annotate(a, 0.96, 0.04, "test condition", scale, ha="right",
                        color="0.3")
    axs[0].set_ylabel("$y/H$")
    st.paper_legend(axs[0], scale, loc="center left",
                    handles=_role_handles(scale, ("truth", "learned")))
    return axs


# --------------------------------------------------------------------------
# N2a -- u(y) at the one drive the main figure carries
# --------------------------------------------------------------------------

def plot_N2a(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 9.0, axes_width=9.5)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    drive_panel(ax, MAIN_DRIVE, scale, show_init=True, keyed=True, title="")
    ax.set_ylabel("$y/H$")
    st.paper_legend(ax, scale, loc="upper right",
                    handles=_role_handles(scale))
    return pn.finish(fig, ax, FIG, "N2a", save, dpi, own)


# --------------------------------------------------------------------------
# N2b -- flow curve
# --------------------------------------------------------------------------

def _flow_curve():
    drives, q_truth, q_seeds = ld.evp_flow_curve()
    keep = drives <= CEILING
    return (drives[keep], np.abs(q_truth[keep]), np.abs(q_seeds[:, keep]))


def plot_N2b(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(12.0, 9.0, axes_width=9.5)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    d, qt, qs = _flow_curve()
    lo, hi, mid = qs.min(axis=0), qs.max(axis=0), qs.mean(axis=0)
    learned_c = ROLE["learned"][1]["color"]

    ax.fill_between(d, lo, hi, color=learned_c, alpha=0.30, zorder=2)
    _role_plot(ax, d, mid, "learned", scale, k=1.2, zorder=3,
               label=LEARNED_LABEL)
    _role_plot(ax, d, qt, "truth", scale, k=1.1, zorder=4,
               label=ROLE["truth"][0])
    ms = 20 * scale
    for gx, q in zip(d, qt):
        trained = gx in TRAINING_DRIVES
        ax.plot(gx, q, "o", ms=ms, color=st.C_TRUTH,
                mfc=st.C_TRUTH if trained else "white", mew=2.6 * scale,
                zorder=5,
                label=("Training conditions" if (trained and gx == 1.8) else
                       ("Test conditions" if (not trained and gx == 0.5)
                        else None)))
    ax.axvline(ld.EVP_G_C, lw=_lw(scale, YIELD_RULE_K), zorder=1, **YIELD_RULE)
    ax.set_yscale("log")
    ax.set_xlim(0.3, 5.25)
    ax.set_ylim(3e-10, 3.0)
    ax.set_xlabel("$G_x$")
    ax.set_ylabel("$|Q|$")
    ax.annotate(rf"$G_c = {ld.EVP_G_C}$", xy=(ld.EVP_G_C, 1e-5),
                xytext=(9 * scale, 0), textcoords="offset points",
                color=st.C_TRUTHFAM, va="center",
                fontsize=st.NOTEBOOK_EFFECTIVE["tick_labelsize_major"] * scale)
    order = [LEARNED_LABEL, ROLE["truth"][0], "Training conditions",
             "Test conditions"]
    h, l = ax.get_legend_handles_labels()
    by_label = dict(zip(l, h))
    st.paper_legend(ax, scale, loc="lower right",
                    handles=[by_label[k] for k in order if k in by_label])
    if own and save:
        write_caption_note()
    return pn.finish(fig, ax, FIG, "N2b", save, dpi, own)


def caption_note() -> str:
    """What N2b no longer says on the face of the panel."""
    d, qt, _ = _flow_curve()
    i13 = int(np.argmin(np.abs(d - 1.3)))
    return f"""N2b -- caption notes
=====================

Sub-yield arrest
----------------
Below the critical drive the channel does not creep, it stops.  At
G_x = 1.3, one rung under G_c, the truth flux is |Q| = {qt[i13]:.2e} after
30 lambda, which is round-off, not slow flow; the five trained closures sit on
the same floor.  The near-vertical rise at G_c is that arrest ending, not a
resolution artefact.

Where the curve stops
---------------------
The flow curve is drawn to G_x = {CEILING:g} because that is the per-seed
well-posedness ceiling (SN8c): above it at least one seed loses positive
definiteness of A within the evaluation horizon, so the ensemble is not
defined there.  The shaded band beyond the ceiling has been dropped from the
panel; this is where that fact now lives.

Symbols
-------
G_c = {ld.EVP_G_C} is the critical drive, the dash-dotted vertical; it is drawn
like the dash-dotted yield surfaces y = +-tau_y/G_x in N2a, both marking where
the material yields.  Filled circles are the three training conditions
(G_x = 1.8, 2.5, 4.0); open circles are test conditions the closures never saw.
The trained curve is five independently seeded networks whose flow curves
are indistinguishable at plot resolution, not a single run.  The line is
the mean over the five seeds and the shaded band is their full range; the
band is not keyed in the legend because it is invisible at every yielded
drive -- the seed-to-seed spread of |Q| is under 0.4% of the mean
everywhere above G_c, and the one visibly wider rung (G_x = 1.3, spread
1.9e-09 to 2.2e-09) is scatter on the round-off floor, not flow.
"""


def write_caption_note():
    out = ld.dp.out_dir(FIG) / "N2b_caption_note.txt"
    out.write_text(caption_note())
    print(f"[N2b] {out}", flush=True)
    return out


# --------------------------------------------------------------------------
# assembled figure
# --------------------------------------------------------------------------

def plot_N2(save=True, dpi=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20.0, 8.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.24,
                          left=0.055, right=0.985, top=0.90, bottom=0.13)
    with pn.uniform_scale(0.78) as scale:
        for tag, slot, fn in (("(a)", gs[0, 0], plot_N2a),
                              ("(b)", gs[0, 1], plot_N2b)):
            a = fig.add_subplot(slot)
            fn(ax=a)
            pn.panel_tag(a, tag, scale, loc="outside")
    if save:
        print(f"[N2_full] {pn.save_panel(fig, FIG, 'N2_full', dpi)}",
              flush=True)
    return fig
