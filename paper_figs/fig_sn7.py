"""Figure SN7 (supplement S14-S15) -- EVP profiles, arrest, training.

Everything is read from the five-seed evaluation store
(``analysis_pub_readback/evp_fix_seed_eval``), which ran truth and all five
learned closures on one protocol: 15 lambda, extended to 30 lambda at the three
sub-yield drives, early stopping off.  Training used drives {1.8, 2.5, 4.0} at
3 lambda.

Four panels: (a) u(y) at four drives, (b) the sub-yield arrest transient,
    (c) the training loss, (d) the four scalars as a 4x2: stage-1
    absolute on the left, theta-block ratio on the right.
    Two earlier panels were cut and are kept callable at
the bottom of this module:

* the 12-drive u(y) grid -- half the rungs are arrested profiles whose shape is
  round-off residual at 1e-11, and the informative drives are panel (a);
* plug half-width against drive -- the arrest cliff at G_c and the ceiling
  above G_x = 5 are already in the main figure's Q(G_x).

Nomenclature follows N2: capital ``G_x``, ``G_c``, and the N1 role names
"Ground truth" / "Trained TBNN" / "TBNN prediction".
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo
from . import loaders as ld
from . import panels as pn
from . import style as st

FIG = "SN7"

LADDER = tuple(float(g) for g in ld.evp_ladder())
TRAINING_DRIVES = (1.8, 2.5, 4.0)
ARREST_DRIVES = (0.5, 1.0, 1.3)
LAM = ld.EVP_TRUTH["lam"]
TAU_Y = ld.EVP_TRUTH["tau_y"]

TRUTH_LABEL = "Ground truth"
LEARNED_LABEL = "TBNN prediction"

ARREST_COLOR = {0.5: "#4c72b0", 1.0: "#55a868", 1.3: "#9467bd"}


def _lw(scale, k=1.0):
    return k * st.BASE_LINEWIDTH * scale


def _profile(arm: str, gx: float):
    return geo.evp_profile_with_walls(ld.evp_profile(arm, gx)["u"])


# --------------------------------------------------------------------------
# SN7a -- u(y) at the four drives the text discusses
# --------------------------------------------------------------------------

def plot_SN7a(ax=None, save=True, dpi=None):
    """The four drives at full size, without the initial closure.

    N2a carries one drive, G_x = 4, and adds the initial TBNN there; the other
    three drives -- including the held-out G_x = 5 -- are only here.
    """
    from . import fig_n2 as n2

    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(1, 24.0, 9.0, ncols=len(n2.DRIVES),
                                       sharey=True, axes_width=7.0,
                                       gridspec_kw=dict(wspace=0.14))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 1, len(n2.DRIVES), sharey=True, wspace=0.12)
        scale = pn.adopt(axs[0])
        for a in axs[1:]:
            pn.adopt(a)

    n2.drive_grid(axs, n2.DRIVES, scale, show_init=False)

    if own:
        pn.tidy(fig)
        if save:
            print(f"[SN7a] {pn.save_panel(fig, FIG, 'SN7a', dpi)}", flush=True)
            write_notes()
        return fig
    return axs


# --------------------------------------------------------------------------
# SN7b -- sub-yield arrest, |Q|(t) to 30 lambda
# --------------------------------------------------------------------------

def _ring_extrema(q) -> np.ndarray:
    """Turning points of |Q| on this arm: peaks and notches, own times."""
    a = np.abs(np.asarray(q))
    pk = np.where((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]))[0] + 1
    dip = np.where((a[1:-1] < a[:-2]) & (a[1:-1] <= a[2:]))[0] + 1
    return np.unique(np.concatenate([pk, dip]))


def _marker_indices(q) -> np.ndarray:
    """Peaks, notches, and one midpoint on each descent.

    Everything is this arm's own |Q|.  A notch marker sits in that arm's
    valley, not on the truth clock -- a shared-time sample in the notch
    would read as a timing error.
    """
    ext = _ring_extrema(q)
    a = np.abs(np.asarray(q))
    extra = []
    for i, j in zip(ext[:-1], ext[1:]):
        if j - i < 4 or a[i] <= a[j]:
            continue
        extra.append(i + (j - i) // 2)
    if not extra:
        return ext
    return np.unique(np.concatenate([ext, np.asarray(extra, int)]))


def _learned_markers(ax, t, q, color, scale):
    """Learned arm as open markers, no line -- N1e's convention.

    Truth and the five closures agree to within a line width, so drawing both
    as lines in one colour hides the truth underneath.
    """
    from .fig_n1 import MARKER_SIZE

    k = _marker_indices(q)
    return ax.plot(np.asarray(t)[k], np.abs(np.asarray(q))[k],
                   linestyle="none", marker="o", mfc="none", mec=color,
                   ms=MARKER_SIZE * scale,
                   mew=st.NOTEBOOK_EFFECTIVE["marker_edge_width"] * scale
                   * 1.3)[0]


def _arrest_handles(scale):
    """Two role proxies plus one per drive, all without a seed count."""
    from matplotlib.lines import Line2D

    from .fig_n1 import MARKER_SIZE

    mew = st.NOTEBOOK_EFFECTIVE["marker_edge_width"] * scale * 1.3
    out = [Line2D([], [], color="0.25", lw=_lw(scale, 1.0), label=TRUTH_LABEL),
           Line2D([], [], color="none", marker="o", mfc="none", mec="0.25",
                  ms=MARKER_SIZE * scale, mew=mew, label=LEARNED_LABEL)]
    out += [Line2D([], [], color=ARREST_COLOR[gx], lw=_lw(scale, 1.0),
                   label=rf"$G_x = {gx:g}$") for gx in ARREST_DRIVES]
    return out


def plot_SN7b(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 8.0)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    for gx in ARREST_DRIVES:
        c = ARREST_COLOR[gx]
        d = ld.evp_profile("truth", gx)
        ax.plot(d["t"] / LAM, np.abs(d["Q"]), "-", color=c,
                lw=_lw(scale, 1.0), zorder=3)
        for sd in ld.EVP_SEEDS:
            s = ld.evp_profile(sd, gx)
            _learned_markers(ax, s["t"] / LAM, s["Q"], c, scale)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 30.0)
    # First ring peaks at |Q| = 0.18 / 0.37 / 0.48; the old top of 1e-1
    # clipped them, which is why early time looked empty.  Top is 2 so the
    # G_x = 1.3 peak is not sitting on the spine.
    ax.set_ylim(1e-11, 2.0)
    ax.set_xlabel(r"$t / \lambda$")
    ax.set_ylabel("$|Q|$")
    st.legend(ax, scale, handles=_arrest_handles(scale), loc="upper right",
              ncol=1,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.80)
    return pn.finish(fig, ax, FIG, "SN7b", save, dpi, own)


# --------------------------------------------------------------------------
# SN7c / SN7d -- training loss, then the four scalars
# --------------------------------------------------------------------------

LOSS_MAIN_YLIM = (1e-5, 4e5)
#: Floor bottom is the 4.4e-12 landing (the "zero" of the broken axis).
#: Upper is raised so the last few incumbent points have headroom above it.
LOSS_FLOOR = 4.376059e-12
LOSS_FLOOR_YLIM = (LOSS_FLOOR, 3e-8)

PARAM_SERIES = (("Gp", r"$G_p$"), ("lam", r"$\lambda$"),
                ("nu_s", r"$\eta_s$"), ("tau_y", r"$\tau_y$"))

SEED_COLOR = {"s1": "#1b9e77", "s2": "#d95f02", "s3": "#7570b3",
              "s4": "#e7298a", "s5": "#1f77b4"}
SEED_LABEL = {sd: f"Seed {i}"
              for i, sd in enumerate(ld.EVP_SEEDS, start=1)}


def _incumbents(loss) -> np.ndarray:
    """Indices of L-BFGS-B accepted positions, not line-search trials."""
    keep, best = [], np.inf
    for i, L in enumerate(np.asarray(loss, float)):
        if L <= best:
            best = L
            keep.append(i)
    return np.asarray(keep, int)


def _break_marks(upper, lower, scale):
    """Diagonal ticks that mark a broken axis, with the spines opened."""
    kw = dict(marker=[(-1, -0.7), (1, 0.7)], markersize=11 * scale,
              linestyle="none", color="0.15", mec="0.15",
              mew=1.5 * scale, clip_on=False, zorder=5)
    upper.plot([0, 1], [0, 0], transform=upper.transAxes, **kw)
    lower.plot([0, 1], [1, 1], transform=lower.transAxes, **kw)
    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)
    upper.tick_params(bottom=False, labelbottom=False, which="both")
    lower.tick_params(top=False, which="both")


def accepted_ratio_spread() -> dict:
    """Per-parameter max |recovered/truth - 1| over every accepted state."""
    spread = {key: 0.0 for key, _ in PARAM_SERIES}
    for sd in ld.EVP_SEEDS:
        a = ld.accepted_progress(ld.evp_progress(sd))
        for key, _ in PARAM_SERIES:
            r = np.abs(a[key] / ld.EVP_TRUTH[key] - 1.0)
            spread[key] = max(spread[key], float(r.max()))
    return spread


def ratio_ylim() -> tuple[float, float]:
    """Tight shared ratio range, from the worst accepted |ratio-1|."""
    half = max(accepted_ratio_spread().values()) * 1.15
    return 1.0 - half, 1.0 + half


def scalar_ylim() -> tuple[float, float]:
    """Wide log range of stage-1 incumbent *physical* values (notes only)."""
    p = ld.evp_progress("s1")
    s1 = p["stage"] == "stage1"
    k = _incumbents(p["loss"][s1])
    lo, hi = np.inf, 0.0
    for key, _ in PARAM_SERIES:
        v = p[key][s1][k]
        lo = min(lo, float(v.min()))
        hi = max(hi, float(v.max()))
    return lo / 1.25, hi * 1.25


def _stage1_incumbents():
    p = ld.evp_progress("s1")
    s1 = p["stage"] == "stage1"
    k = _incumbents(p["loss"][s1])
    return p, s1, k, p["step"][s1][k], p["loss"][s1][k]


def _sn7c_limits():
    n1 = int((ld.evp_progress("s1")["stage"] == "stage1").sum())
    last = max(ld.evp_progress(sd)["step"][-1] for sd in ld.EVP_SEEDS)
    first_th = min(
        float(ld.accepted_progress(ld.evp_progress(sd))["step"][
            ld.accepted_progress(ld.evp_progress(sd))["stage"] != "stage1"][0])
        for sd in ld.EVP_SEEDS)
    return n1, first_th, last


def _seed_handles(scale):
    from matplotlib.lines import Line2D

    return [Line2D([], [], color=SEED_COLOR[sd], lw=_lw(scale, 0.9),
                   marker="o", ms=5 * scale, label=SEED_LABEL[sd])
            for sd in ld.EVP_SEEDS]


def _theta_loss_trace(p):
    """Loss after stage 1: every Adam step, resolve as incumbents only.

    ``c0``..``c3`` are real gradient steps.  The resolve blocks are L-BFGS-B
    and the raw rows include bound probes four to six orders above the
    incumbent, so those stay thinned.
    """
    steps, losses = [], []
    stage = np.asarray([str(s) for s in p["stage"]])
    i = 0
    while i < len(stage):
        st = stage[i]
        j = i
        while j < len(stage) and stage[j] == st:
            j += 1
        if st == "stage1":
            i = j
            continue
        sl, ss = p["loss"][i:j], p["step"][i:j]
        if st.startswith("c"):
            steps.append(ss)
            losses.append(sl)
        else:
            k = _incumbents(sl)
            steps.append(ss[k])
            losses.append(sl[k])
        i = j
    return np.concatenate(steps), np.concatenate(losses)


def _draw_loss(main, floor, scale, n1, last):
    from matplotlib.ticker import NullFormatter, NullLocator

    for axis in (main, floor):
        axis.axvspan(0, n1, color="0.90", zorder=0)
        axis.set_xlim(-0.02 * last, 1.03 * last)
    # Stage 1 is bit-identical in every seed: one shared curve.
    _, _, _, st1_step, st1_loss = _stage1_incumbents()
    for axis in (main, floor):
        axis.plot(st1_step, st1_loss, "-", color="0.25",
                  lw=_lw(scale, 0.95), zorder=2)
    for sd in ld.EVP_SEEDS:
        p = ld.evp_progress(sd)
        step, loss = _theta_loss_trace(p)
        for axis in (main, floor):
            axis.plot(step, loss, "-", color=SEED_COLOR[sd],
                      lw=_lw(scale, 0.75), zorder=3)
        a = ld.accepted_progress(p)
        later = a["stage"] != "stage1"
        main.plot(a["step"][later], a["loss"][later], "o",
                  color=SEED_COLOR[sd], ms=5.5 * scale, zorder=4)
    main.set_yscale("log")
    floor.set_yscale("log")
    main.set_ylim(*LOSS_MAIN_YLIM)
    floor.set_ylim(*LOSS_FLOOR_YLIM)
    floor.yaxis.set_minor_locator(NullLocator())
    floor.yaxis.set_minor_formatter(NullFormatter())
    floor.set_yticks([LOSS_FLOOR, 1e-8])
    floor.set_yticklabels([r"$4.4\times10^{-12}$", r"$10^{-8}$"])
    _break_marks(main, floor, scale)
    main.set_ylabel("Loss")
    main.yaxis.set_label_coords(-0.12, 0.42)
    main.tick_params(labelbottom=False)
    floor.set_xlabel("Iteration")
    st.legend(main, scale, handles=_seed_handles(scale), loc="upper right",
              ncol=1,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.68)


def plot_SN7c(ax=None, save=True, dpi=None):
    """Broken-axis training loss, one colour per network seed."""
    own = ax is None
    if own:
        fig, slots, scale = pn.new_stack(
            2, 11.0, 7.6, sharex=True, axes_width=9.5,
            gridspec_kw=dict(hspace=0.055, height_ratios=[1.0, 0.20]))
        main, floor = slots
    else:
        fig = ax.get_figure()
        main, floor = pn.subaxes(ax, 2, 1, sharex=True, hspace=0.055,
                                 height_ratios=[1.0, 0.20])
        scale = pn.adopt(main)
        pn.adopt(floor)
    n1, _, last = _sn7c_limits()
    _draw_loss(main, floor, scale, n1, last)
    if own:
        fig.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.12,
                            hspace=0.055)
        if save:
            print(f"[SN7c] {pn.save_panel(fig, FIG, 'SN7c', dpi)}", flush=True)
            write_notes()
        return fig
    return [main, floor]


def _stage1_param_ylim(key: str) -> tuple[float, float]:
    """Linear range for one scalar from the stage-1 incumbent walk only."""
    p, s1, k, _, _ = _stage1_incumbents()
    v = p[key][s1][k]
    lo = min(float(v.min()), float(ld.EVP_TRUTH[key]))
    hi = max(float(v.max()), float(ld.EVP_TRUTH[key]))
    pad = 0.08 * (hi - lo)
    return max(0.0, lo - pad), hi + pad


def _split_sn7d(axs):
    """Column-wise sharex; shared ratio y on the right only."""
    left, right = axs[0::2], axs[1::2]
    for a in left[1:]:
        a.sharex(left[0])
    for a in right[1:]:
        a.sharex(right[0])
        a.sharey(right[0])
    return left, right


def plot_SN7d(ax=None, save=True, dpi=None):
    """Four scalars as 4x2: stage-1 absolute (left), theta-block ratio (right).

    Columns do not share x.  Stage 1 is one deterministic curve; the five
    network seeds appear only on the right.
    """
    from matplotlib.ticker import FormatStrFormatter

    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(
            4, 13.6, 14.2, ncols=2, sharex=False, axes_width=5.4,
            gridspec_kw=dict(wspace=0.42, hspace=0.14))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 4, 2, sharex=False, wspace=0.40, hspace=0.14)
        scale = pn.adopt(axs[0])
        for a in axs[1:]:
            pn.adopt(a)

    left, right = _split_sn7d(axs)
    n1, first_th, last = _sn7c_limits()
    p, s1, k, st1_step, _ = _stage1_incumbents()
    rlo, rhi = ratio_ylim()

    for axis, (key, lab) in zip(left, PARAM_SERIES):
        axis.set_xlim(-0.6, n1 + 2.0)
        axis.axhline(ld.EVP_TRUTH[key], color=st.C_TRUTH, ls="--",
                     lw=_lw(scale, 0.65), zorder=1)
        axis.plot(st1_step, p[key][s1][k], "-", color="0.25",
                  lw=_lw(scale, 0.95), zorder=2)
        axis.set_ylim(*_stage1_param_ylim(key))
        axis.set_ylabel(lab)

    for axis, (key, lab) in zip(right, PARAM_SERIES):
        axis.set_xlim(first_th * 0.92, last * 1.03)
        axis.axhline(1.0, color=st.C_TRUTH, ls="--",
                     lw=_lw(scale, 0.65), zorder=1)
        for sd in ld.EVP_SEEDS:
            a = ld.accepted_progress(ld.evp_progress(sd))
            later = a["stage"] != "stage1"
            axis.plot(a["step"][later], a[key][later] / ld.EVP_TRUTH[key],
                      "o-", color=SEED_COLOR[sd],
                      ms=5 * scale, lw=_lw(scale, 0.75), zorder=3)
        axis.set_ylabel(f"{lab} / truth")
    right[0].set_ylim(rlo, rhi)
    right[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    for a in (*left[:-1], *right[:-1]):
        a.tick_params(labelbottom=False)
    left[-1].set_xlabel("Iteration")
    right[-1].set_xlabel("Iteration")

    if own:
        pn.tidy(fig)
        if save:
            print(f"[SN7d] {pn.save_panel(fig, FIG, 'SN7d', dpi)}", flush=True)
            write_notes()
        return fig
    return axs


# --------------------------------------------------------------------------
# the numbers the notes quote
# --------------------------------------------------------------------------

def _crossings(t, q):
    """Zero-crossing times of Q, linearly interpolated between samples."""
    s = np.sign(q)
    idx = np.where(np.diff(s) != 0)[0]
    return np.array([t[i] + (t[i + 1] - t[i]) * abs(q[i])
                     / (abs(q[i]) + abs(q[i + 1])) for i in idx])


def arrest_metrics() -> dict:
    """Final |Q|, ringing period and phase agreement at the three sub-yield
    drives, read from the same arrays panel (b) plots."""
    out = {}
    for gx in ARREST_DRIVES:
        d = ld.evp_profile("truth", gx)
        ct = _crossings(d["t"], d["Q"])
        period = 2.0 * np.diff(ct)
        seeds, phase = {}, 0.0
        for sd in ld.EVP_SEEDS:
            s = ld.evp_profile(sd, gx)
            cs = _crossings(s["t"], s["Q"])
            n = min(len(ct), len(cs))
            phase = max(phase, float(np.abs(cs[:n] - ct[:n]).max()))
            seeds[sd] = float(abs(s["Q"][-1]))
        out[gx] = {"q_truth": float(abs(d["Q"][-1])), "q_seeds": seeds,
                   "n_crossings": int(len(ct)),
                   "period": float(period.mean()),
                   "period_lam": float(period.mean() / LAM),
                   "phase_max": phase}
    return out


def training_metrics() -> dict:
    """Stage-1 floor, final loss, gradient count, and final scalars per seed."""
    scal = ld.evp_scalars()
    out = {}
    for sd in ld.EVP_SEEDS:
        p = ld.evp_progress(sd)
        s1 = p["stage"] == "stage1"
        a = ld.accepted_progress(p)
        finals = {k: scal[sd][k] for k, _ in PARAM_SERIES}
        pct = {k: 100.0 * abs(finals[k] - ld.EVP_TRUTH[k]) / ld.EVP_TRUTH[k]
               for k, _ in PARAM_SERIES}
        out[sd] = {"stage1_min": float(p["loss"][s1].min()),
                   "final": float(p["loss"][-1]),
                   "n_grad": int(p["step"][-1]),
                   "n_raw": int(len(p["step"])),
                   "raw_max": float(p["loss"].max()),
                   "accepted_max": float(a["loss"].max()),
                   "n_blocks": int(len(a["step"])),
                   "scalars": finals, "pct_err": pct}
    return out


def training_schedule() -> dict:
    """Per-seed block map: name, step range, who is free."""
    out = {}
    for sd in ld.EVP_SEEDS:
        p = ld.evp_progress(sd)
        stage = np.asarray([str(s) for s in p["stage"]])
        blocks = []
        i = 0
        while i < len(stage):
            st = stage[i]
            j = i
            while j < len(stage) and stage[j] == st:
                j += 1
            if st == "stage1":
                who = "scalars, theta pinned at OB"
            elif st.startswith("c"):
                who = "Adam on theta, scalars pinned"
            else:
                who = "L-BFGS-B scalars, theta pinned"
            blocks.append({"name": st, "lo": int(p["step"][i]),
                           "hi": int(p["step"][j - 1]), "n": j - i,
                           "who": who})
            i = j
        out[sd] = blocks
    return out


def duplication_check() -> dict:
    """Is panel (a) a subset of main-text N2a?

    Compares what each actually draws, not what the brief specced.
    """
    from . import fig_n2 as n2

    shared = [g for g in n2.DRIVES if g == n2.MAIN_DRIVE]
    return {"n2a_drives": (n2.MAIN_DRIVE,), "sn7a_drives": n2.DRIVES,
            "n2a_arms": ("truth", "obinit", "5 seeds"),
            "sn7a_arms": ("truth", "5 seeds"),
            "shared_drives": tuple(shared),
            "sn7a_only": tuple(g for g in n2.DRIVES if g != n2.MAIN_DRIVE),
            "verdict": "not a subset: SN7a adds three drives, one of them "
                       "held out, and drops the initial closure"}


def notes() -> str:
    import textwrap

    def para(text, indent=""):
        return textwrap.fill(" ".join(text.split()), width=78,
                             initial_indent=indent,
                             subsequent_indent=" " * len(indent))

    am, tm, dup = arrest_metrics(), training_metrics(), duplication_check()
    sched = training_schedule()
    drift = ld.evp_theta_drift()
    spread = accepted_ratio_spread()
    rel = {sd: tm[sd]["pct_err"] for sd in ld.EVP_SEEDS}
    worst_sd = max(rel, key=lambda s: max(rel[s].values()))
    worst_k = max(rel[worst_sd], key=lambda k: rel[worst_sd][k])
    worst = rel[worst_sd][worst_k]
    best = min(min(r.values()) for r in rel.values())
    finals = [tm[sd]["final"] for sd in ld.EVP_SEEDS]
    grads = [tm[sd]["n_grad"] for sd in ld.EVP_SEEDS]
    floor = tm["s1"]["stage1_min"]
    same_floor = all(abs(tm[sd]["stage1_min"] - floor) <= 1e-18
                     for sd in ld.EVP_SEEDS)
    names = {"Gp": "G_p", "lam": "lambda", "nu_s": "eta_s", "tau_y": "tau_y"}

    out = [para(
        "Figure SN7 -- EVP profiles, arrest, training.  Four panels: (a) u(y) "
        "at four drives, (b) the sub-yield arrest transient, (c) the training "
        "loss, (d) the four scalars as a 4x2 (stage-1 absolute | theta-block "
        "ratio).  Five network seeds throughout (evp_fix_A_3lam_agn and its "
        "s2..s5 siblings); (c) and the right column of (d) colour them and "
        "name them Seed 1..5.  The seeds differ by the random draw of the "
        "network hidden-layer weights.  Every scalar starts at 1.0 in every "
        "run, so 'initialization' would be ambiguous once (d) also shows "
        "that scalar start."), ""]

    out += [para(
        "PANELS CUT.  The 12-drive u(y) grid is gone: three of its twelve "
        "rungs are below yield, where the profile is not a small flow but "
        "round-off residual at 1e-11, and two more sit above the "
        "well-posedness ceiling, so half the grid was shape-of-noise at "
        "thumbnail size.  The drives that carry the argument are in (a) at "
        "readable size.  Plug half-width against drive is also gone: it showed "
        "the arrest cliff at G_c and the ceiling above G_x = 5, which is what "
        "Q(G_x) in the main figure shows, so it was a second view of two facts "
        "the reader already has.  Both stay callable "
        "(plot_SN7_ladder_grid, plot_SN7_plug_halfwidth) and their last "
        "renders are in retired/."), ""]

    out += [para(
        "DUPLICATION CHECK on the promoted profile panel.  As built, N2a draws "
        f"ONE drive, G_x = {dup['n2a_drives'][0]:g}, with the initial TBNN "
        "alongside truth and the trained seeds -- it is not the four-drive "
        "grid the brief describes, because the main figure was cut back to a "
        "single drive earlier in this revision.  Panel (a) draws "
        + ", ".join(f"G_x = {g:g}" for g in dup["sn7a_drives"])
        + " with truth and the trained seeds only.  So (a) is NOT a subset: it "
        "adds " + ", ".join(f"G_x = {g:g}" for g in dup["sn7a_only"])
        + f", of which G_x = 5 is held out and is also the well-posedness "
        "ceiling, and it drops the initial closure.  The overlap is one cell "
        f"of four: G_x = {dup['shared_drives'][0]:g} repeats the main figure's "
        "truth and trained curves without the initial one.  That cell is kept "
        "so the four drives can be read as a ladder; say the word and it comes "
        "out."), ""]

    out += [para(
        "PANEL (b), WHAT THE RINGING IS.  Below G_c the channel does not creep "
        "to a stop, it rings down.  The sub-yield response is a damped elastic "
        "oscillation: the polymer network is the spring (G_p = 3.2), the fluid "
        "carries the inertia (density 1.0) and the solvent is the damper "
        f"(eta_s = 0.8).  Measured period {am[0.5]['period']:.3f} time units, "
        f"{am[0.5]['period_lam']:.2f} lambda, steady to better than 1% over "
        "the whole window at both G_x = 0.5 and 1.0.  For scale, the "
        "fundamental standing shear wave of a channel of half-width 1 at speed "
        "sqrt(G_p/rho) = 1.789 has period 4H/c = 2.236 time units, 7% from the "
        "measurement -- close enough to say the oscillation is the material's "
        "elastic mode and not a solver artefact."), ""]

    out += [para(
        "Truth and all five learned closures ring at the same frequency and in "
        "phase.  Zero crossings of Q agree to at most "
        f"{max(v['phase_max'] for v in am.values()):.1e} time units, which is "
        "0.5% of a period, and the count is identical arm by arm: "
        + ", ".join(f"{am[g]['n_crossings']} at G_x = {g:g}"
                    for g in ARREST_DRIVES)
        + ".  Amplitude decays through eight decades with no pedestal; at "
        "30 lambda it is still decaying, so these are upper bounds, not "
        "asymptotes.  Final |Q|:"), ""]
    for gx in ARREST_DRIVES:
        v = am[gx]
        lo = min(v["q_seeds"].values())
        hi = max(v["q_seeds"].values())
        out.append(f"  G_x = {gx:<4g} truth {v['q_truth']:.3e}   "
                   f"learned {lo:.3e} to {hi:.3e}")
    out += ["", para(
        f"So |Q| is below 1e-9 at G_x = 0.5 and 1.0 and "
        f"{am[1.3]['q_truth']:.1e} at G_x = 1.3, the rung just under G_c; the "
        "spread over the five closures is 0.1% at the two lower drives and 8% "
        "at 1.3, which is scatter on a round-off floor rather than "
        "disagreement about flow."), ""]

    out += [para(
        "PANEL (b), HOW IT IS DRAWN.  Ground truth is the solid line and each "
        "learned closure is open markers with no line, N1e's convention, "
        "labelled TBNN prediction.  All five closures are drawn, and they "
        "overlap into what looks like one marker series -- that coincidence "
        "is the result, and the count is here rather than on the panel.  "
        "Markers are that arm's own turning points of |Q| (peaks and "
        "notches) plus one midpoint on each descent, so the valleys are "
        "on the panel and not only the envelope.  They are not a "
        "shared-time stride through the notch: a marker at the truth dip "
        "time sits on the wall of the learned notch and reads as a "
        "disagreement that is a 0.001-lambda timing offset."), ""]

    out += [para(
        "PANELS (c) AND (d) are separate figures.  (c) is the broken-axis "
        "loss.  (d) is a 4x2: rows G_p, lambda, eta_s, tau_y; columns are "
        "two different regimes and do not share x.  Legend: Seed 1..5 = "
        "s1..s5, theta_seed 0, 2, 3, 4, 5 -- the random draw of the network "
        "hidden-layer weights, not a scalar start (every scalar starts at "
        "1.0).  Stage 1 is deterministic -- L-BFGS-B on the scalars with "
        "theta pinned at Oldroyd-B -- so the five runs are bit-identical "
        "there and the seed enters only the later theta blocks."), ""]

    out += [para(
        "PANEL (c), LOSS AXIS.  Stage 1 incumbent sequence (24 accepted "
        "positions, 1.19e5 down to 4.376e-12; six line-search trials that "
        "went up are dropped), drawn once in grey.  The grey shade is kept; "
        "the 'stage 1' text is not.  Upper strip 1e-5 to 4e5; floor strip "
        f"bottom is the 4.4e-12 landing (the literal zero of the broken "
        f"axis), upper {LOSS_FLOOR_YLIM[1]:.0e}, drawn as a thin band "
        "(height ratio 0.20) so both 1e-8 and 4.4e-12 fit as ticks and "
        "the break gap is just wide enough for the 1e-8 label.  After "
        "stage 1 the coloured traces are every Adam step in c0..c3 "
        "(60 steps each; c0 is 31-90, which is the gap the previous "
        "draw left empty) and the resolve blocks as incumbents only -- "
        "those are L-BFGS-B and the raw rows include bound probes.  "
        "Markers are still the accepted block-ends.  One colour per "
        f"seed.  Raw max of the logged trials is {tm['s1']['raw_max']:.1e}."), ""]

    ends = ", ".join(f"{sd} at {sched[sd][-1]['hi']}" for sd in ld.EVP_SEEDS)
    out += [para(
        "TRAINING SCHEDULE.  Never both free.  The order is identical in "
        "every seed: stage1 (scalars, theta pinned at Oldroyd-B), c0 "
        "(Adam on theta, scalars pinned), then four resolve/c sandwiches "
        "(L-BFGS-B on the scalars, then Adam on theta).  stage1 is always "
        "steps 1-30 and c0 is always 31-90.  What varies seed-to-seed is "
        "how long each resolve runs, so the later c blocks slide and the "
        f"runs end at different steps ({ends})."), ""]
    out.append(f"  {'seed':<6}{'block':<10}{'steps':>12}{'n':>5}  who")
    for sd in ld.EVP_SEEDS:
        for k, b in enumerate(sched[sd]):
            tag = sd if k == 0 else ""
            out.append(f"  {tag:<6}{b['name']:<10}{b['lo']:>5}-{b['hi']:<6}"
                       f"{b['n']:5d}  {b['who']}")
        out.append("")

    out += [para(
        "PANEL (d), LEFT COLUMN -- stage 1 only.  x is iterations 0 to the "
        "end of stage 1 (~30), not extended to 400.  y is the absolute "
        "parameter, linear, dashed truth.  One grey curve, not five: theta "
        "is pinned at Oldroyd-B, so the five runs are bit-identical and the "
        "seed has not entered yet.  No title, no shade, no legend (the "
        "seed colours live on (c)).  The job is the walk from the agnostic "
        "start (all four at 1.0) onto truth, including the overshoot to "
        "the bounds (G_p to 10, lambda and eta_s to 5, tau_y to 3)."), ""]

    out += [para(
        "PANEL (d), RIGHT COLUMN -- theta blocks.  x is the first accepted "
        "theta-block state through the last iteration (~90-402), independent "
        "of the left column.  y is recovered/truth, linear, one shared "
        f"range {ratio_ylim()[0]:.5f} to {ratio_ylim()[1]:.5f} from the "
        "worst-case accepted |ratio-1| (lambda), not autoscaled per row -- "
        f"eta_s moves +-{100*spread['nu_s']:.3f}% and lambda "
        f"+-{100*spread['lam']:.3f}%, and independent autoscaling would "
        "draw them equally wide and invert the message.  Five seed curves, "
        "accepted-state markers, no legend (see (c)), no title, no shade.  "
        "Worst final is "
        f"{names[worst_k]} in {worst_sd} at {worst:.3f}%; best "
        f"{best:.4f}%."), ""]

    out += [para(
        f"Stage 1 (L-BFGS-B on the four scalars, theta held at OB-init) bottoms "
        f"out at {floor:.4e}"
        + (" in every one of the five seeds, to the last digit stored"
           if same_floor else " (seeds differ)")
        + ".  That identity is not luck and is not a shared random seed: "
        "theta is pinned at the reconstructed Oldroyd-B initialisation, "
        "the four scalars start at 1.0, and L-BFGS-B is deterministic, so "
        "stage 1 is the same problem in every run.  The network-weight "
        "draw (theta_seed 0, 2, 3, 4, 5) enters only when the later theta "
        "blocks release the hidden-layer weights."), ""]

    out += ["Per-seed stage-1 loss, final loss, and gradient count:"]
    for sd in ld.EVP_SEEDS:
        out.append(f"  {sd}  stage-1 {tm[sd]['stage1_min']:.6e}   "
                   f"final {tm[sd]['final']:.4e}   "
                   f"{tm[sd]['n_grad']} gradients   "
                   f"({tm[sd]['n_blocks']} block ends of "
                   f"{tm[sd]['n_raw']} logged rows)")
    out += ["", para(
        f"Final losses run {min(finals):.1e} to {max(finals):.1e} in "
        f"{min(grads)}-{max(grads)} gradients, nine orders above the stage-1 "
        "floor."), ""]

    out += ["Per-seed final scalars and % errors "
            "(truth G_p=3.2, lambda=0.7, eta_s=0.8, tau_y=1.45):"]
    hdr = (f"  {'seed':<6}{'G_p':>10}{'%':>8}{'lambda':>10}{'%':>8}"
           f"{'eta_s':>10}{'%':>8}{'tau_y':>10}{'%':>8}")
    out.append(hdr)
    for sd in ld.EVP_SEEDS:
        v, e = tm[sd]["scalars"], tm[sd]["pct_err"]
        out.append(f"  {sd:<6}{v['Gp']:10.6f}{e['Gp']:8.4f}"
                   f"{v['lam']:10.6f}{e['lam']:8.4f}"
                   f"{v['nu_s']:10.6f}{e['nu_s']:8.4f}"
                   f"{v['tau_y']:10.6f}{e['tau_y']:8.4f}")
    out += ["", para(
        f"Worst final is {names[worst_k]} in {worst_sd} at {worst:.3f}%; "
        f"best {best:.4f}%.  Accepted-state spreads (max |ratio-1| over the "
        "whole trajectory, all seeds): "
        + ", ".join(f"{names[k]} +-{100*spread[k]:.3f}%"
                    for k, _ in PARAM_SERIES) + "."), ""]

    dim = next(iter(drift.values()))["dim"]
    a1 = ld.accepted_progress(ld.evp_progress("s1"))
    steps = ", ".join(str(int(s)) for s in a1["step"])
    stages = ", ".join(str(s) for s in a1["stage"])
    out += [para(
        "THETA DRIFT, ||theta - theta_OB||_2 vs iteration -- still no panel, "
        "reported here.  progress.csv does not store theta, so the series is "
        f"the five saved stage checkpoints only (dim = {dim}).  Stage 1 is "
        "machine-epsilon from the reconstructed OB-init (theta frozen).  "
        "Resolve checkpoints are after the Adam theta block plus the scalar "
        f"re-solve.  Accepted block-end steps for seed 1 are [{steps}] at "
        f"stages {stages}; only stage1 and the four resolve archives have a "
        "theta pickle."), ""]
    out.append(f"  {'seed':<6}{'th_seed':>8}"
               + "".join(f"{tag:>12}" for tag in ld.EVP_THETA_STAGES))
    for sd in ld.EVP_SEEDS:
        d = drift[sd]
        out.append(f"  {sd:<6}{d['theta_seed']:>8}"
                   + "".join(f"{d['stages'][tag]:12.3e}"
                             for tag in ld.EVP_THETA_STAGES))
    out += [""]

    out += [para(
        "LAMBDA SPREAD.  Lambda has the widest spread of the four parameters "
        f"here (~+-{100*spread['lam']:.2f}% vs ~+-{100*spread['nu_s']:.2f}% "
        "for eta_s).  Far too small to claim anything from, but record it: "
        "lambda is also the softest direction in the Giesekus schedule "
        "scatter and the FENE-P recovery."), ""]

    out += [para(
        "MANUSCRIPT SENTENCE.  The theta blocks degrade the training loss by "
        "nine orders while every scalar stays within 0.1% of truth.  At "
        "OB-init the closure already represents the Saramito truth exactly, "
        "so releasing theta can only move away from an exact optimum; the "
        "blocks are included to show the recovery does not depend on freezing "
        "theta there."), ""]

    out += [para(
        "NOMENCLATURE.  This figure now writes G_x and G_c, matching N2, and "
        "names the arms 'Ground truth' / 'Trained TBNN' in (a) and 'Ground "
        "truth' / 'TBNN prediction' in (b)."), ""]
    out += [ceiling_caption(), ""]
    return "\n".join(out)


def ceiling_caption() -> str:
    """Caption text for the main EVP figure's shaded region, and the Methods
    sentence on the evaluation horizon.

    Both are what is left of retired SN8: (c) was the well-posedness ceiling
    panel and (a) the forward sensitivity sweep.  A reader who sees the grey
    band in N2b will ask what sets it, so the numbers live here rather than in
    a panel of their own.
    """
    import textwrap

    def para(text):
        return textwrap.fill(" ".join(text.split()), width=78)

    s = ld.evp_stability_ladder()
    g = s["g"]
    i6 = int(np.argmin(np.abs(g - 6.0)))
    past = s["min_eig"][:, g > 5.0]
    ceilings = ", ".join(f"{c:.1f}" for c in s["ceiling"])
    sens = ld.evp_sensitivity()
    peaks = {h: (float(v["g"][int(np.argmax(np.abs(v["logderiv"])))]),
                 float(np.abs(v["logderiv"]).max())) for h, v in sens.items()}

    out = ["CARRIED OVER FROM RETIRED SN8", "",
           para(
        "Caption text for the well-posedness ceiling -- the shaded region "
        f"above G_x = 5.  Above G_x = 5 some seeds lose well-posedness.  At "
        f"G_x = 6 the seeds reach max|tau_d| ~ "
        f"{np.nanmax(s['td'][:, i6]):.0f} against "
        f"{s['truth_td'][i6]:.2f} for truth, while min eig(A) stays strictly "
        f"positive ({np.nanmin(past):.0e} to {np.nanmax(past):.0e}) -- a "
        "dynamic instability of the coupled system, not a representational "
        f"limit of the closure.  Per-seed contiguous ceiling "
        f"{{{ceilings}}}, non-monotone.  No selection is made on the seeds "
        "that reach 6.0."), "",
           para(
        "Methods sentence on the evaluation horizon.  The sensitivity of the "
        "flux to the yield stress peaks at a different drive for a 3-lambda "
        f"horizon (|d ln Q / d ln tau_y| = {peaks[3.0][1]:.1f} at "
        f"G_x = {peaks[3.0][0]:g}) than for 15 lambda "
        f"({peaks[15.0][1]:.1f} at G_x = {peaks[15.0][0]:g}), so which drive "
        "looks informative is a property of the protocol.  That is why every "
        "reported evaluation uses one fixed common 15-lambda ladder with early "
        "stopping off."), "",
           para(
        "Both were panels of SN8, which is retired: the sensitivity sweep is a "
        "forward property of the Saramito truth with no TBNN in it, and the "
        "ceiling panel was a numerical-failure diagnosis.  The numbers above "
        "are read live through the same loaders those panels used, from "
        f"{ld.dp.path('evp_seed_eval_summary').name} and "
        f"{ld.dp.path('evp_phaseB').name}, so they cannot drift.  The retired "
        "code is paper_figs.obsolete.fig_sn8 and the last renders are in "
        "final_figures/obsolete/.")]
    return "\n".join(out)


def write_notes():
    out = ld.dp.out_dir(FIG) / "SN7_notes.txt"
    out.write_text(notes())
    print(f"[SN7_notes] {out}", flush=True)
    return out


# --------------------------------------------------------------------------
# assembled figure
# --------------------------------------------------------------------------

def plot_SN7(save=True, dpi=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(26.0, 36.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.72, 1.05, 1.85],
                          width_ratios=[1.0, 1.05], hspace=0.28,
                          wspace=0.20, left=0.05, right=0.985, top=0.96,
                          bottom=0.05)
    with pn.uniform_scale(0.50) as scale:
        axs = plot_SN7a(ax=fig.add_subplot(gs[0, :]))
        pn.panel_tag(axs[0], "(a)", scale, loc="outside")
        a = fig.add_subplot(gs[1, 0])
        plot_SN7b(ax=a)
        pn.panel_tag(a, "(b)", scale, loc="outside")
        axs = plot_SN7c(ax=fig.add_subplot(gs[1, 1]))
        pn.panel_tag(axs[0], "(c)", scale, loc="outside")
        axs = plot_SN7d(ax=fig.add_subplot(gs[2, :]))
        pn.panel_tag(axs[0], "(d)", scale, loc="outside")
    if save:
        print(f"[SN7_full] {pn.save_panel(fig, FIG, 'SN7_full', dpi)}",
              flush=True)
    return fig


# --------------------------------------------------------------------------
# RETIRED.  Kept callable; not in the assembled figure and not in the index.
# Their last renders are in final_figures/SN7/retired/.
#
# plot_SN7_ladder_grid: u(y) on all twelve rungs.  Cut because the three
# sub-yield rungs are round-off residual at 1e-11 and two more are past the
# well-posedness ceiling, so half the grid showed the shape of noise; the
# informative drives are panel (a) at full size.
#
# plot_SN7_plug_halfwidth: kinematic plug half-width against drive.  Cut
# because Q(G_x) in the main figure already shows the arrest cliff at G_c and
# the ceiling above G_x = 5.
# --------------------------------------------------------------------------

def plot_SN7_ladder_grid(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, axs, scale = pn.new_stack(2, 26.0, 11.0, ncols=6, sharey=True,
                                       axes_width=3.4,
                                       gridspec_kw=dict(wspace=0.16,
                                                        hspace=0.42))
    else:
        fig = ax.get_figure()
        axs = pn.subaxes(ax, 2, 6, sharey=True, wspace=0.16, hspace=0.42)
        scale = pn.adopt(axs[0])
        for a in axs[1:]:
            pn.adopt(a)

    fs = st.NOTEBOOK_EFFECTIVE["tick_labelsize_minor"] * scale
    for k, (a, gx) in enumerate(zip(axs, LADDER)):
        n_nan = 0
        for j, sd in enumerate(ld.EVP_SEEDS):
            if ld.evp_nan(sd, gx):
                n_nan += 1
                continue
            yy, us = _profile(sd, gx)
            a.plot(us, yy, "-", color=st.C_LEARN, lw=_lw(scale, 0.8),
                   alpha=0.85, zorder=3)
        yy, ut = _profile("truth", gx)
        a.plot(ut, yy, "--", color=st.C_TRUTH, lw=_lw(scale, 0.9), zorder=4)
        if gx > TAU_Y:
            for s in (-1, 1):
                a.axhline(s * TAU_Y / gx, color=st.C_TRUTHFAM, ls="-.",
                          lw=_lw(scale, 0.55), zorder=2)
        a.set_ylim(-1.0, 1.0)
        a.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        a.set_xlim(left=0.0)
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        a.xaxis.get_offset_text().set_fontsize(fs)
        a.set_title(f"$G_x = {gx:g}$" + ("  (training)"
                                         if gx in TRAINING_DRIVES else ""),
                    fontsize=st.NOTEBOOK_EFFECTIVE["axes_label_fontsize"]
                    * scale * 0.95, pad=5 * scale)
        if k >= 6:
            a.set_xlabel("$u_x$")
        if k % 6 == 0:
            a.set_ylabel("$y$")
        if gx in ARREST_DRIVES:
            pn.annotate(a, 0.5, 0.82, "arrested\n(round-off residual)", scale,
                        ha="center", va="center", color="0.45",
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
        if n_nan:
            pn.annotate(a, 0.96, 0.04, f"{n_nan}/5 seeds NaN", scale,
                        ha="right", color="0.2")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=st.C_TRUTH, ls="--", lw=_lw(scale, 0.9),
               label=TRUTH_LABEL),
        Line2D([], [], color=st.C_LEARN, ls="-", lw=_lw(scale, 0.8),
               label=LEARNED_LABEL),
        Line2D([], [], color=st.C_TRUTHFAM, ls="-.", lw=_lw(scale, 0.55),
               label=r"$\pm\tau_y/G_x$")]
    lg = axs[0].legend(handles=handles, loc="center", frameon=False,
                       fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"]
                       * scale * 0.85, handlelength=2.2, borderpad=0.2,
                       labelspacing=0.35)
    lg.set_zorder(6)
    pn.annotate(axs[0], 0.5, 0.06, "unmarked drives are held out", scale,
                ha="center", va="bottom", color="0.35")

    if own:
        pn.tidy(fig)
        if save:
            print(f"[retired] {pn.save_panel(fig, FIG, 'SN7_retired_ladder_grid', dpi)}",
                  flush=True)
        return fig
    return axs


def plot_SN7_plug_halfwidth(ax=None, save=True, dpi=None):
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(11.0, 8.0)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    p = ld.evp_plug_ladder()
    g, yielded = p["g"], p["g"] > TAU_Y
    ax.axvspan(0.3, ld.EVP_G_C, color="0.90", zorder=0)
    ax.axvspan(5.0, 6.2, color="0.90", zorder=0)
    ax.plot(g[yielded], p["analytic"][yielded], "-", color=st.C_TRUTHFAM,
            lw=_lw(scale, 1.0), zorder=2, label=r"analytic $\tau_y/G_x$")
    # A NaN'd forward leaves stale arrays behind, so those rungs are broken
    # out of the line rather than bridged.
    learned = np.where(p["nan"], np.nan, p["learned"])
    for j, sd in enumerate(ld.EVP_SEEDS):
        ax.plot(g, learned[j], "-", color=st.C_LEARN, lw=_lw(scale, 0.7),
                alpha=0.8, zorder=3,
                label=LEARNED_LABEL if j == 0 else None)
    ax.plot(g, p["truth"], "o", ms=11 * scale, mfc="white",
            color=st.C_TRUTH, mew=1.6 * scale, zorder=4, label=TRUTH_LABEL)

    dy = 2.0 / 64
    within = (g > ld.EVP_G_C) & (g <= 5.0)
    worst = np.nanmax(np.abs(learned[:, within] - p["truth"][within])) / dy
    at_gc = np.nanmax(np.abs(learned[:, g == ld.EVP_G_C]
                             - p["truth"][g == ld.EVP_G_C])) / dy
    ax.set_xlim(0.3, 6.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("$G_x$")
    ax.set_ylabel("kinematic plug half-width")
    ax.axvline(ld.EVP_G_C, color=st.C_TRUTHFAM, ls="-.", lw=_lw(scale, 0.8),
               zorder=1)
    pn.annotate(ax, 0.035, 0.55, "arrested\n(no plug edge)", scale,
                ha="left", va="bottom", color="0.35")
    pn.annotate(ax, 0.985, 0.055, "past the\nceiling", scale, ha="right",
                va="bottom", color="0.35")
    pn.annotate(ax, 0.98, 0.94,
                f"learned equals truth cell-for-cell over\n"
                rf"$G_c < G_x \leq 5$ (worst {worst:.0f} of "
                rf"$\Delta y = {dy:.4f}$);" "\n"
                rf"at $G_c$ itself two seeds are {at_gc:.0f} cells wide",
                scale, ha="right", va="top", color="0.3")
    st.legend(ax, scale, loc="center right", ncol=1,
              fontsize=st.NOTEBOOK_RCPARAMS["legend.fontsize"] * scale * 0.85)
    return pn.finish(fig, ax, FIG, "SN7_retired_plug_halfwidth", save, dpi,
                     own)
