"""Generated tables for the supplement.

Table S2 is the row-per-run version of SN5: every FENE-P run in the condition
ladder, its four recovered parameters against truth, how far its battery
residual sits above the noise floor, and which family the battery selected.

The EVP robustness table is the agnostic half of the eight-arm matrix that
used to be the SN8b scatter: drive set x training horizon, one row per arm,
signed recovery error per scalar.  B-R initializations are recorded in the
notes, not in the table.  It is rendered twice -- as an image styled like the
N1d parameter table, so it can be dropped into the EVP supplement figure, and
as markdown plus LaTeX here.
"""
from __future__ import annotations

import os

import numpy as np

from . import data_paths as dp
from . import loaders as ld
from . import panels as pn
from .fig_sn5 import CONDITIONS, REPS

TABLE_DIR = dp.OUT_ROOT / "tables"
PARAMS = (("eta_p", r"\eta_p", "2.24"), ("lam", r"\lambda", "0.7"),
          ("nu_s", r"\nu_s", "0.8"), ("Lsq", "L^2", "12"))
CONTROL = "clean_analytic_fene_p"


def _row(target: str, condition: str) -> dict:
    rec = ld.fene_recovery(target)
    err = ld.fene_recovery_errors(target)
    winner, margin = ld.battery_winner(target)
    try:
        loss = ld.fene_summary(target)["loss_final"]
    except KeyError:
        loss = None
    return {"condition": condition, "run": target,
            "label": ld.FENE_LABEL.get(target, target),
            "representative": REPS.get(condition) == target,
            "values": {k: rec[k] for k, _, _ in PARAMS},
            "errors": {k: 100 * err[k] for k, _, _ in PARAMS},
            "loss_final": loss, "resid_x_floor": rec["mse"] / ld.NOISE_FLOOR,
            "winner": winner, "margin": margin}


def table_S2_rows() -> list[dict]:
    rows = [_row(t, cond) for cond, members in CONDITIONS for t in members]
    rows.append(_row(CONTROL, "clean analytic control"))
    return rows


def _fmt(v, spec="{:.3f}"):
    return "--" if v is None else spec.format(v)


def _markdown(rows: list[dict]) -> str:
    cols = (["condition", "run"] +
            [c for k, _, _ in PARAMS for c in (k, "err %")] +
            ["final loss", "resid / floor", "BIC winner", "margin"])
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    out = [head, sep]
    prev = None
    for r in rows:
        cond = r["condition"] if r["condition"] != prev else ""
        prev = r["condition"]
        name = r["label"] + (" \\*" if r["representative"] else "")
        cells = []
        for k, _, _ in PARAMS:
            cells += [_fmt(r["values"][k]), f"{r['errors'][k]:+.1f}"]
        out.append("| " + " | ".join(
            [cond, name] + cells +
            [_fmt(r["loss_final"], "{:.3f}"),
             f"{r['resid_x_floor']:.2f}", r["winner"],
             f"{r['margin']:,.2f}"]) + " |")
    return "\n".join(out)


def _latex(rows: list[dict]) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{FENE-P parameter recovery across training conditions."
             r" Truth $(\eta_p,\lambda,\nu_s,L^2) = (2.24, 0.7, 0.8, 12)$."
             r" ``resid'' is the battery mean squared residual in units of the"
             r" $9\times10^{-4}$ noise floor; $\Delta$BIC margin is over the"
             r" runner-up family. Starred rows are the representatives shown"
             r" in Fig.~S11.}",
             r"\label{tab:fenep_conditions}",
             r"\begin{tabular}{llrrrrrrrrrrl}", r"\toprule",
             r"condition & run & $\eta_p$ & \% & $\lambda$ & \% & $\nu_s$ & \%"
             r" & $L^2$ & \% & loss & resid & winner (margin) \\",
             r"\midrule"]
    prev = None
    for r in rows:
        if r["condition"] != prev and prev is not None:
            lines.append(r"\midrule")
        cond = (r["condition"].replace("+", r"$+$").replace("_", r"\_")
                if r["condition"] != prev else "")
        prev = r["condition"]
        name = r["label"].replace("_", r"\_") + (r"$^\star$"
                                                 if r["representative"] else "")
        cells = []
        for k, _, _ in PARAMS:
            cells += [_fmt(r["values"][k], "{:.3f}"),
                      f"{r['errors'][k]:+.1f}"]
        win = r["winner"].replace("FENEPConformation", "FENE-P")
        lines.append(" & ".join(
            [cond, name] + cells +
            [_fmt(r["loss_final"], "{:.2f}"), f"{r['resid_x_floor']:.1f}",
             f"{win} ({r['margin']:,.0f})"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_table_S2(save: bool = True) -> str:
    rows = table_S2_rows()
    ctrl = rows[-1]
    body = [
        "# Table S2 — FENE-P recovery across conditions",
        "",
        "Truth: `eta_p 2.24`, `lam 0.7`, `nu_s 0.8`, `L^2 12` (the battery "
        "stores `extension_length` = L, squared here). `resid / floor` is the "
        "battery MSE divided by the 9e-4 noise floor. `\\*` marks the "
        "representative network of each condition in figure SN5, chosen by "
        "lowest mean absolute relative error over the four parameters.",
        "",
        _markdown(rows),
        "",
        f"Clean-analytic control: all four parameters within "
        f"{max(abs(v) for v in ctrl['errors'].values()):.2f}% of truth at "
        f"{ctrl['resid_x_floor']:.2f}x the noise floor, so the battery "
        f"recovers the generating model when the data are not filtered "
        f"through a network.",
        "",
        "```latex",
        _latex(rows),
        "```",
        "",
    ]
    text = "\n".join(body)
    if save:
        os.makedirs(TABLE_DIR, exist_ok=True)
        path = TABLE_DIR / "tableS2.md"
        with open(path, "w") as fh:
            fh.write(text)
        print(f"[tableS2] {path}", flush=True)
    return text


# --------------------------------------------------------------------------
# EVP robustness: the eight-arm matrix, as a table instead of a scatter
# --------------------------------------------------------------------------

#: Where the image goes.  SN8 is no longer a figure; this is a panel for the
#: EVP supplement figure and keeps its old folder so nothing else moves.
EVP_FIG = "SN8"
EVP_PANEL = "SN8b_robustness_table"
EVP_TABLE = "tables/tableS_evp_robustness.md"

#: The four scalars: archive key, mathtext for the image, LaTeX symbol, and the
#: plain name the markdown uses.  The solvent viscosity is spelled as in SN6.
EVP_SCALARS = (("Gp", r"$G_p$", "G_p", "Gp"),
               ("lam", r"$\lambda$", r"\lambda", "lam"),
               ("nu_s", r"$\eta_s$", r"\eta_s", "eta_s"),
               ("tau_y", r"$\tau_y$", r"\tau_y", "tau_y"))
HORIZON_TEXT = {"3lam": r"$3\lambda$", "7lam": r"$7\lambda$"}
HORIZON_MD = {"3lam": "3 lambda", "7lam": "7 lambda"}
INIT_TEXT = {"agn": "agnostic (1.0)", "br": "B-R bracket"}


def evp_robustness_rows() -> list[dict]:
    """One row per arm: the three axes, then signed recovery error in %."""
    arms = ld.evp_arm_matrix()
    rows = []
    for name, rec in arms.items():
        drives = ", ".join(f"{d:g}" for d in rec["drives"])
        rows.append({"arm": name, "group": rec["group"],
                     "drives": drives,
                     "drive_label": rf"$G_x = {drives}$",
                     "drive_set": f"{rec['group']}: {drives}",
                     "horizon": rec["horizon"], "init": rec["init"],
                     "loss": rec["loss"], "n_grads": rec["n_grads"],
                     "init_vals": rec["init_vals"],
                     "init_method": rec["init_method"],
                     "init_note": rec["init_note"],
                     "err": {k: 100.0 * rec["rel"][k]
                             for k, _, _, _ in EVP_SCALARS}})
    return rows


def evp_table_rows() -> list[dict]:
    """The published table: agnostic arms only, no initialization column."""
    return [r for r in evp_robustness_rows() if r["init"] == "agn"]


def evp_robustness_worst(rows=None) -> tuple[str, str, float]:
    """Arm, scalar and signed value of the worst-magnitude cell, in %."""
    rows = rows or evp_robustness_rows()
    cells = [(r["arm"], k, r["err"][k]) for r in rows for k, _, _, _ in EVP_SCALARS]
    return max(cells, key=lambda c: abs(c[2]))


def plot_evp_robustness_table(ax=None, save=True, dpi=None):
    """Agnostic arms as an N1d-styled table image: drives x horizon."""
    own = ax is None
    if own:
        fig, ax, scale = pn.new_panel(9.0, 3.4, axes_width=8.2)
    else:
        fig, scale = ax.get_figure(), pn.adopt(ax)

    rows = evp_table_rows()
    # Typographic minus in the image only; the markdown and LaTeX stay ASCII.
    body = [[r["drive_label"], HORIZON_TEXT[r["horizon"]]]
            + [f"{r['err'][k]:+.4f}".replace("-", "\u2212")
               for k, _, _, _ in EVP_SCALARS]
            for r in rows]
    columns = (["drives", "horizon"]
               + [f"{lab} (%)" for _, lab, _, _ in EVP_SCALARS])
    pn.table_panel(ax, columns, body, scale,
                   col_widths=[0.30, 0.12] + [0.145] * 4,
                   row_height=2.1, font_k=0.48)
    if own and save:
        write_evp_robustness_notes()
        make_table_S_evp_robustness()
    return pn.finish(fig, ax, EVP_FIG, EVP_PANEL, save, dpi, own)


def _evp_markdown(rows: list[dict]) -> str:
    cols = (["drives", "horizon"]
            + [f"{md} err %" for _, _, _, md in EVP_SCALARS]
            + ["final loss", "gradients"])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = [f"{r['err'][k]:+.4f}" for k, _, _, _ in EVP_SCALARS]
        out.append("| " + " | ".join(
            [f"Gx = {r['drives']}", HORIZON_MD[r["horizon"]]]
            + cells + [f"{r['loss']:.2e}", str(r["n_grads"])]) + " |")
    return "\n".join(out)


def _evp_latex(rows: list[dict]) -> str:
    arm, key, val = evp_robustness_worst(rows)
    sym = dict((k, tex) for k, _, tex, _ in EVP_SCALARS)[key]
    safe_arm = arm.replace("_", r"\_")
    worst = (rf"Worst cell: ${val:+.4f}$\% in ${sym}$, arm {safe_arm}.")
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Robustness of the EVP scalar recovery to the training"
             r" protocol. Four agnostic arms: drive set $\times$ training"
             r" horizon, every scalar started at $1.0$."
             r" Cells are signed recovery error in per cent against"
             r" $(G_p, \lambda, \eta_s, \tau_y) = (3.2, 0.7, 0.8, 1.45)$. "
             + worst + r"}",
             r"\label{tab:evp_robustness}",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"$G_x$ & horizon & $G_p$ & $\lambda$ & $\eta_s$"
             r" & $\tau_y$ \\",
             r"\midrule"]
    for i, r in enumerate(rows):
        if i and rows[i - 1]["group"] != r["group"]:
            lines.append(r"\midrule")
        cells = [f"{r['err'][k]:+.4f}" for k, _, _, _ in EVP_SCALARS]
        lines.append(" & ".join(
            [rf"$G_x = {r['drives']}$", HORIZON_TEXT[r["horizon"]]]
            + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_table_S_evp_robustness(save: bool = True) -> str:
    rows = evp_table_rows()
    arm, key, val = evp_robustness_worst(rows)
    label = dict((k, md) for k, _, _, md in EVP_SCALARS)[key]
    body = [
        "# Table S — EVP recovery is robust to the training protocol",
        "",
        "Truth: `Gp 3.2`, `lam 0.7`, `eta_s 0.8`, `tau_y 1.45`. Four agnostic "
        "arms = {A, B} x {3 lambda, 7 lambda}, every scalar started at 1.0, "
        "with A = {1.8, 2.5, 4} and B = {1.6, 1.8, 2.5} as the training "
        "drives and the horizon as the length of the training window. Cells "
        "are signed recovery error in per cent; `final loss` and `gradients` "
        "are the arm's own optimisation, not a comparison across arms (each "
        "arm fits a different data set). The four B-R arms are in the notes, "
        "not in this table.",
        "",
        _evp_markdown(rows),
        "",
        f"Worst cell among the 16 shown: `{label}` at "
        f"{val:+.4f}% in `{arm}`. No shown arm misses any scalar by more than "
        f"{abs(val):.3f}%, so neither the drive set nor the training horizon "
        f"changes the answer.",
        "",
        "The same table as an image, styled as the N1d parameter table: "
        f"[../{EVP_FIG}/{EVP_PANEL}.jpg](../{EVP_FIG}/{EVP_PANEL}.jpg). "
        f"Provenance and the B-R numbers: "
        f"[../{EVP_FIG}/{EVP_PANEL}_notes.txt](../{EVP_FIG}/"
        f"{EVP_PANEL}_notes.txt).",
        "",
        "```latex",
        _evp_latex(rows),
        "```",
        "",
    ]
    text = "\n".join(body)
    if save:
        os.makedirs(TABLE_DIR, exist_ok=True)
        path = dp.OUT_ROOT / EVP_TABLE
        with open(path, "w") as fh:
            fh.write(text)
        print(f"[tableS_evp_robustness] {path}", flush=True)
    return text


def evp_robustness_notes() -> str:
    import textwrap

    def para(text, indent=""):
        return textwrap.fill(" ".join(text.split()), width=78,
                             initial_indent=indent,
                             subsequent_indent=" " * len(indent))

    all_rows = evp_robustness_rows()
    shown = evp_table_rows()
    arm, key, val = evp_robustness_worst(shown)
    all_arm, all_key, all_val = evp_robustness_worst(all_rows)
    label = dict((k, md) for k, _, _, md in EVP_SCALARS)[key]
    all_label = dict((k, md) for k, _, _, md in EVP_SCALARS)[all_key]
    br = [r for r in all_rows if r["init"] == "br"]
    note = br[0]["init_note"]
    out = [para(
        "EVP robustness table -- provenance.  The published table is the "
        "agnostic half of the eight-arm matrix that was the SN8b scatter: "
        "four rows, {A, B} x {3 lambda, 7 lambda}, every scalar started at "
        "1.0.  The initialization column is gone.  Read from "
        "evp_fix_eval/eval_summary.json (registry key evp_8arm_summary), "
        "field recovery[k].signed_rel, times 100.  Truth (Gp, lam, eta_s, "
        "tau_y) = (3.2, 0.7, 0.8, 1.45).  The solvent viscosity is written "
        "eta_s here, as in SN6; the archives call it nu_s."), ""]
    out += [para(
        f"Worst-magnitude cell among the 16 shown: {label} at {val:+.4f}% in "
        f"{arm}.  Every other shown cell is smaller, so neither the drive set "
        f"nor the training horizon moves any scalar by more than "
        f"{abs(val):.3f}%."), ""]
    out += [para(
        "\"B-R\" is the Buckingham-Reiner bracket initialization.  It is not one "
        "fixed start: the bracket is computed from each arm's own two highest "
        "training drives, so all four B-R arms begin somewhere different.  The "
        "run archive records it per arm under br_init, flagged verbatim: "
        f"\"{note}\".  The values, as initialized (after clipping to the "
        "optimiser bounds):"), ""]
    for r in br:
        v = r["init_vals"]
        out.append(f"  {r['arm']:22s} Gp {v['Gp']:.4f}  lam {v['lam']:.4f}  "
                   f"eta_s {v['nu_s']:.4f}  tau_y {v['tau_y']:.4f}")
    a3 = [r for r in br if r["arm"] == "evp_fix_A_3lam_br"][0]["init_vals"]
    d_tau = 100.0 * (a3["tau_y"] / ld.EVP_TRUTH["tau_y"] - 1.0)
    d_gp = 100.0 * (a3["Gp"] / ld.EVP_TRUTH["Gp"] - 1.0)
    out += ["", para(
        "The A / 3-lambda arm is the one usually quoted, "
        f"({a3['Gp']:.3f}, {a3['lam']:.1f}, {a3['nu_s']:.3f}, "
        f"{a3['tau_y']:.3f}): tau_y starts {d_tau:+.0f}% off truth and Gp "
        f"{d_gp:+.0f}%.  That bias is the point of the arm -- a wrong but "
        "physically motivated start, not a claim about the material.  The "
        "agnostic arms start every scalar at 1.0 (method neutral_ones).  "
        "They are the four rows in the table.  Over all eight arms the "
        f"worst cell is still {all_label} at {all_val:+.4f}% in {all_arm}; "
        "that cell is a B-R arm and is not in the published table."), ""]
    out += [para(
        "The final-loss column is each arm's own objective on its own data, so "
        "it is not comparable across arms: A and B train on different drives "
        "and 3 lambda and 7 lambda integrate different windows.  It is in the "
        "markdown table for traceability only and is not in the image."), ""]
    out += [para(
        "The image is styled as the N1d parameter table (teal header, "
        "alternating row shading) so it can be dropped into the EVP "
        "supplement figure as a panel.  Regenerate with "
        "paper_figs.plot_evp_robustness_table(); the markdown and LaTeX come "
        "from paper_figs.make_table_S_evp_robustness()."), ""]
    return "\n".join(out)


def write_evp_robustness_notes():
    out = dp.out_dir(EVP_FIG) / f"{EVP_PANEL}_notes.txt"
    out.write_text(evp_robustness_notes())
    print(f"[{EVP_PANEL}] {out}", flush=True)
    return out
