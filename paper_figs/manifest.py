"""What each panel is, where it goes, and which archive it reads.

This is the single description of the figure set; ``make_all_figures.py``
renders from it and writes ``final_figures/figures.md`` from it, so the index
cannot drift from the code that produced the images.
"""
from __future__ import annotations

from typing import NamedTuple


class Panel(NamedTuple):
    panel: str          # file stem under final_figures/<figure>/
    call: str           # importable from paper_figs
    what: str           # one line, what the panel shows
    source: str         # the archive(s) it reads, by registry key


class Figure(NamedTuple):
    figure: str         # final_figures/<figure>/
    manuscript: str
    title: str
    panels: tuple[Panel, ...]


FIGURES: tuple[Figure, ...] = (
    Figure("N1", "Fig 5", "Giesekus contraction, transfer and recovery", (
        Panel("N1a", "plot_N1a",
              "contraction $u_x$, truth above and learned s4 below",
              "gie_passc_fields"),
        Panel("N1b", "plot_N1b",
              "tr A along the centreline, ground truth / trained / initial",
              "gie_passc_fields, gie_init_traj"),
        Panel("N1c", "plot_N1c",
              "BIC model selection over five families, Giesekus wins",
              "battery target gie_A_s4"),
        Panel("N1d", "plot_N1d",
              "recovered constitutive parameters vs truth, table. Caption: "
              "TBNN selected by lowest final training loss; all three "
              "schedules in the SI table",
              "battery target gie_A_s4"),
        Panel("N1e", "plot_N1e",
              "cavity transfer: steady centreline $u_x/U_{\\rm lid}$ profiles "
              "above, the $\\max A_{xx}$ transient below, "
              "De = 0.20 / 0.35 / 0.50. Caption material and the verified "
              "numbers: N1e_caption_notes.txt",
              "cavity transfer_prod ladder"),
        Panel("N1e_horizontal", "plot_N1e_horizontal",
              "same two sub-panels side by side, for a wide slot in the "
              "layout; identical curves, encoding and keys",
              "cavity transfer_prod ladder"),
        Panel("N1_full", "plot_N1", "assembled figure", "as above"),
    )),
    Figure("N2", "Fig 6", "EVP yield stress", (
        Panel("N2a", "plot_N2a",
              "training profile $u(y)$ at $G_x$ = 4 (untitled, so the drive "
              "belongs in the caption): ground truth, the trained "
              "TBNN (five seeds, overplotted), and the initial TBNN at its own "
              "amplitude; the dash-dotted rules are the yield surface "
              "$y = \\pm\\tau_y/G_x$",
              "evp_fix_seed_eval, evp_obinit (run G3)"),
        Panel("N2b", "plot_N2b",
              "flow curve $|Q|(G_x)$ with the arrest cliff at $G_c$, marked by "
              "the same dash-dotted rule N2a uses for the yield surface; "
              "filled circles are training conditions, open circles test "
              "conditions. Arrest, ceiling and seed-spread numbers: "
              "N2b_caption_note.txt",
              "evp_seed_eval_summary"),
        Panel("N2_full", "plot_N2", "assembled figure", "as above"),
    )),
    Figure("SN1", "S7", "Contraction fields and errors", (
        Panel("SN1a", "plot_SN1a",
              "$u_y$, truth above the centreline and trained TBNN below, "
              "split as in N1a; the field is antisymmetric, so the halves "
              "carry opposite signs",
              "gie_passc_fields"),
        Panel("SN1b", "plot_SN1b",
              "absolute difference in $x$-velocity, titled above the axes, "
              "ROI band outlined", "gie_passc_fields"),
        Panel("SN1c", "plot_SN1c",
              "absolute difference in $y$-velocity, titled above the axes, "
              "ROI band outlined", "gie_passc_fields"),
        Panel("SN1d", "plot_SN1d",
              "tr A, truth above the centreline and trained TBNN below",
              "gie_passc_fields"),
        Panel("SN1e", "plot_SN1e",
              "absolute difference in tr A, ROI band outlined. "
              "Relative L2 numbers: SN1_metrics.txt",
              "gie_passc_fields"),
        Panel("SN1_full", "plot_SN1", "assembled figure", "as above"),
    )),
    Figure("SN2", "S8", "Training convergence and model extraction", (
        Panel("SN2a", "plot_SN2a", "loss vs iteration, three schedules",
              "gie progress.csv"),
        Panel("SN2b", "plot_SN2b", "$G_p$, $\\lambda$, $\\eta_s$ vs iteration",
              "gie progress.csv"),
        Panel("SN2d", "plot_SN2d",
              "AOS stress response of the trained TBNN, candidate family fits "
              "overlaid",
              "battery target + replay (derive_aos.py)"),
        Panel("SN2_full", "plot_SN2", "assembled figure", "as above"),
    )),
    Figure("SN3", "S9", "Cavity transfer detail", (
        Panel("SN3a", "plot_SN3a",
              "$A_{xx}$ with streamlines, ground truth; the arm is a title "
              "above the axes so it can be cropped for a caption",
              "cavity transfer_prod"),
        Panel("SN3b", "plot_SN3b",
              "$A_{xx}$ with streamlines, TBNN prediction",
              "cavity transfer_prod"),
        Panel("SN3c", "plot_SN3c",
              "absolute difference in $A_{xx}$, titled above the axes; "
              "relative $L_2$ printed on the panel",
              "cavity transfer_prod"),
        Panel("SN3d", "plot_SN3d",
              "centreline profiles, the source computation for N1e(i)",
              "cavity transfer_prod"),
        Panel("SN3e", "plot_SN3e", "steadiness histories",
              "cavity diagnostics.npz"),
        Panel("SN3f", "plot_SN3f",
              "vortex strength, eye position and SPD margin vs De",
              "cavity result.json"),
        Panel("SN3g", "plot_SN3g",
              r"speed $|\mathbf{u}|/U_{\rm lid}$ with streamlines, ground "
              "truth", "cavity transfer_prod"),
        Panel("SN3h", "plot_SN3h",
              r"speed $|\mathbf{u}|/U_{\rm lid}$ with streamlines, TBNN "
              "prediction", "cavity transfer_prod"),
        Panel("SN3i", "plot_SN3i",
              r"absolute difference in velocity magnitude, "
              r"$|\Delta\mathbf{u}|/U_{\rm lid}$, rms printed "
              "on the panel", "cavity transfer_prod + metrics.json"),
        Panel("SN3_full", "plot_SN3", "assembled figure", "as above"),
    )),
    Figure("SN4", "S10", "FENE-P single-rate recovery", (
        Panel("SN4a", "plot_SN4a",
              "contraction $u_x$, truth above and learned below",
              "fene truth/learned traj (run G4)"),
        Panel("SN4b", "plot_SN4b",
              "tr A along the centreline, ground truth / initial / trained, "
              "as N1b", "fene truth / init / learned traj"),
        Panel("SN4c", "plot_SN4c",
              r"$\Delta$BIC over the five families, FENE-P selected, as N1c",
              "battery targets"),
        Panel("SN4d", "plot_SN4d",
              "four-parameter recovery across the five seeds",
              "battery targets + summary.json"),
        Panel("SN4d_table", "plot_SN4d_table",
              "alternate to SN4d: truth against the seed mean $\\pm$ s.d., "
              "styled as N1d; the $\\pm$ is the spread over the five seeds at "
              "one fixed schedule", "battery targets"),
        Panel("SN4e", "plot_SN4e",
              "training curves, five seeds: loss and $\\eta_p$ against "
              "iteration, side by side at N1e_horizontal's proportions",
              "fene progress.csv"),
        Panel("SN4_full", "plot_SN4", "assembled figure", "as above"),
    )),
    Figure("SN5", "S11", "FENE-P condition ladder", (
        Panel("SN5a", "plot_SN5a",
              r"representative $\Delta$BIC per condition, linear axis, groups "
              "labelled by drive as in (b) and (c); the selected family is the "
              "gap at zero in each group. Conditions, margins and the dropped "
              "legacy arm: SN5_notes.txt",
              "battery targets"),
        Panel("SN5b", "plot_SN5b",
              "$L^2$ readback error, all runs, grouped by drive with $n$ per "
              "group; the sole Linear PTT verdict is the open square in the "
              "Linear PTT blue of (a). Group means sit above the axes",
              "run summary.json"),
        Panel("SN5c", "plot_SN5c",
              "mechanism: distribution of truth $\\mathrm{tr}\\,\\mathbf{A}/L^2$ "
              "over the fluid cells of the final frame, one histogram per "
              "condition, all normalised by the truth $L^2 = 12$ so the FENE-P "
              "pole is at 1.0. The single-rate and $+$pressure conditions share "
              "one truth field, so their curves coincide: SN5_notes.txt",
              "run arrays.npz (truth $x_1$ cloud) + summary.json"),
        Panel("SN5_full", "plot_SN5", "assembled figure", "as above"),
    )),
    Figure("SN6", "S12", "FENE-P direct calibration", (
        Panel("SN6a", "plot_SN6a",
              "trajectories of the four fitted scalars ($G_p$, $\\lambda$, "
              "$\\eta_s$, $L^2$) against iteration, 2x2, four initializations, "
              "truth on every cell; per-cell linear scales. The excluded I4 "
              "start, the legend numbering and the fit's log-space "
              "parameterisation: SN6_notes.txt", "fene8_direct progress.csv"),
        Panel("SN6a_lsq_log", "plot_SN6a_lsq_log",
              "alternate to SN6a: the $L^2$ cell on a log axis. Not chosen -- "
              "the trajectories span 8 to 50, under one decade, and the log "
              "ticks put no label at the truth", "fene8_direct progress.csv"),
        Panel("SN6b", "plot_SN6b",
              "loss against iteration, same four initializations, separate "
              "figure from (a) on purpose. No floor line: the four final "
              "losses span five decades, values in SN6_notes.txt",
              "fene8_direct progress.csv"),
        Panel("SN6_full", "plot_SN6",
              "layout preview of the two panels, parameters first", "as above"),
    )),
    Figure("SN7", "S13", "EVP profiles, arrest, training", (
        Panel("SN7a", "plot_SN7a",
              "$u(y)$ at $G_x$ = 1.8, 2.5, 4, 5, ground truth vs the trained "
              "TBNN, with $G_x$ = 5 marked as a test condition. Not a subset "
              "of N2a, which carries $G_x$ = 4 alone plus the initial closure: "
              "SN7_notes.txt", "evp_fix_seed_eval"),
        Panel("SN7b", "plot_SN7b",
              "$|Q|(t)$ to 30$\\lambda$ at $G_x$ = 0.5, 1, 1.3, the sub-yield "
              "ring-down. Ground truth solid, learned as open markers at the "
              "own peaks, notches and one descent midpoint (N1e's encoding); period and final $|Q|$ in "
              "SN7_notes.txt", "evp_fix_seed_eval"),
        Panel("SN7c", "plot_SN7c",
              "broken-axis training loss, one colour per network seed. "
              "Stage 1 (shared, deterministic) descends to "
              "$4.4\\times10^{-12}$. SN7_notes.txt",
              "evp_fix progress.csv"),
        Panel("SN7d", "plot_SN7d",
              "$G_p$, $\\lambda$, $\\eta_s$, $\\tau_y$ as a 4x2: stage-1 "
              "absolute (one curve, linear) | theta-block ratio to truth "
              "(five seeds, shared linear $y$). SN7_notes.txt",
              "evp_fix progress.csv"),
        Panel("SN7_full", "plot_SN7", "assembled figure", "as above"),
    )),
)

TABLES = (("tables/tableS2.md", "make_table_S2", "Table S2",
           "FENE-P recovery across conditions, markdown + LaTeX",
           "battery targets + run summary.json"),
          ("SN8/SN8b_robustness_table.jpg", "plot_evp_robustness_table",
           "Table S (EVP)",
           "four agnostic EVP arms as a table image, styled as N1d: "
           "$G_x$ / horizon against signed recovery error in $G_p$, "
           "$\\lambda$, $\\eta_s$, $\\tau_y$. B-R arms in the notes only. "
           "SN8b_robustness_table_notes.txt",
           "evp_8arm_summary"),
          ("tables/tableS_evp_robustness.md",
           "make_table_S_evp_robustness", "Table S (EVP)",
           "the same four-arm agnostic table as markdown + LaTeX",
           "evp_8arm_summary"))

#: Retired, kept on disk and callable, deliberately out of the index above.
OBSOLETE = (
    ("SN8a", "forward sensitivity calculation, not a learning result",
     "$|\\mathrm{d}\\ln Q/\\mathrm{d}\\ln\\tau_y|$ against drive is a property "
     "of the Saramito truth model; no TBNN enters it, and the link it implies "
     "to our recovery was never tested. One Methods sentence survives, in "
     "SN7_notes.txt: the sensitivity peak moves with the horizon, which is why "
     "all evaluations use one fixed 15$\\lambda$ ladder."),
    ("SN8b", "converted to a table image",
     "the eight-arm robustness matrix is now SN8/SN8b_robustness_table.jpg "
     "plus tables/tableS_evp_robustness.md; the scatter put 32 points and a "
     "three-line tick label where seven columns do the job."),
    ("SN8c", "numerical-failure diagnosis, superseded by caption text",
     "the per-seed well-posedness ceiling is now caption text for the main EVP "
     "figure's shaded region, in SN7_notes.txt."),
    ("SN8_full", "assembled figure of the three panels above", "nothing left "
     "to assemble."),
    ("SN7 12-drive ladder grid", "half the rungs were round-off residual",
     "callable as plot_SN7_ladder_grid; render in final_figures/SN7/retired/."),
    ("SN7 plug half-width", "second view of N2b's arrest cliff and ceiling",
     "callable as plot_SN7_plug_halfwidth; render in "
     "final_figures/SN7/retired/."),
)
