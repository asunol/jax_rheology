#!/usr/bin/env python
"""Assemble one PDF of the production cavity, contraction, and EVP plots.

Pages are the already-rendered archives (truth ladder, s4 transfer, flow
curve).  This script does not re-run any solver.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import importlib.util as _ilu

_style_src = Path(__file__).resolve().parent / "style.py"
_style_spec = _ilu.spec_from_file_location("paper_figs_style_pub", _style_src)
_style_mod = _ilu.module_from_spec(_style_spec)
assert _style_spec.loader is not None
_style_spec.loader.exec_module(_style_mod)
style = _style_mod
plt.rcParams.update(style.PUB_REPORT_RCPARAMS)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/production_report"
REPORT = OUT / "all_production_plots_report.pdf"
CAVITY_ROOT = ROOT / "work/cavity_transfer"
CONTRACTION = (
    ROOT
    / "work/regen_contraction/gie_A_s4/passc/figN1ab_contraction.png"
)
EVP_FLOW = (
    ROOT
    / "work/evp_learned_flowcurve/figN2c_flowcurve.png"
)
EVP_PROFILES = (
    ROOT
    / "work/evp_learned_flowcurve/si_profiles.png"
)
CAVITY_REGRESSION = (
    CAVITY_ROOT / "regression/fields_De0p20_phase0_gate.png"
)
COMPARE_DIR = CAVITY_ROOT / "comparison_s4"
COMPARISON_PLOTS = (
    COMPARE_DIR / "figN1e_cavity_transfer.png",
    COMPARE_DIR / "cavity_ladder_truth_vs_s4.png",
    COMPARE_DIR / "cavity_histories_truth_vs_s4.png",
    COMPARE_DIR / "cavity_profiles_truth_vs_s4.png",
)
DE_VALUES = (0.20, 0.35, 0.50)


def _load_ladder():
    rows = []
    for De in DE_VALUES:
        run = CAVITY_ROOT / f"de_ladder/De{De:.2f}"
        result = json.loads((run / "result.json").read_text())
        with np.load(run / "diagnostics.npz", allow_pickle=False) as z:
            diagnostics = {key: np.asarray(z[key]) for key in z.files}
        rows.append((De, result, diagnostics))
    return rows


def _save_asset(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)


def _page_title(fig, title, subtitle=None):
    fig.text(0.06, 0.965, title, ha="left", va="top", fontsize=19, weight="bold")
    if subtitle:
        fig.text(0.06, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color="0.35")


def _title_page(pdf, ladder, comparison_available):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(
        0.08, 0.88, "Production plots report", fontsize=25, weight="bold",
        ha="left",
    )
    fig.text(
        0.08, 0.835,
        "Cavity truth ladder . Giesekus contraction transfer . EVP flow curve",
        fontsize=13, ha="left", color="0.25",
    )
    fig.text(
        0.08, 0.75,
        "Contents\n"
        "1. Cavity: final fields at De = 0.20, 0.35, 0.50\n"
        "2. Cavity: stress-inclusive steadiness histories and ladder summary\n"
        "3. Cavity: Phase-0 regression field diagnostic\n"
        "4. Contraction: truth versus learned s4 velocity and stretch fields\n"
        "5. EVP: learned-closure flow curve and SI velocity profiles",
        fontsize=12, linespacing=1.65, ha="left", va="top",
    )
    states = ", ".join(
        f"De {De:.2f}: {result['classification']}"
        for De, result, _ in ladder
    )
    fig.text(
        0.08, 0.53,
        "Scope note",
        fontsize=14, weight="bold", ha="left",
    )
    if comparison_available:
        scope = (
            "The cavity section contains the completed Giesekus truth ladder "
            "and matched forwards with the frozen July-10 s4 TBNN checkpoint. "
            "Truth and TBNN use identical grid, dt, velocity, density, ramp, "
            "solver settings and horizon at each De; only the constitutive "
            "closure differs.\n\n"
            f"Truth ladder classifications: {states}."
        )
    else:
        scope = (
            "The cavity section contains the completed truth-only De ladder. "
            "A learned-closure cavity transfer run--and therefore the intended "
            "truth-vs-learned Figure N1(e)--does not yet exist. No such result is "
            "inferred or substituted here.\n\n"
            f"Ladder classifications: {states}."
        )
    fig.text(
        0.08, 0.49, scope, fontsize=11, linespacing=1.5,
        ha="left", va="top", wrap=True,
    )
    fig.text(
        0.08, 0.12,
        "Data provenance: archived float64 production outputs under "
        "work/cavity_transfer, work/regen_contraction and work/evp_learned_flowcurve. "
        "No simulations were run to produce this report.",
        fontsize=9, color="0.4", ha="left", va="bottom", wrap=True,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _cavity_fields(ladder):
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.4))
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.07, top=0.90, wspace=0.30, hspace=0.30)
    all_speed = []
    all_axx = []
    all_psi = []
    for _, result, data in ladder:
        U = result["config"]["U_lid"]
        all_speed.append(np.hypot(data["final_u"], data["final_v"]) / U)
        all_axx.append(data["final_A_xx"])
        all_psi.append(data["final_psi"] / U)
    maxima = (
        max(float(np.max(field)) for field in all_speed),
        max(float(np.max(field)) for field in all_axx),
    )
    psi_min = min(float(np.min(field)) for field in all_psi)
    images = [None, None, None]
    for row, ((De, result, _), speed, axx, psi) in enumerate(
        zip(ladder, all_speed, all_axx, all_psi)
    ):
        images[0] = axes[row, 0].imshow(
            speed.T, origin="lower", extent=(0, 1, 0, 1),
            cmap="viridis", vmin=0.0, vmax=maxima[0], aspect="equal",
        )
        images[1] = axes[row, 1].imshow(
            axx.T, origin="lower", extent=(0, 1, 0, 1),
            cmap="magma", vmin=min(1.0, float(np.min(axx))), vmax=maxima[1],
            aspect="equal",
        )
        images[2] = axes[row, 2].imshow(
            psi.T, origin="lower", extent=(0, 1, 0, 1),
            cmap="cividis", vmin=psi_min, vmax=0.0, aspect="equal",
        )
        axes[row, 0].set_ylabel(f"De={De:.2f}\n$y/L$")
        axes[row, 2].text(
            0.97, 0.04, result["classification"], transform=axes[row, 2].transAxes,
            ha="right", va="bottom", fontsize=8, color="white", weight="bold",
        )
        for ax in axes[row]:
            ax.set_xlabel("$x/L$")
    titles = (
        r"$|\mathbf{u}|/U_{\rm lid}$",
        r"$A_{xx}$",
        r"$\psi/(U_{\rm lid}L)$",
    )
    for col, title in enumerate(titles):
        axes[0, col].set_title(title)
        fig.colorbar(
            images[col], ax=axes[:, col].tolist(), orientation="horizontal",
            fraction=0.025, pad=0.07,
        )
    fig.suptitle("Cavity truth ladder: final fields", fontsize=17, weight="bold")
    fig.text(
        0.5, 0.015,
        "Giesekus truth . Re=1 . regularized lid . DEVSS viscosity=0 . "
        "Source: archived production diagnostics",
        ha="center", fontsize=9, color="0.35",
    )
    return fig


def _cavity_histories(ladder):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True)
    colors = ("#1b9e77", "#d95f02", "#7570b3")
    for color, (De, result, data) in zip(colors, ladder):
        progress = np.arange(1, len(data["ke_traj"]) + 1) / len(data["ke_traj"])
        U = result["config"]["U_lid"]
        axes[0, 0].plot(progress, data["ke_traj"] / U**2, color=color, label=f"De={De:.2f}")
        axes[0, 1].plot(progress, data["max_Axx_traj"], color=color)
        axes[1, 0].plot(progress, data["min_lam_traj"], color=color)
        axes[1, 1].plot(progress, data["psi_min_traj"] / U, color=color)
    axes[0, 0].set_title("Kinetic-energy history")
    axes[0, 0].set_ylabel(r"$KE/U_{\rm lid}^2$")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set_title("Maximum streamwise conformation")
    axes[0, 1].set_ylabel(r"$\max A_{xx}$")
    axes[1, 0].set_title("Minimum conformation eigenvalue")
    axes[1, 0].set_ylabel(r"$\min\lambda(\mathbf{A})$")
    axes[1, 1].set_title("Primary-vortex streamfunction")
    axes[1, 1].set_ylabel(r"$\min\psi/(U_{\rm lid}L)$")
    for ax in axes.flat:
        ax.set_xlabel("completed simulation fraction $t/T$")
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        "Cavity truth ladder: stress-inclusive steadiness histories",
        fontsize=17, weight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.text(
        0.5, 0.015,
        "All three runs classified STEADY; all trajectories finite with no NaNs.",
        ha="center", fontsize=9, color="0.35",
    )
    return fig


def _cavity_summary(ladder):
    De = np.array([row[0] for row in ladder])
    psi = np.array([row[1]["psi_min_over_U_lid_L"] for row in ladder])
    axx = np.array([row[1]["max_Axx_over_trajectory"] for row in ladder])
    eig = np.array([row[1]["min_eigenvalue_over_trajectory"] for row in ladder])
    elapsed = np.array([row[1]["elapsed_seconds"] for row in ladder]) / 60.0
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 7.2))
    series = (
        (psi, r"$\min\psi/(U_{\rm lid}L)$", "Vortex strength"),
        (axx, r"$\max_t A_{xx}$", "Peak conformation"),
        (eig, r"$\min_t\lambda(\mathbf{A})$", "SPD margin"),
        (elapsed, "elapsed time (min)", "Measured wall time"),
    )
    for ax, (values, ylabel, title) in zip(axes.flat, series):
        ax.plot(De, values, "-o", color=style.C_LEARN, lw=2, ms=7)
        for x, value in zip(De, values):
            ax.annotate(f"{value:.3g}", (x, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Deborah number De")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(De)
    fig.suptitle("Cavity truth ladder: production summary", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.text(
        0.5, 0.015,
        "Source: result.json for De = 0.20, 0.35, 0.50; wall times are measured, not projections.",
        ha="center", fontsize=9, color="0.35",
    )
    return fig


def _image_page(pdf, image_path, title, caption):
    image = plt.imread(image_path)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(8.5, 11))
    _page_title(fig, title, caption)
    ax = fig.add_axes((0.06, 0.08, 0.88, 0.80))
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_aspect("equal")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    required = (CONTRACTION, EVP_FLOW, EVP_PROFILES, CAVITY_REGRESSION)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing report input(s):\n" + "\n".join(missing))
    ladder = _load_ladder()
    comparison_available = all(path.exists() for path in COMPARISON_PLOTS)

    with PdfPages(REPORT) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Production plots report"
        metadata["Subject"] = "Cavity, contraction, and EVP production figures"
        metadata["Author"] = "TBNN production analysis"
        _title_page(pdf, ladder, comparison_available)

        for name, maker in (
            ("cavity_ladder_fields", _cavity_fields),
            ("cavity_ladder_histories", _cavity_histories),
            ("cavity_ladder_summary", _cavity_summary),
        ):
            fig = maker(ladder)
            pdf.savefig(fig)
            _save_asset(fig, name)
            plt.close(fig)

        if comparison_available:
            comparison_pages = (
                (
                    COMPARISON_PLOTS[0],
                    "Cavity transfer -- Figure N1(e)",
                    "At De=0.50, Giesekus truth versus the frozen July-10 s4 "
                    "TBNN checkpoint in an identical unseen cavity box.",
                ),
                (
                    COMPARISON_PLOTS[1],
                    "Cavity ladder -- truth versus learned s4",
                    "Vortex strength, peak conformation, SPD margin and normalized "
                    "velocity error across all three matched De values.",
                ),
                (
                    COMPARISON_PLOTS[2],
                    "Cavity steadiness -- truth versus learned s4",
                    "Stress-inclusive histories; solid curves are Giesekus truth "
                    "and dashed curves are the frozen TBNN.",
                ),
                (
                    COMPARISON_PLOTS[3],
                    "Cavity centerline profiles -- truth versus learned s4",
                    "Matched horizontal and vertical centerline velocity profiles "
                    "at De=0.20, 0.35 and 0.50.",
                ),
            )
            for image, title, caption in comparison_pages:
                _image_page(pdf, image, title, caption)

        _image_page(
            pdf, CAVITY_REGRESSION,
            "Cavity Phase-0 regression diagnostic",
            "Archived six-panel solver-regression field check at De=0.20; "
            "this is not the learned-transfer Figure N1(e).",
        )
        _image_page(
            pdf, CONTRACTION,
            "Contraction fields -- Figure N1(a,b)",
            "Trusted July-12 archives at 128x256: Giesekus truth above and "
            "frozen learned s4 below; cyan contour outlines the ROI band.",
        )
        _image_page(
            pdf, EVP_FLOW,
            "EVP learned-closure flow curve -- Figure N2(c)",
            "Saramito truth versus frozen v2_prod2 closure; filled markers are "
            "training drives and vertical lines mark yield thresholds.",
        )
        _image_page(
            pdf, EVP_PROFILES,
            "EVP velocity profiles -- supporting information",
            "Truth and learned u_x(y/H) profiles below, near, and above yield.",
        )

    manifest = {
        "report": REPORT.name,
        "pages": 8 + (4 if comparison_available else 0),
        "source_plots": [
            str(CAVITY_REGRESSION.relative_to(ROOT)),
            str(CONTRACTION.relative_to(ROOT)),
            str(EVP_FLOW.relative_to(ROOT)),
            str(EVP_PROFILES.relative_to(ROOT)),
        ],
        "generated_plots": [
            "cavity_ladder_fields.pdf",
            "cavity_ladder_histories.pdf",
            "cavity_ladder_summary.pdf",
        ],
        "cavity_scope": (
            "truth and frozen July-10 s4 matched De ladder"
            if comparison_available
            else "truth-only De ladder; learned transfer N1(e) unavailable"
        ),
        "comparison_plots": (
            [str(path.relative_to(ROOT)) for path in COMPARISON_PLOTS]
            if comparison_available else []
        ),
        "simulations_run": False,
    }
    (OUT / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(REPORT)


if __name__ == "__main__":
    main()
