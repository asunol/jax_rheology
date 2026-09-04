#!/usr/bin/env python
"""Compare archived Giesekus-truth and frozen July-10 s4 cavity forwards.

Reads the De ladder under ``work/cavity_transfer``, writes the N1e
transfer figure plus ladder / history / profile panels, and dumps
``transfer_metrics.json``.  Nothing here retrains: the s4 checkpoint is
loaded only to rebuild polymer stress from the stored conformation.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repo_paths import bootstrap, REPO_ROOT
bootstrap()
from jax_rheology.models import tbnn_memory as tb
from tbnn_diff_rheo_adapter import load_tbnn_checkpoint

ROOT = REPO_ROOT
TRUTH_ROOT = ROOT / "work/cavity_transfer/de_ladder"
LEARNED_ROOT = ROOT / "work/cavity_transfer/learned_de_ladder"
OUT = ROOT / "work/cavity_transfer/comparison"
CHECKPOINT = ROOT / "gie_prod_rerun/gie_A_s4/theta_checkpoint.npz"
DE_VALUES = (0.20, 0.35, 0.50)
COLORS = ("#1b9e77", "#d95f02", "#7570b3")
IDENTICAL_KEYS = (
    "de_label", "cells", "dt", "T", "total_steps", "inner_steps",
    "outer_steps", "U_lid", "ramp_time", "density", "Re", "L", "lid_type",
    "convection", "devss_viscosity", "solver_type", "solver_tol",
    "solver_maxiter", "record_diagnostics", "eta0_truth", "float64",
)


def _load_run(root: Path, De: float):
    directory = root / f"De{De:.2f}"
    result = json.loads((directory / "result.json").read_text())
    with np.load(directory / "diagnostics.npz", allow_pickle=False) as archive:
        diagnostics = {key: np.asarray(archive[key]) for key in archive.files}
    return result, diagnostics


def _relative_l2(model, truth):
    denominator = float(np.linalg.norm(truth))
    if denominator <= 1e-14 * np.sqrt(np.size(truth)):
        return None
    return float(np.linalg.norm(model - truth) / denominator)


def _learned_stress(data, checkpoint):
    A = tuple(
        jnp.asarray(data[f"final_A_{key}"], dtype=jnp.float64)
        for key in ("xx", "xy", "yy", "zz")
    )
    with np.load(CHECKPOINT, allow_pickle=False) as archive:
        bound_c = float(archive["ckpt_bound_c"])
    K, _, _ = tb.tbnn_K_and_frozen(
        *A, checkpoint["theta"], bound_c,
        anchored=True, mobility="softplus", yield_mode="off", kappa=1.0,
    )
    return tuple(np.asarray(checkpoint["Gp"] * component) for component in K)


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _collect():
    checkpoint = load_tbnn_checkpoint(str(CHECKPOINT))
    rows = []
    metrics = {}
    for De in DE_VALUES:
        truth_result, truth = _load_run(TRUTH_ROOT, De)
        learned_result, learned = _load_run(LEARNED_ROOT, De)
        config_identity = {
            key: truth_result["config"][key] == learned_result["config"][key]
            for key in IDENTICAL_KEYS
        }
        if not all(config_identity.values()):
            failed = [key for key, okay in config_identity.items() if not okay]
            raise RuntimeError(f"non-identical truth/TBNN config at De={De}: {failed}")
        if learned_result["classification"] != "STEADY":
            raise RuntimeError(
                f"learned De={De:.2f} is {learned_result['classification']}, not STEADY"
            )
        truth_tau = (
            3.2 * (truth["final_A_xx"] - 1.0),
            3.2 * truth["final_A_xy"],
            3.2 * (truth["final_A_yy"] - 1.0),
            3.2 * (truth["final_A_zz"] - 1.0),
        )
        learned_tau = _learned_stress(learned, checkpoint)
        U = truth_result["config"]["U_lid"]
        du = learned["final_u"] - truth["final_u"]
        dv = learned["final_v"] - truth["final_v"]
        velocity_rms = float(np.sqrt(np.mean(du**2 + dv**2)))
        component_metrics = {}
        for key in ("u", "v", "A_xx", "A_xy", "A_yy", "A_zz"):
            component_metrics[f"relative_L2_{key}"] = _relative_l2(
                learned[f"final_{key}"], truth[f"final_{key}"]
            )
        for label, learned_field, truth_field in zip(
            ("tau_xx", "tau_xy", "tau_yy", "tau_zz"), learned_tau, truth_tau
        ):
            component_metrics[f"relative_L2_{label}"] = _relative_l2(
                learned_field, truth_field
            )
            component_metrics[f"absolute_RMS_error_{label}"] = float(
                np.sqrt(np.mean((learned_field - truth_field) ** 2))
            )
        record = {
            "config_identity": config_identity,
            "config_identity_pass": True,
            "truth_classification": truth_result["classification"],
            "learned_classification": learned_result["classification"],
            "velocity_rms": velocity_rms,
            "velocity_rms_over_U_lid": velocity_rms / U,
            "truth_psi_min_over_UL": truth_result["psi_min_over_U_lid_L"],
            "learned_psi_min_over_UL": learned_result["psi_min_over_U_lid_L"],
            "psi_min_over_UL_error": (
                learned_result["psi_min_over_U_lid_L"]
                - truth_result["psi_min_over_U_lid_L"]
            ),
            "truth_eye": truth_result["eye"],
            "learned_eye": learned_result["eye"],
            "eye_distance": float(
                np.linalg.norm(
                    np.asarray(learned_result["eye"])
                    - np.asarray(truth_result["eye"])
                )
            ),
            "truth_max_Axx": truth_result["max_Axx_final"],
            "learned_max_Axx": learned_result["max_Axx_final"],
            "truth_min_eigenvalue": truth_result["min_eigenvalue_over_trajectory"],
            "learned_min_eigenvalue": learned_result["min_eigenvalue_over_trajectory"],
            "truth_nan": truth_result["nan"],
            "learned_nan": learned_result["nan"],
            **component_metrics,
        }
        metrics[f"De{De:.2f}"] = record
        rows.append((De, truth_result, truth, learned_result, learned, record))
    return rows, metrics


def _fig_n1e(row):
    De, truth_result, truth, learned_result, learned, record = row
    truth_axx = truth["final_A_xx"]
    learned_axx = learned["final_A_xx"]
    vmax = max(float(np.max(truth_axx)), float(np.max(learned_axx)))
    levels = np.linspace(min(float(np.min(truth_axx)), float(np.min(learned_axx))), vmax, 18)
    n = truth_axx.shape[0]
    x = (np.arange(n) + 0.5) / n
    y = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, y, indexing="ij")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), sharex=True, sharey=True)
    for ax, data, velocity, title in (
        (axes[0], truth_axx, (truth["final_u"], truth["final_v"]), "Giesekus truth"),
        (axes[1], learned_axx, (learned["final_u"], learned["final_v"]), "Frozen TBNN s4"),
    ):
        contour = ax.contourf(X, Y, data, levels=levels, cmap="viridis", extend="max")
        ax.streamplot(
            x, y, velocity[0].T, velocity[1].T,
            color="white", density=0.8, linewidth=0.45, arrowsize=0.55,
        )
        ax.set_title(title)
        ax.set_xlabel("$x/L$")
        ax.set_aspect("equal")
    axes[0].set_ylabel("$y/L$")
    fig.colorbar(contour, ax=axes.tolist(), fraction=0.025, pad=0.025, label=r"$A_{xx}$")
    fig.suptitle(
        f"Figure N1(e): unseen cavity transfer at De={De:.2f}",
        fontsize=16, weight="bold",
    )
    fig.text(
        0.5, 0.01,
        f"velocity RMS/U={record['velocity_rms_over_U_lid']:.3%} · "
        f"$\\psi_{{min}}/(UL)$ truth={truth_result['psi_min_over_U_lid_L']:.5f}, "
        f"TBNN={learned_result['psi_min_over_U_lid_L']:.5f} · "
        f"eye truth=({truth_result['eye'][0]:.3f},{truth_result['eye'][1]:.3f}), "
        f"TBNN=({learned_result['eye'][0]:.3f},{learned_result['eye'][1]:.3f})",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.13, top=0.84, wspace=0.14)
    return fig


def _fig_ladder_summary(rows):
    De = np.asarray([row[0] for row in rows])
    truth_psi = np.asarray([row[1]["psi_min_over_U_lid_L"] for row in rows])
    learned_psi = np.asarray([row[3]["psi_min_over_U_lid_L"] for row in rows])
    truth_axx = np.asarray([row[1]["max_Axx_final"] for row in rows])
    learned_axx = np.asarray([row[3]["max_Axx_final"] for row in rows])
    truth_eig = np.asarray([row[1]["min_eigenvalue_over_trajectory"] for row in rows])
    learned_eig = np.asarray([row[3]["min_eigenvalue_over_trajectory"] for row in rows])
    velocity_error = np.asarray([row[5]["velocity_rms_over_U_lid"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 7.2))
    pairs = (
        (truth_psi, learned_psi, r"$\min\psi/(U_{\rm lid}L)$", "Vortex strength"),
        (truth_axx, learned_axx, r"$\max A_{xx}$", "Peak conformation"),
        (truth_eig, learned_eig, r"$\min_t\lambda(\mathbf{A})$", "SPD margin"),
    )
    for ax, (truth_values, learned_values, ylabel, title) in zip(axes.flat[:3], pairs):
        ax.plot(De, truth_values, "-o", color="black", label="Giesekus truth")
        ax.plot(De, learned_values, "--s", color="#d1495b", label="frozen TBNN s4")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Deborah number De")
        ax.set_xticks(De)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)
    axes[1, 1].plot(De, 100.0 * velocity_error, "-o", color="#d1495b")
    axes[1, 1].set_title("Velocity transfer error")
    axes[1, 1].set_ylabel("RMS velocity error / $U_{lid}$ (%)")
    axes[1, 1].set_xlabel("Deborah number De")
    axes[1, 1].set_xticks(De)
    axes[1, 1].grid(True, alpha=0.25)
    fig.suptitle(
        "Cavity ladder: Giesekus truth versus frozen July-10 s4",
        fontsize=16, weight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.text(
        0.5, 0.012,
        "Identical grid, dt, U_lid, density, ramp, solver, horizon and DEVSS=0 at each De.",
        ha="center", fontsize=9, color="0.35",
    )
    return fig


def _fig_histories(rows):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True)
    for color, (De, truth_result, truth, _, learned, _) in zip(COLORS, rows):
        progress = np.arange(1, len(truth["ke_traj"]) + 1) / len(truth["ke_traj"])
        U = truth_result["config"]["U_lid"]
        for data, linestyle, label_suffix in (
            (truth, "-", "truth"), (learned, "--", "TBNN"),
        ):
            label = f"De={De:.2f} {label_suffix}"
            axes[0, 0].plot(progress, data["ke_traj"] / U**2, color=color, ls=linestyle, label=label)
            axes[0, 1].plot(progress, data["max_Axx_traj"], color=color, ls=linestyle)
            axes[1, 0].plot(progress, data["min_lam_traj"], color=color, ls=linestyle)
            axes[1, 1].plot(progress, data["psi_min_traj"] / U, color=color, ls=linestyle)
    labels = (
        ("Kinetic energy", r"$KE/U_{\rm lid}^2$"),
        ("Peak conformation", r"$\max A_{xx}$"),
        ("SPD margin", r"$\min\lambda(\mathbf{A})$"),
        ("Vortex strength", r"$\min\psi/(U_{\rm lid}L)$"),
    )
    for ax, (title, ylabel) in zip(axes.flat, labels):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("completed simulation fraction $t/T$")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        "Stress-inclusive histories: truth (solid) versus TBNN (dashed)",
        fontsize=16, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _fig_profiles(rows):
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 10.0))
    for row_index, (De, _, truth, _, learned, record) in enumerate(rows):
        n = truth["final_u"].shape[0]
        coordinate = (np.arange(n) + 0.5) / n
        midpoint = n // 2
        axes[row_index, 0].plot(
            truth["final_u"][midpoint, :], coordinate, color="black", label="truth"
        )
        axes[row_index, 0].plot(
            learned["final_u"][midpoint, :], coordinate, "--", color="#d1495b",
            label="TBNN s4",
        )
        axes[row_index, 1].plot(
            coordinate, truth["final_v"][:, midpoint], color="black", label="truth"
        )
        axes[row_index, 1].plot(
            coordinate, learned["final_v"][:, midpoint], "--", color="#d1495b",
            label="TBNN s4",
        )
        axes[row_index, 0].set_ylabel(f"De={De:.2f}\n$y/L$")
        axes[row_index, 0].set_xlabel(r"$u(x=0.5,y)$")
        axes[row_index, 1].set_xlabel("$x/L$")
        axes[row_index, 1].set_ylabel(r"$v(x,y=0.5)$")
        axes[row_index, 0].text(
            0.03, 0.05, f"RMS/U={record['velocity_rms_over_U_lid']:.2%}",
            transform=axes[row_index, 0].transAxes, fontsize=8,
        )
        for ax in axes[row_index]:
            ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    fig.suptitle("Cavity centerline velocity profiles", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, metrics = _collect()
    _save(_fig_n1e(rows[-1]), "figN1e_cavity_transfer")
    _save(_fig_ladder_summary(rows), "cavity_ladder_truth_vs_s4")
    _save(_fig_histories(rows), "cavity_histories_truth_vs_s4")
    _save(_fig_profiles(rows), "cavity_profiles_truth_vs_s4")
    payload = {
        "closure": "July-10 Giesekus s4",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "model": "tbnn_potential_logconf_bk_v2",
        "frozen_forward_only": True,
        "metrics": metrics,
    }
    (OUT / "transfer_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
