#!/usr/bin/env python
"""Forward-only Q(g_x) sweep for Saramito truth and a frozen learned closure.

The default ``v2_prod2`` checkpoint is a superseded run, kept for comparison
only; it must not be quoted as a paper result. The published
elastoviscoplastic results come from ``tbnn_evp_data/evp_fix/``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evp_baseline_diag as phase0
import fwd_yield_sweep as fys
from jax_rheology.models import registry as cr
from jax_rheology.models import tbnn_memory as tb
from visco_opt_tbnn_run import theta_from_named_arrays


from repo_paths import bootstrap, FROZEN_MEM, REPO_ROOT
bootstrap()
ROOT = FROZEN_MEM
CHECKPOINT = ROOT / "tbnn_evp_data/v2_yield/v2_prod2/theta_checkpoint.npz"
CONFIG = CHECKPOINT.with_name("config.json")
GX_ALL = np.asarray(fys.SWEEP_GX_ALL, dtype=np.float64)
GX_BY_JOB = {"A": fys.JOB_A_GX, "B": fys.JOB_B_GX}
PROFILE_GX = (1.0, 1.45, 1.8, 4.0)
TRUTH_TAU_Y = fys.TRUTH_TAU_Y


def _tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_learned():
    z = np.load(CHECKPOINT, allow_pickle=False)
    heads = [str(head) for head in z["ckpt_heads"]]
    nlayers = {head: int(n) for head, n in zip(heads, z["ckpt_nlayers"])}
    theta = theta_from_named_arrays(z, heads, nlayers)
    values = {
        "Gp": float(z["ckpt_Gp_fit"]),
        "lam": float(z["ckpt_lam_fit"]),
        "nu_s": float(z["ckpt_nu_s"]),
        "tau_y": float(z["ckpt_tau_y_fit"]),
        "bound_c": float(z["ckpt_bound_c"]),
        "kappa": float(z["ckpt_kappa_final"]),
    }
    params = {
        "Gp": jnp.asarray(values["Gp"], dtype=jnp.float64),
        "lam": jnp.asarray(values["lam"], dtype=jnp.float64),
        "tau_y": jnp.asarray(values["tau_y"], dtype=jnp.float64),
        "theta": theta,
        "tbnn_bound_c": values["bound_c"],
        "tbnn_kappa": values["kappa"],
    }
    return params, values


def _central_halfwidth(unyielded: np.ndarray, dy: float) -> float:
    center = len(unyielded) // 2
    if not bool(unyielded[center]):
        return 0.0
    lo = hi = center
    while lo > 0 and unyielded[lo - 1]:
        lo -= 1
    while hi + 1 < len(unyielded) and unyielded[hi + 1]:
        hi += 1
    return 0.5 * (hi - lo + 1) * dy


def _measure_stress_yield(out, cfg, Gp: float, tau_y: float):
    fields = [
        np.asarray(out[key][-1]).mean(axis=0)
        for key in ("A_xx_traj", "A_xy_traj", "A_yy_traj", "A_zz_traj")
    ]
    tau_d = np.asarray(tb.saramito_tau_d_norm(
        *(jnp.asarray(field) for field in fields), Gp
    ))
    unyielded = tau_d <= tau_y
    return {
        "plug_halfwidth": _central_halfwidth(
            unyielded, float(cfg["Ly"]) / int(cfg["Ny"])
        ),
        "yielded_fraction": float((~unyielded).mean()),
        "tau_d_profile": tau_d,
    }


def _run_one(g_x: float, closure: str, out_dir: Path, learned):
    # Locked production config is authoritative: Ny=128, Nx=32, dt=2.5e-3,
    # inner=10, solver_tol=1e-8. The sweep helper only supplies chunking and
    # the fixed 15-lambda convergence horizon.
    cfg = phase0.locked_cfg()
    cfg["g_x"] = float(g_x)
    if closure == "truth":
        model = cr.get_model("saramito_logconf_bk_v2")
        params = {
            "Gp": jnp.asarray(fys.TRUTH_GP, dtype=jnp.float64),
            "lam": jnp.asarray(fys.TRUTH_LAM, dtype=jnp.float64),
            "tau_y": jnp.asarray(fys.TRUTH_TAU_Y, dtype=jnp.float64),
        }
        Gp, tau_y, nu_s = fys.TRUTH_GP, fys.TRUTH_TAU_Y, fys.TRUTH_NUS
    else:
        params, values = learned
        model = cr.get_model("tbnn_potential_yield_logconf_bk_v2")
        Gp, tau_y, nu_s = values["Gp"], values["tau_y"], values["nu_s"]
    cfg["nu_s"] = nu_s
    grid, model, initial_state, permeability = fys._build_channel_with_model(cfg, model)
    out = fys._evolve_to_steady(
        cfg, model, params, grid, initial_state, permeability, float(g_x), closure
    )
    metrics = dict(out["metrics"])
    metrics.update(_measure_stress_yield(out, cfg, Gp, tau_y))
    metrics.update(
        closure=closure,
        Gp=float(Gp),
        tau_y=float(tau_y),
        nu_s=float(nu_s),
        checkpoint=str(CHECKPOINT) if closure == "learned" else None,
    )
    tau_d = metrics.pop("tau_d_profile")
    path = out_dir / f"{closure}_gx{_tag(g_x)}.npz"
    np.savez_compressed(
        path,
        g_x=np.float64(g_x),
        y=np.asarray(metrics["y"], dtype=np.float64),
        u_profile=np.asarray(metrics["u_profile"], dtype=np.float64),
        tau_d_profile=tau_d,
        Q_history=np.asarray(metrics["Q_hist"], dtype=np.float64),
        final_A_xx=np.asarray(out["A_xx_traj"][-1]),
        final_A_xy=np.asarray(out["A_xy_traj"][-1]),
        final_A_yy=np.asarray(out["A_yy_traj"][-1]),
        final_A_zz=np.asarray(out["A_zz_traj"][-1]),
    )
    metrics.pop("y")
    metrics.pop("u_profile")
    metrics.pop("Q_hist")
    with (out_dir / f"{closure}_gx{_tag(g_x)}.json").open("w") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
    print(
        f"[{closure} g={g_x:g}] Q={metrics['Q']:.8e} "
        f"plug={metrics['plug_halfwidth']:.5f} yielded={metrics['yielded_fraction']:.1%} "
        f"conv={metrics['steady_converged']} ratio={metrics['conv_ratio']:.3e} "
        f"T/lambda={metrics['T_final'] / fys.TRUTH_LAM:.2f}",
        flush=True,
    )


def run_job(job: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    learned = _load_learned()
    for g_x in GX_BY_JOB[job]:
        for closure in ("truth", "learned"):
            _run_one(float(g_x), closure, out_dir, learned)


def _load_records(out_dir: Path):
    records = {}
    for closure in ("truth", "learned"):
        for g_x in GX_ALL:
            path = out_dir / f"{closure}_gx{_tag(float(g_x))}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            records[(closure, float(g_x))] = json.loads(path.read_text())
    return records


def aggregate(out_dir: Path):
    records = _load_records(out_dir)
    rows = []
    for g_x in GX_ALL:
        for closure in ("truth", "learned"):
            row = records[(closure, float(g_x))]
            rows.append({
                key: row[key]
                for key in (
                    "closure", "g_x", "Q", "plug_halfwidth", "yielded_fraction",
                    "steady_converged", "conv_ratio", "stop_reason", "T_final",
                    "min_eigA", "max_trA", "any_nan", "walltime_s",
                )
            })
    with (out_dir / "sweep_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    raw = {"g_x": GX_ALL}
    for closure in ("truth", "learned"):
        closure_rows = [records[(closure, float(g_x))] for g_x in GX_ALL]
        for key in (
            "Q", "plug_halfwidth", "yielded_fraction", "steady_converged",
            "conv_ratio", "T_final", "min_eigA", "max_trA", "any_nan",
        ):
            raw[f"{closure}_{key}"] = np.asarray([row[key] for row in closure_rows])
        raw[f"{closure}_u_profiles"] = np.stack([
            np.load(out_dir / f"{closure}_gx{_tag(float(g_x))}.npz")["u_profile"]
            for g_x in GX_ALL
        ])
    np.savez_compressed(out_dir / "sweep_raw.npz", **raw)

    learned_tau = records[("learned", float(GX_ALL[-1]))]["tau_y"]
    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    for closure, color, marker, label in (
        ("truth", "black", "o", "Saramito truth"),
        ("learned", "#d95f02", "s", "frozen learned closure"),
    ):
        q_values = raw[f"{closure}_Q"]
        ax.plot(GX_ALL, q_values, color=color, linewidth=2, label=label)
        ax.scatter(
            GX_ALL, q_values, marker=marker, s=40, facecolors="white",
            edgecolors=color, linewidths=1.3, zorder=3,
        )
        trained = np.isin(GX_ALL, np.asarray(phase0.DRIVES))
        ax.scatter(
            GX_ALL[trained], q_values[trained], marker=marker, s=45,
            facecolors=color, edgecolors=color, linewidths=1.0, zorder=4,
        )
    ax.scatter([], [], marker="o", s=40, facecolors="black", edgecolors="black",
               label="training drive")
    ax.axvline(TRUTH_TAU_Y, color="black", linestyle=":", linewidth=1.2,
               label=rf"truth $g_c={TRUTH_TAU_Y:.2f}$")
    ax.axvline(learned_tau, color="#d95f02", linestyle=":", linewidth=1.2,
               label=rf"learned $g_c={learned_tau:.2f}$")
    ax.axhspan(-1e-4, 1e-4, color="0.9", zorder=0, label="near-zero flow")
    ax.set(xlabel=r"$g_x$", ylabel=r"$Q$", title="EVP channel flow curve")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figN2c_flowcurve.pdf")
    fig.savefig(out_dir / "figN2c_flowcurve.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), sharey=True)
    for ax, g_x in zip(axes.ravel(), PROFILE_GX):
        idx = int(np.flatnonzero(np.isclose(GX_ALL, g_x))[0])
        for closure, color, linestyle, label in (
            ("truth", "black", "-", "truth"),
            ("learned", "#d95f02", "--", "learned"),
        ):
            ax.plot(raw[f"{closure}_u_profiles"][idx], np.load(
                out_dir / f"{closure}_gx{_tag(g_x)}.npz"
            )["y"] - 1.0, color=color, linestyle=linestyle, linewidth=2, label=label)
        ax.set_title(rf"$g_x={g_x:g}$")
        ax.set_xlabel(r"$u_x$")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel(r"$y/H$")
    axes[1, 0].set_ylabel(r"$y/H$")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "si_profiles.pdf")
    fig.savefig(out_dir / "si_profiles.png", dpi=300)
    plt.close(fig)

    manifest = {
        "kind": "evp_v2_prod2_forward_flowcurve",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "checkpoint_config": str(CONFIG),
        "g_x": GX_ALL.tolist(),
        "config": {
            "source": "evp_baseline_diag.locked_cfg",
            "Nx": 32, "Ny": 128, "dt": 2.5e-3, "inner_steps": 10,
            "solver_tol": 1e-8, "T_max_lambda": fys.T_MAX_LAM,
            "convergence_window_lambda": fys.CONV_WINDOW_LAM,
            "convergence_rtol": fys.CONV_RTOL,
            "float64": bool(jax.config.read("jax_enable_x64")),
        },
        "outputs": [
            "sweep_raw.npz", "sweep_summary.csv", "figN2c_flowcurve.pdf",
            "figN2c_flowcurve.png", "si_profiles.pdf", "si_profiles.png",
            "QSWEEP_FINDINGS.md",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[aggregate] wrote production artifacts to {out_dir}", flush=True)


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("A", "B"))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "work/evp_learned_flowcurve",
    )
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(parser)
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("float64 must be enabled")
    if args.aggregate == (args.job is not None):
        parser.error("choose exactly one of --job or --aggregate")
    if args.job:
        run_job(args.job, args.out_dir)
    else:
        aggregate(args.out_dir)


if __name__ == "__main__":
    main()
