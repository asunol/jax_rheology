#!/usr/bin/env python
"""Carry a trained contraction closure into the lid-driven cavity, forward only.

Runs the frozen seed-s4 closure through the same Deborah-number ladder as the
truth solver, in the same boxes, so the two are directly comparable. Nothing
is retrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from repo_paths import bootstrap, REPO_ROOT
bootstrap()
ROOT = REPO_ROOT

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_rheology.diagnostics import cavity as cd
from jax_rheology.forward import cavity as cf
from jax_rheology.geometries import cavity as cav
from jax_rheology.models import registry as cr
from jax_rheology.models import tbnn_memory as _tbnn_closure  #  noqa: F401

import cavity_transfer_truth_ladder as truth_ladder
from tbnn_diff_rheo_adapter import load_tbnn_checkpoint


CHECKPOINT = ROOT / "gie_prod_rerun/gie_A_s4/theta_checkpoint.npz"
SUMMARY = ROOT / "gie_prod_rerun/gie_A_s4/summary.json"
EXPECTED_CHECKPOINT_SHA256 = (
    "a7bed94d1ce87b66766281cb1ee2bc6ab52f7479d069ec5d1686af722eb80aee"
)
MODEL_NAME = "tbnn_potential_logconf_bk_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_verify_checkpoint():
    checkpoint_sha256 = _sha256(CHECKPOINT)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA256 mismatch: {checkpoint_sha256} != "
            f"{EXPECTED_CHECKPOINT_SHA256}"
        )
    summary = json.loads(SUMMARY.read_text())
    if not summary.get("run_completed"):
        raise RuntimeError("July-10 s4 run is not marked complete")
    if summary["truth_model"] != "giesekus" or summary["scheme"] != "s4":
        raise RuntimeError("checkpoint provenance is not July-10 Giesekus s4")
    checkpoint = load_tbnn_checkpoint(str(CHECKPOINT))
    if checkpoint["truth_model"] != "giesekus":
        raise RuntimeError("checkpoint truth-model tag is not Giesekus")
    with np.load(CHECKPOINT, allow_pickle=False) as archive:
        bound_c = float(archive["ckpt_bound_c"])
    expected = {
        "Gp": float(summary["Gp_fit"]),
        "lam": float(summary["lam_fit"]),
        "nu_s": float(summary["nu_s_fit"]),
    }
    for key, value in expected.items():
        if not np.isclose(checkpoint[key], value, rtol=0.0, atol=1e-14):
            raise RuntimeError(f"checkpoint/summary scalar mismatch for {key}")
    checkpoint["bound_c"] = bound_c
    checkpoint["sha256"] = checkpoint_sha256
    return checkpoint


def resolved_config(de: float, checkpoint: dict, timing: bool) -> dict:
    config = truth_ladder.resolved_config(de, timing=timing)
    config["model"] = MODEL_NAME
    config["polymer_params"] = {
        "Gp": checkpoint["Gp"],
        "lam": checkpoint["lam"],
        "tbnn_bound_c": checkpoint["bound_c"],
    }
    config["base_viscosity"] = checkpoint["nu_s"]
    config["checkpoint"] = str(CHECKPOINT.relative_to(ROOT))
    config["checkpoint_sha256"] = checkpoint["sha256"]
    config["campaign"] = "July-10 production rerun"
    config["scheme"] = "s4"
    config["theta_frozen"] = True
    eta_p = checkpoint["Gp"] * checkpoint["lam"]
    eta_0 = checkpoint["nu_s"] + eta_p
    config["eta_p_learned"] = eta_p
    config["eta_0_learned"] = eta_0
    config["eta_p_relerr_vs_truth"] = (
        eta_p / (truth_ladder.TRUTH["Gp"] * truth_ladder.TRUTH["lam"]) - 1.0
    )
    config["eta_0_relerr_vs_truth"] = eta_0 / truth_ladder.ETA0_TRUTH - 1.0
    config["Re_eff_learned"] = (
        config["density"] * config["U_lid"] * config["L"] / eta_0
    )
    return config


def build_and_evolve(config: dict, checkpoint: dict):
    grid = cav.make_cavity_grid(L=config["L"], cells_per_side=config["cells"])
    model = cr.get_model(config["model"])
    state, perm_f, bc_spec = cav.build_cavity_viscoelastic_state(
        grid, U_lid=config["U_lid"], model=model
    )
    params = {
        "Gp": jnp.asarray(checkpoint["Gp"], jnp.float64),
        "lam": jnp.asarray(checkpoint["lam"], jnp.float64),
        "theta": checkpoint["theta"],
        "tbnn_bound_c": jnp.asarray(checkpoint["bound_c"], jnp.float64),
    }
    final, out = cf.evolve_cavity(
        state,
        grid,
        density=config["density"],
        dt=config["dt"],
        inner_steps=config["inner_steps"],
        outer_steps=config["outer_steps"],
        U_lid=config["U_lid"],
        ramp_time=config["ramp_time"],
        lid_type=config["lid_type"],
        perm_f=perm_f,
        bc_spec=bc_spec,
        model=model,
        polymer_params=params,
        base_viscosity=config["base_viscosity"],
        solver_type=config["solver_type"],
        solver_tol=config["solver_tol"],
        solver_maxiter=config["solver_maxiter"],
        convection=config["convection"],
        record_diagnostics=True,
        devss_viscosity=config["devss_viscosity"],
    )
    jax.block_until_ready(out["ke_traj"])
    return grid, final, out


def run(de: float, out_dir: Path, timing: bool):
    checkpoint = load_and_verify_checkpoint()
    config = resolved_config(de, checkpoint, timing=timing)
    mode = "timing" if timing else "learned"
    print(f"[{mode}] config={json.dumps(config, sort_keys=True)}", flush=True)
    started = time.perf_counter()
    grid, final, out = build_and_evolve(config, checkpoint)
    elapsed = time.perf_counter() - started

    if timing:
        result = {
            "config": config,
            "elapsed_seconds": elapsed,
            "projected_production_seconds": (
                elapsed * int(round(truth_ladder.T_FINAL / config["dt"]))
                / config["total_steps"]
            ),
        }
        result["projected_production_hours"] = (
            result["projected_production_seconds"] / 3600.0
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"timing_De{de:.2f}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(f"[timing] result={json.dumps(result, sort_keys=True)}", flush=True)
        return

    ke = np.asarray(out["ke_traj"], dtype=np.float64)
    max_axx = np.asarray(out["max_Axx_traj"], dtype=np.float64)
    min_lam = np.asarray(out["min_lam_traj"], dtype=np.float64)
    any_nan = np.asarray(out["any_nan_traj"], dtype=bool)
    ramp = np.asarray(out["ramp_traj"], dtype=np.float64)
    psi_min = truth_ladder.psi_history(out, grid)
    verdict = cd.classify_steadiness(ke, max_axx, psi_min)

    u = np.asarray(final.velocity[0].array.data, dtype=np.float64)
    v = np.asarray(final.velocity[1].array.data, dtype=np.float64)
    axx = np.asarray(final.memory_fields[0].array.data, dtype=np.float64)
    axy = np.asarray(final.memory_fields[1].array.data, dtype=np.float64)
    ayy = np.asarray(final.memory_fields[2].array.data, dtype=np.float64)
    azz = np.asarray(final.memory_fields[3].array.data, dtype=np.float64)
    final_psi, eye, psi = cd.psi_min_and_center_nodal(u, v, grid)

    run_dir = out_dir / f"De{de:.2f}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": config,
        "classification": verdict,
        "elapsed_seconds": elapsed,
        "nan": bool(np.any(any_nan)),
        "min_eigenvalue_over_trajectory": float(np.nanmin(min_lam)),
        "max_Axx_over_trajectory": float(np.nanmax(max_axx)),
        "max_Axx_final": float(np.nanmax(axx)),
        "psi_min_final": float(final_psi),
        "psi_min_over_U_lid_L": float(final_psi / config["U_lid"]),
        "eye": [float(eye[0]), float(eye[1])],
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        run_dir / "diagnostics.npz",
        ke_traj=ke,
        max_Axx_traj=max_axx,
        min_lam_traj=min_lam,
        any_nan_traj=any_nan,
        ramp_traj=ramp,
        psi_min_traj=psi_min,
        final_u=u,
        final_v=v,
        final_A_xx=axx,
        final_A_xy=axy,
        final_A_yy=ayy,
        final_A_zz=azz,
        final_psi=psi,
    )
    print(f"[learned] result={json.dumps(result, sort_keys=True)}", flush=True)
    if verdict == "BLEW_UP" or result["nan"] or result["min_eigenvalue_over_trajectory"] <= 0:
        raise RuntimeError(f"hard stop: unhealthy learned run at De={de:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("timing", "production"), default="production")
    parser.add_argument(
        "--de", type=float, choices=tuple(truth_ladder.CONFIGS), required=True
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "work/cavity_transfer/learned_de_ladder",
    )
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(parser)
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("float64 is not enabled")
    run(
        args.de,
        args.out_dir if args.mode == "production" else args.out_dir / "timing",
        timing=args.mode == "timing",
    )


if __name__ == "__main__":
    main()
