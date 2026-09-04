#!/usr/bin/env python
"""Truth-Giesekus timing probes and De ladder for cavity transfer."""
from __future__ import annotations

import argparse
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


TRUTH = {"Gp": 3.2, "lam": 0.7, "nu_s": 0.8, "alpha": 0.30}
ETA0_TRUTH = TRUTH["nu_s"] + TRUTH["Gp"] * TRUTH["lam"]
LENGTH = 1.0
RE = 1.0
T_FINAL = 4.0
N_FRAMES = 200
CONFIGS = {
    0.20: {"cells": 96, "dt": 5.0e-5},
    0.35: {"cells": 128, "dt": 2.0e-6},
    0.50: {"cells": 128, "dt": 2.0e-6},
}


def resolved_config(de: float, timing: bool) -> dict:
    base = CONFIGS[de]
    u_lid = de * LENGTH / TRUTH["lam"]
    rho = RE * ETA0_TRUTH / (u_lid * LENGTH)
    total_steps = 2000 if timing else int(round(T_FINAL / base["dt"]))
    if total_steps % N_FRAMES:
        raise ValueError(f"{total_steps=} is not divisible by {N_FRAMES=}")
    return {
        "de_label": de,
        "cells": base["cells"],
        "dt": base["dt"],
        "T": total_steps * base["dt"],
        "total_steps": total_steps,
        "inner_steps": total_steps // N_FRAMES,
        "outer_steps": N_FRAMES,
        "U_lid": u_lid,
        "ramp_time": TRUTH["lam"],
        "density": rho,
        "Re": RE,
        "L": LENGTH,
        "lid_type": "regularized",
        "convection": "upwind",
        "devss_viscosity": 0.0,
        "solver_type": "bicgstab",
        "solver_tol": 1.0e-11,
        "solver_maxiter": 300,
        "record_diagnostics": True,
        "model": "giesekus_logconf_bk_v2",
        "polymer_params": {
            "Gp": TRUTH["Gp"],
            "lam": TRUTH["lam"],
            "alpha": TRUTH["alpha"],
        },
        "base_viscosity": TRUTH["nu_s"],
        "eta0_truth": ETA0_TRUTH,
        "float64": bool(jax.config.read("jax_enable_x64")),
    }


def build_and_evolve(config: dict):
    grid = cav.make_cavity_grid(L=config["L"], cells_per_side=config["cells"])
    model = cr.get_model(config["model"])
    state, perm_f, bc_spec = cav.build_cavity_viscoelastic_state(
        grid, U_lid=config["U_lid"], model=model
    )
    params = {
        "Gp": jnp.asarray(TRUTH["Gp"], jnp.float64),
        "lam": jnp.asarray(TRUTH["lam"], jnp.float64),
        "alpha": jnp.asarray(TRUTH["alpha"], jnp.float64),
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


def run_timing(de: float, out_dir: Path):
    config = resolved_config(de, timing=True)
    print(f"[timing] config={json.dumps(config, sort_keys=True)}", flush=True)

    t0 = time.perf_counter()
    build_and_evolve(config)
    compile_plus_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    build_and_evolve(config)
    repeat_seconds = time.perf_counter() - t0

    production_steps = int(round(T_FINAL / config["dt"]))
    projected_seconds = repeat_seconds * production_steps / config["total_steps"]
    result = {
        "config": config,
        "compile_plus_first_2000_seconds": compile_plus_first,
        "repeat_2000_seconds": repeat_seconds,
        "production_steps": production_steps,
        "projected_production_seconds": projected_seconds,
        "projected_production_hours": projected_seconds / 3600.0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"timing_De{de:.2f}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[timing] result={json.dumps(result, sort_keys=True)}", flush=True)


def psi_history(out: dict, grid) -> np.ndarray:
    u_traj = np.asarray(out["u_traj"])
    v_traj = np.asarray(out["v_traj"])
    values = []
    for u, v in zip(u_traj, v_traj):
        if np.all(np.isfinite(u)) and np.all(np.isfinite(v)):
            psi_min, _, _ = cd.psi_min_and_center_nodal(u, v, grid)
        else:
            psi_min = float("nan")
        values.append(psi_min)
    return np.asarray(values, dtype=np.float64)


def run_ladder(de: float, out_dir: Path):
    config = resolved_config(de, timing=False)
    print(f"[ladder] config={json.dumps(config, sort_keys=True)}", flush=True)
    t0 = time.perf_counter()
    grid, final, out = build_and_evolve(config)
    elapsed = time.perf_counter() - t0

    ke = np.asarray(out["ke_traj"], dtype=np.float64)
    max_axx = np.asarray(out["max_Axx_traj"], dtype=np.float64)
    min_lam = np.asarray(out["min_lam_traj"], dtype=np.float64)
    any_nan = np.asarray(out["any_nan_traj"], dtype=bool)
    ramp = np.asarray(out["ramp_traj"], dtype=np.float64)
    psi_min = psi_history(out, grid)
    verdict = cd.classify_steadiness(ke, max_axx, psi_min)

    u = np.asarray(final.velocity[0].array.data)
    v = np.asarray(final.velocity[1].array.data)
    axx = np.asarray(final.memory_fields[0].array.data)
    axy = np.asarray(final.memory_fields[1].array.data)
    ayy = np.asarray(final.memory_fields[2].array.data)
    azz = np.asarray(final.memory_fields[3].array.data)
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
        "psi_min_over_U_lid_L": float(final_psi / (config["U_lid"] * LENGTH)),
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
    print(f"[ladder] result={json.dumps(result, sort_keys=True)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("timing", "ladder"), required=True)
    parser.add_argument("--de", type=float, choices=tuple(CONFIGS), required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "work" / "cavity_transfer",
    )
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(parser)
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("float64 is not enabled")
    if args.mode == "timing":
        run_timing(args.de, args.out_dir / "timing")
    else:
        run_ladder(args.de, args.out_dir / "de_ladder")


if __name__ == "__main__":
    main()
