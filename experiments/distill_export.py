#!/usr/bin/env python
"""Export a library TBNN to a float32 stress-trace npz.

Traces evaluated in the solver environment differ from the rheometry
environment's paper path by at most 2e-6 (1 ULP in float32, from a different
XLA build); this is therefore not the route used for Table S1.

Runs in the solver environment at JAX's default float32. Loads a frozen
instantaneous checkpoint through
``jax_rheology.models.tbnn_instantaneous.build_tbnn_bounded_model``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Evaluator is the f32 side of the distill split. A leaked x64 flag must
# not promote the traces.
os.environ.pop("JAX_ENABLE_X64", None)

from repo_paths import FROZEN_INST, bootstrap  # noqa: E402

bootstrap()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", False)
_DTYPE = jnp.asarray(1.0).dtype
if _DTYPE != jnp.float32:
    raise RuntimeError(
        f"distill_export is f32-only; jnp.asarray(1.0).dtype={_DTYPE}"
    )

from jax_rheology.models.tbnn_instantaneous import (  # noqa: E402
    build_tbnn_bounded_model,
)

# Main-CLI grid (model_selection_tbnn.py generate_single_curve_from_dir).
T_END = 16.0
N_TIME = 120
AMPLITUDE = 10.0
FREQUENCY = 1.0
NOISE_LEVEL = 0.0


def _params_path(trajectory_data_dir: Path, base_name: str) -> Path:
    converted = trajectory_data_dir / base_name.replace(".pkl", "_converted.pkl")
    original = trajectory_data_dir / base_name
    if converted.is_file():
        return converted
    return original


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_tbnn(run_dir: Path, checkpoint: str = "final"):
    """Pickle + library constructor. Architecture hardcodes match the driver."""
    trajectory_data_dir = run_dir / "trajectory_data"
    if not trajectory_data_dir.is_dir():
        raise FileNotFoundError(f"missing {trajectory_data_dir}")

    if checkpoint == "final":
        params_file = _params_path(trajectory_data_dir, "final_tbnn_params.pkl")
        metadata_prefix = "final"
    elif checkpoint == "initial":
        params_file = _params_path(trajectory_data_dir, "initial_tbnn_params.pkl")
        metadata_prefix = "initial"
    elif checkpoint == "stage1":
        params_file = _params_path(trajectory_data_dir, "stage1_tbnn_params.pkl")
        metadata_prefix = "stage1"
    else:
        raise ValueError(f"unsupported checkpoint {checkpoint!r}")

    if not params_file.is_file():
        raise FileNotFoundError(f"parameters file not found: {params_file}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, message=".*named_shape.*"
        )
        with params_file.open("rb") as f:
            tbnn_params = pickle.load(f)

    visc_file = trajectory_data_dir / f"{metadata_prefix}_viscosities.npy"
    eta_min = 1e-2
    eta_max = 10.0
    if visc_file.is_file():
        viscosities = np.load(visc_file)
        eta_min = max(float(np.min(viscosities)) * 0.1, 1e-3)
        eta_max = float(np.max(viscosities)) * 2.0

    hidden_units = [16]
    M = 12
    model = build_tbnn_bounded_model(
        hidden_units=hidden_units,
        M=M,
        eta_min=eta_min,
        eta_max=eta_max,
        gamma_ref=1.0,
        s_floor=0.35,
        alpha_temp=0.8,
        freeze_eta0=True,
        eta0_fixed=1.0,
        eta0_eps=1e-5,
        mu_min_gamma=0.1,
        mu_max_gamma=10.0,
        gate_gamma=0.1,
        gate_width_z=0.5,
        log_head=True,
        log_mixing="add",
    )
    return {
        "tbnn_model": model,
        "tbnn_params": tbnn_params,
        "params_file": params_file,
        "checkpoint_sha256": _sha256_file(params_file),
        "hidden_units": hidden_units,
        "M": M,
        "eta_min": eta_min,
        "eta_max": eta_max,
        "checkpoint": checkpoint,
    }


def _compute_velocity_gradient_for_shear(gamma_dot):
    return jnp.array([[0.0, gamma_dot], [0.0, 0.0]])


def _compute_S_R_from_grad(grad):
    S = 0.5 * (grad + grad.T)
    R = 0.5 * (grad - grad.T)
    return S, R


def _compute_invariants_2d(S, R):
    I1 = jnp.trace(S @ S)
    I2 = jnp.trace(R @ R)
    return jnp.array([I1, I2])


def _eta_from_tbnn_scalar(gamma_abs, invariants, tbnn_model, params):
    gamma_grid = jnp.asarray([[gamma_abs]])
    inv_grid = invariants.reshape(2, 1, 1)
    prms = params if ("params" in params) else {"params": params}
    eta_11 = tbnn_model.apply(prms, gamma_grid, inv_grid)
    return eta_11[0, 0]


def evaluate_traces(tbnn_model, tbnn_params, *, amplitude=AMPLITUDE,
                    frequency=FREQUENCY, t_end=T_END, n_time=N_TIME):
    """Homogeneous-shear eta and tau on the main-CLI grid. Library apply only."""
    time = np.linspace(0.0, t_end, n_time, dtype=np.float32)
    time_j = jnp.asarray(time)
    g_t = amplitude * jnp.sin(frequency * time_j)

    def eta_of_t(gamma_signed):
        grad = _compute_velocity_gradient_for_shear(gamma_signed)
        S, R = _compute_S_R_from_grad(grad)
        inv = _compute_invariants_2d(S, R)
        return _eta_from_tbnn_scalar(
            jnp.abs(gamma_signed), inv, tbnn_model, tbnn_params
        )

    eta_t = jax.vmap(eta_of_t)(g_t)
    stress = eta_t * g_t
    return {
        "time": np.asarray(time, dtype=np.float32),
        "gamma_dot": np.asarray(g_t, dtype=np.float32),
        "eta": np.asarray(eta_t, dtype=np.float32),
        "stress": np.asarray(stress, dtype=np.float32),
        "initial_condition": np.zeros((3, 3), dtype=np.float32),
    }


def write_npz(path: Path, arrays: dict, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload.update(
        {
            "dtype": np.asarray("float32"),
            "amplitude": np.float32(meta["amplitude"]),
            "frequency": np.float32(meta["frequency"]),
            "t_end": np.float32(meta["t_end"]),
            "n_time": np.int32(meta["n_time"]),
            "noise_level": np.float32(meta["noise_level"]),
            "run_dir": np.asarray(meta["run_dir"]),
            "run_name": np.asarray(meta["run_name"]),
            "checkpoint": np.asarray(meta["checkpoint"]),
            "params_file": np.asarray(meta["params_file"]),
            "checkpoint_sha256": np.asarray(meta["checkpoint_sha256"]),
            "hidden_units": np.asarray(meta["hidden_units"], dtype=np.int32),
            "M": np.int32(meta["M"]),
            "library_module": np.asarray(
                "jax_rheology.models.tbnn_instantaneous"
            ),
            "evaluator": np.asarray("experiments/distill_export.py"),
        }
    )
    np.savez(path, **payload)
    return path


def verify_roundtrip(arrays: dict, path: Path) -> None:
    """(ii) in-memory evaluator arrays vs np.load of the file just written."""
    z = np.load(path)
    fails = []
    for key in ("time", "gamma_dot", "eta", "stress", "initial_condition"):
        a = np.asarray(arrays[key])
        b = np.asarray(z[key])
        if a.shape != b.shape or a.dtype != b.dtype or np.any(a != b):
            if a.shape != b.shape or a.dtype != b.dtype:
                fails.append(f"{key}: shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
            else:
                i = int(np.argmax(a.reshape(-1) != b.reshape(-1)))
                fails.append(f"{key}: index {i} {a.reshape(-1)[i]!r} vs {b.reshape(-1)[i]!r}")
        else:
            print(f"[roundtrip] {key}: bitwise 0  shape={a.shape} dtype={a.dtype}",
                  flush=True)
    if fails:
        raise SystemExit("npz roundtrip not bitwise 0: " + "; ".join(fails))
    print("[roundtrip] PASS", flush=True)


def export_run(run_dir: Path, out_path: Path, *, checkpoint: str = "final",
               verify: bool = False) -> Path:
    run_dir = Path(run_dir).resolve()
    loaded = load_frozen_tbnn(run_dir, checkpoint=checkpoint)
    arrays = evaluate_traces(loaded["tbnn_model"], loaded["tbnn_params"])
    meta = {
        "amplitude": AMPLITUDE,
        "frequency": FREQUENCY,
        "t_end": T_END,
        "n_time": N_TIME,
        "noise_level": NOISE_LEVEL,
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "checkpoint": loaded["checkpoint"],
        "params_file": str(loaded["params_file"]),
        "checkpoint_sha256": loaded["checkpoint_sha256"],
        "hidden_units": loaded["hidden_units"],
        "M": loaded["M"],
    }
    write_npz(out_path, arrays, meta)
    print(
        f"[distill_export] wrote {out_path} dtype=float32 "
        f"n={arrays['time'].shape[0]} sha={loaded['checkpoint_sha256'][:12]} "
        f"run={run_dir.name}",
        flush=True,
    )
    if verify:
        verify_roundtrip(arrays, Path(out_path))
    return Path(out_path)


def resolve_run_dir(run_dir=None, run_name=None) -> Path:
    if run_dir:
        return Path(run_dir).resolve()
    if not run_name:
        raise SystemExit("need --run-dir or --run-name")
    return (FROZEN_INST / "tbnn_debug_results_constriction_new" / run_name).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate frozen TBNN via library code; write f32 npz"
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--out", required=True, metavar="NPZ")
    parser.add_argument("--verify-roundtrip", action="store_true",
                        help="bitwise-compare in-memory arrays to the written npz")
    args = parser.parse_args(argv)
    run_dir = resolve_run_dir(args.run_dir, args.run_name)
    if not (run_dir / "trajectory_data").is_dir():
        raise SystemExit(f"missing frozen run {run_dir}")
    export_run(run_dir, Path(args.out), checkpoint=args.checkpoint,
               verify=args.verify_roundtrip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
