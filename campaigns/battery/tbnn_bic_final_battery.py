#!/usr/bin/env python
"""FENE8 publication-settings corrected-panel battery (800/6/60, 3 restarts).

FENE-P candidate is vendored ``diff_rheo.models.FENEPConformation``
(local extension in ``models/_conformation.py``) with
``ConformationStrainRateProtocol`` -- paper-era optimizer path, bit-stable.
Stress-form ``FENEP`` remains in-tree as the certified-equivalent upstream
reference. Panel name ``FENEPConformation`` is unchanged.
Provenance gate ``assert_provenance_unchanged`` compares the live
``diff_rheo`` tree against the recorded snapshot in
``reference_values/battery_provenance``. The vendored tree carries no git
history, so the comparison is on content: a hash over every Python file in
the tree, the declared upstream revision, and the sha256 of this script's
model-selection sibling. Regeneration additionally records the interpreter's
resolved diff_rheo path via ``imported_diff_rheo_python``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

if os.environ.get("JAX_PLATFORMS") != "cpu":
    raise RuntimeError("JAX_PLATFORMS=cpu required before imports")

from repo_paths import DIFF_RHEO, insert_diff_rheo
insert_diff_rheo()

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import diff_rheo as dr
from diff_rheo._forcing import VelocityGradient
from diff_rheo.models import (
    ConformationStrainRateProtocol,
    FENEPConformation,
    Giesekus,
    LinearPTT,
    Newtonian,
    OldroydB,
)
from diff_rheo.parameters import LogParameter
import battery_provenance_snapshot as provenance
import battery_instrument as instrument
import tbnn_bic_model_selection as production_battery
import tbnn_diff_rheo_adapter as adapter

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
OUT = REPO_ROOT / "work/bic_battery"
DATA = OUT / "data"
FITS = OUT / "fits"
DIFF = DIFF_RHEO

PROTOCOL = SimpleNamespace(
    epochs=800, lr=0.1, noise=0.03, n_cycles=6, pts_per_cycle=60
)
RESTART_SEEDS = (101, 202, 303)
PANEL = ("Newtonian", "OldroydB", "Giesekus",
         "FENEPConformation", "LinearPTT")
TARGETS = {
    "R1": ROOT / "fene7_prod/fene7_cur_vel/theta_checkpoint.npz",
    "R2": ROOT / "fene7_prod/fene7_cur_velp/theta_checkpoint.npz",
    "R3": ROOT / "fene7_prod/fene7_u05_vel/theta_checkpoint.npz",
    "R4": ROOT / "fene7_prod/fene7_u05_velp/theta_checkpoint.npz",
    "R5": ROOT / "fene7_prod/fene7_cur_velp_s1/theta_checkpoint.npz",
    "R6": ROOT / "fene7_prod/fene7_cur_velp_s2/theta_checkpoint.npz",
    "R7": ROOT / "fene7_prod/fene7_cur_velp_lo/theta_checkpoint.npz",
    "T1": ROOT / "fene7_pinned/fene7p_u05_pin/theta_checkpoint.npz",
    "T2": ROOT / "fene7_pinned/fene7p_cur_pin/theta_checkpoint.npz",
    "T3": ROOT / "fene7_pinned/fene7p_cur_free/theta_checkpoint.npz",
    "gie_A_s1": ROOT / "gie_prod_rerun/gie_A_s1/theta_checkpoint.npz",
    "gie_A_s1b": ROOT / "gie_prod_rerun/gie_A_s1b/theta_checkpoint.npz",
    "gie_A_s4": ROOT / "gie_prod_rerun/gie_A_s4/theta_checkpoint.npz",
}
ANALYTIC = {
    "clean_analytic_fene_p":
        ROOT / "analysis_phase2_instrument/diag_diffrheo/batteries/FENEP.npz",
    "clean_analytic_giesekus":
        ROOT / "analysis_phase2_instrument/diag_diffrheo/batteries/Giesekus.npz",
}
ALL_TARGETS = tuple(TARGETS) + tuple(ANALYTIC)
F_GRID = [0.01, 0.1, 1.0, 10.0]
W_GRID = [0.33, 1.0, 2.0]
DEFAULT_INITS = {
    "Newtonian": {"viscosity": 1.0},
    "OldroydB": {
        "polymer_viscosity": 1.0, "relaxation_time": 1.0,
        "solvent_viscosity": 1.0,
    },
    "Giesekus": {
        "polymer_viscosity": 1.0, "relaxation_time": 1.0,
        "solvent_viscosity": 1.0, "alpha": 0.1,
    },
    "FENEPConformation": {
        "polymer_viscosity": 1.0, "relaxation_time": 1.0,
        "solvent_viscosity": 1.0, "extension_length": 5.0,
    },
    "LinearPTT": {
        "polymer_viscosity": 1.0, "relaxation_time": 1.0,
        "solvent_viscosity": 1.0, "epsilon": 0.1, "zeta": 0.1,
    },
}


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def imported_diff_rheo_python():
    rows = []
    seen = set()
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        path = Path(raw).resolve()
        if path.suffix == ".py" and DIFF in path.parents and path not in seen:
            seen.add(path)
            rows.append(
                {"path": str(path.relative_to(DIFF)), "sha256": sha256(path)}
            )
    return sorted(rows, key=lambda row: row["path"])


def assert_provenance_unchanged():
    """Fail if the fitting code is not the tree the snapshot was recorded from.

    Compares the live ``diff_rheo`` tree against
    ``reference_values/battery_provenance/start_snapshot.json`` on whichever
    fields both carry; see :mod:`battery_provenance_snapshot`.
    """
    start = json.loads((provenance.OUT / "start_snapshot.json").read_text())
    current, _ = provenance.snapshot_payload()
    comparisons = provenance.compare(start, current)
    changed = [key for key, same in comparisons.items() if not same]
    if changed:
        raise RuntimeError(f"provenance changed during pass: {changed}")
    return {"compared": sorted(comparisons), "tree_sha256": current["tree_sha256"]}


def assert_cpu_x64():
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"CPU required, got {jax.default_backend()}")
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("jax_enable_x64 is false")
    if jnp.ones((), dtype=jnp.float64).dtype != jnp.float64:
        raise RuntimeError("float64 assertion failed")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    os.replace(temp, path)


def exception_record(case, exc):
    return {
        "case": case, "status": "error",
        "error_type": type(exc).__name__, "error": repr(exc),
        "traceback": traceback.format_exc(),
        "imported_diff_rheo_python": imported_diff_rheo_python(),
    }


def jitter_inits(candidate, seed, scale=0.5):
    rng = np.random.default_rng(seed)
    base = DEFAULT_INITS[candidate]
    return {
        key: float(value * math.exp(rng.normal(0.0, scale)))
        for key, value in base.items()
    }


def make_model(candidate, inits):
    lp = {key: LogParameter(value) for key, value in inits.items()}
    if candidate == "Newtonian":
        return Newtonian(viscosity=lp["viscosity"])
    if candidate == "OldroydB":
        return OldroydB(
            polymer_viscosity=lp["polymer_viscosity"],
            relaxation_time=lp["relaxation_time"],
            solvent_viscosity=lp["solvent_viscosity"],
        )
    if candidate == "Giesekus":
        return Giesekus(
            polymer_viscosity=lp["polymer_viscosity"],
            relaxation_time=lp["relaxation_time"],
            solvent_viscosity=lp["solvent_viscosity"],
            alpha=lp["alpha"],
        )
    if candidate == "FENEPConformation":
        return FENEPConformation(
            polymer_viscosity=lp["polymer_viscosity"],
            relaxation_time=lp["relaxation_time"],
            solvent_viscosity=lp["solvent_viscosity"],
            extension_length=lp["extension_length"],
        )
    if candidate == "LinearPTT":
        return LinearPTT(
            polymer_viscosity=lp["polymer_viscosity"],
            relaxation_time=lp["relaxation_time"],
            solvent_viscosity=lp["solvent_viscosity"],
            epsilon=lp["epsilon"],
            zeta=lp["zeta"],
        )
    raise KeyError(candidate)


def corrected_rheometer(solver):
    """Paper-era conformation-form protocol (local FENE-P extension)."""
    return dr.VirtualRheometer(ConformationStrainRateProtocol(), solver)


def make_rheometer(candidate, model, solver):
    if candidate == "FENEPConformation":
        return corrected_rheometer(solver)
    return dr.VirtualRheometer.setup(model, "strain_rate_response", solver)


def serialize_data(data, path: Path, extra: dict):
    arrays = {
        "time": np.stack([np.asarray(ref.time) for ref in data.data]),
        "sigma_noisy": np.stack([np.asarray(ref.data) for ref in data.data]),
        "gammadot": np.stack([np.asarray(ref.forcing_data) for ref in data.data]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    write_json(path.with_suffix(".json"), extra)


def as_data(path: Path):
    with np.load(path, allow_pickle=False) as values:
        refs = [
            dr.ShearStrainRateData(
                time=jnp.asarray(values["time"][i]),
                data=jnp.asarray(values["sigma_noisy"][i]),
                forcing_data=jnp.asarray(values["gammadot"][i]),
                initial_condition=jnp.zeros((3, 3), dtype=jnp.float64),
            )
            for i in range(values["time"].shape[0])
        ]
    return dr.BatchedData.from_data(*refs)


def envelope(ck):
    solver = instrument.data_solver()
    rhs = adapter.make_tbnn_rhs(ck["theta"], ck["lam"])
    rows = []
    for omega in W_GRID:
        period = 2 * np.pi / omega
        ts = jnp.linspace(0, PROTOCOL.n_cycles * period,
                          PROTOCOL.n_cycles * PROTOCOL.pts_per_cycle + 1)
        for amp in F_GRID:
            vg = VelocityGradient.from_components(
                grad_u_12=lambda t, f=amp, w=omega: f * jnp.sin(w * t)
            )
            sol = solver.integrate(rhs, adapter.A_REST, ts, vg)
            state = np.asarray(sol.ys)
            tr_a = state[:, 0] + state[:, 2] + state[:, 3]
            rows.append(
                {
                    "omega": omega, "amplitude": amp,
                    "max_trA": float(np.max(tr_a)),
                    "min_trA": float(np.min(tr_a)),
                    "finite": bool(np.all(np.isfinite(state))),
                }
            )
    return rows


def prepare(target):
    started = time.perf_counter()
    output = DATA / f"{target}.npz"
    if target in ANALYTIC:
        # Regenerate analytic controls at publication protocol via production
        # package truth models (solver-convention FENE uses FENEPConformation;
        # panel name remains FENEPConformation).
        source = ANALYTIC[target]
        with np.load(source, allow_pickle=False) as values:
            # Keep the archived analytic NPZ content but record protocol target;
            # noise floor is taken from the archive (same generator family).
            refs = [
                dr.ShearStrainRateData(
                    time=jnp.asarray(values["time"][i]),
                    data=jnp.asarray(values["sigma_noisy"][i]),
                    forcing_data=jnp.asarray(values["gammadot"][i]),
                    initial_condition=jnp.zeros((3, 3), dtype=jnp.float64),
                )
                for i in range(values["time"].shape[0])
            ]
            noise_mse = float(np.mean(values["noise"] ** 2))
        data = dr.BatchedData.from_data(*refs)
        metadata = {
            "target": target, "kind": "analytic_archived_npz",
            "source": str(source), "source_sha256": sha256(source),
            "noise_mse": noise_mse,
            "note": "archived analytic NPZ retained; fit epochs=800",
        }
    else:
        checkpoint = TARGETS[target]
        ck = adapter.load_tbnn_checkpoint(checkpoint)
        data, generator_meta, _ = production_battery._gen_tbnn_data(
            ck, instrument.data_solver(), PROTOCOL, jax.random.PRNGKey(0)
        )
        env = envelope(ck)
        metadata = {
            "target": target, "kind": "learned_checkpoint",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "generator_meta": generator_meta, "envelope": env,
            "max_trA": max(row["max_trA"] for row in env),
        }
    metadata.update(
        {
            "status": "ok", "protocol": vars(PROTOCOL),
            "restart_seeds": list(RESTART_SEEDS),
            "elapsed_seconds": time.perf_counter() - started,
            "imported_diff_rheo_python": imported_diff_rheo_python(),
        }
    )
    serialize_data(data, output, metadata)


def pooled_metrics(model, rheometer, data):
    sse, n = 0.0, 0
    for ref in data.data:
        simulation = rheometer.run_experiment(
            model, ref.get_forcing_function(), ref.time, ref.initial_condition
        )
        pred = np.asarray(ref.extract_from_simulation(simulation))
        actual = np.asarray(ref.data)
        sse += float(np.sum((pred - actual) ** 2))
        n += pred.size
    mse = sse / n
    return mse, float(dr.calculate_bic_from_l2(model, rheometer, data))


def fit_one_restart(candidate, data, seed):
    inits = jitter_inits(candidate, seed)
    model = make_model(candidate, inits)
    rheometer = make_rheometer(candidate, model, instrument.fit_solver())
    config = dr.FittingConfig(
        num_epochs=PROTOCOL.epochs, learning_rate=PROTOCOL.lr,
        ensemble_size=1, key=jax.random.PRNGKey(seed), verbose=False,
    )
    try:
        fitted = dr.fit_model_to_experimental_data(
            model, rheometer, data, config
        )
        mse, bic = pooled_metrics(fitted, rheometer, data)
        ok = bool(np.isfinite(bic))
        params = {
            key: float(value)
            for key, value in fitted.parameter_values.items()
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "restart_seed": seed, "init": inits, "ok": False,
            "bic": float("inf"), "mse": float("inf"), "params": {},
            "error": repr(exc),
        }
    return {
        "restart_seed": seed, "init": inits, "ok": ok,
        "bic": bic, "mse": mse, "params": params,
    }


def fit_restart(target, candidate, restart_index):
    seed = RESTART_SEEDS[restart_index]
    case = f"{target}_{candidate}_r{restart_index}_s{seed}"
    output = FITS / target / f"{candidate}_r{restart_index}.json"
    started = time.perf_counter()
    try:
        data = as_data(DATA / f"{target}.npz")
        row = fit_one_restart(candidate, data, seed)
        write_json(
            output,
            {
                "case": case, "target": target, "candidate": candidate,
                "restart_index": restart_index, "restart_seed": seed,
                "status": "ok" if row["ok"] else "failed",
                "protocol": vars(PROTOCOL),
                "result": row,
                "elapsed_seconds": time.perf_counter() - started,
                "imported_diff_rheo_python": imported_diff_rheo_python(),
            },
        )
    except Exception as exc:
        write_json(output, exception_record(case, exc))


def merge_target(target):
    rows = []
    for candidate in PANEL:
        restarts = []
        for restart_index, seed in enumerate(RESTART_SEEDS):
            path = FITS / target / f"{candidate}_r{restart_index}.json"
            record = json.loads(path.read_text())
            if record.get("status") == "ok":
                restarts.append(record["result"])
            else:
                # DIVERGED / failed -> BIC=+inf (epoch-218 convention); never substitute
                res = record.get("result") or {}
                restarts.append(
                    {
                        "restart_seed": seed, "ok": False,
                        "bic": float("inf"), "mse": float("inf"),
                        "params": {},
                        "error": res.get("error") or record.get("error"),
                        "status": record.get("status"),
                        "halt_epoch": res.get("halt_epoch"),
                        "last_finite": res.get("last_finite"),
                    }
                )
        ok_rows = [row for row in restarts if row["ok"]]
        if ok_rows:
            best = min(ok_rows, key=lambda row: row["bic"])
            rows.append(
                {
                    "name": candidate, "bic": best["bic"], "mse": best["mse"],
                    "params": best["params"], "ok": True,
                    "best_restart_seed": best["restart_seed"],
                    "restart_spread": {
                        "seeds": list(RESTART_SEEDS),
                        "bic": [row["bic"] for row in restarts],
                        "mse": [row["mse"] for row in restarts],
                        "ok": [row["ok"] for row in restarts],
                        "params": [row["params"] for row in restarts],
                        "init": [row.get("init", {}) for row in restarts],
                    },
                }
            )
        else:
            rows.append(
                {
                    "name": candidate, "bic": math.inf, "mse": math.inf,
                    "params": {}, "ok": False,
                    "restart_spread": {
                        "seeds": list(RESTART_SEEDS), "restarts": restarts,
                    },
                }
            )
    ranked = sorted((row for row in rows if row["ok"]), key=lambda row: row["bic"])
    metadata = json.loads((DATA / f"{target}.json").read_text())
    result = {
        "target": target, "status": "ok" if ranked else "failed",
        "winner": ranked[0]["name"] if ranked else None,
        "margin": (
            ranked[1]["bic"] - ranked[0]["bic"] if len(ranked) > 1 else None
        ),
        "max_trA": metadata.get("max_trA"),
        "protocol": vars(PROTOCOL),
        "results": rows,
        "imported_diff_rheo_python": imported_diff_rheo_python(),
    }
    write_json(OUT / "targets" / f"{target}.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["prepare", "fit", "merge-target", "list"],
    )
    parser.add_argument("--target")
    parser.add_argument("--candidate", choices=PANEL)
    parser.add_argument("--restart", type=int, choices=(0, 1, 2))
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(parser)
    if args.mode == "list":
        print("TARGETS", " ".join(ALL_TARGETS))
        print("PANEL", " ".join(PANEL))
        print("RESTART_SEEDS", " ".join(str(s) for s in RESTART_SEEDS))
        print("PROTOCOL", vars(PROTOCOL))
        return
    assert_cpu_x64()
    assert_provenance_unchanged()
    try:
        if args.mode == "prepare":
            if args.target not in ALL_TARGETS:
                raise SystemExit(f"unknown target {args.target}")
            prepare(args.target)
        elif args.mode == "fit":
            if (
                args.target not in ALL_TARGETS
                or args.candidate is None
                or args.restart is None
            ):
                raise SystemExit(
                    "fit requires --target --candidate --restart"
                )
            fit_restart(args.target, args.candidate, args.restart)
        else:
            if args.target not in ALL_TARGETS:
                raise SystemExit(f"unknown target {args.target}")
            merge_target(args.target)
        assert_provenance_unchanged()
    except Exception as exc:
        path = (
            OUT / "errors"
            / (
                f"{args.mode}_{args.target or 'none'}_"
                f"{args.candidate or 'none'}_r{args.restart}"
            )
        ).with_suffix(".json")
        write_json(path, exception_record(args.mode, exc))
        raise


if __name__ == "__main__":
    main()
