#!/usr/bin/env python
"""CPU-only zero-dimensional interrogation of a trained FENE-P closure.

This analysis-only script never imports or calls the CFD solver.  It reuses the
production battery's checkpoint adapter, DiffraxSolver settings, data generator,
candidate constructors, optimizer, and BIC implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

if os.environ.get("JAX_PLATFORMS") != "cpu":
    raise RuntimeError("JAX_PLATFORMS=cpu must be set before starting this script")

from repo_paths import insert_diff_rheo
insert_diff_rheo()

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import diff_rheo as dr
from diff_rheo._forcing import VelocityGradient
from diff_rheo.models import FENEP
from diff_rheo.parameters import LogParameter

import tbnn_bic_model_selection as battery
import tbnn_diff_rheo_adapter as adapter

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / "work/bic_instrument"
TRUTH = dict(Gp=3.2, lam=0.7, nu_s=0.8, eta_p=2.24, Lsq=12.0)
F_GRID = [0.01, 0.1, 1.0, 10.0]
W_GRID = [0.33, 1.0, 2.0]

CLEAN = {
    "fene_p": ROOT / "fene_repr_probe/A/ckpt_fene_p.npz",
    "giesekus": ROOT / "fene_repr_probe/A/ckpt_giesekus.npz",
    "linear_ptt": ROOT / "fene_repr_probe/A/ckpt_linear_ptt.npz",
    "oldroyd_b": ROOT / "fene_repr_probe/A/ckpt_oldroyd_b.npz",
}
RUNS = {
    "R1": ("fene7_cur_vel", ROOT / "fene7_prod/fene7_cur_vel"),
    "R2": ("fene7_cur_velp", ROOT / "fene7_prod/fene7_cur_velp"),
    "R3": ("fene7_u05_vel", ROOT / "fene7_prod/fene7_u05_vel"),
    "R4": ("fene7_u05_velp", ROOT / "fene7_prod/fene7_u05_velp"),
    "R5": ("fene7_cur_velp_s1", ROOT / "fene7_prod/fene7_cur_velp_s1"),
    "R6": ("fene7_cur_velp_s2", ROOT / "fene7_prod/fene7_cur_velp_s2"),
    "R7": ("fene7_cur_velp_lo", ROOT / "fene7_prod/fene7_cur_velp_lo"),
    "T1": ("fene7p_u05_pin", ROOT / "fene7_pinned/fene7p_u05_pin"),
    "T2": ("fene7p_cur_pin", ROOT / "fene7_pinned/fene7p_cur_pin"),
    "T3": ("fene7p_cur_free", ROOT / "fene7_pinned/fene7p_cur_free"),
}


def assert_cpu_x64() -> None:
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"CPU backend required, got {jax.default_backend()}")
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("jax_enable_x64 is false")
    if jnp.ones((), dtype=jnp.float64).dtype != jnp.float64:
        raise RuntimeError("float64 assertion failed")


def args_record() -> SimpleNamespace:
    return SimpleNamespace(
        epochs=500, lr=0.1, noise=0.03, n_cycles=4, pts_per_cycle=20
    )


def data_solver() -> dr.DiffraxSolver:
    return dr.DiffraxSolver(
        solver="tsit5", rtol=1e-8, atol=1e-8, dt0=1e-3,
        max_steps=4_000_000, throw=False
    )


def fit_solver() -> dr.DiffraxSolver:
    return dr.DiffraxSolver(
        solver="tsit5", rtol=1e-6, atol=1e-6, dt0=1e-3,
        max_steps=1_000_000, throw=False
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fp:
        json.dump(value, fp, indent=2, default=float)
    os.replace(tmp, path)


def exception_record(case: str, exc: BaseException) -> dict:
    return {
        "case": case,
        "status": "error",
        "error_type": type(exc).__name__,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }


def interrogate_envelopes(ck: dict, label: str, persist_points: bool = True) -> list[dict]:
    """Re-integrate every exact production leg and persist its trace envelope."""
    solver = data_solver()
    rhs = adapter.make_tbnn_rhs(ck["theta"], ck["lam"])
    summaries: list[dict] = []
    point_path = OUT / "trajectories" / f"tra_{label}.csv"
    point_path.parent.mkdir(parents=True, exist_ok=True)
    point_fp = point_path.open("w", newline="") if persist_points else None
    writer = None
    if point_fp is not None:
        writer = csv.DictWriter(
            point_fp,
            fieldnames=["case", "omega", "f", "sample", "time", "A_xx",
                        "A_xy", "A_yy", "A_zz", "trA", "finite"],
        )
        writer.writeheader()
    try:
        for omega in W_GRID:
            period = 2.0 * np.pi / omega
            ts = jnp.linspace(0.0, 4.0 * period, 4 * 20 + 1)
            for f in F_GRID:
                vg = VelocityGradient.from_components(
                    grad_u_12=lambda t, f=f, w=omega: f * jnp.sin(w * t)
                )
                sol = solver.integrate(rhs, adapter.A_REST, ts, vg)
                A = np.asarray(sol.ys, dtype=np.float64)
                trA = A[:, 0] + A[:, 2] + A[:, 3]
                finite = np.isfinite(A).all(axis=1) & np.isfinite(trA)
                summaries.append(
                    {
                        "case": label,
                        "omega": omega,
                        "f": f,
                        "n": int(len(ts)),
                        "finite": bool(finite.all()),
                        "max_trA": float(np.max(trA[finite])) if finite.any() else None,
                        "min_trA": float(np.min(trA[finite])) if finite.any() else None,
                    }
                )
                if writer is not None:
                    for i, (t, avec, tr, ok) in enumerate(
                        zip(np.asarray(ts), A, trA, finite)
                    ):
                        writer.writerow(
                            {
                                "case": label, "omega": omega, "f": f,
                                "sample": i, "time": float(t),
                                "A_xx": float(avec[0]), "A_xy": float(avec[1]),
                                "A_yy": float(avec[2]), "A_zz": float(avec[3]),
                                "trA": float(tr), "finite": bool(ok),
                            }
                        )
    finally:
        if point_fp is not None:
            point_fp.close()
    summary_path = OUT / "trajectories" / f"tra_summary_{label}.csv"
    with summary_path.open("w", newline="") as fp:
        writer2 = csv.DictWriter(fp, fieldnames=list(summaries[0]))
        writer2.writeheader()
        writer2.writerows(summaries)
    return summaries


def serializable_fit(result: dict) -> dict:
    return {
        "name": result["name"],
        "bic": result["bic"],
        "mse": result["mse"],
        "params": result["params"],
        "ok": result["ok"],
        "k_reported": len(
            [k for k in result["params"] if k != "observation_noise"]
        ),
    }


def run_clean(model: str) -> None:
    case = f"clean_{model}"
    out = OUT / "clean_battery" / f"{case}.json"
    try:
        ck_path = CLEAN[model]
        if not ck_path.is_file():
            raise FileNotFoundError(ck_path)
        ck = adapter.load_tbnn_checkpoint(ck_path)
        a = args_record()
        key = jax.random.PRNGKey(0)
        # This is the production generator itself, including its seeded 0.03 noise.
        data, meta, _ = battery._gen_tbnn_data(ck, data_solver(), a, key)
        results = []
        for name, maker in battery._candidate_models().items():
            results.append(
                serializable_fit(
                    battery._fit_one(name, maker, data, fit_solver(), a, key)
                )
            )
        finite = [r for r in results if r["ok"]]
        winner = min(finite, key=lambda r: r["bic"])["name"] if finite else None
        envelopes = interrogate_envelopes(ck, case)
        write_json(
            out,
            {
                "case": case,
                "status": "ok",
                "checkpoint": str(ck_path),
                "protocol": {
                    "F_GRID": F_GRID, "W_GRID": W_GRID, "noise": 0.03,
                    "epochs": 500, "lr": 0.1, "n_cycles": 4,
                    "pts_per_cycle": 20,
                    "data_solver": {
                        "solver": "tsit5", "rtol": 1e-8, "atol": 1e-8,
                        "dt0": 1e-3, "max_steps": 4_000_000,
                    },
                    "fit_solver": {
                        "solver": "tsit5", "rtol": 1e-6, "atol": 1e-6,
                        "dt0": 1e-3, "max_steps": 1_000_000,
                    },
                },
                "generator_meta": meta,
                "winner": winner,
                "max_trA": max(
                    (x["max_trA"] for x in envelopes if x["max_trA"] is not None),
                    default=None,
                ),
                "results": results,
            },
        )
    except Exception as exc:  # record-don't-fail
        write_json(out, exception_record(case, exc))


def run_clean_candidate(model: str, candidate: str) -> None:
    """Persist one exact clean-battery candidate fit (timeout-safe split)."""
    case = f"clean_{model}_{candidate}"
    out = OUT / "clean_battery/parts" / f"{case}.json"
    try:
        ck_path = CLEAN[model]
        ck = adapter.load_tbnn_checkpoint(ck_path)
        a = args_record()
        key = jax.random.PRNGKey(0)
        data, meta, _ = battery._gen_tbnn_data(ck, data_solver(), a, key)
        makers = battery._candidate_models()
        if candidate not in makers:
            raise KeyError(f"unknown candidate {candidate!r}")
        result = serializable_fit(
            battery._fit_one(candidate, makers[candidate], data, fit_solver(), a, key)
        )
        write_json(
            out,
            {
                "case": case, "status": "ok", "checkpoint": str(ck_path),
                "generator_meta": meta, "result": result,
            },
        )
    except Exception as exc:  # record-don't-fail
        write_json(out, exception_record(case, exc))


def merge_clean(model: str) -> None:
    """Merge split candidate records and persist all exact trA trajectories."""
    case = f"clean_{model}"
    out = OUT / "clean_battery" / f"{case}.json"
    try:
        results = []
        reused = []
        pr1b = read_json(OUT / "clean_battery/pr1b_fenep_self_fit.json")
        for candidate in battery._candidate_models():
            part = read_json(
                OUT / "clean_battery/parts" / f"{case}_{candidate}.json"
            )
            # A completed fit failure is still a valid battery row (ok=False,
            # BIC=inf); retain it rather than treating it as a missing job.
            if part and isinstance(part.get("result"), dict):
                results.append(part["result"])
                continue
            # PR1b used the identical seed-0 clean-FENE data and standard
            # constructors for these two fits; reuse rather than recompute.
            pr1b_name = {
                "Newtonian": "Newtonian_standard_init",
                "FENEP": "FENEP_agnostic_init",
            }.get(candidate)
            prior = next(
                (
                    r for r in (pr1b or {}).get("results", [])
                    if r.get("name") == pr1b_name and r.get("ok")
                ),
                None,
            )
            if prior is None:
                raise RuntimeError(f"missing/failed split record for {candidate}")
            prior = dict(prior)
            prior["name"] = candidate
            results.append(prior)
            reused.append({"candidate": candidate, "source": "PR1b identical fit"})
        ck_path = CLEAN[model]
        ck = adapter.load_tbnn_checkpoint(ck_path)
        envelopes = interrogate_envelopes(ck, case)
        finite = [r for r in results if r["ok"]]
        winner = min(finite, key=lambda r: r["bic"])["name"] if finite else None
        write_json(
            out,
            {
                "case": case, "status": "ok", "checkpoint": str(ck_path),
                "protocol": {
                    "F_GRID": F_GRID, "W_GRID": W_GRID, "noise": 0.03,
                    "epochs": 500, "lr": 0.1, "n_cycles": 4,
                    "pts_per_cycle": 20,
                },
                "winner": winner,
                "reused_identical_fit_records": reused,
                "max_trA": max(
                    (x["max_trA"] for x in envelopes if x["max_trA"] is not None),
                    default=None,
                ),
                "results": results,
            },
        )
    except Exception as exc:  # record-don't-fail
        write_json(out, exception_record(case, exc))


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_pr1b() -> None:
    case = "pr1b_fenep_self_fit"
    out = OUT / "clean_battery" / f"{case}.json"
    try:
        ck = adapter.load_tbnn_checkpoint(CLEAN["fene_p"])
        a = args_record()
        key = jax.random.PRNGKey(0)
        data, _, _ = battery._gen_tbnn_data(ck, data_solver(), a, key)
        makers = {
            "FENEP_truth_init": lambda: FENEP(
                polymer_viscosity=LogParameter(TRUTH["eta_p"]),
                relaxation_time=LogParameter(TRUTH["lam"]),
                solvent_viscosity=LogParameter(TRUTH["nu_s"]),
                extension_length=LogParameter(np.sqrt(TRUTH["Lsq"])),
            ),
            "FENEP_agnostic_init": battery._candidate_models()["FENEP"],
            "Newtonian_standard_init": battery._candidate_models()["Newtonian"],
        }
        results = []
        for name, maker in makers.items():
            r = battery._fit_one(name, maker, data, fit_solver(), a, key)
            results.append(serializable_fit(r))
        truth_fit = next(r for r in results if r["name"] == "FENEP_truth_init")
        drift = {}
        truth_params = {
            "polymer_viscosity": TRUTH["eta_p"],
            "relaxation_time": TRUTH["lam"],
            "solvent_viscosity": TRUTH["nu_s"],
            "extension_length": np.sqrt(TRUTH["Lsq"]),
        }
        for name, value in truth_params.items():
            fitted = truth_fit["params"].get(name)
            drift[name] = {
                "truth": value,
                "fit": fitted,
                "relative_drift": (
                    abs(fitted - value) / abs(value) if fitted is not None else None
                ),
            }
        write_json(
            out,
            {
                "case": case, "status": "ok", "results": results,
                "truth_init_parameter_drift": drift,
            },
        )
    except Exception as exc:  # record-don't-fail
        write_json(out, exception_record(case, exc))


def _linear_leg(ck: dict, omega: float) -> tuple[float, float]:
    """Return last-two-cycle in-phase and quadrature total viscosities."""
    f = 0.01
    period = 2.0 * np.pi / omega
    ts = jnp.linspace(0.0, 4.0 * period, 4 * 20 + 1)
    vg = VelocityGradient.from_components(
        grad_u_12=lambda t, f=f, w=omega: f * jnp.sin(w * t)
    )
    sol = data_solver().integrate(
        adapter.make_tbnn_rhs(ck["theta"], ck["lam"]),
        adapter.A_REST, ts, vg
    )
    A = np.asarray(sol.ys)
    K, _, _ = adapter.tbnn_K_and_frozen(
        A[:, 0], A[:, 1], A[:, 2], A[:, 3], ck["theta"]
    )
    t = np.asarray(ts)
    gd = f * np.sin(omega * t)
    sigma = ck["Gp"] * np.asarray(K[1]) + ck["nu_s"] * gd
    use = t >= (2.0 * period - 1e-12)
    X = np.column_stack(
        [np.sin(omega * t[use]), np.cos(omega * t[use]), np.ones(use.sum())]
    )
    coef, _, _, _ = np.linalg.lstsq(X, sigma[use], rcond=None)
    return float(coef[0] / f), float(coef[1] / f)


def _fit_ob_response(in_phase: np.ndarray, quadrature: np.ndarray) -> dict:
    omega = np.asarray(W_GRID)

    def residual(logp: np.ndarray) -> np.ndarray:
        eta_p, lam, nu_s = np.exp(logp)
        x = omega * lam
        pred_in = nu_s + eta_p / (1.0 + x * x)
        pred_quad = -eta_p * x / (1.0 + x * x)
        return np.concatenate([pred_in - in_phase, pred_quad - quadrature])

    fit = least_squares(
        residual, np.log([TRUTH["eta_p"], TRUTH["lam"], TRUTH["nu_s"]]),
        method="trf"
    )
    eta_p, lam, nu_s = np.exp(fit.x)
    return {
        "eta_p_eff": float(eta_p),
        "lam_eff": float(lam),
        "nu_s_eff": float(nu_s),
        "cost": float(fit.cost),
        "optimality": float(fit.optimality),
        "success": bool(fit.success),
        "message": fit.message,
    }


def _linearized_response(ck: dict) -> tuple[np.ndarray, np.ndarray]:
    """Frequency response from the exact Jacobian of the deployed closure at I."""
    y0 = adapter.A_REST

    def relax(y):
        return jnp.stack(
            adapter.tbnn_source_R(
                y[0], y[1], y[2], y[3], ck["lam"], ck["theta"]
            )
        )

    def kxy(y):
        return adapter.tbnn_K_and_frozen(
            y[0], y[1], y[2], y[3], ck["theta"]
        )[0][1]

    jac_r = np.asarray(jax.jacfwd(relax)(y0), dtype=np.float64)
    cvec = ck["Gp"] * np.asarray(jax.jacfwd(kxy)(y0), dtype=np.float64)
    # dA_xy/dt|stretch = gammadot*A_yy, hence B=[0,1,0,0] at I.
    bvec = np.array([0.0, 1.0, 0.0, 0.0])
    response = []
    for omega in W_GRID:
        h = cvec @ np.linalg.solve(
            1j * omega * np.eye(4) - jac_r, bvec
        ) + ck["nu_s"]
        response.append(h)
    return np.real(response), np.imag(response)


def run_learned() -> None:
    rows = []
    for rid, (run_name, run_dir) in RUNS.items():
        rec = {"run_id": rid, "run": run_name, "status": "pending"}
        try:
            if not (run_dir / "DONE").is_file():
                rec.update(status="skipped_not_complete")
                rows.append(rec)
                continue
            ck_path = run_dir / "theta_checkpoint.npz"
            ck = adapter.load_tbnn_checkpoint(ck_path)
            envelopes = interrogate_envelopes(ck, rid)
            X0 = jnp.zeros((1, 3), dtype=jnp.float64)
            heads = adapter.tbnn_heads(ck["theta"], X0)
            in_phase, quadrature = zip(*[_linear_leg(ck, w) for w in W_GRID])
            ob = _fit_ob_response(np.asarray(in_phase), np.asarray(quadrature))
            lin_in, lin_quad = _linearized_response(ck)
            ob_lin = _fit_ob_response(lin_in, lin_quad)
            rec.update(
                status="ok",
                checkpoint=str(ck_path),
                Gp=ck["Gp"], lam=ck["lam"], nu_s=ck["nu_s"],
                eta_p=ck["Gp"] * ck["lam"],
                m0_I=float(np.asarray(heads[3])[0]),
                m1_I=float(np.asarray(heads[4])[0]),
                linear_in_phase=list(in_phase),
                linear_quadrature=list(quadrature),
                jacobian_in_phase=lin_in.tolist(),
                jacobian_quadrature=lin_quad.tolist(),
                max_trA=max(
                    (x["max_trA"] for x in envelopes if x["max_trA"] is not None),
                    default=None,
                ),
                linear_response_method="analytic Jacobian at A=I",
                eta_p_eff=ob_lin["eta_p_eff"],
                lam_eff=ob_lin["lam_eff"],
                nu_s_eff=ob_lin["nu_s_eff"],
                jacobian_ob_cost=ob_lin["cost"],
                eta_p_eff_f001=ob["eta_p_eff"],
                lam_eff_f001=ob["lam_eff"],
                nu_s_eff_f001=ob["nu_s_eff"],
                f001_ob_cost=ob["cost"],
                f001_fit_success=ob["success"],
                f001_fit_message=ob["message"],
            )
        except Exception as exc:  # record-don't-fail
            rec.update(exception_record(rid, exc), run_id=rid, run=run_name)
        rows.append(rec)
        write_json(OUT / "learned" / f"{rid}.json", rec)
    write_json(OUT / "learned" / "p3_rows.json", rows)


def write_environment() -> None:
    import diffrax
    import equinox
    import optax
    import scipy

    write_json(
        OUT / "environment.json",
        {
            "status": "ok",
            "python": sys.version,
            "platform": platform.platform(),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(x) for x in jax.devices()],
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "versions": {
                "jax": jax.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "diffrax": diffrax.__version__,
                "equinox": equinox.__version__,
                "optax": optax.__version__,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "clean", "clean-candidate", "clean-merge",
            "pr1b", "learned", "environment",
        ],
    )
    parser.add_argument("--model", choices=sorted(CLEAN))
    parser.add_argument("--candidate", choices=list(battery._candidate_models()))
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(parser)
    OUT.mkdir(parents=True, exist_ok=True)
    assert_cpu_x64()
    if args.mode == "clean":
        if args.model is None:
            parser.error("--model is required for clean mode")
        run_clean(args.model)
    elif args.mode == "clean-candidate":
        if args.model is None or args.candidate is None:
            parser.error("--model and --candidate are required")
        run_clean_candidate(args.model, args.candidate)
    elif args.mode == "clean-merge":
        if args.model is None:
            parser.error("--model is required")
        merge_clean(args.model)
    elif args.mode == "pr1b":
        run_pr1b()
    elif args.mode == "learned":
        run_learned()
    else:
        write_environment()


if __name__ == "__main__":
    main()
