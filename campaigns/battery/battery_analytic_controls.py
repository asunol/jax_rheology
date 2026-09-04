#!/usr/bin/env python
"""Read-only diagnosis of diff_rheo candidate conventions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

if os.environ.get("JAX_PLATFORMS") != "cpu":
    raise RuntimeError("JAX_PLATFORMS=cpu required before imports")

from repo_paths import DIFF_RHEO, insert_diff_rheo
insert_diff_rheo()

IMPORT_START = time.perf_counter()
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax

import diff_rheo as dr
from diff_rheo._core import data_fitting_loss
from diff_rheo._data_types import ShearStrainRateData
from diff_rheo._forcing import VelocityGradient
from diff_rheo.models import FENEP, Giesekus, LinearPTT, Newtonian, OldroydB
from diff_rheo.parameters import LogParameter

import battery_instrument as instrument
import tbnn_bic_model_selection as battery

IMPORT_SECONDS = time.perf_counter() - IMPORT_START
ROOT = Path(__file__).resolve().parent
DIFF_TREE = DIFF_RHEO
OUT = ROOT.parents[1] / "work/bic_analytic_controls"
BATTERIES = OUT / "batteries"
FITS = OUT / "fits"
TRUTH = {
    "Gp": 3.2, "lam": 0.7, "nu_s": 0.8, "eta_p": 2.24,
    "alpha": 0.3, "Lsq": 12.0, "epsilon": 0.25,
}
FAMILIES = ("Newtonian", "OldroydB", "Giesekus", "FENEP", "LinearPTT")


def assert_cpu_x64():
    if jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"):
        raise RuntimeError("CPU float64 assertion failed")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float) + "\n")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(DIFF_TREE), *args], text=True
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def environment():
    import diff_rheo._core as core
    import diff_rheo.models._viscoelastic as ve

    workspace = {
        "_core.py": DIFF_TREE / "src/diff_rheo/_core.py",
        "_viscoelastic.py": DIFF_TREE / "src/diff_rheo/models/_viscoelastic.py",
    }
    installed = {
        "_core.py": Path(core.__file__).resolve(),
        "_viscoelastic.py": Path(ve.__file__).resolve(),
    }
    rows = []
    for key in workspace:
        rows.append(
            {
                "source": key,
                "workspace_path": str(workspace[key].resolve()),
                "workspace_sha256": sha256(workspace[key]),
                "installed_path": str(installed[key]),
                "installed_sha256": sha256(installed[key]),
                "same_file": workspace[key].resolve() == installed[key],
                "same_hash": sha256(workspace[key]) == sha256(installed[key]),
            }
        )
    accelerator_packages = {}
    for name in (
        "jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda11-plugin",
        "jax-cuda11-pjrt", "jax-metal",
    ):
        try:
            accelerator_packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            accelerator_packages[name] = None
    write_json(
        OUT / "environment.json",
        {
            "status": "ok", "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "diffrax": importlib.metadata.version("diffrax"),
            "equinox": importlib.metadata.version("equinox"),
            "optax": importlib.metadata.version("optax"),
            "diff_rheo": importlib.metadata.version("diff-rheo"),
            "diff_rheo_package_file": str(Path(dr.__file__).resolve()),
            "backend": jax.default_backend(),
            "devices": [str(x) for x in jax.devices()],
            "accelerator_packages": accelerator_packages,
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "jax_enable_x64": jax.config.read("jax_enable_x64"),
            "imports_seconds": IMPORT_SECONDS,
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_origin": git_value("config", "--get", "remote.origin.url"),
            "package_metadata": dict(importlib.metadata.metadata("diff-rheo")),
            "source_comparison": rows,
        },
    )


def leg_axes():
    for omega in battery.W_GRID:
        period = 2.0 * np.pi / omega
        ts = jnp.linspace(0.0, 4.0 * period, 81)
        for amp in battery.F_GRID:
            yield float(omega), float(amp), ts


def stretch(gd, y):
    axx, axy, ayy, azz = y
    return jnp.asarray([2 * gd * axy, gd * ayy, 0.0, 0.0])


def solver_rhs(family):
    def rhs(t, y, vg):
        gd = vg.gradient(t)[0, 1]
        axx, axy, ayy, azz = y
        B = jnp.asarray([axx - 1, axy, ayy - 1, azz - 1])
        if family == "OldroydB":
            G = B
        elif family == "Giesekus":
            bxx, bxy, byy, bzz = B
            BB = jnp.asarray(
                [
                    bxx * bxx + bxy * bxy, bxy * (bxx + byy),
                    bxy * bxy + byy * byy, bzz * bzz,
                ]
            )
            G = B + TRUTH["alpha"] * BB
        elif family == "FENEP":
            tr_a = axx + ayy + azz
            d = TRUTH["Lsq"] - tr_a
            dfl = 1e-3 * TRUTH["Lsq"]
            deff = dfl + dfl * jax.nn.softplus((d - dfl) / dfl)
            f = TRUTH["Lsq"] / deff
            a = TRUTH["Lsq"] / (TRUTH["Lsq"] - 3)
            G = jnp.asarray([f * axx - a, f * axy, f * ayy - a, f * azz - a])
        elif family == "LinearPTT":
            factor = 1 + TRUTH["epsilon"] * (axx + ayy + azz - 3)
            G = factor * B
        else:
            raise ValueError(family)
        return stretch(gd, y) - G / TRUTH["lam"]

    return rhs


def stress_from_A(family, A, gd):
    if family == "FENEP":
        tr_a = A[:, 0] + A[:, 2] + A[:, 3]
        d = TRUTH["Lsq"] - tr_a
        dfl = 1e-3 * TRUTH["Lsq"]
        deff = dfl + dfl * np.logaddexp(0, (d - dfl) / dfl)
        kxy = TRUTH["Lsq"] / deff * A[:, 1]
    else:
        kxy = A[:, 1]
    return TRUTH["Gp"] * kxy + TRUTH["nu_s"] * gd


def generate_family(family):
    started = time.perf_counter()
    rng = jax.random.PRNGKey(0)
    arrays = {k: [] for k in ("omega", "amp", "time", "gammadot",
                                      "sigma_clean", "noise", "sigma_noisy")}
    solver = instrument.data_solver()
    for omega, amp, ts in leg_axes():
        gd = np.asarray(amp * jnp.sin(omega * ts))
        if family == "Newtonian":
            sigma = TRUTH["nu_s"] * gd
        else:
            vg = VelocityGradient.from_components(
                grad_u_12=lambda t, f=amp, w=omega: f * jnp.sin(w * t)
            )
            sol = solver.integrate(
                solver_rhs(family), jnp.asarray([1.0, 0.0, 1.0, 1.0]), ts, vg
            )
            sigma = stress_from_A(family, np.asarray(sol.ys), gd)
        rng, sub = jax.random.split(rng)
        noise = 0.03 * np.asarray(jax.random.normal(sub, sigma.shape))
        for key, value in (
            ("omega", np.full(81, omega)), ("amp", np.full(81, amp)),
            ("time", np.asarray(ts)), ("gammadot", gd),
            ("sigma_clean", sigma), ("noise", noise),
            ("sigma_noisy", sigma + noise),
        ):
            arrays[key].append(value)
    result = {k: np.stack(v) for k, v in arrays.items()}
    BATTERIES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BATTERIES / f"{family}.npz", **result)
    write_json(
        BATTERIES / f"{family}_summary.json",
        {
            "status": "ok", "family": family,
            "generation_seconds": time.perf_counter() - started,
            "imports_seconds": IMPORT_SECONDS,
            "noise_mse": float(np.mean(result["noise"] ** 2)),
            "shape": list(result["sigma_noisy"].shape),
        },
    )


def as_data(z):
    refs = []
    for i in range(z["time"].shape[0]):
        refs.append(
            ShearStrainRateData(
                time=jnp.asarray(z["time"][i]),
                data=jnp.asarray(z["sigma_noisy"][i]),
                forcing_data=jnp.asarray(z["gammadot"][i]),
                initial_condition=jnp.zeros((3, 3), dtype=jnp.float64),
            )
        )
    return dr.BatchedData.from_data(*refs)


def truth_model(family):
    if family == "Newtonian":
        return Newtonian(viscosity=LogParameter(0.8))
    shared = {
        "polymer_viscosity": LogParameter(2.24),
        "relaxation_time": LogParameter(0.7),
        "solvent_viscosity": LogParameter(0.8),
    }
    if family == "OldroydB":
        return OldroydB(**shared)
    if family == "Giesekus":
        return Giesekus(**shared, alpha=LogParameter(0.3))
    if family == "FENEP":
        return FENEP(**shared, extension_length=LogParameter(np.sqrt(12.0)))
    if family == "LinearPTT":
        return LinearPTT(
            **shared, epsilon=LogParameter(0.25), zeta=LogParameter(1e-12)
        )
    raise ValueError(family)


def pooled_metrics(model, rheo, data):
    sse, n = 0.0, 0
    for ref in data.data:
        sim = rheo.run_experiment(
            model, ref.get_forcing_function(), ref.time, ref.initial_condition
        )
        pred = np.asarray(ref.extract_from_simulation(sim))
        target = np.asarray(ref.data)
        sse += float(np.sum((pred - target) ** 2))
        n += pred.size
    mse = sse / n
    k = len(
        [x for x in jax.tree_util.tree_leaves(model) if eqx.is_inexact_array(x)]
    )
    return {
        "MSE": mse, "k": k,
        "BIC": n * (np.log(2 * np.pi * max(mse, np.finfo(float).tiny)) + 1)
        + k * np.log(n),
        "params": {key: float(value) for key, value in model.parameter_values.items()},
    }


def fit_family(family):
    started = time.perf_counter()
    with np.load(BATTERIES / f"{family}.npz", allow_pickle=False) as z:
        noise_mse = float(np.mean(z["noise"] ** 2))
        data = as_data(z)
    load_seconds = time.perf_counter() - started
    model = truth_model(family)
    rheo = dr.VirtualRheometer.setup(
        model, "strain_rate_response", instrument.fit_solver()
    )
    initial = pooled_metrics(model, rheo, data)
    optimizer = optax.adam(0.1)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    progress, divergence = [], None
    fit_started = time.perf_counter()
    for epoch in range(500):
        t0 = time.perf_counter()
        (loss, _), grads = eqx.filter_value_and_grad(
            data_fitting_loss, has_aux=True
        )(model, rheo, data.fitting_schedule(instrument.args_record(), epoch),
          ensemble_size=1)
        grad_leaves = [
            x for x in jax.tree_util.tree_leaves(grads) if eqx.is_inexact_array(x)
        ]
        grad_norm = jnp.sqrt(sum(jnp.sum(x * x) for x in grad_leaves))
        jax.block_until_ready(loss)
        jax.block_until_ready(grad_norm)
        if not np.isfinite(float(loss)) or not np.isfinite(float(grad_norm)):
            divergence = {
                "epoch": epoch + 1, "loss": float(loss),
                "grad_norm": float(grad_norm),
            }
            break
        updates, state = optimizer.update(grads, state, model)
        model = eqx.apply_updates(model, updates)
        if epoch == 0 or (epoch + 1) % 25 == 0:
            rec = {
                "epoch": epoch + 1, "loss": float(loss),
                "grad_norm": float(grad_norm),
                "epoch_seconds": time.perf_counter() - t0,
            }
            progress.append(rec)
            print(json.dumps(rec), flush=True)
    fit_seconds = time.perf_counter() - fit_started
    final = (
        pooled_metrics(model, rheo, data) if divergence is None
        else {"MSE": math.inf, "BIC": math.inf, "params": {}}
    )
    write_json(
        FITS / f"{family}.json",
        {
            "status": "ok" if divergence is None else "failed",
            "family": family, "truth": TRUTH, "noise_mse": noise_mse,
            "threshold": 2 * noise_mse, "initial": initial, "post_fit": final,
            "floor_hit_initial": initial["MSE"] <= 2 * noise_mse,
            "floor_hit_post_fit": final["MSE"] <= 2 * noise_mse,
            "divergence": divergence, "progress": progress,
            "timing": {
                "imports_seconds": IMPORT_SECONDS, "load_seconds": load_seconds,
                "fit_seconds": fit_seconds,
                "script_seconds": time.perf_counter() - started,
            },
        },
    )


def native_fenep_trajectory(vg, ts):
    model = truth_model("FENEP")
    rheo = dr.VirtualRheometer.setup(
        model, "strain_rate_response", instrument.data_solver()
    )
    ref = ShearStrainRateData(
        time=ts, data=jnp.zeros_like(ts), forcing_data=jnp.zeros_like(ts),
        initial_condition=jnp.zeros((3, 3), dtype=jnp.float64),
    )
    sim = rheo.run_experiment(
        model, vg, ref.time, ref.initial_condition
    )
    return np.asarray(ref.extract_from_simulation(sim))


def stress_peterlin_rhs(t, y, vg, L):
    eta, lam = TRUTH["eta_p"], TRUTH["lam"]
    L2 = L * L
    a = L2 / (L2 - 3)
    tau = jnp.asarray([[y[0], y[1], 0.0], [y[1], y[2], 0.0], [0.0, 0.0, y[3]]])
    grad = vg.gradient(t)
    g = grad + grad.T
    f = (L2 + lam / (a * eta) * jnp.trace(tau)) / (L2 - 3)
    H = tau @ grad + grad.T @ tau - f / lam * tau + f * eta / lam * g
    c1 = lam / (a * eta * (L2 - 3))
    beta = c1 / (lam * f)
    Q = lam * tau + a * eta * jnp.eye(3)
    trdot = jnp.trace(H) / (1 - beta * jnp.trace(Q))
    dot = H + beta * Q * trdot
    return jnp.asarray([dot[0, 0], dot[0, 1], dot[1, 1], dot[2, 2]])


def stress_peterlin_rhs_matrix(t, tau, vg, L):
    vec = jnp.asarray([tau[0, 0], tau[0, 1], tau[1, 1], tau[2, 2]])
    dot = stress_peterlin_rhs(t, vec, vg, L)
    return jnp.asarray(
        [[dot[0], dot[1], 0.0], [dot[1], dot[2], 0.0], [0.0, 0.0, dot[3]]]
    )


def hypothesis_rhs(kind, L2):
    def rhs(t, y, vg):
        gd = vg.gradient(t)[0, 1]
        axx, axy, ayy, azz = y
        tr_a = axx + ayy + azz
        f = L2 / (L2 - tr_a)
        a = L2 / (L2 - 3)
        B = jnp.asarray([axx - 1, axy, ayy - 1, azz - 1])
        FA = jnp.asarray([f * axx - a, f * axy, f * ayy - a, f * azz - a])
        if kind == "a_equals_1":
            G = jnp.asarray([f * axx - 1, f * axy, f * ayy - 1, f * azz - 1])
        elif kind == "relax_f_stress_hookean":
            G = FA
        elif kind == "relax_hookean_stress_f":
            G = B
        elif kind == "fene_cr":
            G = f * B
        elif kind == "b_parameter":
            b = L2 - 3
            fb = b / (b + 3 - tr_a)
            G = jnp.asarray([fb * axx - 1, fb * axy, fb * ayy - 1, fb * azz - 1])
        else:
            G = FA
        return stretch(gd, y) - G / TRUTH["lam"]

    return rhs


def hypothesis_stress(kind, A, gd, L2):
    tr_a = A[:, 0] + A[:, 2] + A[:, 3]
    f = L2 / (L2 - tr_a)
    a = L2 / (L2 - 3)
    if kind in ("relax_f_stress_hookean", "fene_cr"):
        kxy = A[:, 1] if kind == "relax_f_stress_hookean" else f * A[:, 1]
    elif kind == "relax_hookean_stress_f":
        kxy = f * A[:, 1]
    elif kind == "b_parameter":
        b = L2 - 3
        kxy = b / (b + 3 - tr_a) * A[:, 1]
    else:
        kxy = f * A[:, 1]
    return TRUTH["Gp"] * kxy + TRUTH["nu_s"] * gd


def reproduce_fenep():
    solver = instrument.data_solver()
    variants = (
        ("exact_stress_peterlin", None, np.sqrt(12.0)),
        ("solver_convention", "solver", 12.0),
        ("a_equals_1", "a_equals_1", 12.0),
        ("extension_as_Lsq", "solver", 144.0),
        ("relax_f_stress_hookean", "relax_f_stress_hookean", 12.0),
        ("relax_hookean_stress_f", "relax_hookean_stress_f", 12.0),
        ("fene_cr", "fene_cr", 12.0),
        ("b_parameter", "b_parameter", 12.0),
    )
    rows = []
    pointwise = []
    native_model = truth_model("FENEP")
    native_instance = native_model.get_instance()
    for leg, (omega, amp, ts) in enumerate(leg_axes()):
        vg = VelocityGradient.from_components(
            grad_u_12=lambda t, f=amp, w=omega: f * jnp.sin(w * t)
        )
        native = native_fenep_trajectory(vg, ts)
        gd = np.asarray(amp * jnp.sin(omega * ts))
        for name, kind, extent in variants:
            if name == "exact_stress_peterlin":
                sol = solver.integrate(
                    lambda t, y, forcing: stress_peterlin_rhs_matrix(
                        t, y, forcing, extent
                    ),
                    jnp.zeros((3, 3)), ts, vg,
                )
                pred = np.asarray(sol.ys)[:, 0, 1] + TRUTH["nu_s"] * gd
            else:
                sol = solver.integrate(
                    hypothesis_rhs(kind, extent),
                    jnp.asarray([1.0, 0.0, 1.0, 1.0]), ts, vg,
                )
                pred = hypothesis_stress(kind, np.asarray(sol.ys), gd, extent)
            err = pred - native
            rows.append(
                {
                    "leg": leg, "omega": omega, "amp": amp, "variant": name,
                    "max_abs_error": float(np.max(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err * err))),
                }
            )
        for sample, t in enumerate(np.linspace(0.0, 1.0, 11)):
            y = jnp.asarray(
                [0.2 + .03 * sample, -0.1 + .01 * sample,
                 0.15 - .005 * sample, 0.05 + .002 * sample]
            )
            native_rhs = native_instance.extra_stress_response_rhs(
                t, jnp.asarray(
                    [[y[0], y[1], 0.0], [y[1], y[2], 0.0], [0.0, 0.0, y[3]]]
                ), vg
            )
            native_vec = jnp.asarray(
                [native_rhs[0, 0], native_rhs[0, 1],
                 native_rhs[1, 1], native_rhs[2, 2]]
            )
            derived = stress_peterlin_rhs(t, y, vg, np.sqrt(12.0))
            pointwise.append(
                {
                    "leg": leg, "sample": sample,
                    "max_abs_rhs_error": float(
                        np.max(np.abs(np.asarray(native_vec - derived)))
                    ),
                }
            )
    fields = list(rows[0])
    with (OUT / "d3_reproduction.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "d3_rhs_identity.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(pointwise[0]))
        writer.writeheader()
        writer.writerows(pointwise)
    summary = []
    for name, _, _ in variants:
        subset = [r for r in rows if r["variant"] == name]
        summary.append(
            {
                "variant": name,
                "max_abs_error_all_legs": max(r["max_abs_error"] for r in subset),
                "pooled_rmse": float(
                    np.sqrt(np.mean([r["rmse"] ** 2 for r in subset]))
                ),
                "machine_precision_match": max(
                    r["max_abs_error"] for r in subset
                ) < 1e-9,
            }
        )
    write_json(
        OUT / "d3_summary.json",
        {
            "status": "ok", "variants": summary,
            "max_pointwise_rhs_error": max(
                r["max_abs_rhs_error"] for r in pointwise
            ),
            "identity": (
                "stress-form Peterlin FENE-P with f determined by tr(tau) "
                "and the material derivative of 1/f retained"
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["environment", "d3", "prepare", "fit"])
    parser.add_argument("--family", choices=FAMILIES)
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(parser)
    assert_cpu_x64()
    try:
        if args.mode == "environment":
            environment()
        elif args.mode == "d3":
            reproduce_fenep()
        elif args.mode == "prepare":
            if args.family is None:
                parser.error("--family required")
            generate_family(args.family)
        else:
            if args.family is None:
                parser.error("--family required")
            fit_family(args.family)
    except Exception as exc:
        tag = args.mode + (f"_{args.family}" if args.family else "")
        write_json(
            OUT / f"error_{tag}.json",
            {"status": "error", "error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
