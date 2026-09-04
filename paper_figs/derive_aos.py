#!/usr/bin/env python
"""Forward-evaluate the BIC battery candidates on their own AOS protocol.

The battery stores only the fitted parameters, not the predicted stress trace,
so panel SN2d has to replay each candidate once.  This is a pure forward
replay: parameters are read frozen from the target rollup and no optimiser is
constructed.  As a check, the pooled MSE and BIC of the replay are compared
against the values the battery recorded.

diff_rheo lives in its own conda environment, so this runs as a subprocess
under that interpreter rather than being imported by ``loaders``; the result is
cached as an npz and everything downstream reads the cache.

    JAX_PLATFORMS=cpu <diff_rheo python> paper_figs/derive_aos.py \
        --target gie_A_s4 --out paper_figs_derived/aos_gie_A_s4.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from repo_paths import bootstrap  # noqa: E402
bootstrap()

import numpy as np  # noqa: E402

import battery_instrument as instrument  # noqa: E402
import tbnn_bic_final_battery as battery  # noqa: E402


def replay(target: str) -> dict[str, np.ndarray]:
    data = battery.as_data(battery.DATA / f"{target}.npz")
    rollup = json.load(open(battery.OUT / "targets" / f"{target}.json"))
    with np.load(battery.DATA / f"{target}.npz") as raw:
        out = {k: raw[k] for k in ("time", "sigma_noisy", "gammadot")}

    checks = {}
    for row in rollup["results"]:
        name = row["name"]
        if not row["ok"]:
            continue
        model = battery.make_model(name, row["params"])
        rheometer = battery.make_rheometer(
            name, model, instrument.fit_solver())
        pred = [
            np.asarray(ref.extract_from_simulation(
                rheometer.run_experiment(model, ref.get_forcing_function(),
                                         ref.time, ref.initial_condition)))
            for ref in data.data
        ]
        out[f"pred_{name}"] = np.stack(pred)
        mse, bic = battery.pooled_metrics(model, rheometer, data)
        checks[name] = {"mse": mse, "mse_stored": row["mse"],
                        "bic": bic, "bic_stored": row["bic"]}
        print(f"{name:20s} mse {mse:.8e} (stored {row['mse']:.8e})",
              flush=True)
    out["checks_json"] = np.array(json.dumps(checks))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(ap)
    arrays = replay(args.target)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
