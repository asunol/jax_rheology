#!/usr/bin/env python
"""Read a trained instantaneous closure back out as a classical model.

Produces Fig. 4 and the Table S1 row for one run: the closure is probed
under oscillatory shear, then Carreau-Yasuda and Newtonian models are
fitted to the response and compared by BIC. Fitting happens in
``tbnn_model_selection/model_selection_tbnn.py``; the trained checkpoint
is read only.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._dispatch import apply_precision, bootstrap, peek_config

# Frozen paper Table S1 row 1 (printed to 6 decimals).
_PAPER_R1 = {
    "n_gt": 0.7,
    "n_learned": 0.700706,
    "lam_gt": 5.0,
    "lam_learned": 4.941240,
}

# The model-selection child runs in the rheometry environment
# (environment_diff_rheo.yml); TBNN_PYDR names its interpreter.
_PYDR = os.environ.get("TBNN_PYDR")


_INST_RUNS = {
    "iteration_12_20251008_050525": "fig2_instantaneous_demo/cy_n0p7_lam5",
    "cy_param_01_20251011_032016": "fig4_table_s1_cy_distillation/cy_n0p6_lam3",
    "cy_param_04_20251011_032025": "fig4_table_s1_cy_distillation/cy_n0p6_lam1_centers_frozen",
    "cy_param_06_20251011_032025": "fig4_table_s1_cy_distillation/cy_n0p6_lam7_centers_frozen",
    "cy_param_07_20251011_032036": "fig4_table_s1_cy_distillation/cy_n0p8_lam5",
    "cy_param_09_20251011_032025": "fig4_table_s1_cy_distillation/cy_n0p8_lam10",
    "cy_param_11_20251011_032025": "fig4_table_s1_cy_distillation/cy_n0p8_lam15",
}


def _resolve_data_root(raw: str | None) -> Path | None:
    if not raw:
        return None
    if raw in ("bundle", "data_bundle"):
        from repo_paths import REPO_ROOT
        return REPO_ROOT / "data_bundle"
    return Path(raw)


def _frozen_run_dir(data: dict, data_root: Path | None = None) -> Path:
    folder = data.get("frozen_dir", "iteration_12_20251008_050525")
    if data_root is not None:
        rel = _INST_RUNS.get(folder)
        if rel:
            return data_root / rel
        return data_root / folder
    from repo_paths import FROZEN_INST
    return FROZEN_INST / "tbnn_debug_results_constriction_new" / folder


def _params_pkl(run_dir: Path) -> Path:
    for p in (
        run_dir / "trajectory_data" / "final_tbnn_params.pkl",
        run_dir / "trained_params.pkl",
        run_dir / "final_tbnn_params.pkl",
    ):
        if p.is_file():
            return p
    return run_dir / "trajectory_data" / "final_tbnn_params.pkl"


def _distill(data: dict, run_dir: Path) -> dict:
    from fit_tbnn_to_viscosity import fit_tbnn_to_viscosity

    pkl = _params_pkl(run_dir)
    with open(pkl, "rb") as f:
        saved = pickle.load(f)
    arch = data.get("architecture", [16])
    res = fit_tbnn_to_viscosity(
        hidden_units=arch,
        M=int(data.get("M", 12)),
        freeze_centers=bool(data.get("freeze_centers", False)),
        freeze_eta0=True,
        eta0_fixed=1.0,
        s_floor=float(data.get("s_floor", 0.35)),
        alpha_temp=float(data.get("alpha_temp", 0.8)),
        init_mode="from_saved",
        saved_params=saved,
        etainf_cy=float(data.get("ref_eta_inf", 0.02)),
        eta0_cy=float(data.get("ref_eta_0", 1.0)),
        lam_cy=float(data.get("ref_lambda", 5.0)),
        n_cy=float(data.get("ref_n", 0.7)),
        a_cy=float(data.get("ref_a", 2.0)),
        show_plots=False,
        num_steps=5,
    )
    return {
        "final_total_loss": float(res["loss_hist"][-1]),
        "data_loss": float(res["data_loss_hist"][-1]),
    }


def _bic(iteration_folder: str) -> dict:
    if not _PYDR:
        raise SystemExit(
            "the model-selection step runs in the rheometry environment. "
            "Set TBNN_PYDR to a Python from environment_diff_rheo.yml."
        )
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "True"
    env.setdefault("PYTHONPATH", f"{_ROOT}:{_ROOT}/jax-cfd:{_ROOT}/jax_ib")
    out_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "table_s1_bic"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "bic.log"
    with log.open("w") as fh:
        rc = subprocess.call(
            [_PYDR, "-u", str(_ROOT / "tbnn_model_selection" / "model_selection_tbnn.py"),
             iteration_folder],
            cwd=str(out_dir),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    text = log.read_text()
    rec = {"rc": rc, "log": str(log)}
    m_n = re.search(r"\(power index\):\s*([0-9.eE+-]+)", text)
    m_lam = re.search(r"\(time constant\):\s*([0-9.eE+-]+)", text)
    rec["fit_n"] = float(m_n.group(1)) if m_n else None
    rec["fit_lam"] = float(m_lam.group(1)) if m_lam else None
    return rec


def main():
    bootstrap()
    parser = argparse.ArgumentParser(
        description="Fig 4 / Table S1 distill (phase-3a + BIC row)")
    parser.add_argument("--config", required=True, metavar="YAML")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("TBNN_DATA_BUNDLE") or os.environ.get("DATA_ROOT"),
        help="Unpacked data_bundle/ (or the tokens bundle / data_bundle). "
             "BIC subprocess still uses the frozen iteration folder name.",
    )
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(parser)
    data = peek_config()
    apply_precision(data)
    run_dir = _frozen_run_dir(data, _resolve_data_root(args.data_root))
    if not _params_pkl(run_dir).is_file():
        raise SystemExit(f"missing checkpoint under {run_dir}")
    print(f"[distill] run_dir={run_dir} pkl={_params_pkl(run_dir)}", flush=True)

    d = _distill(data, run_dir)
    print(f"[distill] fit_tbnn_to_viscosity final_total_loss={d['final_total_loss']:.6e} "
          f"data_loss={d['data_loss']:.6e}", flush=True)

    b = _bic(data.get("frozen_dir", "iteration_12_20251008_050525"))
    n_gt = float(data.get("ref_n", _PAPER_R1["n_gt"]))
    lam_gt = float(data.get("ref_lambda", _PAPER_R1["lam_gt"]))
    n_learned = b["fit_n"]
    lam_learned = b["fit_lam"]
    print()
    print("run   n_gt    n_learned     lam_gt   lam_learned")
    print(f"R1    {n_gt}    {n_learned:.6f}     {lam_gt}     {lam_learned:.6f}")
    print()
    print(f"[recorded paper] n_learned={_PAPER_R1['n_learned']:.6f} "
          f"lam_learned={_PAPER_R1['lam_learned']:.6f}")
    if n_learned is not None and lam_learned is not None:
        # Compare at 6 printed decimals, matching the paper table.
        def same_printed(a, b, nd=6):
            return abs(float(a) - float(b)) <= 1.5 * 10 ** (-nd)
        ok_n = same_printed(n_learned, _PAPER_R1["n_learned"])
        ok_l = same_printed(lam_learned, _PAPER_R1["lam_learned"])
        print(f"[D6] n match={ok_n}  lam match={ok_l} (6-decimal printed precision)")
        if not (ok_n and ok_l):
            raise SystemExit(2)
    else:
        raise SystemExit(f"BIC parse failed rc={b['rc']} log={b['log']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
