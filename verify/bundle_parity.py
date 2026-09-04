"""Regenerate the paper figures from the data bundle and check them.

The viscoelastic and elastoviscoplastic figures must reproduce the published
bytes, so each regenerated file is compared to its recorded sha256 in
``verify/figure_hashes.json``, including both supporting tables. The
instantaneous figures are float32 and are not a hash set; they are checked by
loading the deposited helper fields, summaries, and checkpoints.

Three of the notes files record the data paths they were built from, so they
are compared after normalizing those lines away.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DST = Path(__file__).resolve().parents[1]
OUT = DST / "verify_data" / "bundle_parity"


def bundle_root() -> Path:
    """The unpacked Zenodo deposit."""
    for name in ("DATA_ROOT", "TBNN_DATA_BUNDLE"):
        raw = os.environ.get(name)
        if raw and Path(raw).expanduser().is_dir():
            return Path(raw).expanduser()
    local = DST / "data_bundle"
    if local.is_dir():
        return local
    raise RuntimeError(
        "no data bundle found; set DATA_ROOT to the unpacked Zenodo deposit"
    )


def instantaneous_helper(bundle: Path) -> dict:
    fig2 = bundle / "fig2_instantaneous_demo" / "cy_n0p7_lam5"
    import sys
    camp = str(DST / "campaigns")
    if camp not in sys.path:
        sys.path.insert(0, camp)
    from plot_tbnn_training_results import load_trajectory_data
    data = load_trajectory_data(str(fig2))
    need = ("reference_velocity_x", "reference_velocity_y",
            "updated_tbnn_velocity_x", "updated_tbnn_velocity_y")
    missing_fields = [k for k in need if k not in data]
    pkls = sorted(p.relative_to(bundle).as_posix()
                  for p in bundle.rglob("trained_params.pkl"))
    summaries = sorted(p.relative_to(bundle).as_posix()
                       for p in bundle.rglob("iteration_summary.txt"))
    fig3 = bundle / "fig3_porous_transfer" / "porous_production_fields.npz"
    return {
        "fig2_loaded": not missing_fields,
        "missing_fields": missing_fields,
        "n_trained_params": len(pkls),
        "n_summaries": len(summaries),
        "fig3_exists": fig3.is_file(),
        "ok": (not missing_fields) and len(pkls) >= 7 and fig3.is_file(),
        "note": "field and checkpoint load, not a figure hash set",
    }


def run(check: dict, *, run_cmd, sha256_file, env_common, py: str,
        compare_to_recorded, load_recorded) -> dict:
    bundle = bundle_root()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_out = OUT / "final_figures"
    if fig_out.exists():
        import shutil
        shutil.rmtree(fig_out)
    env = env_common()
    env["JAX_PLATFORMS"] = "cpu"
    env["DATA_ROOT"] = str(bundle)
    env["TBNN_DATA_BUNDLE"] = str(bundle)
    env["TBNN_FIG_OUT"] = str(fig_out)
    run_cmd(
        [py, "-u", str(DST / "experiments" / "figures.py"),
         "--config", str(DST / "experiments" / "configs" / "figures.yaml")],
        env=env,
        log_path=OUT / "figures.log",
    )
    mem = compare_to_recorded(
        fig_out, load_recorded(), check["expected"]["table_paths"]
    )
    mem["ok"] = (
        mem["identical"] == int(check["expected"]["identical"])
        and mem["tables_identical"] == int(check["expected"]["tables_identical"])
    )
    inst = instantaneous_helper(bundle)
    (OUT / "parity.json").write_text(
        json.dumps({"memory": mem, "instantaneous": inst}, indent=2, default=str) + "\n"
    )
    ok = bool(mem["ok"] and inst["ok"])
    return {
        "id": "bundle_parity",
        "got": {
            "identical": mem["identical"],
            "tables_identical": mem["tables_identical"],
            "instantaneous_ok": inst["ok"],
            "n_trained_params": inst["n_trained_params"],
        },
        "pass": ok,
        "comparison": "recorded-hashes",
        "expected": {
            "identical": check["expected"]["identical"],
            "tables_identical": check["expected"]["tables_identical"],
            "instantaneous": "visual/helper",
        },
        "note": check["expected"].get("note"),
    }
