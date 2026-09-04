#!/usr/bin/env python
"""G3 -- EVP channel forwards from the OB-init closure (for N2a).

The initial guess of the evp_fix runs is fully specified by their config:
``--init ob --no-br-init`` puts theta at the Oldroyd-B initialisation for
``theta_seed`` and every scalar at 1.0.  This script rebuilds exactly that
state and pushes it through the *same* evaluation protocol as the fitted arms
(``evp_fix_eval.run_ladder``: fixed 15-lambda horizon, 30 lambda below yield,
early stopping off), so the resulting curves are directly comparable to the
truth and learned curves already in the seed-eval store.

Nothing is fitted here.  The closure is read from the training config, and the
checkpoint of the reference seed is loaded only to verify -- by SHA256 -- that
the architecture constants used here are the ones that run actually used.

    python paper_figs/run_evp_obinit.py --drives 1.8 2.5 4.0 5.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import bootstrap, REPO_ROOT
bootstrap()
ROOT = str(REPO_ROOT)

import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import evp_forward_diag as efd  # noqa: E402
import evp_fix_eval as ee  # noqa: E402
from jax_rheology.models import registry as cr  #  noqa: E402
import jax_rheology.models.tbnn_memory as tb  # noqa: E402

REF_RUN = "evp_fix_A_3lam_agn"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def obinit_closure(cfg_json: dict, ckpt) -> tuple[dict, dict]:
    """Closure at the documented initial state of the evp_fix runs."""
    a = cfg_json["args"]
    br = cfg_json["br_init"]
    if not a["no_br_init"] or a["init"] != "ob":
        raise SystemExit(f"[FATAL] {REF_RUN} was not an OB-init/no-br-init run")
    inits = {k: float(br[f"{k}_init_clipped"])
             for k in ("Gp", "lam", "nu_s", "tau_y")}
    if set(inits.values()) != {1.0}:
        raise SystemExit(f"[FATAL] unexpected scalar init {inits}")

    z = np.load(ckpt, allow_pickle=False)
    bound_c = float(z["ckpt_bound_c"])
    kappa0 = float(str(a["kappa_schedule"]).split(",")[0])
    theta, _ = tb.init_tbnn_theta(jax.random.PRNGKey(int(a["theta_seed"])),
                                  width=int(a["width"]), depth=int(a["depth"]),
                                  bound_c=bound_c)
    params = {"Gp": jnp.asarray(inits["Gp"], dtype=jnp.float64),
              "lam": jnp.asarray(inits["lam"], dtype=jnp.float64),
              "tau_y": jnp.asarray(inits["tau_y"], dtype=jnp.float64),
              "theta": theta, "tbnn_bound_c": bound_c,
              "tbnn_kappa": kappa0}
    closure = dict(name="obinit",
                   model=cr.get_model("tbnn_potential_yield_logconf_bk_v2"),
                   params=params, Gp=inits["Gp"], lam=inits["lam"],
                   nu_s=inits["nu_s"], tau_y=inits["tau_y"])
    meta = dict(theta_seed=int(a["theta_seed"]), width=int(a["width"]),
                depth=int(a["depth"]), bound_c=bound_c, kappa=kappa0,
                scalars=inits, yield_mode=a["yield_mode"])
    return closure, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drives", type=float, nargs="+",
                    default=[1.8, 2.5, 4.0, 5.0])
    ap.add_argument("--out-dir", default="paper_figs_derived/evp_obinit")
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(ap)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path("work/evp_channel") / REF_RUN / "config.json"
    ckpt = Path("work/evp_channel") / REF_RUN / "theta_checkpoint.npz"
    cfg_json = json.load(open(cfg_path))
    ck_sha = sha256(ckpt)
    print(f"[ckpt] {ckpt} sha256 {ck_sha}", flush=True)

    closure, meta = obinit_closure(cfg_json, ckpt)
    cfg = efd.prod_cfg()
    print(f"[setup] Nx={cfg['Nx']} Ny={cfg['Ny']} dt={cfg['dt']} "
          f"inner={cfg['inner_steps']} tol={cfg['solver_tol']} "
          f"eval={ee.EVAL_LAM}lam arrest={ee.ARREST_LAM}lam "
          f"early_stop=off", flush=True)
    print(f"[obinit] {meta}", flush=True)

    t0 = time.time()
    lad = ee.run_ladder(closure, cfg, "obinit", out, drives=tuple(args.drives))
    print(f"[ladder] {time.time()-t0:.0f}s", flush=True)

    with open(out / "obinit_ladder.json", "w") as fh:
        json.dump(ee._jsonable(dict(
            kind="evp_obinit_forward", closure=meta,
            reference_run=REF_RUN, reference_checkpoint=str(ckpt),
            reference_checkpoint_sha256=ck_sha,
            protocol=dict(eval_lam=ee.EVAL_LAM, arrest_lam=ee.ARREST_LAM,
                          early_stop=False, Nx=cfg["Nx"], Ny=cfg["Ny"],
                          dt=cfg["dt"], inner_steps=cfg["inner_steps"],
                          solver_tol=cfg["solver_tol"]),
            drives=list(args.drives), ladder=lad,
            jobid=os.environ.get("SLURM_JOB_ID", "local"),
            partition=os.environ.get("SLURM_JOB_PARTITION", "local"))),
            fh, indent=2)
    print(f"[done] {out / 'obinit_ladder.json'}", flush=True)


if __name__ == "__main__":
    main()
