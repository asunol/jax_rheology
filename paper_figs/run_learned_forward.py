#!/usr/bin/env python
"""Forward-only contraction run from a frozen checkpoint.

Used for the FENE-P representative network, whose learned contraction fields
were never archived (the archived ``ctr_fene_*`` fields belong to the pressure
campaign, not to a headline single-rate run).  Truth and OB-init trajectories
for the same configuration already exist and are *not* recomputed: the config
hash of this run is checked against the archived campaign manifest first, and
the run aborts if they differ.

No optimiser, no gradients: theta and the three scalars are read from the
checkpoint, whose SHA256 is verified against the value passed on the command
line before anything is built.

    python paper_figs/run_learned_forward.py \
        --campaign-root fene7_prod --run fene7_u05_vel \
        --expect-sha256 <sha> --shared analysis_pub_readback/_campaign_fene/regen \
        --out paper_figs_derived/fene_repr_forward.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import bootstrap, REPO_ROOT
bootstrap()
ROOT = str(REPO_ROOT)

import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import regen_contraction as R  # noqa: E402  (reused verbatim: build/evolve/pack)
from jax_rheology.models import tbnn_memory as tb  #  noqa: E402
from visco_opt_tbnn_run import theta_from_named_arrays  # noqa: E402
import visco_opt_tbnn_contraction_run as C  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--expect-sha256", required=True)
    ap.add_argument("--shared", required=True,
                    help="archived campaign regen dir whose cfg_hash must match")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="3 outer steps, for a login-node plumbing check")
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(ap)

    run_dir = os.path.join(args.campaign_root, args.run)
    ckpt = os.path.join(run_dir, "theta_checkpoint.npz")
    got = sha256(ckpt)
    if got != args.expect_sha256:
        raise SystemExit(f"[FATAL] {ckpt} sha256 {got} != expected "
                         f"{args.expect_sha256}")
    print(f"[ckpt] {ckpt} sha256 verified {got}", flush=True)

    a = json.load(open(os.path.join(run_dir, "summary.json")))["args"]
    cfg_hash = R._cfg_hash({k: a[k] for k in (
        "H", "ratio", "L_up", "L_down", "nx", "ny", "U", "dt", "inner",
        "outer", "ramp_time", "truth_model")})
    shared_manifest = json.load(
        open(os.path.join(args.shared, "regen_manifest.json")))
    if cfg_hash != shared_manifest["cfg_hash"]:
        raise SystemExit(
            f"[FATAL] cfg_hash {cfg_hash} != archived truth/init "
            f"{shared_manifest['cfg_hash']}; truth and OB-init would have to be "
            f"regenerated as well")
    print(f"[cfg] hash {cfg_hash} matches {args.shared}", flush=True)

    if args.smoke:
        a = dict(a)
        a["outer"] = 3

    grid, tbnn_model, tbnn_state, perm, bc = R._build(a, C.TBNN_NAME)
    evolve = R._make_evolver(a, tbnn_model, grid, perm, bc)
    pressure_on = True
    factor = float(a["density"]) / (int(a["inner"]) * float(a["dt"]))
    Xc, Yc = grid.mesh(grid.cell_center)
    tap_idx = [C.bilinear_idx(np.asarray(Xc)[:, 0], np.asarray(Yc)[0, :], x, y)
               for (x, y) in C.TAPS_PHYS]

    z = np.load(ckpt, allow_pickle=False)
    heads = [str(h) for h in z["ckpt_heads"]]
    nlayers = {h: int(n) for h, n in zip(heads, z["ckpt_nlayers"])}
    theta = theta_from_named_arrays(z, heads, nlayers)
    Gp, lam, nu_s = (float(z["ckpt_Gp_fit"]), float(z["ckpt_lam_fit"]),
                     float(z["ckpt_nu_s"]))
    params = dict(Gp=jnp.asarray(Gp), lam=jnp.asarray(lam), theta=theta,
                  tbnn_bound_c=float(a["bound_c"]))
    print(f"[frozen] Gp={Gp:.6f} lam={lam:.6f} nu_s={nu_s:.6f}", flush=True)

    t0 = time.time()
    final, out = evolve(tbnn_state, params, nu_s)
    out["u_traj"].block_until_ready()
    packed = R._pack(final, out, grid, a, pressure_on, tap_idx, factor)
    axes = R._grid_axes(grid)
    meta = dict(H=float(a["H"]), R=float(a["ratio"]), U=float(a["U"]))
    R._save(args.out, packed, axes, meta)
    print(f"[learned {args.run}] {time.time()-t0:.1f}s "
          f"max|u|={np.abs(packed['u']).max():.4f} "
          f"maxA_xx={packed['A_xx'].max():.4f}", flush=True)

    manifest = os.path.splitext(args.out)[0] + "_manifest.json"
    with open(manifest, "w") as fh:
        json.dump(dict(kind="contraction_learned_forward_only", run=args.run,
                       campaign_root=args.campaign_root, checkpoint=ckpt,
                       checkpoint_sha256=got, cfg_hash=cfg_hash,
                       shared_truth_init=args.shared,
                       scalars=dict(Gp_fit=Gp, lam_fit=lam, nu_s_fit=nu_s),
                       optimiser=False, smoke=bool(args.smoke),
                       jobid=os.environ.get("SLURM_JOB_ID", "local"),
                       partition=os.environ.get("SLURM_JOB_PARTITION", "local")),
                  fh, indent=2)
    print(f"[manifest] {manifest}", flush=True)


if __name__ == "__main__":
    main()
