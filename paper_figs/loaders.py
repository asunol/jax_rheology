"""Every read of every artifact goes through here.

Loaders resolve keys through :mod:`paper_figs.data_paths` only, cache by key,
and assert the array shapes that the figure code relies on.  A shape that has
drifted therefore fails at load with the registry key in the message rather
than producing a quietly wrong panel.
"""
from __future__ import annotations

import csv
import functools
import hashlib
import json

import numpy as np

from . import data_paths as dp

# --------------------------------------------------------------------------
# Ground-truth parameter values.
# --------------------------------------------------------------------------
GIE_TRUTH = {"Gp": 3.2, "lam": 0.7, "nu_s": 0.8, "alpha": 0.30, "eta_p": 2.24}
FENE_TRUTH = {"eta_p": 2.24, "lam": 0.7, "nu_s": 0.8, "Lsq": 12.0}
EVP_TRUTH = {"Gp": 3.2, "lam": 0.7, "nu_s": 0.8, "tau_y": 1.45}
NOISE_FLOOR = 0.03 ** 2          # 9e-4: the design noise level, squared
EVP_G_C = 1.45

GIE_S4_CKPT_SHA256 = (
    "a7bed94d1ce87b66766281cb1ee2bc6ab52f7479d069ec5d1686af722eb80aee")

BIC_FAMILIES = ("Newtonian", "OldroydB", "OldroydB",
                "Giesekus", "FENEPConformation", "LinearPTT")
FAMILY_ORDER = ("Newtonian", "OldroydB", "FENEPConformation", "LinearPTT",
                "Giesekus")
FAMILY_LABEL = {
    "Newtonian": "Newtonian",
    "OldroydB": "Oldroyd-B",
    "Giesekus": "Giesekus",
    "FENEPConformation": "FENE-P",
    "LinearPTT": "Linear PTT",
}


def sha256(key: str) -> str:
    h = hashlib.sha256()
    with open(dp.path(key), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@functools.lru_cache(maxsize=None)
def load_json(key: str) -> dict:
    with open(dp.path(key)) as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=None)
def load_npz(key: str, *want: str) -> dict:
    """Load selected keys (or all) from an npz, as a plain dict of arrays."""
    with np.load(dp.path(key), allow_pickle=False) as z:
        names = want or tuple(z.files)
        missing = [n for n in names if n not in z.files]
        if missing:
            raise KeyError(f"{key}: npz lacks {missing}; has {list(z.files)}")
        return {n: np.asarray(z[n]) for n in names}


@functools.lru_cache(maxsize=None)
def load_pickle(key: str):
    import pickle

    with open(dp.path(key), "rb") as fh:
        return pickle.load(fh)


@functools.lru_cache(maxsize=None)
def load_csv(key: str) -> dict:
    """Read a progress CSV into a dict of float arrays (non-numeric kept str)."""
    with open(dp.path(key), newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{key}: empty CSV")
    out: dict[str, np.ndarray] = {}
    for col in rows[0]:
        vals = [r[col] for r in rows]
        try:
            out[col] = np.array([float(v) if v not in ("", None) else np.nan
                                 for v in vals])
        except (TypeError, ValueError):
            out[col] = np.array(vals, dtype=object)
    return out


def _assert_shape(key: str, name: str, arr, shape):
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"{key}:{name} shape {arr.shape}, expected {shape}")


# --------------------------------------------------------------------------
# Giesekus contraction
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def gie_config() -> dict:
    """Resolved July-10 production config, from the Pass C provenance file."""
    prov = load_json("gie_passc_provenance")
    cfg = prov["resolved_config"]
    assert int(cfg["nx"]) == 128 and int(cfg["ny"]) == 256, cfg
    return cfg


@functools.lru_cache(maxsize=1)
def gie_fields() -> dict:
    """Final truth / learned-s4 contraction fields + ROI, on the 128x256 grid."""
    d = load_npz("gie_passc_fields")
    for name in ("truth_u", "truth_v", "truth_A_xx", "truth_A_xy",
                 "truth_A_yy", "truth_A_zz", "learned_u", "learned_v",
                 "learned_A_xx", "learned_A_xy", "learned_A_yy",
                 "learned_A_zz", "roi_weight", "roi_activation"):
        _assert_shape("gie_passc_fields", name, d[name], (128, 256))
    _assert_shape("gie_passc_fields", "xc", d["xc"], (128,))
    _assert_shape("gie_passc_fields", "yc", d["yc"], (256,))
    return dict(d)


@functools.lru_cache(maxsize=1)
def gie_init_fields() -> dict:
    """Final frame of the OB-init contraction forward.

    The archive is the July-12 shared Giesekus regeneration
    (``_campaign_giesekus/regen/init_traj.npz``), whose manifest records
    ``OB-init theta seed=0 gauge Gp=lam=nu_s=1`` at config hash 1ac16f0810ee --
    the same hash as the gie_A_s4 resolved config.  The top-level ``u``, ``v``,
    ``A_xx``, ``A_yy``, ``A_zz`` entries are the final frame, so no 400-frame
    trajectory is decompressed.
    """
    man = load_json("gie_truth_manifest")
    if "OB-init" not in man.get("init", ""):
        raise ValueError(f"gie_init_traj manifest is not OB-init: {man}")
    if man["cfg_hash"] != load_json("gie_s4_learned_manifest")["cfg_hash"]:
        raise ValueError("init/learned archive config hashes differ")
    d = load_npz("gie_init_traj", "u", "v", "A_xx", "A_yy", "A_zz",
                 "xc", "yc", "xfu", "yfu", "H", "R", "U")
    for name in ("u", "v", "A_xx", "A_yy", "A_zz"):
        _assert_shape("gie_init_traj", name, d[name], (128, 256))
    return dict(d)


@functools.lru_cache(maxsize=1)
def gie_checkpoint_scalars() -> dict:
    """Trained (Gp, lam, nu_s) per training schedule, from each checkpoint."""
    out = {}
    for run in ("gie_A_s1", "gie_A_s1b", "gie_A_s4"):
        z = load_npz(f"gie_ckpt_{run}", "ckpt_Gp_fit", "ckpt_lam_fit",
                     "ckpt_nu_s", "ckpt_loss")
        out[run] = {"Gp": float(z["ckpt_Gp_fit"]),
                    "lam": float(z["ckpt_lam_fit"]),
                    "nu_s": float(z["ckpt_nu_s"]),
                    "loss": float(z["ckpt_loss"])}
    return out


GIE_SCHEDULES = ("gie_A_s1", "gie_A_s1b", "gie_A_s4")
#: Reader-facing names.  The run tags s1 / s1b / s4 carry no meaning outside
#: the campaign directory, so figures number the schedules instead.
GIE_SCHEDULE_LABEL = {"gie_A_s1": "Training 1", "gie_A_s1b": "Training 2",
                      "gie_A_s4": "Training 3"}


@functools.lru_cache(maxsize=None)
def _progress(key: str) -> dict:
    d = load_csv(key)
    step = d["step"]
    if np.any(np.diff(step) < 0):
        raise ValueError(f"{key}: step is not monotone")
    stage = np.asarray([str(s) for s in d["stage"]], dtype=object)
    accepted = np.array([i == len(stage) - 1 or stage[i + 1] != s
                         for i, s in enumerate(stage)])
    return {"step": step, "loss": d["loss"], "Gp": d["Gp"], "lam": d["lam"],
            "nu_s": d["nu_s"], "eta_p": d["Gp"] * d["lam"], "stage": stage,
            "accepted": accepted,
            "is_scalar_block": np.asarray([s.startswith("sc") for s in stage])}


def accepted_progress(p: dict) -> dict:
    """Training history restricted to the state accepted at each block end."""
    m = p["accepted"]
    return {k: v[m] for k, v in p.items()
            if k not in ("accepted", "is_scalar_block")}


def gie_progress(run: str) -> dict:
    """Training history of one Giesekus schedule.

    ``stage`` alternates ``sc<k>`` (scalar block) and ``th<k>`` (network
    block) and ``step`` is the cumulative optimiser step, which is monotone.

    Inside a ``sc`` block the rows are *trial* evaluations of the derivative-
    free scalar search, so they include probes at the parameter bounds (0.02
    and 20) whose loss is four orders of magnitude above the incumbent.  The
    ``accepted`` mask selects the last row of each block, i.e. the state the
    run actually carried forward; that is the convergence history.
    """
    return _progress(f"gie_progress_{run}")


# --------------------------------------------------------------------------
# BIC battery
# --------------------------------------------------------------------------

def battery_target(target: str) -> dict:
    return load_json(f"battery_target_{target}")


def battery_rows(target: str) -> dict:
    """family name -> {bic, mse, params, best_restart_seed, restart_spread}."""
    d = battery_target(target)
    return {r["name"]: r for r in d["results"]}


def battery_delta_bic(target: str) -> dict:
    """DeltaBIC of each family relative to the winner (winner = 0)."""
    rows = battery_rows(target)
    best = min(r["bic"] for r in rows.values())
    return {name: r["bic"] - best for name, r in rows.items()}


def battery_winner(target: str) -> tuple[str, float]:
    d = battery_target(target)
    return d["winner"], float(d["margin"])


def gie_battery_scalars(target: str) -> dict:
    """Giesekus scalars read back from a finished network by the BIC battery.

    The battery reports the fit in its own parameterisation (solvent and
    polymer viscosities, relaxation time, alpha); G_p is not stored and is
    recovered as eta_p / lambda.  These are independent of the trained scalars
    in progress.csv: the battery refits the whole family to the network's
    stress response, so it can land somewhere else.
    """
    p = battery_rows(target)["Giesekus"]["params"]
    lam = float(p["relaxation_time"])
    eta_p = float(p["polymer_viscosity"])
    return {"Gp": eta_p / lam, "lam": lam,
            "nu_s": float(p["solvent_viscosity"]), "eta_p": eta_p,
            "alpha": float(p["alpha"])}


def battery_restart_spread(target: str, family: str) -> dict:
    """Per-restart fits of one family: seeds, BIC, MSE and recovered params.

    The rollup stores this as parallel lists keyed by field (not, as
    a list of per-seed records); it is transposed
    here into one record per seed.
    """
    row = battery_rows(target)[family]
    s = row["restart_spread"]
    return {
        "seeds": [int(v) for v in s["seeds"]],
        "bic": [float(v) for v in s["bic"]],
        "mse": [float(v) for v in s["mse"]],
        "ok": [bool(v) for v in s["ok"]],
        "params": list(s["params"]),
        "init": list(s["init"]),
        "best_restart_seed": int(row["best_restart_seed"]),
    }


@functools.lru_cache(maxsize=None)
def battery_aos(target: str) -> dict:
    """AOS protocol arrays plus each candidate's forward-replayed prediction.

    The replay is produced by ``paper_figs/derive_aos.py`` under the diff_rheo
    environment and cached; see that script for the provenance check against
    the battery's own recorded MSE/BIC.
    """
    key = f"aos_{target}"
    z = load_npz(key)
    n_legs, n_pts = z["time"].shape
    for name in ("sigma_noisy", "gammadot"):
        _assert_shape(key, name, z[name], (n_legs, n_pts))
    out = {k: v for k, v in z.items() if k != "checks_json"}
    out["checks"] = json.loads(str(z["checks_json"]))
    meta = load_json(f"battery_datameta_{target}")
    out["legs"] = meta["generator_meta"]      # amplitude f and frequency omega
    return out


# --------------------------------------------------------------------------
# Cavity transfer
# --------------------------------------------------------------------------

def cavity_metrics() -> dict:
    return load_json("cavity_transfer_metrics")["metrics"]


def cavity_run(arm: str, de: float) -> dict:
    tag = f"De{de:.2f}"
    cfg = load_json(f"cavity_{arm}_config_{tag}")
    res = load_json(f"cavity_{arm}_result_{tag}")
    n = int(cfg["cells"])
    diag = load_npz(f"cavity_{arm}_diag_{tag}")
    for name in ("final_u", "final_v", "final_A_xx", "final_A_xy",
                 "final_A_yy", "final_A_zz"):
        _assert_shape(f"cavity_{arm}_diag_{tag}", name, diag[name], (n, n))
    _assert_shape(f"cavity_{arm}_diag_{tag}", "final_psi", diag["final_psi"],
                  (n - 1, n - 1))
    return {"config": cfg, "result": res, "diag": diag}


def cavity_centreline(arm: str, de: float):
    """u along the vertical centreline and v along the horizontal one.

    This is the computation SN3d plots and N1e(i) reuses; both call here.
    """
    run = cavity_run(arm, de)
    n = int(run["config"]["cells"])
    coord = (np.arange(n) + 0.5) / n * float(run["config"]["L"])
    mid = n // 2
    return {
        "coord": coord,
        "u_of_y": np.asarray(run["diag"]["final_u"])[mid, :],
        "v_of_x": np.asarray(run["diag"]["final_v"])[:, mid],
        "U_lid": float(run["config"]["U_lid"]),
    }


def cavity_history(arm: str, de: float) -> dict:
    """Steadiness diagnostics vs time for one cavity run.

    The npz stores one sample per outer step and no time vector, so t is
    reconstructed from the config as ``(i+1) * T / outer_steps``.
    """
    run = cavity_run(arm, de)
    cfg, diag = run["config"], run["diag"]
    n_out = int(cfg["outer_steps"])
    t = (np.arange(n_out) + 1) * float(cfg["T"]) / n_out
    for name in ("ke_traj", "max_Axx_traj", "psi_min_traj", "min_lam_traj"):
        _assert_shape(f"cavity_{arm}_diag_De{de:.2f}", name, diag[name],
                      (n_out,))
    scale = float(cfg["U_lid"]) * float(cfg["L"])
    return {"t": t, "ke": diag["ke_traj"], "max_Axx": diag["max_Axx_traj"],
            "psi_min": diag["psi_min_traj"], "min_lam": diag["min_lam_traj"],
            "psi_min_over_UL": diag["psi_min_traj"] / scale,
            "any_nan": bool(np.any(diag["any_nan_traj"]))}


def cavity_ladder_metrics() -> dict:
    """Per-De scalars quoted in N1e and SN3f, straight from transfer_metrics."""
    m = cavity_metrics()
    de = list(dp.CAVITY_DE)
    out = {"De": np.array(de)}
    for arm in ("truth", "learned"):
        for field, key in (("max_Axx", "max_Axx"),
                           ("psi_min_over_UL", "psi_min_over_UL"),
                           ("min_eig", "min_eigenvalue")):
            out[f"{arm}_{field}"] = np.array(
                [m[f"De{d:.2f}"][f"{arm}_{key}"] for d in de])
        out[f"{arm}_eye"] = np.array([m[f"De{d:.2f}"][f"{arm}_eye"]
                                      for d in de])
    out["velocity_rms_over_U"] = np.array(
        [m[f"De{d:.2f}"]["velocity_rms_over_U_lid"] for d in de])
    out["relative_L2_A_xx"] = np.array(
        [m[f"De{d:.2f}"]["relative_L2_A_xx"] for d in de])
    return out


def cavity_centreline_scalar(arm: str, de: float) -> float:
    """Minimum of u along the vertical centreline, scaled by the lid speed.

    This is the standard lid-driven-cavity centreline benchmark scalar (the
    counter-flow extremum below the primary vortex).  N1e(i) plots it; SN3d
    plots the profile it is taken from, through the same loader.
    """
    c = cavity_centreline(arm, de)
    return float(np.min(c["u_of_y"]) / c["U_lid"])


# --------------------------------------------------------------------------
# FENE-P
# --------------------------------------------------------------------------

FENE_SINGLE_RATE_CANDIDATES = ("R3", "fene8_u05_s1", "fene8_u05_s2",
                               "fene8_u05_s3", "fene8_u05_s4")


def fene_recovery(target: str) -> dict:
    """FENE-P winner params for a target, with L^2 = extension_length^2."""
    rows = battery_rows(target)
    r = rows["FENEPConformation"]
    p = r["params"]
    return {
        "eta_p": float(p["polymer_viscosity"]),
        "lam": float(p["relaxation_time"]),
        "nu_s": float(p["solvent_viscosity"]),
        "Lsq": float(p["extension_length"]) ** 2,
        "L": float(p["extension_length"]),
        "bic": float(r["bic"]), "mse": float(r["mse"]),
        "best_restart_seed": r.get("best_restart_seed"),
    }


def fene_recovery_errors(target: str) -> dict:
    rec = fene_recovery(target)
    return {k: (rec[k] - FENE_TRUTH[k]) / FENE_TRUTH[k]
            for k in ("eta_p", "lam", "nu_s", "Lsq")}


def fene_mare(target: str) -> float:
    """Mean absolute relative error over the four FENE-P parameters."""
    e = fene_recovery_errors(target)
    return float(np.mean([abs(v) for v in e.values()]))


#: Battery target name -> registry stem of the training run behind it.
FENE_PROGRESS_KEY = {"R3": "fene_progress_R3",
                     "fene8_u05_s1": "fene_progress_fene8_u05_s1",
                     "fene8_u05_s2": "fene_progress_fene8_u05_s2",
                     "fene8_u05_s3": "fene_progress_fene8_u05_s3",
                     "fene8_u05_s4": "fene_progress_fene8_u05_s4"}
FENE_LABEL = {"R3": "R3", "fene8_u05_s1": "s1", "fene8_u05_s2": "s2",
              "fene8_u05_s3": "s3", "fene8_u05_s4": "s4"}


@functools.lru_cache(maxsize=None)
def fene_progress(target: str) -> dict:
    """Training history of one FENE-P run, same block structure as Giesekus."""
    return _progress(FENE_PROGRESS_KEY[target])


@functools.lru_cache(maxsize=None)
def fene_summary(target: str) -> dict:
    return load_json(f"fene_summary_{target}")


def fene_config(target: str) -> dict:
    return fene_summary(target)["args"]


@functools.lru_cache(maxsize=None)
def fene_truth_state_cloud(target: str) -> dict:
    """Final-frame truth ``tr A`` field of one FENE-P training run.

    ``arrays.npz`` stores the first TBNN invariant ``x1 = tr A - 3`` of the
    *truth* conformation at the last solver frame, flattened over the 128x256
    grid (visco_opt_tbnn_contraction_run.py:1331-1355).  For a dual-rate run
    the driver's per-rate loop leaves the archived frame at the *last* rate in
    ``U_list``, i.e. U = 4; that is recorded in the returned ``U`` so callers
    cannot mistake it for the U = 0.5 frame.
    """
    z = load_npz(f"fene_arrays_{target}", "x1")
    x1 = np.asarray(z["x1"], float)
    if x1.shape != (128 * 256,):
        raise ValueError(f"fene_arrays_{target}: x1 shape {x1.shape}")
    s = fene_summary(target)
    trA = x1.reshape(128, 256) + 3.0
    return {"trA": trA, "U": float(s["U_list"][-1]),
            "U_list": [float(u) for u in s["U_list"]],
            "Lsq_truth": float(s["Lsq_truth"])}


@functools.lru_cache(maxsize=1)
def fene_repr_fields() -> dict:
    """Truth, OB-init and learned contraction fields of the representative run.

    Truth and OB-init come from the archived campaign regeneration; the learned
    arm is the forward-only run produced by ``paper_figs/run_learned_forward.py``
    at the same config hash.
    """
    want = ("u", "v", "A_xx", "A_yy", "A_zz", "xc", "yc", "xfu", "yfu",
            "H", "R", "U")
    out = {}
    for arm, key in (("truth", "fene_truth_traj"), ("init", "fene_init_traj"),
                     ("learned", "fene_repr_forward")):
        z = load_npz(key, *want)
        for name in ("u", "v", "A_xx", "A_yy", "A_zz"):
            _assert_shape(key, name, z[name], (128, 256))
        for name in ("u", "v", "A_xx", "A_yy", "A_zz"):
            out[f"{arm}_{name}"] = z[name]
        # A_xy is stored only along the trajectory; the final frame is the last.
        out.update({k: z[k] for k in ("xc", "yc", "xfu", "yfu", "H", "R", "U")})
    return out


def direct_progress(fit: str) -> dict:
    d = load_csv(f"direct_progress_{fit}")
    for col in ("nfev", "loss", "Gp", "lam", "nu_s", "Lsq"):
        if col not in d:
            raise KeyError(f"direct_progress_{fit}: no column {col}; "
                           f"has {list(d)}")
    return d


# --------------------------------------------------------------------------
# EVP channel
# --------------------------------------------------------------------------

EVP_SEEDS = ("s1", "s2", "s3", "s4", "s5")


def evp_summary() -> dict:
    return load_json("evp_seed_eval_summary")


def evp_drive_tag(gx: float) -> str:
    for k, v in dp.EVP_DRIVES.items():
        if abs(k - gx) < 1e-9:
            return v
    raise KeyError(f"no eval store for g_x={gx}")


def evp_obinit_ladder() -> dict:
    """Provenance + per-drive results of the OB-init forwards (run G3)."""
    p = dp.path("evp_obinit_eval") / "obinit_ladder.json"
    with open(p) as fh:
        return json.load(fh)


def evp_profile(arm: str, gx: float) -> dict:
    """Final u(y) profile and Q(t) for ``arm`` in {'truth','s1'..'s5','obinit'}."""
    tag = evp_drive_tag(gx)
    if arm == "obinit":
        p = dp.path("evp_obinit_eval") / f"obinit_gx{tag}.npz"
        with np.load(p, allow_pickle=False) as z:
            d = {k: np.asarray(z[k]) for k in ("u_prof", "Q", "t")}
    else:
        key = f"evp_truth_gx{tag}" if arm == "truth" else f"evp_{arm}_gx{tag}"
        d = load_npz(key, "u_prof", "Q", "t")
    if d["u_prof"].ndim != 2 or d["u_prof"].shape[1] != 64:
        raise ValueError(f"{key}: u_prof {d['u_prof'].shape}, expected (T, 64)")
    return {"u": np.asarray(d["u_prof"])[-1], "u_prof": np.asarray(d["u_prof"]),
            "Q": np.asarray(d["Q"]), "t": np.asarray(d["t"])}


def evp_flow_curve():
    """g_x -> (Q_truth, [Q_learned per seed])."""
    s = evp_summary()
    drives = sorted(float(k) for k in s["truth"])
    q_truth = np.array([s["truth"][f"{g:g}"]["Q"] for g in drives], float)
    q_seeds = np.array([[s["seeds"][sd]["flow_curve"][f"{g:g}"]["Q_learned"]
                         for g in drives] for sd in EVP_SEEDS], float)
    return np.array(drives), q_truth, q_seeds


def evp_scalars():
    """Per-seed recovered (Gp, lam, nu_s, tau_y)."""
    s = evp_summary()
    return {sd: {k: float(s["seeds"][sd]["scalars"][k])
                 for k in ("Gp", "lam", "nu_s", "tau_y")} for sd in EVP_SEEDS}


def evp_ladder() -> np.ndarray:
    """The 12-drive evaluation ladder, as stored in the seed-eval config."""
    return np.array(evp_summary()["config"]["ladder"], float)


def evp_nan(seed: str, gx: float) -> bool:
    """True if that seed's forward hit NaN at this drive (arrays then stale)."""
    return bool(evp_summary()["seeds"][seed]["flow_curve"][f"{gx:g}"]["any_nan"])


def evp_plug_ladder() -> dict:
    """Kinematic plug half-width vs drive: truth, per seed, and analytic."""
    s, g = evp_summary(), evp_ladder()
    truth = np.array([s["truth"][f"{x:g}"]["kinematic"]["halfwidth"]
                      for x in g], float)
    learn = np.array([[s["seeds"][sd]["flow_curve"][f"{x:g}"]["kin_plug_learned"]
                       for x in g] for sd in EVP_SEEDS], float)
    analytic = np.where(g > EVP_TRUTH["tau_y"], EVP_TRUTH["tau_y"] / g, np.nan)
    nan = np.array([[evp_nan(sd, x) for x in g] for sd in EVP_SEEDS])
    return {"g": g, "truth": truth, "learned": learn, "analytic": analytic,
            "nan": nan}


def evp_stability_ladder() -> dict:
    """max|tau_d| and min eig(A) vs drive, truth and per seed, + ceilings."""
    s, g = evp_summary(), evp_ladder()
    def _f(rec, k):
        v = rec.get(k)
        return np.nan if v is None else float(v)
    truth_td = np.array([_f(s["truth"][f"{x:g}"], "max_td") for x in g])
    truth_me = np.array([_f(s["truth"][f"{x:g}"], "min_eig") for x in g])
    fc = {sd: s["seeds"][sd]["flow_curve"] for sd in EVP_SEEDS}
    td = np.array([[_f(fc[sd][f"{x:g}"], "max_td") for x in g]
                   for sd in EVP_SEEDS])
    me = np.array([[_f(fc[sd][f"{x:g}"], "min_eig") for x in g]
                   for sd in EVP_SEEDS])
    ceil = np.array([float(s["seeds"][sd]["well_posed_gx_ceiling"])
                     for sd in EVP_SEEDS])
    return {"g": g, "truth_td": truth_td, "truth_min_eig": truth_me,
            "td": td, "min_eig": me, "ceiling": ceil,
            "nan": np.array([[evp_nan(sd, x) for x in g] for sd in EVP_SEEDS])}


@functools.lru_cache(maxsize=None)
def evp_progress(seed: str) -> dict:
    """Training history of one EVP seed, one row per gradient evaluation.

    The schedule is ``stage1`` (L-BFGS-B on the four scalars, theta held at
    OB-init) followed by four ``c{0..3}`` blocks of Adam on theta, each closed
    by a ``resolve{0..3}`` L-BFGS-B re-solve of the scalars.  ``block_end``
    marks the last row of each block, i.e. the state carried forward.
    """
    d = load_csv(f"evp_progress_{seed}")
    for col in ("step", "loss", "L_vel", "L_Q", "Gp", "lam", "nu_s", "tau_y",
                "stage"):
        if col not in d:
            raise KeyError(f"evp_progress_{seed}: no column {col}; has {list(d)}")
    if np.any(np.diff(d["step"]) <= 0):
        raise ValueError(f"evp_progress_{seed}: step is not strictly increasing")
    stage = np.asarray([str(s) for s in d["stage"]], dtype=object)
    d = dict(d)
    d["stage"] = stage
    d["block_end"] = np.array([i == len(stage) - 1 or stage[i + 1] != s
                               for i, s in enumerate(stage)])
    # Same meaning as the Giesekus runs' mask, so ``accepted_progress`` works
    # on this dict too: inside a stage1/resolve block most rows are L-BFGS-B
    # line-search trials, and only the last is the state carried forward.
    d["accepted"] = d["block_end"]
    return d


def _flatten_theta(theta) -> np.ndarray:
    """Concatenate W, b per head, sorted, as float64."""
    parts = []
    for head in sorted(theta):
        for W, b in theta[head]:
            parts.append(np.asarray(W, dtype=np.float64).ravel())
            parts.append(np.asarray(b, dtype=np.float64).ravel())
    return np.concatenate(parts)


#: Stage checkpoints that store a full theta.  progress.csv has no theta, so
#: ||theta - theta_OB|| is available only at these block-end archives.
EVP_THETA_STAGES = ("stage1", "resolve0", "resolve1", "resolve2", "resolve3")


@functools.lru_cache(maxsize=None)
def evp_theta_drift() -> dict:
    """||theta - theta_OB||_2 at the five saved stage checkpoints.

    Theta is reconstructed at the documented OB-init (width/depth/bound_c
    and theta_seed from each run's config) and compared to the archived
    stage pickles.  Not a per-iteration series: the optimiser never wrote
    theta into progress.csv.
    """
    import jax
    from jax_rheology.models import tbnn_memory as tb

    out = {}
    for sd in EVP_SEEDS:
        cfg = load_json(f"evp_config_{sd}")["args"]
        bound_c = float(load_npz(f"evp_ckpt_{sd}", "ckpt_bound_c")["ckpt_bound_c"])
        theta_ob, _ = tb.init_tbnn_theta(
            jax.random.PRNGKey(int(cfg["theta_seed"])),
            width=int(cfg["width"]), depth=int(cfg["depth"]),
            bound_c=bound_c)
        ref = _flatten_theta(theta_ob)
        stages = {}
        for tag in EVP_THETA_STAGES:
            fit = load_pickle(f"evp_stage_{sd}_{tag}")
            flat = _flatten_theta(fit["theta"])
            if flat.shape != ref.shape:
                raise ValueError(
                    f"evp_stage_{sd}_{tag}: theta dim {flat.size} != "
                    f"OB dim {ref.size}")
            stages[tag] = float(np.linalg.norm(flat - ref))
        out[sd] = {"theta_seed": int(cfg["theta_seed"]),
                   "dim": int(ref.size), "bound_c": bound_c,
                   "stages": stages}
    return out


EVP_ARMS = ("evp_fix_A_3lam_agn", "evp_fix_A_3lam_br",
            "evp_fix_A_7lam_agn", "evp_fix_A_7lam_br",
            "evp_fix_B_3lam_agn", "evp_fix_B_3lam_br",
            "evp_fix_B_7lam_agn", "evp_fix_B_7lam_br")
EVP_ARM_DRIVES = {"A": (1.8, 2.5, 4.0), "B": (1.6, 1.8, 2.5)}


def evp_arm_matrix() -> dict:
    """8-arm scalar recovery: drive set x training horizon x scalar init."""
    d = load_json("evp_8arm_summary")["arms"]
    missing = [a for a in EVP_ARMS if a not in d]
    if missing:
        raise KeyError(f"evp_8arm_summary: arms {missing} absent; has {list(d)}")
    out = {}
    for a in EVP_ARMS:
        rec = d[a]
        if rec["status"] != "ok":
            raise ValueError(f"{a}: status {rec['status']}")
        _, _, group, horizon, init = a.split("_")
        # ``br_init`` is present on every arm: on the agnostic ones it records
        # ``neutral_ones``, on the bracket ones the Buckingham-Reiner start
        # computed from that arm's own two highest drives, so the numbers
        # differ arm by arm.
        b = rec["br_init"]
        out[a] = {"group": group, "horizon": horizon, "init": init,
                  "drives": EVP_ARM_DRIVES[group],
                  "loss": float(rec["loss_final"]),
                  "n_grads": int(rec["n_grads"]),
                  "init_method": str(b["method"]),
                  "init_note": str(b["note"]),
                  "init_vals": {k: float(b[f"{k}_init_clipped"])
                                for k in ("Gp", "lam", "nu_s", "tau_y")},
                  "rel": {k: float(rec["recovery"][k]["signed_rel"])
                          for k in ("Gp", "lam", "nu_s", "tau_y")}}
    return out


def evp_sensitivity() -> dict:
    """|d ln Q / d ln tau_y| vs drive, per evaluation horizon (phase-B sweep)."""
    s = load_json("evp_phaseB")["sensitivity"]
    out: dict[float, dict[str, np.ndarray]] = {}
    for key, rec in s.items():
        horizon = float(key.split("|")[0])
        out.setdefault(horizon, {"g": [], "logderiv": []})
        out[horizon]["g"].append(float(rec["g_x"]))
        out[horizon]["logderiv"].append(float(rec["logderiv"]))
    for horizon, rec in out.items():
        order = np.argsort(rec["g"])
        rec["g"] = np.array(rec["g"], float)[order]
        rec["logderiv"] = np.array(rec["logderiv"], float)[order]
    return out
