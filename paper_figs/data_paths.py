"""Single registry of every data artifact used by the paper figures.

Nothing else in ``paper_figs`` may hardcode a path.  ``loaders.py`` resolves
keys through :func:`path`; figure modules never see a filesystem path at all.

Every entry in :data:`REGISTRY` is ``stat``'d at import.  A missing artifact
raises :class:`MissingArtifact` naming the registry key, so the failure points
straight at the thing that has to be regenerated.

Every key below traces to a recorded production artifact. Artifacts that must
not be quoted as results are deliberately absent; see :data:`FORBIDDEN` for
the block list that :func:`path` enforces.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from repo_paths import REPO_ROOT, FROZEN_MEM

ROOT = FROZEN_MEM
OUT_ROOT = Path(os.environ["TBNN_FIG_OUT"]) if os.environ.get("TBNN_FIG_OUT") else (
    REPO_ROOT / "final_figures")
DERIVED_ROOT = REPO_ROOT / "paper_figs_derived"


def _bundle_root() -> Path | None:
    raw = os.environ.get("TBNN_DATA_BUNDLE") or os.environ.get("DATA_ROOT")
    if not raw:
        return None
    if raw in ("bundle", "data_bundle"):
        return REPO_ROOT / "data_bundle"
    p = Path(raw)
    if p.is_dir() and (
        (p / "table_s2_bic_battery").exists() or p.name == "data_bundle"
    ):
        return p.resolve()
    return None


class MissingArtifact(FileNotFoundError):
    """A registered artifact is absent from the filesystem."""


class ForbiddenArtifact(RuntimeError):
    """A DEAD / DO-NOT-QUOTE artifact was requested."""


# Substrings that must never appear in a resolved path: superseded runs and
# diagnostics that must not be quoted as results.
FORBIDDEN = (
    "v2_yield/v2_prod2",
    "evp_v2_prod2_flowcurve",
    "analysis_fene7/",
    "corrected_battery/",
    "quarantine/",
    "cavity_outputs/bsd_level4",
    "cavity_outputs/diag_de05",
    "fene8_interim_parameters.md",
    "evp_thinit_eval/walkback.json",
    "check2_gsk_vs_sar",
    "evp_thinit_gsk_s",
)

# Exceptions: the July-comparison archive lives under corrected_battery/ but is
# the *only* quotable source for the archived margins (
# Sec.1 "July-comparison column fix").  It is provenance, not a model-selection
# quote.
FORBIDDEN_EXCEPTIONS = (
    "analysis_phase2_instrument/corrected_battery/giesekus_winners.csv",
)

_GIE_RUNS = ("gie_A_s1", "gie_A_s1b", "gie_A_s4")
_FENE8_RUNS = (
    "fene8_u05_s1", "fene8_u05_s2", "fene8_u05_s3", "fene8_u05_s4",
    "fene8_leg_s3", "fene8_leg_s4",
    "fene8_bal_s0", "fene8_bal_s1", "fene8_bal_s2", "fene8_bal_s3",
    "fene8_bal_s4",
)
_FENE7_RUNS = {
    "R3": "fene7_u05_vel",
    "R4": "fene7_u05_velp",
    "R1": "fene7_cur_vel",
    "R2": "fene7_cur_velp",
    "R5": "fene7_cur_velp_s1",
    "R6": "fene7_cur_velp_s2",
    "R7": "fene7_cur_velp_lo",
}
_DIRECT_FITS = ("I1", "I2", "I3", "I4", "I5")
_EVP_SEEDS = {
    "s1": "evp_fix_A_3lam_agn",
    "s2": "evp_fix_A_3lam_agn_s2",
    "s3": "evp_fix_A_3lam_agn_s3",
    "s4": "evp_fix_A_3lam_agn_s4",
    "s5": "evp_fix_A_3lam_agn_s5",
}
# Drive -> filename tag.
EVP_DRIVES = {
    0.5: "0p5", 1.0: "1", 1.3: "1p3", 1.45: "1p45", 1.6: "1p6", 1.8: "1p8",
    2.5: "2p5", 4.0: "4", 4.5: "4p5", 5.0: "5", 5.5: "5p5", 6.0: "6",
}
CAVITY_DE = (0.20, 0.35, 0.50)


def _build() -> dict[str, Path]:
    r: dict[str, Path] = {}

    # ---------------- Giesekus contraction (N1, SN1, SN2) ----------------
    pr = ROOT / "analysis_pub_readback"
    r["gie_passc_fields"] = pr / "gie_A_s4/passc/passc_final_fields_s4.npz"
    r["gie_passc_provenance"] = pr / "gie_A_s4/passc/resolved_config.json"
    r["gie_truth_traj"] = pr / "_campaign_giesekus/regen/truth_traj.npz"
    r["gie_init_traj"] = pr / "_campaign_giesekus/regen/init_traj.npz"
    r["gie_truth_manifest"] = pr / "_campaign_giesekus/regen/regen_manifest.json"
    r["gie_s4_learned_traj"] = pr / "gie_A_s4/regen/learned_traj.npz"
    r["gie_s4_learned_manifest"] = pr / "gie_A_s4/regen/regen_manifest.json"
    for run in _GIE_RUNS:
        base = ROOT / "gie_prod_rerun" / run
        r[f"gie_ckpt_{run}"] = base / "theta_checkpoint.npz"
        r[f"gie_summary_{run}"] = base / "summary.json"
        r[f"gie_progress_{run}"] = base / "progress.csv"
        r[f"gie_arrays_{run}"] = base / "arrays.npz"

    # ---------------- BIC battery (N1c, SN2d/e, SN4c, SN5a, Table S2) ----
    fb = ROOT / "analysis_phase2_instrument/final_battery"
    r["battery_report"] = fb / "final_report_PR18-PR21.md"
    r["battery_runner"] = ROOT / "tbnn_bic_final_battery.py"
    r["battery_assemble"] = ROOT / "phase2_fene8_final_assemble.py"
    r["july_giesekus_winners_csv"] = (
        ROOT / "analysis_phase2_instrument/corrected_battery/giesekus_winners.csv")
    for target in list(_GIE_RUNS) + list(_FENE8_RUNS) + list(_FENE7_RUNS) + [
            "T1", "T2", "T3", "clean_analytic_giesekus", "clean_analytic_fene_p"]:
        r[f"battery_target_{target}"] = fb / "targets" / f"{target}.json"
        r[f"battery_fits_{target}"] = fb / "fits" / target
        # AOS protocol arrays the candidates were fitted to: time, sigma_noisy,
        # gammadot, 12 legs x 361 samples.
        r[f"battery_data_{target}"] = fb / "data" / f"{target}.npz"
        r[f"battery_datameta_{target}"] = fb / "data" / f"{target}.json"

    # ---------------- FENE-P networks (SN4, SN5, Table S2) ---------------
    for run in _FENE8_RUNS:
        base = ROOT / "fene8_prod" / run
        r[f"fene_summary_{run}"] = base / "summary.json"
        r[f"fene_progress_{run}"] = base / "progress.csv"
        r[f"fene_arrays_{run}"] = base / "arrays.npz"
    for label, sub in _FENE7_RUNS.items():
        base = ROOT / "fene7_prod" / sub
        r[f"fene_summary_{label}"] = base / "summary.json"
        r[f"fene_progress_{label}"] = base / "progress.csv"
        r[f"fene_arrays_{label}"] = base / "arrays.npz"
    r["fene_invocation_recovery"] = ROOT / "fene8_prod/invocation_recovery.md"
    r["fene_campaign_inventory"] = ROOT / "fene8_campaign_inventory.md"
    r["fene_truth_traj"] = pr / "_campaign_fene/regen/truth_traj.npz"
    r["fene_init_traj"] = pr / "_campaign_fene/regen/init_traj.npz"
    r["fene_campaign_manifest"] = pr / "_campaign_fene/regen/regen_manifest.json"

    # ---------------- FENE-P direct calibration (SN6) --------------------
    for fit in _DIRECT_FITS:
        base = ROOT / "fene8_direct" / f"direct_u05_{fit}"
        r[f"direct_progress_{fit}"] = base / "progress.csv"
        r[f"direct_summary_{fit}"] = base / "summary.json"
    r["direct_gate_u05_summary"] = (
        ROOT / "fene8_direct/direct_gate_u05/summary.json")
    r["direct_driver"] = ROOT / "visco_opt_fenep_direct_contraction_run.py"
    # Why the I4 start is excluded from SN6: the U=4-only adjoint probe at that
    # start, and the forensics write-up of the two dual-rate I4 failures.
    _i4 = ROOT / "fene8_direct/i4_forensics"
    r["direct_i4_u4_probe"] = _i4 / "u4_grad_probe/summary.json"
    r["direct_i4_forensics"] = _i4 / "i4_forensics.md"
    r["direct_i4_equal_log"] = _i4 / "a1_a2/36139679_fene8dir.out"

    # ---------------- Cavity transfer (N1e, SN3) -------------------------
    cav = ROOT / "cavity_outputs/transfer_prod"
    for de in CAVITY_DE:
        for arm, sub in (("truth", "de_ladder"),
                         ("learned", "learned_s4_de_ladder")):
            d = cav / sub / f"De{de:.2f}"
            r[f"cavity_{arm}_config_De{de:.2f}"] = d / "config.json"
            r[f"cavity_{arm}_result_De{de:.2f}"] = d / "result.json"
            r[f"cavity_{arm}_diag_De{de:.2f}"] = d / "diagnostics.npz"
    r["cavity_transfer_metrics"] = cav / "comparison_s4/transfer_metrics.json"

    # ---------------- EVP channel (N2, SN7, SN8) -------------------------
    se = pr / "evp_fix_seed_eval"
    r["evp_seed_eval_summary"] = se / "seed_eval_summary.json"
    r["evp_truth_ladder"] = se / "truth_ladder.json"
    for gx, t in EVP_DRIVES.items():
        r[f"evp_truth_gx{t}"] = se / f"truth_gx{t}.npz"
        for seed, run in _EVP_SEEDS.items():
            r[f"evp_{seed}_gx{t}"] = se / f"{run}_gx{t}.npz"
    for seed, run in _EVP_SEEDS.items():
        base = ROOT / "tbnn_evp_data/evp_fix" / run
        r[f"evp_ckpt_{seed}"] = base / "theta_checkpoint.npz"
        r[f"evp_progress_{seed}"] = base / "progress.csv"
        r[f"evp_batch_metrics_{seed}"] = base / "batch_metrics.json"
        r[f"evp_config_{seed}"] = base / "config.json"
        for tag in ("stage1", "resolve0", "resolve1", "resolve2",
                    "resolve3", "final"):
            r[f"evp_stage_{seed}_{tag}"] = base / f"ckpt_stage{tag}.pkl"
    r["evp_targets_A_3lam"] = ROOT / "tbnn_evp_data/evp_fix_targets/targets_A_3lam.json"
    r["evp_phaseB"] = pr / "evp_forward_diag/phaseB.json"
    r["evp_8arm_summary"] = pr / "evp_fix_eval/eval_summary.json"
    r["evp_remediation_validation"] = (
        ROOT / "tbnn_evp_data/evp_final_prod/phase0_remediation_12.json")

    # ---------------- source files cited in verification -----------------
    r["src_contraction_geometry"] = ROOT / "jax_rheology/contraction_geometry.py"
    r["src_regen_contraction"] = ROOT / "regen_contraction.py"
    r["src_evp_fix_seed_eval"] = ROOT / "evp_fix_seed_eval.py"
    r["src_evp_forward_diag"] = ROOT / "evp_forward_diag.py"
    r["src_evp_final_phase0"] = ROOT / "evp_final_phase0.py"
    r["src_pub_style"] = ROOT / "pub_style.py"
    r["src_paper_plots_nb"] = ROOT / "paper_plots.ipynb"
    r["src_passc_figures"] = ROOT / "passc_contraction_figures.py"
    r["src_cavity_comparison"] = ROOT / "cavity_transfer_comparison.py"
    return r


REGISTRY: dict[str, Path] = _build()

# Forward runs and derived caches written by the figure helpers.  Validated
# on access, not at import, because they may not exist yet.
DEFERRED: dict[str, Path] = {
    "gie_init_final_fields": DERIVED_ROOT / "gie_init_final_fields.npz",
    "fene_repr_forward": DERIVED_ROOT / "fene_repr_forward.npz",
    "evp_obinit_eval": DERIVED_ROOT / "evp_obinit",
}
# Candidate AOS replays, one per battery target (paper_figs/derive_aos.py).
for _t in ("gie_A_s4",):
    DEFERRED[f"aos_{_t}"] = DERIVED_ROOT / f"aos_{_t}.npz"

_BUNDLE = _bundle_root()
if _BUNDLE is not None:
    _man = json.loads((Path(__file__).resolve().parent / "bundle_manifest.json").read_text())
    for _k, _rel in _man.get("keys", {}).items():
        _dest = _BUNDLE / _rel
        if _k in REGISTRY:
            REGISTRY[_k] = _dest
        else:
            DEFERRED[_k] = _dest
    for _k in _man.get("t3_skipped_keys", []):
        if _k in REGISTRY:
            DEFERRED[_k] = REGISTRY.pop(_k)


def _validate() -> list[str]:
    missing = []
    for key, p in REGISTRY.items():
        try:
            os.stat(p)
        except OSError:
            missing.append(f"{key} -> {p}")
    return missing


_MISSING = _validate()
if _MISSING:
    raise MissingArtifact(
        "paper_figs.data_paths: %d registered artifact(s) absent:\n  %s"
        % (len(_MISSING), "\n  ".join(_MISSING)))


def path(key: str, *, required: bool = True) -> Path:
    """Resolve a registry key to an absolute path."""
    if key in REGISTRY:
        p = REGISTRY[key]
    elif key in DEFERRED:
        p = DEFERRED[key]
        if required:
            try:
                os.stat(p)
            except OSError as exc:
                raise MissingArtifact(
                    f"deferred artifact '{key}' not produced yet: {p}") from exc
    else:
        raise KeyError(
            f"unknown registry key '{key}'; register it in "
            f"paper_figs/data_paths.py")
    s = str(p)
    if not any(s.endswith(e) for e in FORBIDDEN_EXCEPTIONS):
        for bad in FORBIDDEN:
            if bad in s:
                raise ForbiddenArtifact(
                    f"registry key '{key}' resolves into a DEAD / DO-NOT-QUOTE "
                    f"artifact ({bad})")
    return p


def out_dir(figure_id: str) -> Path:
    d = OUT_ROOT / figure_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def derived_dir() -> Path:
    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)
    return DERIVED_ROOT


def keys(prefix: str = "") -> list[str]:
    return sorted(k for k in REGISTRY if k.startswith(prefix))
