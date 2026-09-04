#!/usr/bin/env python3
"""Assemble the FENE-P battery report from the fit artifacts on disk.

Reads the completed battery output and writes one markdown report covering
the legacy versus balanced training arms, single-rate velocity-only recovery,
the direct differentiable calibration, and the publication-settings tables.
CPU-only: it reduces existing results and runs no fits. The report path is
``REPORT`` below.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
OUT = REPO_ROOT / "work/bic_battery"
REPORT = OUT / "final_report_PR18-PR21.md"
NOISE_DESIGN = 0.03 ** 2
TRUTH = dict(eta_p=2.24, lam=0.7, nu_s=0.8, Lsq=12.0)


def load_target(tid: str) -> dict:
    return json.loads((OUT / "targets" / f"{tid}.json").read_text())


def fenep_row(tid: str) -> dict:
    t = load_target(tid)
    fen = next(r for r in t["results"] if r["name"] == "FENEPConformation")
    p = fen["params"]
    noise = NOISE_DESIGN
    meta_p = OUT / "data" / f"{tid}.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("noise_mse") is not None:
            noise = float(meta["noise_mse"])
    return {
        "target": tid,
        "winner": t["winner"],
        "margin": float(t["margin"]),
        "eta_p": float(p["polymer_viscosity"]),
        "lam": float(p["relaxation_time"]),
        "nu_s": float(p["solvent_viscosity"]),
        "Lsq": float(p["extension_length"]) ** 2,
        "mse": float(fen["mse"]),
        "noise_mse": noise,
        "resid_x": float(fen["mse"]) / noise,
        "restart_spread": fen.get("restart_spread"),
        "protocol": t.get("protocol"),
    }


def mean_sd(vals):
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def fmt(x, digs=6):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.4e}"
    return f"{x:.{digs}g}"


def direct_summary():
    """PR20 table from fene8_direct summaries + I4 forensics amendment."""
    regimes = [
        ("u05", "single-rate", "direct_u05", "direct_gate_u05"),
        ("dual_legacy", "dual legacy-weighting", "direct_dual_legacy",
         "direct_gate_dual_legacy"),
        ("dual_equal", "dual equal-weighting", "direct_dual_equal",
         "direct_gate_dual_equal"),
    ]
    starts = ["I1", "I2", "I3", "I4", "I5"]
    rows = []
    gates = {}
    for key, label, pref, gate in regimes:
        gp = ROOT / "fene8_direct" / gate / "summary.json"
        gates[key] = json.loads(gp.read_text()) if gp.exists() else None
        for st in starts:
            name = f"{pref}_{st}"
            p = ROOT / "fene8_direct" / name
            if not (p / "summary.json").exists():
                rows.append({
                    "run": name, "regime": label, "start": st,
                    "status": "DID NOT START / FAILED @ init",
                    "note": "see fene8_direct/i4_forensics/",
                })
                continue
            s = json.loads((p / "summary.json").read_text())
            fin = s["final"]
            rows.append({
                "run": name, "regime": label, "start": st,
                "status": "DONE",
                "nit": s.get("nit"), "nfev": s.get("nfev"),
                "converged": s.get("converged"),
                "loss_final": s.get("loss_final"),
                "eta_p": fin.get("eta_p"), "lam": fin.get("lam"),
                "nu_s": fin.get("nu_s"), "Lsq": fin.get("Lsq"),
                "init": s.get("init"),
            })
    return gates, rows


def main() -> int:
    # Require bal_s3/s4 readback DONE (dependency should guarantee this).
    for r in ("fene8_bal_s3", "fene8_bal_s4"):
        if not (OUT / "readback" / r / "DONE").exists():
            raise SystemExit(f"[HALT] readback not DONE for {r}")
        if not (OUT / "targets" / f"{r}.json").exists():
            raise SystemExit(f"[HALT] missing targets/{r}.json")

    legacy_ids = ["R2", "R5", "R6", "fene8_leg_s3", "fene8_leg_s4"]
    bal_ids = [f"fene8_bal_s{i}" for i in range(5)]
    u05_ids = ["R3"] + [f"fene8_u05_s{i}" for i in range(1, 5)]
    gie_ids = ["gie_A_s1", "gie_A_s1b", "gie_A_s4"]
    clean_id = "clean_analytic_fene_p"

    legacy = [fenep_row(t) for t in legacy_ids]
    bal = [fenep_row(t) for t in bal_ids]
    u05 = [fenep_row(t) for t in u05_ids]
    gie = [fenep_row(t) for t in gie_ids]
    clean = fenep_row(clean_id)

    def arm_stats(rows, key):
        vals = [r[key] for r in rows]
        m, s = mean_sd(vals)
        return m, s, vals

    lines = []
    lines.append("# FENE8 final report -- PR18-PR21")
    lines.append("")
    lines.append(
        f"Assembled {datetime.now(timezone.utc).isoformat()} from on-disk "
        f"final-settings battery (800/6/60, three restarts) and direct-fit "
        f"summaries. Answers **only** the pre-registered reads in "
        f"`preregistered_reads.md`. Vocabulary: this file is the sole "
        f"**Final** report."
    )
    lines.append("")
    lines.append("Supporting maps (not registered reads): "
                 "`fene8_campaign_inventory.md`, `WHERE_EVERYTHING_IS.md`, "
                 "`fene8_interim_parameters.md` (superseded by this file), "
                 "`fene8_direct/i4_forensics/`.")
    lines.append("")

    # ---- legacy arm versus balanced arm ----
    lines.append("## PR18 -- legacy arm vs balanced arm")
    lines.append("")
    lines.append(
        "Legacy arm n=5: R2, R5, R6, `fene8_leg_s3`, `fene8_leg_s4`. "
        "Balanced arm n=5: `fene8_bal_s0`..`s4`. Shared seed labels "
        "{0..4} where applicable (R2 has no seed twin; R5<->seed1, R6<->seed2, "
        "leg_s3<->3, leg_s4<->4)."
    )
    lines.append("")
    lines.append("| arm | mean+/-sd etap | mean+/-sd lam | mean+/-sd nus | mean+/-sd L^2 | "
                 "mean+/-sd residual/noise | mean+/-sd margin |")
    lines.append("|-----|------------|-----------|------------|------------|"
                 "------------------------|----------------|")
    for name, rows in (("legacy", legacy), ("balanced", bal)):
        cells = []
        for k in ("eta_p", "lam", "nu_s", "Lsq", "resid_x", "margin"):
            m, s, _ = arm_stats(rows, k)
            cells.append(f"{fmt(m)}+/-{fmt(s)}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### Per-seed rows")
    lines.append("")
    lines.append("| run | arm | etap | lam | nus | L^2 | residxfloor | margin |")
    lines.append("|-----|-----|----|---|----|----|-------------|--------|")
    for r in legacy:
        lines.append(
            f"| {r['target']} | legacy | {fmt(r['eta_p'])} | {fmt(r['lam'])} | "
            f"{fmt(r['nu_s'])} | {fmt(r['Lsq'])} | {fmt(r['resid_x'])} | "
            f"{fmt(r['margin'])} |"
        )
    for r in bal:
        lines.append(
            f"| {r['target']} | balanced | {fmt(r['eta_p'])} | {fmt(r['lam'])} | "
            f"{fmt(r['nu_s'])} | {fmt(r['Lsq'])} | {fmt(r['resid_x'])} | "
            f"{fmt(r['margin'])} |"
        )
    lines.append("")
    # paired deltas on seeds 1-4 where both arms exist
    pair_map = {
        1: ("R5", "fene8_bal_s1"),
        2: ("R6", "fene8_bal_s2"),
        3: ("fene8_leg_s3", "fene8_bal_s3"),
        4: ("fene8_leg_s4", "fene8_bal_s4"),
    }
    lines.append("### Paired deltas (balanced - legacy) on shared seeds 1-4")
    lines.append("")
    lines.append("| seed | legacy | balanced | Deltaetap | Deltalam | Deltanus | DeltaL^2 | Deltaresidx | Deltamargin |")
    lines.append("|------|--------|----------|-----|----|-----|-----|---------|---------|")
    for seed, (leg_id, bal_id) in pair_map.items():
        L = next(r for r in legacy if r["target"] == leg_id)
        B = next(r for r in bal if r["target"] == bal_id)
        lines.append(
            f"| {seed} | {leg_id} | {bal_id} | "
            f"{fmt(B['eta_p']-L['eta_p'])} | {fmt(B['lam']-L['lam'])} | "
            f"{fmt(B['nu_s']-L['nu_s'])} | {fmt(B['Lsq']-L['Lsq'])} | "
            f"{fmt(B['resid_x']-L['resid_x'])} | {fmt(B['margin']-L['margin'])} |"
        )
    lines.append("")
    # variance comparison
    lines.append("### Variance comparison (sample variance of joint-fit params)")
    lines.append("")
    lines.append("| param | var(legacy) | var(balanced) |")
    lines.append("|-------|-------------|---------------|")
    for k in ("eta_p", "lam", "nu_s", "Lsq", "resid_x", "margin"):
        vl = statistics.variance([r[k] for r in legacy]) if len(legacy) > 1 else float("nan")
        vb = statistics.variance([r[k] for r in bal]) if len(bal) > 1 else float("nan")
        lines.append(f"| {k} | {fmt(vl)} | {fmt(vb)} |")
    lines.append("")
    lines.append(
        "Basin telemetry / legacy seed-0 basin recurrence: balanced seed 0 "
        f"(`fene8_bal_s0`) joint fit "
        f"(etap={fmt(bal[0]['eta_p'])}, lam={fmt(bal[0]['lam'])}, "
        f"nus={fmt(bal[0]['nu_s'])}, L^2={fmt(bal[0]['Lsq'])}); "
        "training-incident lineage in "
        "`fene8_prod/fene8_bal_s0/PROVENANCE_NOTE.md`."
    )
    lines.append(
        f"Balanced-arm training survival count: "
        f"{sum(1 for i in range(5) if (ROOT/'fene8_prod'/f'fene8_bal_s{i}'/'DONE').exists())}"
        f" / 5 DONE."
    )
    lines.append("")

    # ---- single-rate velocity-only recovery ----
    lines.append("## PR19 -- single-rate velocity-only recovery")
    lines.append("")
    lines.append("n=5: R3, `fene8_u05_s1`..`s4`. Nothing excluded.")
    lines.append("")
    lines.append("| run | etap | lam | nus | L^2 | residxfloor | margin |")
    lines.append("|-----|----|---|----|----|-------------|--------|")
    for r in u05:
        lines.append(
            f"| {r['target']} | {fmt(r['eta_p'])} | {fmt(r['lam'])} | "
            f"{fmt(r['nu_s'])} | {fmt(r['Lsq'])} | {fmt(r['resid_x'])} | "
            f"{fmt(r['margin'])} |"
        )
    lines.append("")
    cells = []
    for k in ("eta_p", "lam", "nu_s", "Lsq"):
        m, s, vals = arm_stats(u05, k)
        cells.append(f"{k}: {fmt(m)} +/- {fmt(s)}")
    lines.append("Mean +/- s.d.: " + "; ".join(cells) + ".")
    # flag outliers by |z|>2 on each param if n>=3
    lines.append("")
    lines.append("Outlier flags (|value-mean| > 2.sd, per parameter; empty if none):")
    for k in ("eta_p", "lam", "nu_s", "Lsq", "resid_x", "margin"):
        m, s, _ = arm_stats(u05, k)
        if s == 0 or math.isnan(s):
            continue
        outs = [r["target"] for r in u05 if abs(r[k] - m) > 2 * s]
        lines.append(f"- {k}: {', '.join(outs) if outs else '(none)'}")
    lines.append("")

    # ---- direct differentiable FENE-P calibration ----
    lines.append("## PR20 -- direct differentiable FENE-P calibration")
    lines.append("")
    lines.append(
        "Truth (etap, lam, nus, L^2) = (2.24, 0.7, 0.8, 12). "
        "Dual I4 rows amended by `fene8_direct/i4_forensics/pr20_amendment_i4.md`."
    )
    lines.append("")
    gates, drows = direct_summary()
    lines.append("### Truth-init gates")
    lines.append("")
    lines.append("| gate | pass | loss_final | max param drift % |")
    lines.append("|------|------|------------|-------------------|")
    for key, gname in (
        ("u05", "direct_gate_u05"),
        ("dual_legacy", "direct_gate_dual_legacy"),
        ("dual_equal", "direct_gate_dual_equal"),
    ):
        g = gates.get(key)
        if not g:
            lines.append(f"| {gname} | MISSING | -- | -- |")
            continue
        tg = g.get("truth_init_gate") or {}
        lines.append(
            f"| {gname} | {tg.get('pass')} | {fmt(g.get('loss_final'))} | "
            f"{fmt(tg.get('max_param_drift_pct'))} |"
        )
    lines.append("")
    lines.append("### Production starts")
    lines.append("")
    lines.append("| run | regime | start | status | nit | converged | "
                 "final (etap, lam, nus, L^2) | loss_final |")
    lines.append("|-----|--------|-------|--------|-----|-----------|"
                 "------------------------|------------|")
    truth_counts = {}
    for row in drows:
        if row["status"] != "DONE":
            lines.append(
                f"| {row['run']} | {row['regime']} | {row['start']} | "
                f"{row['status']} | -- | -- | -- | -- |"
            )
            continue
        near = all(
            abs(row[k] - TRUTH[k]) / TRUTH[k] < 1e-2
            for k in ("eta_p", "lam", "nu_s", "Lsq")
        )
        tiny = row["loss_final"] is not None and row["loss_final"] < 1e-9
        if near and tiny:
            truth_counts[row["regime"]] = truth_counts.get(row["regime"], 0) + 1
        params = (
            f"({fmt(row['eta_p'])}, {fmt(row['lam'])}, "
            f"{fmt(row['nu_s'])}, {fmt(row['Lsq'])})"
        )
        lines.append(
            f"| {row['run']} | {row['regime']} | {row['start']} | DONE | "
            f"{row['nit']} | {row['converged']} | {params} | "
            f"{fmt(row['loss_final'])} |"
        )
    lines.append("")
    lines.append(
        "Starts converging to truth (loss < 1e-9 and all params within 1%): "
        + ", ".join(f"{k}: {v}" for k, v in sorted(truth_counts.items()))
        + "."
    )
    lines.append("")
    lines.append(
        "Regime (b) dual-rate velocity+pressure legacy normalization: "
        "numbers reported as-is; interpretation flag per preregistered read."
    )
    lines.append("")
    # fold I4 amendment pointer
    amend = ROOT / "fene8_direct/i4_forensics/pr20_amendment_i4.md"
    if amend.exists():
        lines.append("### I4 amendment (verbatim path)")
        lines.append("")
        lines.append(f"See `{amend.relative_to(ROOT)}`.")
        lines.append("")

    # ---- publication-settings battery tables ----
    lines.append("## PR21 -- publication-settings battery tables")
    lines.append("")
    lines.append(
        "Protocol: 800 epochs, 6 cycles, 60 points/leg, three optimizer "
        "restarts (seeds 101/202/303). Prior 500/4/20 numbers are superseded."
    )
    lines.append("")
    all_targets = (
        ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "T1", "T2", "T3"]
        + [f"fene8_u05_s{i}" for i in range(1, 5)]
        + ["fene8_leg_s3", "fene8_leg_s4"]
        + [f"fene8_bal_s{i}" for i in range(5)]
        + gie_ids
        + ["clean_analytic_fene_p", "clean_analytic_giesekus"]
    )
    lines.append("| target | winner | margin | FENEP etap | FENEP lam | FENEP nus | "
                 "FENEP L^2 | FENEP mse | residxfloor | BIC spread (FENEP) |")
    lines.append("|--------|--------|--------|----------|---------|----------|"
                 "----------|-----------|-------------|---------------------|")
    for tid in all_targets:
        tp = OUT / "targets" / f"{tid}.json"
        if not tp.exists():
            lines.append(f"| {tid} | MISSING | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        t = load_target(tid)
        fen = next((r for r in t["results"] if r["name"] == "FENEPConformation"), None)
        if fen is None:
            lines.append(
                f"| {tid} | {t.get('winner')} | {fmt(t.get('margin'))} | "
                f"-- | -- | -- | -- | -- | -- | -- |"
            )
            continue
        p = fen["params"]
        noise = NOISE_DESIGN
        mp = OUT / "data" / f"{tid}.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
            if meta.get("noise_mse") is not None:
                noise = float(meta["noise_mse"])
        spread = fen.get("restart_spread") or {}
        bics = spread.get("bic") or []
        bic_note = (
            f"[{fmt(min(bics))} ... {fmt(max(bics))}]" if bics else "--"
        )
        lines.append(
            f"| {tid} | {t['winner']} | {fmt(t['margin'])} | "
            f"{fmt(p['polymer_viscosity'])} | {fmt(p['relaxation_time'])} | "
            f"{fmt(p['solvent_viscosity'])} | {fmt(p['extension_length']**2)} | "
            f"{fmt(fen['mse'])} | {fmt(fen['mse']/noise)} | {bic_note} |"
        )
    lines.append("")
    lines.append("### Giesekus margins (publication settings) beside July values")
    lines.append("")
    # July values from interim_report_1 if present
    july = {
        "gie_A_s1": 6748.110607570718,
        "gie_A_s1b": 7713.416761387069,
        "gie_A_s4": 8401.796897681634,
    }
    lines.append("| target | margin (800/6/60, this battery) | July matched column |")
    lines.append("|--------|--------------------------------:|--------------------:|")
    for tid in gie_ids:
        r = next(x for x in gie if x["target"] == tid)
        lines.append(
            f"| {tid} | {fmt(r['margin'])} | {fmt(july.get(tid))} |"
        )
    lines.append("")
    lines.append(
        f"Clean analytic FENE-P control: winner={clean['winner']}, "
        f"margin={fmt(clean['margin'])}, "
        f"params=({fmt(clean['eta_p'])}, {fmt(clean['lam'])}, "
        f"{fmt(clean['nu_s'])}, {fmt(clean['Lsq'])}), "
        f"residxfloor={fmt(clean['resid_x'])}."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### QA notes")
    lines.append("")
    lines.append(
        "`fene8_bal_s3` Giesekus Adam (seed 101) diverged at epoch 112 "
        "(nonfinite loss/grad; alpha~=1) under unchanged 800/6/60 -- a "
        "far-from-family symptom recorded as DIVERGED/BIC=+inf per the "
        "epoch-218 convention; see "
        "`fene8_direct/i4_forensics/bal_s3_giesekus_debug/RECOVERY_NOTE.md`."
    )
    lines.append("")
    lines.append(
        f"Job / host: SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'local')}; "
        f"noise floor for learned targets = design 0.03^2 = {NOISE_DESIGN}."
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    done = OUT / "FINAL_REPORT_DONE"
    done.write_text(
        f"{os.environ.get('SLURM_JOB_ID', 'local')}\n{REPORT}\n"
    )
    print(f"[ok] wrote {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
