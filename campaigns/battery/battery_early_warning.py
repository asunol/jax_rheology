#!/usr/bin/env python
"""Read a battery pass while it is still running and flag discrepancies.

Numbers only: reports what the fits have produced so far and which values
look inconsistent, so a long pass can be abandoned early. Writes an interim
markdown report; it changes nothing.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

if os.environ.get("JAX_PLATFORMS") != "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
FINAL = REPO_ROOT / "work/bic_battery"
PRIOR = ROOT / "analysis_phase2_instrument/corrected_battery"
FENE = [f"R{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 4)]
GIE = ["gie_A_s1", "gie_A_s1b", "gie_A_s4"]
CONTROLS = ["clean_analytic_fene_p", "clean_analytic_giesekus"]


def read_json(path):
    return json.loads(path.read_text())


def main():
    prior_gie = {
        row["target"]: row
        for row in csv.DictReader((PRIOR / "giesekus_winners.csv").open())
    }
    rows = []
    verdict_changes = []
    control_rows = []
    gie_rows = []
    for target in FENE + GIE + CONTROLS:
        cur = read_json(FINAL / "targets" / f"{target}.json")
        old = read_json(PRIOR / "targets" / f"{target}.json")
        row = {
            "target": target,
            "winner_800_6_60": cur.get("winner"),
            "margin_800_6_60": cur.get("margin"),
            "winner_500_4_20": old.get("winner"),
            "margin_500_4_20": old.get("margin"),
            "verdict_changed": cur.get("winner") != old.get("winner"),
        }
        # restart spread for winner
        win = next(
            (r for r in cur.get("results", []) if r.get("name") == cur.get("winner")),
            None,
        )
        if win and "restart_spread" in win:
            bics = win["restart_spread"].get("bic", [])
            row["winner_restart_bic"] = bics
            row["winner_restart_bic_spread"] = (
                (max(bics) - min(bics)) if bics else None
            )
        rows.append(row)
        if target in FENE and row["verdict_changed"]:
            verdict_changes.append(row)
        if target in CONTROLS:
            control_rows.append(row)
        if target in GIE or target == "clean_analytic_giesekus":
            july = prior_gie.get(target, {})
            gie_rows.append(
                {
                    **row,
                    "july_margin_800_6x60": july.get("july_margin_800_6x60"),
                    "july_protocol": july.get("july_protocol"),
                }
            )

    # Control floors: FENE should win analytic FENE; Giesekus should win analytic Gie
    flags = []
    for row in control_rows:
        if row["target"] == "clean_analytic_fene_p":
            if row["winner_800_6_60"] != "FENEPConformation":
                flags.append(
                    f"CONTROL_FLOOR_MISS clean_analytic_fene_p "
                    f"winner={row['winner_800_6_60']}"
                )
        if row["target"] == "clean_analytic_giesekus":
            if row["winner_800_6_60"] != "Giesekus":
                flags.append(
                    f"CONTROL_FLOOR_MISS clean_analytic_giesekus "
                    f"winner={row['winner_800_6_60']}"
                )
    for row in verdict_changes:
        flags.append(
            f"VERDICT_CHANGE {row['target']}: "
            f"{row['winner_500_4_20']}->{row['winner_800_6_60']} "
            f"margins {row['margin_500_4_20']}->{row['margin_800_6_60']}"
        )

    lines = [
        "# Interim report 1 -- early-warning read (2c)",
        "",
        "Status only. Numbers as-is. No interpretation beyond discrepancy flags.",
        "",
        "## Protocol",
        "",
        "- final battery: 800 epochs, 6 cycles, 60 pts/leg, 3 restarts",
        "- prior corrected battery: 500 epochs, 4 cycles, 20 pts/leg",
        "",
        "## FENE7-network family verdicts (R1-R7, T1-T3)",
        "",
        "| target | winner_800_6_60 | margin_800_6_60 | winner_500_4_20 | margin_500_4_20 | changed |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["target"] not in FENE:
            continue
        lines.append(
            f"| {row['target']} | {row['winner_800_6_60']} | "
            f"{row['margin_800_6_60']} | {row['winner_500_4_20']} | "
            f"{row['margin_500_4_20']} | {row['verdict_changed']} |"
        )
    lines += [
        "",
        "## Clean controls",
        "",
        "| target | winner_800_6_60 | margin_800_6_60 | winner_500_4_20 | margin_500_4_20 |",
        "|---|---|---|---|---|",
    ]
    for row in control_rows:
        lines.append(
            f"| {row['target']} | {row['winner_800_6_60']} | "
            f"{row['margin_800_6_60']} | {row['winner_500_4_20']} | "
            f"{row['margin_500_4_20']} |"
        )
    lines += [
        "",
        "## Giesekus margins vs July (matched 800/6/60 column from prior table)",
        "",
        "| target | margin_800_6_60 | margin_500_4_20 | july_margin_800_6x60 |",
        "|---|---|---|---|",
    ]
    for row in gie_rows:
        lines.append(
            f"| {row['target']} | {row['margin_800_6_60']} | "
            f"{row['margin_500_4_20']} | {row.get('july_margin_800_6x60')} |"
        )
    lines += ["", "## Discrepancy flags", ""]
    if flags:
        for flag in flags:
            lines.append(f"- {flag}")
    else:
        lines.append("- none")
    lines.append("")
    text = "\n".join(lines)
    out = FINAL / "interim_report_1.md"
    out.write_text(text)
    (FINAL / "early_warning.json").write_text(
        json.dumps(
            {
                "rows": rows, "gie_rows": gie_rows, "flags": flags,
                "n_verdict_changes_fene7": len(verdict_changes),
            },
            indent=2,
            default=float,
        )
        + "\n"
    )
    print(text)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
