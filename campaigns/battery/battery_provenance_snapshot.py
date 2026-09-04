#!/usr/bin/env python
"""Record and compare the identity of the diff_rheo tree the battery fits with.

The snapshot is content-derived: the sha256 and size of every Python file in
the tree, one ``tree_sha256`` over all of them, the declared upstream
revision, and the sha256 of the production battery script. Comparing a later
snapshot against the recorded one is what stops a battery pass from silently
running against changed fitting code.

The tree is resolved through ``repo_paths.DIFF_RHEO``, so it is the vendored
copy by default and ``TBNN_DIFF_RHEO`` can point it at a separate checkout.
When that checkout is a git repository the snapshot additionally records the
HEAD, the porcelain status, and a hash of the tracked diff; the vendored copy
has no ``.git``, so those fields are absent there and the comparison runs on
the content fields alone.

Snapshots live in ``reference_values/battery_provenance``.
``battery_provenance_check.py`` runs the same logic into a subdirectory of it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT.parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT.parents[1]))
from repo_paths import DIFF_RHEO, REPO_ROOT  # noqa: E402

DIFF = DIFF_RHEO.resolve()
OUT = REPO_ROOT / "reference_values" / "battery_provenance"
PRODUCTION = ROOT / "tbnn_bic_model_selection.py"

#: Upstream revision the vendored tree was taken from, as declared in
#: ``diff_rheo/LICENSE_NOTE``. Recorded so a snapshot says which upstream it
#: corresponds to even though the vendored copy carries no git history.
VENDORED_REVISION = "nat_coms @ 92d9dada"


def is_git_checkout() -> bool:
    """True only when DIFF is its own repository.

    The vendored tree is a plain directory inside this repository, so running
    git there would report *this* repository's HEAD and working-tree status.
    """
    return (DIFF / ".git").exists()


def run(*args: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-C", str(DIFF), *args],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".py":
        relative = path.relative_to(DIFF)
        module_parts = list(relative.with_suffix("").parts)
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        importable = bool(module_parts) and all(part.isidentifier() for part in module_parts)
        return "importable_code" if importable else "python_code_nonmodule_path"
    if suffix in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        return "generated_plot_artifact"
    if suffix in {".txt", ".csv", ".json", ".npz", ".pkl"}:
        return "result_or_data_artifact"
    if suffix == ".ipynb":
        return "notebook_code_artifact"
    if "__pycache__" in path.parts or suffix in {".pyc", ".so"}:
        return "cache_or_build_artifact"
    return "other_untracked_artifact"


def untracked_paths():
    if not is_git_checkout():
        return []
    raw = run("status", "--porcelain=v1", "-z", "--untracked-files=all")
    result = []
    for entry in raw.split("\0"):
        if entry.startswith("?? "):
            result.append((DIFF / entry[3:]).resolve())
    return sorted(result)


def python_paths():
    paths = []
    for path in DIFF.rglob("*.py"):
        if ".git" not in path.parts and "__pycache__" not in path.parts:
            paths.append(path.resolve())
    return sorted(paths)


def tree_sha256(python_rows) -> str:
    """One hash over every Python file in the tree, path and content."""
    h = hashlib.sha256()
    for row in python_rows:
        h.update(row["path"].encode())
        h.update(b"\0")
        h.update(row["sha256"].encode())
        h.update(b"\n")
    return h.hexdigest()


def snapshot_payload():
    git = is_git_checkout()
    status = run("status", "--porcelain=v1", "--untracked-files=all") if git else None
    diff = run("diff", "--no-ext-diff", "--binary", binary=True) if git else b""
    untracked = [
        {
            "path": str(path.relative_to(DIFF)),
            "sha256": sha256(path),
            "size": path.stat().st_size,
            "classification": classify(path),
            "importable_python_flag": classify(path) == "importable_code",
        }
        for path in untracked_paths()
    ]
    python = [
        {
            "path": str(path.relative_to(DIFF)),
            "sha256": sha256(path),
            "size": path.stat().st_size,
            "tracked": not str(
                run("ls-files", "--error-unmatch", str(path.relative_to(DIFF)))
                if subprocess.run(
                    ["git", "-C", str(DIFF), "ls-files", "--error-unmatch",
                     str(path.relative_to(DIFF))],
                    capture_output=True,
                ).returncode == 0 else ""
            ).strip() == "",
        }
        for path in python_paths()
    ]
    try:
        tree_name = str(DIFF.relative_to(REPO_ROOT))
    except ValueError:
        # DIFF was pointed outside the repository via TBNN_DIFF_RHEO; record
        # only its name, never an absolute path.
        tree_name = DIFF.name
    payload = {
        "tree": tree_name,
        "is_git_checkout": git,
        "vendored_revision": VENDORED_REVISION,
        "tree_sha256": tree_sha256(python),
        "n_python_files": len(python),
        "untracked": untracked,
        "python": python,
        "production_battery_sha256": sha256(PRODUCTION),
    }
    if git:
        payload["head"] = run("rev-parse", "HEAD").strip()
        payload["status"] = status
        payload["diff_sha256"] = hashlib.sha256(diff).hexdigest()
        payload["diff_bytes"] = len(diff)
    return payload, diff


def write_csv(path: Path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_snapshot(label: str):
    OUT.mkdir(parents=True, exist_ok=True)
    payload, diff = snapshot_payload()
    if payload.get("head"):
        (OUT / f"{label}_head.txt").write_text(payload["head"] + "\n")
        (OUT / f"{label}_status_porcelain.txt").write_text(payload["status"])
        (OUT / f"{label}_tracked_diff.patch").write_bytes(diff)
    write_csv(OUT / f"{label}_untracked.csv", payload["untracked"])
    write_csv(OUT / f"{label}_all_python_sha256.csv", payload["python"])
    (OUT / f"{label}_snapshot.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return payload


#: Fields compared between the recorded snapshot and a later one. Only those
#: present in both are used, so a content-only snapshot of the vendored tree
#: and a git snapshot of a separate checkout each compare on what they have.
COMPARED = (
    "vendored_revision", "tree_sha256", "n_python_files", "untracked",
    "python", "production_battery_sha256",
    "head", "status", "diff_sha256", "diff_bytes",
)


def compare(start: dict, current: dict) -> dict:
    return {
        key: start[key] == current[key]
        for key in COMPARED
        if key in start and key in current
    }


def final_gate():
    start = json.loads((OUT / "start_snapshot.json").read_text())
    final = write_snapshot("end")
    comparisons = compare(start, final)
    result = {
        "status": "pass" if all(comparisons.values()) else "fail",
        "comparisons": comparisons,
        "tree_sha256_expected": start["tree_sha256"],
        "tree_sha256_observed": final["tree_sha256"],
    }
    (OUT / "final_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "pass":
        raise SystemExit("diff_rheo/prod-battery provenance gate changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["start", "end"])
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(parser)
    if args.mode == "start":
        write_snapshot("start")
    else:
        final_gate()


if __name__ == "__main__":
    main()
