#!/usr/bin/env python
"""Regenerate final_figures/ end to end.

    python make_all_figures.py                 # everything
    python make_all_figures.py N1 SN7          # only those figures
    python make_all_figures.py --panels-only   # skip the assembled figures
    python make_all_figures.py --dpi 600
    python make_all_figures.py --index-only    # rewrite figures.md, render nothing

Nothing here fits, trains or differentiates: every panel reads frozen
archives through ``paper_figs.loaders``.  The run also rewrites
``final_figures/figures.md``, the index of what was produced.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import paper_figs as P  # noqa: E402
from paper_figs import data_paths as dp  # noqa: E402
from paper_figs.manifest import FIGURES, OBSOLETE, TABLES  # noqa: E402


def _human(n: int) -> str:
    return f"{n / 1024:.0f} kB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.1f} MB"


def render(only: list[str], panels_only: bool, dpi: int | None) -> list[dict]:
    rows = []
    for fig in FIGURES:
        if only and fig.figure not in only:
            continue
        for p in fig.panels:
            if panels_only and p.panel.endswith("_full"):
                continue
            fn = getattr(P, p.call)
            t0 = time.time()
            out = fn(dpi=dpi) if dpi else fn()
            plt.close("all")
            path = dp.out_dir(fig.figure) / f"{p.panel}.jpg"
            rows.append({"figure": fig.figure, "manuscript": fig.manuscript,
                         "title": fig.title, "panel": p.panel,
                         "call": p.call, "what": p.what, "source": p.source,
                         "path": path, "bytes": path.stat().st_size,
                         "seconds": time.time() - t0})
            print(f"  {p.panel:<10s} {rows[-1]['seconds']:6.1f} s  "
                  f"{_human(rows[-1]['bytes']):>8s}", flush=True)
            del out
    return rows


def render_tables(only: list[str]) -> list[dict]:
    rows = []
    if only and "tables" not in only:
        return rows
    for rel, call, label, what, source in TABLES:
        getattr(P, call)()
        path = dp.OUT_ROOT / rel
        rows.append({"figure": "tables", "manuscript": label, "title": label,
                     "panel": rel.split("/")[-1], "call": call, "what": what,
                     "source": source, "path": path,
                     "bytes": path.stat().st_size, "seconds": 0.0})
    return rows


def index_rows() -> list[dict]:
    """Manifest plus what is on disk, without rendering.

    Lets a subset run refresh figures.md: the index is a description of the
    manifest, so it does not need the panels to be redrawn.
    """
    rows = []
    for fig in FIGURES:
        for p in fig.panels:
            path = dp.out_dir(fig.figure) / f"{p.panel}.jpg"
            if not path.exists():
                continue
            rows.append({"figure": fig.figure, "manuscript": fig.manuscript,
                         "title": fig.title, "panel": p.panel, "call": p.call,
                         "what": p.what, "source": p.source, "path": path,
                         "bytes": path.stat().st_size, "seconds": 0.0})
    for rel, call, label, what, source in TABLES:
        path = dp.OUT_ROOT / rel
        if not path.exists():
            continue
        rows.append({"figure": "tables", "manuscript": label, "title": label,
                     "panel": rel.split("/")[-1], "call": call, "what": what,
                     "source": source, "path": path,
                     "bytes": path.stat().st_size, "seconds": 0.0})
    return rows


def write_index(rows: list[dict]) -> Path:
    by_fig: dict[str, list[dict]] = {}
    for r in rows:
        by_fig.setdefault(r["figure"], []).append(r)

    lines = ["# final_figures", "",
             "Regenerate with `python make_all_figures.py`; every panel is "
             "also callable on its own, e.g. `from paper_figs import "
             "plot_SN7b; plot_SN7b()`.",
             "Decisions, discrepancies and open questions behind these "
             "panels: [../paper_figs_findings.md](../paper_figs_findings.md).",
             "",
             "| figure | manuscript | panel | call | file | content | "
             "source archive |",
             "|---|---|---|---|---|---|---|"]
    esc = lambda s: s.replace("|", r"\|")   # pipes inside math break the table
    for fig in FIGURES + (None,):
        key = fig.figure if fig is not None else "tables"
        for r in by_fig.get(key, []):
            rel = r["path"].relative_to(dp.OUT_ROOT)
            lines.append(f"| {r['figure']} | {r['manuscript']} | "
                         f"{r['panel']} | `{r['call']}()` | "
                         f"[{rel}]({rel}) | {esc(r['what'])} | "
                         f"{esc(r['source'])} |")
    lines += ["", f"{len(rows)} files, "
              f"{_human(sum(r['bytes'] for r in rows))} total, "
              f"{sum(r['seconds'] for r in rows):.0f} s to render.", ""]
    lines += ["## Retired", "",
              "Not part of the figure set. The images are kept under "
              "`obsolete/` and `<figure>/retired/`, and the code is still "
              "callable; neither is rendered by this script.", "",
              "| was | why | where it went |", "|---|---|---|"]
    for was, why, went in OBSOLETE:
        lines.append(f"| {was} | {esc(why)} | {esc(went)} |")
    lines.append("")
    out = dp.OUT_ROOT / "figures.md"
    out.write_text("\n".join(lines))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("figures", nargs="*", help="subset, e.g. N1 SN7 tables")
    ap.add_argument("--panels-only", action="store_true")
    ap.add_argument("--index-only", action="store_true",
                    help="rewrite figures.md from the manifest, render nothing")
    ap.add_argument("--dpi", type=int, default=None)
    from jax_rheology.io.config import parse_with_config
    a, _cfg = parse_with_config(ap)

    t0 = time.time()
    if a.index_only:
        rows = index_rows()
        print(f"index: {write_index(rows)}")
        print(f"{len(rows)} files in {time.time() - t0:.0f} s")
        return

    rows = render(a.figures, a.panels_only, a.dpi)
    rows += render_tables(a.figures)
    # A subset run leaves the timings of the panels it skipped out of the
    # index, so rebuild it from disk instead of from this run's rows.
    print(f"index: {write_index(rows if not a.figures else index_rows())}")
    print(f"{len(rows)} files in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
