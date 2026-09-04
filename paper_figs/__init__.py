"""Publication figures for the viscoelastic and elastoviscoplastic results
(Figs. 5-6 and Supplementary Figs. S7-S15).

Every panel is a no-argument callable::

    from paper_figs import plot_N1a, plot_N1
    plot_N1a()            # writes final_figures/N1/N1a.jpg, returns the Figure
    plot_N1a(save=False)  # just returns the Figure
    plot_N1a(ax=some_ax)  # draws into an existing axes, returns it

Data locations live only in :mod:`paper_figs.data_paths` and reads happen only
in :mod:`paper_figs.loaders`, so a panel never reaches for a path of its own.
This package only plots: it does not fit, train, or differentiate.
"""
from __future__ import annotations

import os
import sys

from repo_paths import REPO_ROOT
_ROOT = str(REPO_ROOT)
for _p in (_ROOT, os.path.join(_ROOT, "jax_ib"), os.path.join(_ROOT, "jax-cfd")):
    if _p not in sys.path:
        sys.path.append(_p)

try:  # float64 must be set before any jnp import-time work
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)
except Exception:  # pragma: no cover - plotting works without jax
    pass

from . import data_paths, geometry, loaders, panels, style  # noqa: E402
from .style import apply_style, describe  # noqa: E402

__all__ = ["data_paths", "geometry", "loaders", "panels", "style",
           "apply_style", "describe"]


def _export(module_name: str, names: list[str]) -> None:
    import importlib

    mod = importlib.import_module(f".{module_name}", __name__)
    for n in names:
        globals()[n] = getattr(mod, n)
        __all__.append(n)


_export("fig_n1", ["plot_N1a", "plot_N1b", "plot_N1c", "plot_N1d",
                   "plot_N1d_alt", "plot_N1d_schedules", "plot_N1e",
                   "plot_N1e_horizontal", "plot_N1e_ladder", "plot_N1"])
_export("fig_sn1", ["plot_SN1a", "plot_SN1b", "plot_SN1c", "plot_SN1d",
                    "plot_SN1e", "plot_SN1_uy_truth", "plot_SN1_uy_tbnn",
                    "plot_SN1_trA_truth", "plot_SN1_trA_tbnn", "plot_SN1"])
_export("fig_sn2", ["plot_SN2a", "plot_SN2b", "plot_SN2b_3x1", "plot_SN2c",
                    "plot_SN2d", "plot_SN2e", "plot_SN2"])
_export("fig_sn3", ["plot_SN3a", "plot_SN3b", "plot_SN3c", "plot_SN3d",
                    "plot_SN3e", "plot_SN3f", "plot_SN3g", "plot_SN3h",
                    "plot_SN3i", "plot_SN3"])
_export("fig_sn4", ["plot_SN4a", "plot_SN4b", "plot_SN4b_throat", "plot_SN4c",
                    "plot_SN4d", "plot_SN4d_table", "plot_SN4d_table_alt",
                    "plot_SN4e", "plot_SN4e_alt", "plot_SN4",
                    "representative_table"])
_export("fig_sn5", ["plot_SN5a", "plot_SN5b", "plot_SN5c", "plot_SN5",
                    "ladder_table"])
_export("tables", ["make_table_S2", "make_table_S_evp_robustness",
                   "plot_evp_robustness_table", "evp_robustness_rows",
                   "evp_table_rows", "evp_robustness_worst"])
_export("fig_sn6", ["plot_SN6a", "plot_SN6a_lsq_log", "plot_SN6b", "plot_SN6c",
                    "plot_SN6", "summary_table", "i4_exclusion",
                    "loss_floor_spread"])
_export("fig_n2", ["plot_N2a", "plot_N2b", "plot_N2"])
_export("fig_sn7", ["plot_SN7a", "plot_SN7b", "plot_SN7c", "plot_SN7d",
                    "plot_SN7",
                    "plot_SN7_ladder_grid", "plot_SN7_plug_halfwidth",
                    "arrest_metrics", "training_metrics",
                    "duplication_check", "accepted_ratio_spread"])
#: SN8 is retired and its module is not part of this release. Panel (b) is
#: now the EVP robustness table above; (a) and (c) are recorded in
#: ``final_figures/SN7/SN7_notes.txt``.
