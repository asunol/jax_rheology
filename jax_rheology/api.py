"""Public :class:`Simulation` interface: a named geometry plus a named model,
advanced by the finite-volume solvers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class Trajectory:
    """Velocity trajectory plus the runner result dict (for historical save)."""

    array: Any
    results: Optional[dict] = None
    _saver: Optional[Any] = None

    @property
    def shape(self):
        return np.asarray(self.array).shape

    @property
    def dtype(self):
        return np.asarray(self.array).dtype

    @property
    def finite(self) -> bool:
        a = np.asarray(self.array)
        return bool(np.isfinite(a).all())

    def save(self, output_dir: str) -> Path:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if self._saver is not None and self.results is not None:
            self._saver(self.results, output_dir)
            return Path(output_dir)
        dest = Path(output_dir) / "trajectory.npy"
        np.save(dest, np.asarray(self.array))
        return dest


class Simulation:
    """A geometry, a constitutive model, and a time-stepping schedule."""

    def __init__(self, geometry, model, dt, inner_steps, outer_steps, **kwargs):
        self.geometry = geometry
        self.model = model
        self.dt = dt
        self.inner_steps = int(inner_steps)
        self.outer_steps = int(outer_steps)
        self.kwargs = kwargs

    def run(self) -> Trajectory:
        """Advance the named geometry and model; return a velocity trajectory."""
        from jax_rheology.geometries import Constriction, Obstacle, Porous, Contraction
        geom = self.geometry
        if isinstance(geom, Constriction):
            return self._run_gnf("channel_constriction_flow", "run_channel_constriction",
                                 "save_trajectory_file", geom)
        if isinstance(geom, Obstacle):
            return self._run_gnf("channel_obstacle_flow", "run_channel_obstacle",
                                 "save_trajectory_file", geom)
        if isinstance(geom, Porous):
            return self._run_gnf("porous_media_flow", "run_porous_media",
                                 "save_trajectory_file", geom)
        if isinstance(geom, Contraction):
            return self._run_contraction(geom)
        raise TypeError(f"unsupported geometry type {type(geom).__name__}")

    def _run_gnf(self, module_name: str, run_name: str, save_name: str, geom) -> Trajectory:
        from repo_paths import bootstrap
        bootstrap()
        import importlib
        mod = importlib.import_module(module_name)
        cfg = mod.COMMON_CONFIG
        orig_size = cfg.get("domain_size")
        orig_domain = cfg.get("domain")
        try:
            if hasattr(geom, "nx") and hasattr(geom, "ny"):
                cfg["domain_size"] = (int(geom.nx), int(geom.ny))
            if getattr(geom, "domain", None) is not None:
                cfg["domain"] = geom.domain
            model = self.model
            name = model.name if hasattr(model, "name") else str(model)
            params = tuple(model.params) if hasattr(model, "params") else ()
            pg = getattr(geom, "pressure_gradient", None)
            run = getattr(mod, run_name)
            results = run(
                name, *params,
                show_plots=False,
                save_trajectory=False,
                pressure_gradient=pg,
                dt=self.dt,
                inner_steps=self.inner_steps,
                outer_steps=self.outer_steps,
            )
            if results is None:
                raise RuntimeError(f"{run_name} returned None")
            traj = results["trajectory"]
            return Trajectory(
                array=traj,
                results=results,
                _saver=getattr(mod, save_name, None),
            )
        finally:
            if orig_size is not None:
                cfg["domain_size"] = orig_size
            if orig_domain is not None:
                cfg["domain"] = orig_domain

    def _run_contraction(self, geom) -> Trajectory:
        """Contraction forward via planar_contraction + evolve_contraction."""
        from repo_paths import bootstrap
        bootstrap()
        from jax_rheology.geometries import planar_contraction as cg
        from jax_rheology.forward.contraction import evolve_contraction
        from jax_rheology import log_conformation as _lc  # noqa: F401  registers models
        from jax_rheology.models.registry import get_model

        params = dict(self.model.params)
        H = 1.0
        R = float(geom.ratio)
        L_up, L_down = 8.0, 8.0
        cells_per_H = max(4.0, float(geom.ny) / (2.0 * R))
        grid = cg.make_contraction_grid(
            H, L_up, L_down, cells_per_H=cells_per_H, contraction_ratio=R)
        model = get_model("giesekus_logconf_bk_v2")
        initial, perm_f, bc_spec = cg.build_contraction_viscoelastic_state(
            grid, H=H, L_down=L_down, U_inlet=0.0,
            logistic_width=0.05, model=model, contraction_ratio=R,
        )
        polymer = {
            "G_p": float(params["Gp"]),
            "lam": float(params["lam"]),
            "nu_s": float(params["nu_s"]),
            "alpha": float(params.get("alpha", 0.0)),
        }
        final_state, out = evolve_contraction(
            initial, model, polymer, grid,
            density=1.0,
            base_viscosity=float(params["nu_s"]),
            dt=float(self.dt),
            inner_steps=self.inner_steps,
            outer_steps=self.outer_steps,
            U_inlet=float(geom.U),
            ramp_time=float(geom.ramp_time),
            perm_f=perm_f,
            bc_spec=bc_spec,
        )
        u = np.asarray(out["u_traj"])
        v = np.asarray(out["v_traj"])
        array = np.stack([u, v], axis=1)
        return Trajectory(array=array, results={"out": out, "final_state": final_state})
