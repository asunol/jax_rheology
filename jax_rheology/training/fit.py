"""``training.fit`` facade. Translates named specs into existing runner mains.

No numerics are written here. Dispatch is argv injection, same contract as
``experiments/_dispatch``.
"""
from __future__ import annotations

import io
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RoiVelocityPressureLoss:
    """ROI-weighted velocity loss, with an optional pressure term ``w_p``."""

    weight: str = "roi"
    w_p: float = 0.0


@dataclass(frozen=True)
class MaskedFieldRMSE:
    """Field RMSE restricted to an observation mask."""

    observation: Any = None


@dataclass(frozen=True)
class SchemeAlternation:
    """Named training-scheme tag (e.g. ``s1``) for the contraction trainer."""

    name: str = "s1"


@dataclass(frozen=True)
class Adam:
    """Adam step count, learning rate, and warmup/tail window."""

    lr: float = 2.0e-1
    steps: int = 2
    warmup_tail: Tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class PIV:
    """PIV window and overlap used when the trainer downsamples the field."""

    window: Tuple[int, int] = (32, 16)
    overlap: float = 0.75


@dataclass
class FitResult:
    """Parsed trainer losses plus the captured log."""

    loss_init: float
    loss_final: float
    checkpoint: Optional[str] = None
    log: str = ""
    steps: int = 0


def _tee_stdout(buf: io.StringIO):
    class _Tee:
        def __init__(self, primary, secondary):
            self.primary = primary
            self.secondary = secondary

        def write(self, x):
            self.primary.write(x)
            self.secondary.write(x)
            return len(x)

        def flush(self):
            self.primary.flush()
            self.secondary.flush()

        def fileno(self):
            return self.primary.fileno()

    return _Tee(sys.stdout, buf)


def _parse_losses(text: str) -> Tuple[Optional[float], Optional[float]]:
    init = None
    final = None
    m = re.search(r"loss\(init[^)]*\)\s*=\s*([0-9.eE+-]+)", text)
    if m:
        init = float(m.group(1))
    m = re.search(r"Initial loss:\s*([0-9.eE+-]+)", text)
    if m:
        init = float(m.group(1))
    m = re.search(r"Final loss:\s*([0-9.eE+-]+)", text)
    if m:
        final = float(m.group(1))
    if final is None and init is not None:
        final = init
    if init is None and final is not None:
        init = final
    return init, final


def fit(*, geometry, closure, target=None, loss=None, schedule=None,
        optimizer=None, init=None, lr=None, time_budget_s=None,
        out_dir: str = "runs/api_fit", **solver) -> FitResult:
    """Dispatch to the existing instantaneous or contraction trainer."""
    from repo_paths import bootstrap
    bootstrap()
    from jax_rheology.geometries import Constriction, Contraction
    from jax_rheology.closures import MixtureOfSigmoids, TBNN

    if isinstance(geometry, Constriction) or isinstance(closure, MixtureOfSigmoids):
        return _fit_instantaneous(
            geometry, closure, optimizer, lr, out_dir, solver)
    if isinstance(geometry, Contraction) or isinstance(closure, TBNN):
        return _fit_contraction(
            geometry, closure, schedule, init, lr, time_budget_s, out_dir, solver,
            loss=loss)
    raise TypeError(
        f"training.fit: unsupported geometry={type(geometry).__name__} "
        f"closure={type(closure).__name__}"
    )


def _fit_instantaneous(geometry, closure, optimizer, lr, out_dir, solver) -> FitResult:
    from jax_rheology.closures import MixtureOfSigmoids
    steps = int(getattr(optimizer, "steps", None) or 2)
    inner = int(solver.get("inner_steps", 2))
    outer = int(solver.get("outer_steps", 1))
    learning_rate = float(lr if lr is not None else getattr(optimizer, "lr", 2.0e-1))
    argv = [
        "run_tbnn_debug_constriction_cluster_new",
        "1",
        "--num-steps", str(steps),
        "--inner-steps", str(inner),
        "--outer-steps", str(outer),
        "--learning-rate", str(learning_rate),
        "--warmup-steps", "0",
        "--tail-steps", "0",
        "--results-root", str(out_dir),
        "--pressure-gradient", str(getattr(geometry, "pressure_gradient", 5.0)),
    ]
    if isinstance(closure, MixtureOfSigmoids):
        hidden: Sequence[int] = list(closure.hidden or [48, 48])
        argv += ["--architecture", *[str(h) for h in hidden], "--M", str(int(closure.M))]
        if closure.init == "soft_newtonian":
            argv.append("--use-soft-newtonian-init")
    return _run_main("run_tbnn_debug_constriction_cluster_new", argv, steps=steps)


def _fit_contraction(geometry, closure, schedule, init, lr, time_budget_s,
                     out_dir, solver, loss=None) -> FitResult:
    from jax_rheology.closures import TBNN
    scheme = getattr(schedule, "name", None) or "s1"
    inner = int(solver.get("inner_steps", 2))
    outer = int(solver.get("outer_steps", 1))
    nx = int(getattr(geometry, "nx", 32))
    ny = int(getattr(geometry, "ny", 64))
    argv = [
        "visco_opt_tbnn_contraction_run",
        "--scheme", str(scheme),
        "--mem-smoke",
        "--nx", str(nx),
        "--ny", str(ny),
        "--inner", str(inner),
        "--outer", str(outer),
        "--time-budget-s", str(int(time_budget_s or 180)),
        "--out-dir", str(out_dir),
        "--run-name", "api_fit_smoke",
    ]
    if lr is not None:
        argv += ["--lr", str(lr)]
    if isinstance(closure, TBNN):
        argv += ["--width", str(int(closure.width)),
                 "--depth", str(int(closure.depth)),
                 "--seed", str(int(closure.seed))]
    if init:
        if "Gp" in init:
            argv += ["--gp-init", str(init["Gp"])]
        if "lam" in init:
            argv += ["--lam-init", str(init["lam"])]
        if "nu_s" in init or "nus" in init:
            argv += ["--nus-init", str(init.get("nu_s", init.get("nus")))]
    if loss is not None and getattr(loss, "w_p", None) is not None:
        argv += ["--w-p", str(loss.w_p)]
    return _run_main("visco_opt_tbnn_contraction_run", argv, steps=0)


def _run_main(module_name: str, argv: list, steps: int = 0) -> FitResult:
    import importlib
    from repo_paths import bootstrap
    bootstrap()
    old = sys.argv
    buf = io.StringIO()
    sys.argv = list(argv)
    tee = _tee_stdout(buf)
    try:
        sys.stdout = tee
        mod = importlib.import_module(module_name)
        rc = mod.main()
    except SystemExit as e:
        rc = e.code
    finally:
        sys.stdout = sys.__stdout__
        sys.argv = old
    text = buf.getvalue()
    if rc not in (None, 0):
        raise RuntimeError(f"{module_name} main() returned {rc}")
    init, final = _parse_losses(text)
    if init is None or final is None:
        raise RuntimeError(f"{module_name} produced no parseable loss\n{text[-2000:]}")
    return FitResult(loss_init=float(init), loss_final=float(final), log=text,
                     steps=int(steps))
