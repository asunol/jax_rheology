"""Shared entrypoint helpers: resolve the repository root, read ``--config``,
pin floating-point precision from the config, and hand off to the solver.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Mapping, Optional

_PRECISION_APPLIED = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bootstrap() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from repo_paths import bootstrap as _b
    _b()


def peek_config() -> dict:
    """Load the --config YAML (flattened) without consuming the rest of argv."""
    bootstrap()
    from jax_rheology.io.config import load_yaml
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    if not pre_args.config:
        return {}
    return load_yaml(pre_args.config)


def apply_precision(data: Optional[dict] = None) -> None:
    """Pin JAX float64 from the config before any JAX call triggers compilation.

    A ``JAX_ENABLE_X64`` value inherited from the shell cannot silently change
    a run: the environment variable is set or cleared to match the config,
    ``jax.config`` is updated, and the resulting default dtype is asserted.
    """
    global _PRECISION_APPLIED
    if _PRECISION_APPLIED:
        return
    if data is None:
        data = peek_config()
    if not data:
        return
    if "x64" not in data:
        raise SystemExit(
            "config is missing the required 'x64' key. Set 'x64: false' for "
            "generalized-Newtonian runs (single precision) or 'x64: true' for "
            "viscoelastic and rheometry runs (double precision). The key is "
            "required so the precision of a run is recorded in its config "
            "rather than inherited from the shell."
        )
    want = bool(data["x64"])
    if want:
        os.environ["JAX_ENABLE_X64"] = "True"
    else:
        os.environ.pop("JAX_ENABLE_X64", None)
    import jax
    jax.config.update("jax_enable_x64", want)
    import jax.numpy as jnp
    got = jnp.asarray(1.0).dtype
    expected = jnp.float64 if want else jnp.float32
    if got != expected:
        raise RuntimeError(
            f"precision pin failed: config x64={want} but "
            f"jnp.asarray(1.0).dtype={got} "
            f"(jax_enable_x64={jax.config.read('jax_enable_x64')})"
        )
    print(f"[precision] pinned x64={want} dtype={got}", flush=True)
    _PRECISION_APPLIED = True


def run_module(module_name: str):
    bootstrap()
    apply_precision()
    mod = importlib.import_module(module_name)
    return mod.main()


def dispatch(mapping: Mapping[str, str], key: str, value: Optional[str]):
    if value not in mapping:
        raise SystemExit(f"{key}={value!r} not in {sorted(mapping)}")
    rc = run_module(mapping[value])
    raise SystemExit(0 if rc is None else rc)


def inject_positionals(*tokens: str) -> None:
    """Put required positional tokens on argv so YAML can drive the runner.

    ``argparse`` required positionals ignore ``set_defaults``. Tokens already
    present (or ``--help``) are left alone.
    """
    if not tokens:
        return
    if "--help" in sys.argv or "-h" in sys.argv:
        return
    if all(str(t) in sys.argv[1:] for t in tokens):
        return
    sys.argv[1:1] = [str(t) for t in tokens]
