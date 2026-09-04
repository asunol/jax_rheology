"""Config dataclasses + optional YAML loader behind the existing CLIs.

Canonical YAML names are the shared CLI vocabulary; old flag names stay
on argparse as aliases. ``--config`` is optional. A config is applied
first, then CLI flags override it.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# One canonical name per concept. Values are argparse dests that mean the same
# thing on some entrypoint. Destinations that are distinct flags on the same
# parser (e.g. seed vs theta_seed, time_budget_s vs wall_time_s) are not merged.
CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "inner_steps": ("inner_steps", "inner"),
    "outer_steps": ("outer_steps", "outer"),
    "nx": ("nx", "Nx"),
    "ny": ("ny", "Ny"),
    "out_dir": ("out_dir", "output_dir"),
    "learning_rate": ("learning_rate", "lr"),
    "checkpoint_every": ("checkpoint_every", "ckpt_every"),
    "seed": ("seed", "random_seed"),
    "g_x": ("g_x", "g_x_list"),
    "g_x_list": ("g_x_list", "g_x"),
}

_DEST_TO_CANONICAL: dict[str, str] = {
    dest: canon
    for canon, dests in CANONICAL_ALIASES.items()
    for dest in dests
}

# Nested YAML groups. Leaves are flattened onto argparse dests.
_NEST_GROUPS = (
    "truth", "init", "geometry", "solver", "loss", "optim", "runtime", "output",
)


@dataclass(frozen=True)
class RunConfig:
    """Canonical-name view of a parsed CLI, optionally seeded by YAML."""

    values: Mapping[str, Any] = field(repr=False)
    source: Optional[str] = None

    def get(self, name: str, default: Any = None) -> Any:
        if name in self.values:
            return self.values[name]
        dests = CANONICAL_ALIASES.get(name)
        if dests:
            for dest in dests:
                if dest in self.values:
                    return self.values[dest]
        return default

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self.values:
            return self.values[name]
        dests = CANONICAL_ALIASES.get(name)
        if dests:
            for dest in dests:
                if dest in self.values:
                    return self.values[dest]
        raise AttributeError(name)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping. Nested groups are flattened onto argparse dests."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load --config files")
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a mapping, got {type(raw).__name__}")
    return flatten_config(raw)


def flatten_config(raw: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in raw.items():
        if key in ("experiment", "comment", "source") and not prefix:
            continue
        if isinstance(val, Mapping) and (key in _NEST_GROUPS or prefix):
            out.update(flatten_config(val, prefix=f"{prefix}{key}."))
            continue
        out[str(key)] = val
    return out


def canonical_name(dest: str) -> str:
    return _DEST_TO_CANONICAL.get(dest, dest)


def attach_config_flag(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--config`` if the parser does not already declare it."""
    for action in parser._actions:
        if "--config" in action.option_strings or action.dest == "config":
            return parser
    parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional YAML config (canonical names). CLI flags override.",
    )
    return parser


def _parser_dests(parser: argparse.ArgumentParser) -> set[str]:
    return {a.dest for a in parser._actions if a.dest not in ("help",)}


def yaml_to_dests(parser: argparse.ArgumentParser, data: Mapping[str, Any]) -> dict[str, Any]:
    """Map canonical (or raw) YAML keys onto this parser's argparse dests."""
    dests = _parser_dests(parser)
    assigned: dict[str, Any] = {}
    for key, val in data.items():
        key = str(key).replace("-", "_")
        candidates = []
        if key in CANONICAL_ALIASES:
            candidates.extend(CANONICAL_ALIASES[key])
        candidates.append(key)
        canon = _DEST_TO_CANONICAL.get(key)
        if canon:
            candidates.extend(CANONICAL_ALIASES[canon])
        for dest in candidates:
            if dest in dests:
                assigned[dest] = val
                break
    return assigned


def config_from_namespace(args: argparse.Namespace, source: Optional[str] = None) -> RunConfig:
    values = {}
    for dest, val in vars(args).items():
        if dest in ("config", "help"):
            continue
        values[canonical_name(dest)] = val
        if canonical_name(dest) != dest:
            values.setdefault(dest, val)
    return RunConfig(values=values, source=source)


def parse_with_config(
    parser: argparse.ArgumentParser,
    argv: Optional[list[str]] = None,
) -> tuple[argparse.Namespace, RunConfig]:
    """Parse argv. Optional ``--config`` YAML is applied, then CLI overrides."""
    attach_config_flag(parser)
    argv = sys.argv[1:] if argv is None else list(argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    source = pre_args.config
    if source:
        parser.set_defaults(**yaml_to_dests(parser, load_yaml(source)))
    args = parser.parse_args(argv)
    cfg = config_from_namespace(args, source=source)
    return args, cfg
