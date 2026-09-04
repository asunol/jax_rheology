#!/usr/bin/env python
"""Take a provenance snapshot into a separate output directory.

Thin wrapper over :mod:`battery_provenance_snapshot`: the same snapshot and
the same comparison, written under
``reference_values/battery_provenance/recheck/`` so a re-check does not
overwrite the recorded snapshot.
"""
from __future__ import annotations

from pathlib import Path

import battery_provenance_snapshot as impl

ROOT = impl.ROOT
DIFF = impl.DIFF
PRODUCTION = impl.PRODUCTION
OUT = impl.OUT / "recheck"
run = impl.run
sha256 = impl.sha256
classify = impl.classify
untracked_paths = impl.untracked_paths
python_paths = impl.python_paths
write_csv = impl.write_csv


def snapshot_payload():
    old = impl.OUT
    impl.OUT = OUT
    try:
        return impl.snapshot_payload()
    finally:
        impl.OUT = old


def write_snapshot(label: str):
    old = impl.OUT
    impl.OUT = OUT
    try:
        return impl.write_snapshot(label)
    finally:
        impl.OUT = old


def final_gate():
    old = impl.OUT
    impl.OUT = OUT
    try:
        return impl.final_gate()
    finally:
        impl.OUT = old


def main():
    old = impl.OUT
    impl.OUT = OUT
    try:
        return impl.main()
    finally:
        impl.OUT = old


if __name__ == "__main__":
    main()
