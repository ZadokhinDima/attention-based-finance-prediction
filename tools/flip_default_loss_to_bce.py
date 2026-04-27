"""Idempotent: flip default `loss_kind` from "mse" to "bce" in Trainer,
run_hp_search, and run_walk_forward. Old `loss_kind="mse"` callers must
explicitly pass it now.

Usage: .venv/bin/python tools/flip_default_loss_to_bce.py [--dry]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "research.ipynb"

PATCHES: list[tuple[str, str, str]] = [
    (
        "Trainer __init__ default",
        'patience=10, horizons=None, device="cpu", loss_kind="mse")',
        'patience=10, horizons=None, device="cpu", loss_kind="bce")',
    ),
    (
        "run_hp_search default",
        'lookback=None, horizons=None, device=None, loss_kind="mse"):',
        'lookback=None, horizons=None, device=None, loss_kind="bce"):',
    ),
    (
        "run_walk_forward default",
        'lookback=None, horizons=None, device=None, train=True,\n'
        '                     loss_kind="mse"):',
        'lookback=None, horizons=None, device=None, train=True,\n'
        '                     loss_kind="bce"):',
    ),
]


def find_cell_with(nb, needle: str) -> int | None:
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and needle in c.source:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    nb = nbformat.read(NB_PATH, as_version=4)
    changed = False
    for name, old, new in PATCHES:
        idx_new = find_cell_with(nb, new)
        idx_old = find_cell_with(nb, old)
        if idx_new is not None:
            print(f"[skip] {name} — already flipped")
            continue
        if idx_old is None:
            raise RuntimeError(f"[FAIL] {name} — old block not found")
        nb.cells[idx_old].source = nb.cells[idx_old].source.replace(old, new, 1)
        changed = True
        print(f"[patched] {name}")

    if not changed:
        print("Nothing to do.")
        return
    if args.dry:
        print("(dry-run — no file written)")
        return
    nbformat.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    main()
