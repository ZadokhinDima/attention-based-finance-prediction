"""Idempotent edit: switch run_hp_search held-out from `val` only to `val ∪ test`.

Rationale: the `MarketDataset` test split (>VAL_END) was unused during HP search.
Combining val + test gives a larger, more stable held-out for HP picking.

Usage:
    .venv/bin/python tools/update_hp_search_holdout.py [--dry]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "research.ipynb"

OLD_BLOCK = """\
    train_ds = MarketDataset(df_data, lookback=lookback, horizons=horizons, split="train")
    val_ds   = MarketDataset(df_data, lookback=lookback, horizons=horizons, split="val",
                             mean=train_ds.mean, std=train_ds.std, const_mask=train_ds.const_mask)
"""

NEW_BLOCK = """\
    train_ds = MarketDataset(df_data, lookback=lookback, horizons=horizons, split="train")
    # Held-out = val ∪ test (everything after TRAIN_END) — larger, more stable HP signal.
    val_from = str((pd.to_datetime(TRAIN_END) + pd.Timedelta(days=1)).date())
    val_ds   = MarketDataset(df_data, lookback=lookback, horizons=horizons, date_from=val_from,
                             mean=train_ds.mean, std=train_ds.std, const_mask=train_ds.const_mask)
"""

# Also broaden the docstring so future readers know what the held-out is.
OLD_DOC = 'Grid search по `search_grid` на фіксованому train/val split.'
NEW_DOC = ('Grid search по `search_grid`. Train = `MarketDataset(split="train")`; '
           'held-out = всі дані після `TRAIN_END` (val ∪ test).')


def find_hp_search_cell(nb) -> int:
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and "def run_hp_search(" in c.source:
            return i
    raise RuntimeError("Could not find the run_hp_search cell")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    nb = nbformat.read(NB_PATH, as_version=4)
    idx = find_hp_search_cell(nb)
    src = nb.cells[idx].source

    changed = False

    if NEW_BLOCK in src:
        print("run_hp_search already uses val ∪ test held-out — leaving alone.")
    elif OLD_BLOCK in src:
        src = src.replace(OLD_BLOCK, NEW_BLOCK)
        changed = True
        print("Patched val_ds construction.")
    else:
        raise RuntimeError("Could not locate the expected val_ds block — manual review needed.")

    if NEW_DOC in src:
        pass  # already updated
    elif OLD_DOC in src:
        src = src.replace(OLD_DOC, NEW_DOC)
        changed = True
        print("Updated docstring.")

    if not changed:
        print("No changes needed.")
        return

    nb.cells[idx].source = src

    if args.dry:
        print("(dry-run — no file written)")
        return

    nbformat.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    main()
