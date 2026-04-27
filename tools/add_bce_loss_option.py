"""Add `loss_kind` parameter to Trainer, run_hp_search, run_walk_forward.

Defaults to "mse" so the existing notebook flow is unchanged. Opt-in to
direction-classification BCE via `loss_kind="bce"` when calling the helpers.
With BCE: target = (y > 0).float() per horizon, model output is treated as
a logit. DirAcc remains meaningful; MSE/MAE on raw logits become nonsense
and should be ignored for BCE-trained runs.

Idempotent. Usage:
    .venv/bin/python tools/add_bce_loss_option.py [--dry]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "research.ipynb"


# ─── Patch 1: Trainer.__init__ signature + criterion construction ────────────
TRAINER_OLD_SIG = (
    "    def __init__(self, model, train_loader, val_loader, lr=1e-3, max_epochs=50,\n"
    "                 patience=10, horizons=None, device=\"cpu\"):\n"
)
TRAINER_NEW_SIG = (
    "    def __init__(self, model, train_loader, val_loader, lr=1e-3, max_epochs=50,\n"
    "                 patience=10, horizons=None, device=\"cpu\", loss_kind=\"mse\"):\n"
)

TRAINER_OLD_CRIT = "        self.criterion    = nn.MSELoss()\n"
TRAINER_NEW_CRIT = (
    "        assert loss_kind in (\"mse\", \"bce\"), f\"unknown loss_kind: {loss_kind}\"\n"
    "        self.loss_kind    = loss_kind\n"
    "        self.criterion    = nn.MSELoss() if loss_kind == \"mse\" else nn.BCEWithLogitsLoss()\n"
)

TRAINER_OLD_LOSS = (
    "                pred = self.model(x)\n"
    "                loss = self.criterion(pred, y)\n"
)
TRAINER_NEW_LOSS = (
    "                pred = self.model(x)\n"
    "                target = (y > 0).float() if self.loss_kind == \"bce\" else y\n"
    "                loss = self.criterion(pred, target)\n"
)


# ─── Patch 2: run_hp_search signature + Trainer construction ─────────────────
HP_OLD_SIG = (
    "def run_hp_search(model_factory, search_grid, df_data, save_dir,\n"
    "                  filter_fn=None, hp_epochs=30, hp_patience=7, batch_size=64,\n"
    "                  lookback=None, horizons=None, device=None):\n"
)
HP_NEW_SIG = (
    "def run_hp_search(model_factory, search_grid, df_data, save_dir,\n"
    "                  filter_fn=None, hp_epochs=30, hp_patience=7, batch_size=64,\n"
    "                  lookback=None, horizons=None, device=None, loss_kind=\"mse\"):\n"
)

HP_OLD_TRAIN = (
    "        trainer = Trainer(model, train_loader, val_loader, lr=lr,\n"
    "                          max_epochs=hp_epochs, patience=hp_patience,\n"
    "                          horizons=horizons, device=device)\n"
)
HP_NEW_TRAIN = (
    "        trainer = Trainer(model, train_loader, val_loader, lr=lr,\n"
    "                          max_epochs=hp_epochs, patience=hp_patience,\n"
    "                          horizons=horizons, device=device, loss_kind=loss_kind)\n"
)


# ─── Patch 3: run_walk_forward signature + Trainer construction ──────────────
WF_OLD_SIG = (
    "def run_walk_forward(model_factory, best_config, df_data, folds, save_dir,\n"
    "                     val_window=252, max_epochs=50, patience=10, batch_size=64,\n"
    "                     lookback=None, horizons=None, device=None, train=True):\n"
)
WF_NEW_SIG = (
    "def run_walk_forward(model_factory, best_config, df_data, folds, save_dir,\n"
    "                     val_window=252, max_epochs=50, patience=10, batch_size=64,\n"
    "                     lookback=None, horizons=None, device=None, train=True,\n"
    "                     loss_kind=\"mse\"):\n"
)

WF_OLD_TRAIN = (
    "            trainer = Trainer(model, train_loader, val_loader, lr=lr,\n"
    "                              max_epochs=max_epochs, patience=patience,\n"
    "                              horizons=horizons, device=device)\n"
)
WF_NEW_TRAIN = (
    "            trainer = Trainer(model, train_loader, val_loader, lr=lr,\n"
    "                              max_epochs=max_epochs, patience=patience,\n"
    "                              horizons=horizons, device=device, loss_kind=loss_kind)\n"
)


PATCHES = [
    ("Trainer __init__ signature",  TRAINER_OLD_SIG,   TRAINER_NEW_SIG),
    ("Trainer criterion",           TRAINER_OLD_CRIT,  TRAINER_NEW_CRIT),
    ("Trainer _epoch loss",         TRAINER_OLD_LOSS,  TRAINER_NEW_LOSS),
    ("run_hp_search signature",     HP_OLD_SIG,        HP_NEW_SIG),
    ("run_hp_search Trainer call",  HP_OLD_TRAIN,      HP_NEW_TRAIN),
    ("run_walk_forward signature",  WF_OLD_SIG,        WF_NEW_SIG),
    ("run_walk_forward Trainer call", WF_OLD_TRAIN,    WF_NEW_TRAIN),
]


def find_cell_containing(nb, needle: str) -> int | None:
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and needle in c.source:
            return i
    return None


def apply_patches(nb) -> bool:
    changed = False
    for name, old, new in PATCHES:
        idx_with_new = find_cell_containing(nb, new)
        idx_with_old = find_cell_containing(nb, old)

        if idx_with_new is not None:
            print(f"[skip] {name} — already patched")
            continue
        if idx_with_old is None:
            raise RuntimeError(f"[FAIL] {name} — neither old nor new block found; manual review needed")
        nb.cells[idx_with_old].source = nb.cells[idx_with_old].source.replace(old, new, 1)
        changed = True
        print(f"[patched] {name}")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    nb = nbformat.read(NB_PATH, as_version=4)
    changed = apply_patches(nb)

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
