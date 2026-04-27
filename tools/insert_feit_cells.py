"""One-shot, idempotent insertion of the FEiT section into research.ipynb.

Usage:
    .venv/bin/python tools/insert_feit_cells.py          # inserts (skips if already present)
    .venv/bin/python tools/insert_feit_cells.py --dry    # shows where it would insert

The script also rewrites the existing ALL_MODELS cell to add FEiT entries.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "research.ipynb"

# Sentinel string used to detect prior insertion (idempotency).
SENTINEL = "## Frequency Enhanced iTransformer (FEiT)"

CELL_MD_HEADER = """\
## Frequency Enhanced iTransformer (FEiT)

Гібрид iTransformer і FEDformer: токени-змінні (як в iTransformer), але
кожна змінна перед енкодером розбивається на трендовий і волатильний
компоненти, де до волатильного застосовується частотне перетворення
(Fourier або Wavelets). Один спільний енкодер працює над `2·N` токенами.
Декомпозиція виконується **один раз** перед енкодером (на відміну від
FEDformer, де вона повторюється в кожному шарі).
"""

CELL_CODE_MODEL = """\
import sys, os
_ROOT = os.getcwd()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from feit_model import FEiTransformerForecaster, VolatileFeatures

# Smoke test: both `version` × both `pool` on dummy [B, L, N] input.
import torch
SP500_FEAT_IDX = df.columns.get_loc(("SandP500", "Close"))
print(f"SP500_FEAT_IDX (full): {SP500_FEAT_IDX}    (out of {df.shape[1]} features)")

for version in ("Fourier", "Wavelets"):
    for pool in ("sp500_concat", "mean"):
        _m = FEiTransformerForecaster(
            n_features=145, d_model=64, n_heads=8, e_layers=2,
            version=version, modes=32, mode_select="low",
            wavelet_base="db4", wavelet_level=3,
            moving_avg=25, n_horizons=len(HORIZONS), lookback=LOOKBACK,
            target_idx=SP500_FEAT_IDX, pool=pool,
        )
        _y = _m(torch.randn(4, LOOKBACK, 145))
        n_params = sum(p.numel() for p in _m.parameters())
        print(f"version={version:9s}  pool={pool:13s}  output={tuple(_y.shape)}  params={n_params:,}")
        del _m, _y
"""

CELL_MD_FULL = """\
### Пошук гіперпараметрів та Walk-Forward (повна версія)
"""

CELL_CODE_FULL = """\
SP500_IDX_FULL_FEIT   = df.columns.get_loc(("SandP500", "Close"))
SP500_IDX_SIMPLE_FEIT = df_sp.columns.get_loc(("SandP500", "Close"))


def make_feit_factory(target_idx: int):
    def factory(n_features, d_model, n_heads, e_layers, dropout,
                version, modes, wavelet_level, moving_avg, use_norm, pool):
        return FEiTransformerForecaster(
            n_features=n_features, d_model=d_model, n_heads=n_heads,
            e_layers=e_layers, dropout=dropout, use_norm=use_norm, pool=pool,
            moving_avg=moving_avg,
            version=version, modes=modes, mode_select="random",
            wavelet_base="db4", wavelet_level=wavelet_level,
            n_horizons=len(HORIZONS), lookback=LOOKBACK, target_idx=target_idx,
            seed=0,
        )
    return factory


feit_factory_full   = make_feit_factory(SP500_IDX_FULL_FEIT)
feit_factory_simple = make_feit_factory(SP500_IDX_SIMPLE_FEIT)

FEIT_SEARCH = {
    "d_model":       [64, 128],
    "n_heads":       [4, 8],
    "e_layers":      [2, 3],
    "dropout":       [0.1, 0.2],
    "moving_avg":    [13, 25],
    "version":       ["Fourier", "Wavelets"],
    "modes":         [16, 32],
    "wavelet_level": [2, 3],
    "use_norm":      [False, True],
    "pool":          ["sp500_concat", "mean"],
    "lr":            [0.001, 0.0005],
}

# Filter rules:
#   - d_model % n_heads == 0
#   - For Fourier, fix wavelet_level=2 (unused) to deduplicate trials
#   - For Wavelets, fix modes=16 (unused) to deduplicate trials
def _feit_filter(t):
    if t["d_model"] % t["n_heads"] != 0:
        return False
    if t["version"] == "Fourier" and t["wavelet_level"] != 2:
        return False
    if t["version"] == "Wavelets" and t["modes"] != 16:
        return False
    return True

feit_hp_df = run_hp_search(
    feit_factory_full, FEIT_SEARCH, df, os.path.join(RESULTS, "feitransformer"),
    filter_fn=_feit_filter,
)
print("\\nТоп-5 за val_loss:")
display(feit_hp_df.head(5))

feit_results, feit_agg = run_walk_forward(
    feit_factory_full, feit_hp_df.iloc[0], df, folds, os.path.join(RESULTS, "feitransformer"),
)
"""

CELL_MD_SIMPLE = """\
### Спрощений FEiT (тільки S&P 500)
"""

CELL_CODE_SIMPLE = """\
feits_hp_df = run_hp_search(
    feit_factory_simple, FEIT_SEARCH, df_sp, os.path.join(RESULTS, "feitransformer-simple"),
    filter_fn=_feit_filter,
)
print("\\nТоп-5 за val_loss:")
display(feits_hp_df.head(5))

feits_results, feits_agg = run_walk_forward(
    feit_factory_simple, feits_hp_df.iloc[0], df_sp, folds, os.path.join(RESULTS, "feitransformer-simple"),
)
"""

NEW_CELLS = [
    new_markdown_cell(CELL_MD_HEADER),
    new_code_cell(CELL_CODE_MODEL),
    new_markdown_cell(CELL_MD_FULL),
    new_code_cell(CELL_CODE_FULL),
    new_markdown_cell(CELL_MD_SIMPLE),
    new_code_cell(CELL_CODE_SIMPLE),
]


def find_all_models_idx(nb) -> int:
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and c.source.lstrip().startswith("ALL_MODELS = {"):
            return i
    raise RuntimeError("Could not find the ALL_MODELS cell in research.ipynb")


def already_inserted(nb) -> bool:
    return any(SENTINEL in c.source for c in nb.cells if c.cell_type == "markdown")


def update_all_models(nb) -> bool:
    """Add FEiT entries to ALL_MODELS dict if not already present."""
    idx = find_all_models_idx(nb)
    src = nb.cells[idx].source
    if '"FEiT":' in src:
        return False

    insertion = (
        '    "FEiT":         os.path.join(RESULTS, "feitransformer",        "walk_forward.json"),\n'
        '    "FEiT-simple":  os.path.join(RESULTS, "feitransformer-simple", "walk_forward.json"),\n'
    )
    closing = src.rfind("}")
    assert closing != -1, "ALL_MODELS dict has no closing brace"
    nb.cells[idx].source = src[:closing] + insertion + src[closing:]
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="Print actions without writing")
    args = ap.parse_args()

    nb = nbformat.read(NB_PATH, as_version=4)

    if already_inserted(nb):
        print("FEiT section already present — nothing to insert.")
    else:
        idx = find_all_models_idx(nb)
        print(f"Will insert {len(NEW_CELLS)} cells before cell index {idx} (ALL_MODELS).")
        if not args.dry:
            for offset, cell in enumerate(NEW_CELLS):
                nb.cells.insert(idx + offset, cell)

    if update_all_models(nb):
        print("ALL_MODELS cell updated with FEiT entries.")
    else:
        print("ALL_MODELS already references FEiT — leaving alone.")

    if args.dry:
        print("(dry-run — no file written)")
        return

    nbformat.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    main()
