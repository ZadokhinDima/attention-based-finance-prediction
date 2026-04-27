"""Idempotently shrink all *_SEARCH HP grids: smaller dims, more dropout, fewer layers.

Rationale: ~10y train data is too small for the prior over-parameterised search
spaces (d_model up to 256, dropout floor 0.0).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "research.ipynb"


PATCHES: list[tuple[str, str, str]] = [
    (
        "LSTM_SEARCH",
        """LSTM_SEARCH = {
    "hidden_size": [64, 128, 256],
    "n_layers":    [1, 2, 3, 4],
    "dropout":     [0.0, 0.1, 0.2],
    "lr":          [0.001, 0.0005],
}""",
        """LSTM_SEARCH = {
    "hidden_size": [32, 64],
    "n_layers":    [1, 2],
    "dropout":     [0.2, 0.3, 0.4],
    "lr":          [0.001, 0.0005],
}""",
    ),
    (
        "TRANSFORMER_SEARCH",
        """TRANSFORMER_SEARCH = {
    "d_model":  [64, 128, 256],
    "n_heads":  [2, 4],
    "n_layers": [1, 2, 3],
    "dropout":  [0.0, 0.1, 0.2],
    "lr":       [0.001, 0.0005],
}""",
        """TRANSFORMER_SEARCH = {
    "d_model":  [32, 64],
    "n_heads":  [2, 4],
    "n_layers": [1, 2],
    "dropout":  [0.2, 0.3],
    "lr":       [0.001, 0.0005],
}""",
    ),
    (
        "INFORMER_SEARCH",
        """INFORMER_SEARCH = {
    "d_model":  [64, 128],
    "n_heads":  [2, 4],
    "e_layers": [2, 3],
    "factor":   [3, 5],
    "dropout":  [0.0, 0.1, 0.2],
    "lr":       [0.001, 0.0005],
}""",
        """INFORMER_SEARCH = {
    "d_model":  [32, 64],
    "n_heads":  [2, 4],
    "e_layers": [1, 2],
    "factor":   [3, 5],
    "dropout":  [0.2, 0.3],
    "lr":       [0.001, 0.0005],
}""",
    ),
    (
        "FEDFORMER_SEARCH",
        """FEDFORMER_SEARCH = {
    "d_model":  [64, 128],
    "n_heads":  [4, 8],
    "e_layers": [2],
    "modes":    [16, 32],
    "version":  ["Fourier", "Wavelets"],
    "dropout":  [0.0, 0.1],
    "lr":       [0.001, 0.0005],
}""",
        """FEDFORMER_SEARCH = {
    "d_model":  [32, 64],
    "n_heads":  [4, 8],
    "e_layers": [1, 2],
    "modes":    [8, 16],
    "version":  ["Fourier", "Wavelets"],
    "dropout":  [0.1, 0.2],
    "lr":       [0.001, 0.0005],
}""",
    ),
    (
        "ITRANSFORMER_SEARCH",
        """ITRANSFORMER_SEARCH = {
    "d_model":  [64, 128],
    "n_heads":  [4, 8],
    "e_layers": [2, 3],
    "dropout":  [0.0, 0.1],
    "pool":     ["sp500_token", "mean"],
    "lr":       [0.001, 0.0005],
}""",
        """ITRANSFORMER_SEARCH = {
    "d_model":  [32, 64],
    "n_heads":  [4, 8],
    "e_layers": [1, 2],
    "dropout":  [0.2, 0.3],
    "pool":     ["sp500_token", "mean"],
    "lr":       [0.001, 0.0005],
}""",
    ),
    (
        "FEIT_SEARCH",
        """FEIT_SEARCH = {
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
}""",
        """FEIT_SEARCH = {
    "d_model":       [32, 64],
    "n_heads":       [4, 8],
    "e_layers":      [1, 2],
    "dropout":       [0.2, 0.3],
    "moving_avg":    [13, 25],
    "version":       ["Fourier", "Wavelets"],
    "modes":         [8, 16],
    "wavelet_level": [2, 3],
    "use_norm":      [False, True],
    "pool":          ["sp500_concat", "mean"],
    "lr":            [0.001, 0.0005],
}""",
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
            print(f"[skip] {name} — already shrunk")
            continue
        if idx_old is None:
            raise RuntimeError(f"[FAIL] {name} — old block not found; manual review needed")
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
