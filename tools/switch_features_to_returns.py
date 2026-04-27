"""Idempotently switch MarketDataset features from raw OHLCV prices to log-returns,
and update MeanReturnBaseline to consume returns directly.

Targets remain log(P_{t+h}/P_t). Only the input features change.

Usage: .venv/bin/python tools/switch_features_to_returns.py [--dry]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "research.ipynb"


# ─── Patch 1: MarketDataset — insert returns conversion + retain prices ──────
DS_OLD = (
    "        if mean is None:\n"
    "            raw_std    = subset.std()\n"
    "            const_mask = raw_std[raw_std == 0].index\n"
    "            mean       = subset.mean()\n"
    "            std        = raw_std.replace(0, 1)\n"
    "        self.mean       = mean\n"
    "        self.std        = std\n"
    "        self.const_mask = const_mask if const_mask is not None else pd.Index([])\n"
    "\n"
    "        norm = (subset - mean) / std\n"
    "        if len(self.const_mask) > 0:\n"
    "            norm[self.const_mask] = 0.0\n"
    "        norm = norm.fillna(0.0).clip(-10.0, 10.0)\n"
    "\n"
    "        self.data         = norm.values.astype(np.float32)\n"
    "        self.dates        = subset.index\n"
    "        self.target_close = subset[(target_ticker, \"Close\")].values.astype(np.float32)\n"
    "        self.valid_idx    = np.arange(lookback, len(self.data) - self.max_h)\n"
)

DS_NEW = (
    "        # Convert prices/volume to log-returns: log(x_t) - log(x_{t-1}).\n"
    "        # Save target_close prices BEFORE conversion (target Y still uses raw prices).\n"
    "        # Volume can be 0 (holidays etc.) — replace with NaN, then ffill/bfill before log.\n"
    "        target_close_prices = subset[(target_ticker, \"Close\")].astype(np.float32).values\n"
    "        clean   = subset.replace(0, np.nan).ffill().bfill()\n"
    "        returns = np.log(clean).diff().iloc[1:]\n"
    "        returns = returns.replace([np.inf, -np.inf], 0.0).fillna(0.0)\n"
    "        subset  = returns\n"
    "        target_close_prices = target_close_prices[1:]   # align with returns rows\n"
    "\n"
    "        if mean is None:\n"
    "            raw_std    = subset.std()\n"
    "            const_mask = raw_std[raw_std == 0].index\n"
    "            mean       = subset.mean()\n"
    "            std        = raw_std.replace(0, 1)\n"
    "        self.mean       = mean\n"
    "        self.std        = std\n"
    "        self.const_mask = const_mask if const_mask is not None else pd.Index([])\n"
    "\n"
    "        norm = (subset - mean) / std\n"
    "        if len(self.const_mask) > 0:\n"
    "            norm[self.const_mask] = 0.0\n"
    "        norm = norm.fillna(0.0).clip(-10.0, 10.0)\n"
    "\n"
    "        self.data         = norm.values.astype(np.float32)\n"
    "        self.dates        = subset.index\n"
    "        self.target_close = target_close_prices\n"
    "        self.valid_idx    = np.arange(lookback, len(self.data) - self.max_h)\n"
)


# ─── Patch 2: MeanReturnBaseline — drop the price-to-return conversion ───────
BL_OLD = (
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        \"\"\"\n"
    "        x: [B, L, NF] — як у LSTM\n"
    "        \"\"\"\n"
    "        # беремо тільки S&P500: [B, L]\n"
    "        prices = x[:, :, self.sp500_idx]\n"
    "\n"
    "        # лог-ретерни: log(p_t / p_{t-1})\n"
    "        log_returns = torch.log(prices[:, 1:] / prices[:, :-1])  # [B, L-1]\n"
    "\n"
    "        # беремо останні lookback значень\n"
    "        if log_returns.shape[1] < self.lookback:\n"
    "            raise ValueError(\"Sequence too short for given lookback\")\n"
    "\n"
    "        recent = log_returns[:, -self.lookback:]  # [B, lookback]\n"
    "\n"
    "        # середнє по часу\n"
    "        mean_ret = recent.mean(dim=1, keepdim=True)  # [B, 1]\n"
    "\n"
    "        # повторюємо для кожного горизонту\n"
    "        preds = mean_ret.repeat(1, self.n_horizons)  # [B, H]\n"
    "\n"
    "        return preds"
)

BL_NEW = (
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        \"\"\"\n"
    "        x: [B, L, NF] — стандартизовані log-returns (після MarketDataset з returns-фічами).\n"
    "        Беремо колонку S&P500, усереднюємо останні `lookback` returns, повторюємо per horizon.\n"
    "        \"\"\"\n"
    "        returns = x[:, :, self.sp500_idx]                       # [B, L]\n"
    "        if returns.shape[1] < self.lookback:\n"
    "            raise ValueError(\"Sequence too short for given lookback\")\n"
    "        recent   = returns[:, -self.lookback:]                  # [B, lookback]\n"
    "        mean_ret = recent.mean(dim=1, keepdim=True)             # [B, 1]\n"
    "        preds    = mean_ret.repeat(1, self.n_horizons)          # [B, H]\n"
    "        return preds"
)


PATCHES = [
    ("MarketDataset (features → returns)", DS_OLD, DS_NEW),
    ("MeanReturnBaseline (consumes returns)", BL_OLD, BL_NEW),
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
            print(f"[skip] {name} — already patched")
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
