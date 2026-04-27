# CLAUDE.md — Development Context

## Python Execution

Always use the local venv:
```bash
python3 -m venv .venv   # create if missing
.venv/bin/python        # always execute via this
```

## Notebook Structure

`research.ipynb` is a single Jupyter notebook (~78 cells) structured as:

1. **Setup** — imports, constants (`LOOKBACK=60`, `HORIZONS=[1,3,10]`, fold dates)
2. **Data** — yfinance download, `MarketDataset`, train/val/test splits
3. **Trainer** — `Trainer` class with `_clip_grad_norm_safe()` (handles complex-valued FourierBlock gradients)
4. **Helpers** — `run_hp_search`, `run_walk_forward`, `aggregate_metrics`, `plot_walk_forward`, `summary_table`
5. **Models** — one section per architecture, each with:
   - Model class definition
   - Factory closure (`factory(n_features, **hp) → nn.Module`)
   - HP search cell
   - Walk-forward cell

## Model-Specific Notes

### FEDformer
- Imports from `repos/FEDformer` via `sys.path.insert(0, 'repos/FEDformer')`
- Uses `FourierBlock`, `FourierCrossAttention`, `MultiWaveletTransform`, `MultiWaveletCross`
- **FourierBlock constraint**: `n_heads` must equal 8 (hardcoded weight shape in repo)
- **Wavelet param explosion**: default `c=128` → ~50M params with small d_model. Use `wavelet_c=8`
- HP search filter enforces: `d_model % 8 == 0`, `n_heads == 8` for Fourier, `modes == 16` for Wavelet (modes unused in wavelet path, deduplicates trials)
- Encoder-only design: no decomp, just attention + FFN → linear head

### iTransformer
- **Inverted dimensions**: variables become tokens, time steps become embedding dim
- `DataEmbeddingInverted` is inlined (avoids import conflict with FEDformer's same-named modules)
- `x` is permuted `[B, L, N] → [B, N, L]` before the linear embedding
- Pool strategies: `sp500_token` (takes the target variable's token) vs `mean` (average all variable tokens)
- `target_idx` must be computed with `df.columns.get_loc(("SandP500", "Close"))` for the correct column

### Complex Gradients (FEDformer FourierBlock)
`FourierBlock` has `cfloat` (complex) weight tensors. PyTorch's `clip_grad_norm_` crashes on these. The `_clip_grad_norm_safe()` function in the Trainer cell handles this via `torch.view_as_real()`.

## Key Constants

```python
LOOKBACK   = 60       # input sequence length
HORIZONS   = [1, 3, 10]  # prediction horizons (trading days)
TRAIN_END  = "2014-12-31"
VAL_END    = "2015-12-31"
# Walk-forward folds: 11 folds from 2015 to 2025
```

## Results Layout

```
results/
  baseline/
  lstm_full/         lstm_simple/
  transformer_full/  transformer_simple/
  informer_full/     informer_simple/
  fedformer_full/    fedformer_simple/
  itransformer_full/ itransformer_simple/
```

Each directory contains `hparam_search.csv` and `walk_forward.json`.

## Dependencies

Core: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `yfinance`
Extra: `einops` (FEDformer wavelet layers), `sympy` (FEDformer legendre basis)
