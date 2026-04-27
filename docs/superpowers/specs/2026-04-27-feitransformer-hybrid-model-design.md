# Frequency Enhanced iTransformer (FEiT) — Design

**Status:** approved (brainstorm), pending implementation plan
**Date:** 2026-04-27
**Author:** dmytro.zadokhin@grammarly.com (with Claude)

## 1. Motivation

The thesis benchmarks five attention-based architectures for multi-horizon S&P 500 log-return forecasting. The two most recent additions — FEDformer and iTransformer — capture different inductive biases:

- **FEDformer** decomposes each time series into trend + seasonal inside every encoder layer and runs a frequency-domain attention (`FourierBlock` or `MultiWaveletTransform`) on the seasonal residual.
- **iTransformer** inverts the dimension convention: variables become tokens, full lookback histories become embeddings. Attention runs across variables.

The two are complementary rather than competing. This design adds a custom hybrid — **Frequency Enhanced iTransformer (FEiT)** — that keeps iTransformer's variable-as-token framing but front-loads FEDformer's decomposition idea: each variable token is split into a trend half and a frequency-feature half before the encoder. The result is a single shared encoder operating over `2N` tokens.

Decomposition is performed **once at the input**, not at every encoder layer. This is a deliberate simplification of FEDformer's recurrent decomp.

## 2. Architecture

### 2.1 Forward pass

```
x: [B, L, N]                                 # raw OHLCV per variable
   │
   ├─ optional RevIN-like norm (use_norm)
   │
   ├─ series_decomp (moving avg, kernel=moving_avg)
   │     ├─ trend  [B, L, N]
   │     └─ vol    [B, L, N]   (= x - trend)
   │
   │ permute each → [B, N, L]
   │
   ├─ trend embedding:  Linear(L, d_model)               → trend_tok [B, N, d_model]
   │
   ├─ volatile embedding (version=Fourier|Wavelets):
   │     Fourier:   rfft(vol, dim=L) → keep K modes
   │                 → flatten real+imag → Linear(2K, d_model)
   │     Wavelets:  pywt.wavedec(vol, base, level=L_wav)
   │                 → concat coeffs → Linear(coeff_dim, d_model)
   │     →  vol_tok [B, N, d_model]
   │
   ├─ token-type embedding: add learnable [trend_emb] / [vol_emb]
   │
   ├─ stack along token axis: tokens = cat([trend_tok, vol_tok], dim=1) → [B, 2N, d_model]
   │
   ├─ TransformerEncoder (e_layers, n_heads, GELU, norm_first=True) → [B, 2N, d_model]
   │
   ├─ pool:
   │     "sp500_concat": cat(out[:, target_idx], out[:, N + target_idx]) → [B, 2·d_model]
   │     "mean":         out.mean(dim=1)                                  → [B, d_model]
   │
   └─ MLP head → [B, n_horizons]
```

### 2.2 Components

**SeriesDecomp** — Reuse FEDformer's `layers.Autoformer_EncDec.series_decomp` directly. `nn.AvgPool1d` with `kernel=moving_avg`, edge-padded so output length stays `L`. Returns `(seasonal, trend)`. No re-implementation.

**VolatileFeatures** — Self-contained `nn.Module` with a `version` flag.

- *Fourier path*:
  ```python
  X     = torch.fft.rfft(vol, dim=-1)        # [B, N, L//2+1] complex
  X_k   = X[..., self.modes_idx]             # [B, N, K] complex
  feat  = cat([X_k.real, X_k.imag], dim=-1)  # [B, N, 2K]
  out   = self.proj(feat)                    # Linear(2K, d_model)
  ```
  `modes_idx` is fixed at `__init__`. `mode_select="random"` picks `K` random indices via a seeded RNG; `mode_select="low"` picks `[0:K]`. Same semantics as FEDformer's `FourierBlock`.

- *Wavelet path*:
  ```python
  coeffs = pywt.wavedec(vol_np, wavelet_base, level=wavelet_level)
  # coeffs is [cA_n, cD_n, cD_(n-1), ..., cD_1] — concat in this order
  feat   = np.concatenate(coeffs, axis=-1)             # fixed length D
  out    = self.proj(torch.from_numpy(feat).to(device))  # Linear(D, d_model)
  ```
  pywt is numpy-based; we wrap in a CPU numpy roundtrip per batch. Acceptable cost — runs once at input, not per layer. `D` is determined at `__init__` via a dry run on a dummy length-`L` tensor; `proj` is sized accordingly. Note: the notebook's `Trainer` already moves tensors to device via `.to(self.device)`; the wavelet path detaches/numpy-converts the input on the fly, projects on CPU, then moves the resulting feature tensor back to whichever device the rest of the model is on.

**Trend embedding** — `Linear(L, d_model)`, identical to iTransformer's `DataEmbeddingInverted.value_embedding`.

**Token-type embedding** — `nn.Parameter(torch.zeros(2, d_model))`, init `std=0.02`. Index `0` added to all trend tokens, index `1` to all volatile tokens. Disambiguates token roles for the encoder without relying on positional ordering.

**Encoder** — PyTorch `nn.TransformerEncoder`, identical config to the existing `iTransformerForecaster`: `d_model`, `n_heads`, `e_layers`, `dropout`, `dim_feedforward=4·d_model`, GELU, `batch_first=True`, `norm_first=True`, final `LayerNorm`. No custom attention.

**Head** — Two shapes, selected at init by `pool`:
- `pool="sp500_concat"`: `Linear(2·d_model, d_model) → GELU → Linear(d_model, n_horizons)`
- `pool="mean"`: `Linear(d_model, d_model // 2) → GELU → Linear(d_model // 2, n_horizons)`

## 3. Hyperparameters

### 3.1 Constructor signature

```python
FEiTransformerForecaster(
    n_features:    int,
    # iTransformer-shared
    d_model:       int  = 128,
    n_heads:       int  = 4,
    e_layers:      int  = 2,
    dropout:       float = 0.1,
    use_norm:      bool  = False,
    pool:          str   = "sp500_concat",  # | "mean"
    # FEDformer-shared (decomposition + freq)
    moving_avg:    int   = 25,
    version:       str   = "Fourier",       # "Fourier" | "Wavelets"
    modes:         int   = 32,              # Fourier only
    mode_select:   str   = "random",        # "random" | "low"
    wavelet_base:  str   = "db4",           # Wavelets only
    wavelet_level: int   = 3,               # Wavelets only
    # task
    n_horizons:    int   = 3,
    lookback:      int   = 60,
    target_idx:    int   = 0,
)
```

### 3.2 HP search grid

```python
FEIT_SEARCH = dict(
    d_model       = [64, 128],
    n_heads       = [4, 8],
    e_layers      = [2, 3],
    dropout       = [0.1, 0.2],
    moving_avg    = [13, 25],
    version       = ["Fourier", "Wavelets"],
    modes         = [16, 32],          # only varies for Fourier
    wavelet_level = [2, 3],            # only varies for Wavelets
    use_norm      = [False, True],
    pool          = ["sp500_concat", "mean"],
)
```

### 3.3 Filter rules

- `d_model % n_heads == 0`
- For `version == "Fourier"`: fix `wavelet_level == 2` (unused; deduplicates trials)
- For `version == "Wavelets"`: fix `modes == 16` (unused; deduplicates trials)

Same deduplication trick used in the existing `FEDformer` HP search.

## 4. Notebook integration

### 4.1 Cell layout

Insert a new section after iTransformer (cells 67-73), before the final summary cell (75). Six cells, matching the established pattern:

1. **markdown** — `## Frequency Enhanced iTransformer (FEiT)` plus a brief Ukrainian description.
2. **code** — `FEiTransformerForecaster` class + `VolatileFeatures` helper. Smoke test at the bottom: instantiate with both `version` values × both `pool` values on `torch.randn(4, LOOKBACK, 145)`, print output shape and param count.
3. **markdown** — `### Пошук гіперпараметрів та Walk-Forward`.
4. **code** — `feitransformer_factory(...)` closure, `FEIT_SEARCH` grid + filter, `run_hp_search(...)`, `run_walk_forward(...)` for the **full** version.
5. **markdown** — `### Спрощений FEiT (тільки S&P 500)`.
6. **code** — same as cell 4 but for the simple (S&P-only) variant.

### 4.2 Results layout

Matches the existing notebook convention (source of truth: cells 38/40, 62/64, 70/72 — `os.path.join(RESULTS, "<model>")` for full, `<model>-simple` for simple):

```
results/
  feitransformer/
    hparam_search.csv
    walk_forward.json
  feitransformer-simple/
    hparam_search.csv
    walk_forward.json
```

### 4.3 Summary table

Append entries to `ALL_MODELS` in cell 75, using display names (matching the existing `"FEDformer"`, `"LSTM"` style):

```python
"FEiT":         os.path.join(RESULTS, "feitransformer",        "walk_forward.json"),
"FEiT-simple":  os.path.join(RESULTS, "feitransformer-simple", "walk_forward.json"),
```

### 4.4 Dependencies

- Add `pywavelets` to the install cell (cell 1) and to the README's pip line.
- FEDformer repo (already cloned) provides `series_decomp`. iTransformer repo not strictly required (we inline the inverted-embedding pattern as the existing iTransformer cell already does), but its presence does not interfere.

### 4.5 Trainer

No changes. Reuse existing `Trainer` and `_clip_grad_norm_safe()`. The model uses `torch.fft.rfft` with real `Linear` weights — no complex-valued parameters — so safe-clip is harmless overhead, not a requirement.

## 5. Testing & validation

1. **Smoke test** in the model cell: both `version` × both `pool` × random input → output shape `(B, n_horizons)` and param count printed. Param count must be in the same order of magnitude as FEDformer/iTransformer at equal `d_model`.

2. **Forward/backward sanity**: dummy loss → `backward()` → check no NaNs in any gradient.

3. **Decomposition reconstruction check** (dev-only, deletable cell): `trend + seasonal == x` within float tolerance using `series_decomp` on a real batch.

4. **Mode-selection determinism**: two `FEiTransformerForecaster` instances built with the same seed must have identical `modes_idx` for `version="Fourier"` so walk-forward folds train consistent models.

5. **Walk-forward parity gate**: run walk-forward on the **simple** variant first. If MAE is significantly worse than the existing `itransformer_simple` baseline, abort and debug the volatile-features wiring before running the full HP grid.

No new test framework. Notebook does not have one and the thesis does not require it.

## 6. Out of scope

- Per-layer recurrent decomposition (FEDformer-style). Decomp is once at input.
- Frequency-domain attention inside encoder layers (FEDformer's `FourierBlock`). Attention is plain `nn.MultiheadAttention`.
- Cross-attention between trend and volatile streams. The shared encoder handles cross-stream mixing implicitly.
- Two-encoder (separate weights per stream) variants.
- Replacing the existing iTransformer or FEDformer cells. FEiT is purely additive.
