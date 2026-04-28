# TODO

## FEDformer: predict-mean collapse

**Symptom:** all FEDformer HP-search configs converge to predicting `bias ≈ mean(y_train)`. `dir_acc` is bit-identical (0.61883944272995) across 34 different architectures (17k–700k params, Fourier/Wavelets, e_layers=1/2). val_loss varies ~1% (magnitude differs slightly), but sign of every prediction is the same in every model.

**Root cause:** FourierBlock has `cfloat` weights with small init norm → first predictions ~0 → MSE gradient `(pred - y)` is small → AdamW with `weight_decay=1e-3` stuffs those weak weights to zero faster than the gradient can lift them. Encoder dies; head's last `Linear` bias trains to `mean(y_train)` and dominates the output. Sign of output is constant across val samples.

**Why other models are unaffected:** real-valued weights with standard Kaiming init plus residual paths (LSTM hidden state, Transformer skip connections) accumulate signal fast enough to escape weight-decay pull.

**Fix (in order of cheapness):**
1. Parameterize `Trainer(weight_decay=...)` (currently hardcoded `1e-3`) and pass `weight_decay=1e-5` for FEDformer.
2. Lower FEDFORMER_SEARCH dropout floor to `[0.0, 0.1, 0.2]`.
3. Replace CosineAnnealingLR with `ReduceLROnPlateau` for FEDformer so initial LR isn't decayed too fast.
4. (Last resort) override FourierBlock init in vendored repo — Kaiming-style on real/imag halves with std=0.1.

**Diagnostic** (to confirm before fix):
```python
import torch
m = FEDformerForecaster(n_features=145, d_model=32, n_heads=8,
                       e_layers=2, modes=16, version="Wavelets",
                       dropout=0.2, n_horizons=2, lookback=60)
x = torch.randn(8, 60, 145)
print(m(x))   # untrained — should vary across samples; if all ~0, init is the culprit
```

**Status:** known and accepted for current run; revisit after notebook completes.

---

## Per-model vs baseline comparison plots

**Want:** instead of a single ALL_MODELS comparison at the end, draw a comparison plot after each model section showing **baseline + that model (full) + that model (simple)** — three lines per metric/horizon. Lets you see whether each model beats baseline on its own without scanning a crowded global chart.

**Where:** insert one new code cell at the end of each model section (after the simple-version walk-forward). Six cells total: LSTM, Transformer, Informer, FEDformer, iTransformer, FEiT.

**Implementation sketch:**
```python
plot_walk_forward({
    "Baseline":            os.path.join(RESULTS, "baseline", "walk_forward.json"),
    "LSTM":                os.path.join(RESULTS, "lstm", "walk_forward.json"),
    "LSTM-simple":         os.path.join(RESULTS, "lstm-simple", "walk_forward.json"),
}, metrics=("DirAcc",))
```

Helper script: `tools/insert_per_model_comparisons.py` — locate each model's simple-version cell, append the comparison cell after it. Idempotent.

**Status:** to do after current notebook run.
