# Frequency Enhanced iTransformer (FEiT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the FEiT hybrid model (iTransformer × FEDformer decomposition) to `research.ipynb`, with HP search and walk-forward for full and simple variants.

**Architecture:** Model code lives in a sibling `feit_model.py` module for testability. The notebook cell imports from it (a small deviation from the all-inline convention; spec section 4.1 only requires the cell to expose the model class, which `import` satisfies). Standalone runnable test scripts (no pytest framework) validate via plain `assert`. A one-shot, idempotent helper script inserts the new cells into the notebook before the `ALL_MODELS` summary cell.

**Tech Stack:** PyTorch, pywavelets, FEDformer's `series_decomp` (vendored in `repos/FEDformer`), Jupyter `nbformat`.

---

## File Structure

| Path | Purpose |
|------|---------|
| `feit_model.py` (new) | `VolatileFeatures` helper + `FEiTransformerForecaster` model class |
| `tests/test_feit_model.py` (new) | Runnable smoke tests; plain `assert`, no pytest |
| `tools/insert_feit_cells.py` (new) | One-shot script that inserts six new cells into `research.ipynb` (idempotent) |
| `research.ipynb` (modify) | Cell 1 install line; six new cells before `ALL_MODELS`; updated `ALL_MODELS` dict |
| `README.md` (modify) | `pywavelets` in pip line |

---

## Conventions used in every task

- **Working directory** is `/Users/dmytroz/projects/thesis/attention-based-finance-prediction` unless stated otherwise. Use absolute paths in commands so shell state can't drift.
- **Python interpreter** is the project venv: `.venv/bin/python` (per `CLAUDE.md`). Create with `python3 -m venv .venv && .venv/bin/pip install -U pip` if missing.
- **Tests** are run as `.venv/bin/python tests/test_feit_model.py`. The script exits non-zero on `AssertionError`. No pytest, no test runner.
- **Commit cadence** is one commit per task. Commit messages: lowercase imperative, no `Co-Authored-By` (per `CLAUDE.md`).
- **Random seed** for tests: `torch.manual_seed(0)` at the top of every test function.

---

### Task 1: Add `pywavelets` dependency

**Files:**
- Modify: `research.ipynb` (cell 1, the `%pip install` line)
- Modify: `README.md` (pip install line in the "Requirements" section)

- [ ] **Step 1: Inspect the current install line**

```bash
jq -r '.cells[1].source | join("")' research.ipynb
```

Expected output:
```
# Встановлення залежностей (запускати один раз)
%pip install torch numpy pandas matplotlib yfinance pyyaml ipykernel --quiet
```

- [ ] **Step 2: Update cell 1 to include `pywavelets`**

Use the following Python helper (one-shot):

```bash
.venv/bin/python - <<'PY'
import json, pathlib
nb = json.loads(pathlib.Path("research.ipynb").read_text())
src = nb["cells"][1]["source"]
joined = "".join(src) if isinstance(src, list) else src
new = joined.replace(
    "%pip install torch numpy pandas matplotlib yfinance pyyaml ipykernel --quiet",
    "%pip install torch numpy pandas matplotlib yfinance pyyaml ipykernel pywavelets --quiet",
)
assert new != joined, "install line not found — aborting"
nb["cells"][1]["source"] = new.splitlines(keepends=True)
pathlib.Path("research.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("install cell updated")
PY
```

Expected: `install cell updated`

- [ ] **Step 3: Update `README.md` pip line**

Use Edit on [README.md](README.md):

Old: `pip install torch numpy pandas scikit-learn matplotlib yfinance einops sympy`
New: `pip install torch numpy pandas scikit-learn matplotlib yfinance einops sympy pywavelets`

- [ ] **Step 4: Install in venv and verify import**

```bash
.venv/bin/pip install pywavelets
.venv/bin/python -c "import pywt; print(pywt.__version__)"
```

Expected: a version string like `1.5.0` (or newer). No `ImportError`.

- [ ] **Step 5: Commit**

```bash
git add research.ipynb README.md
git commit -m "Add pywavelets dependency for FEiT volatile-features wavelet path"
```

---

### Task 2: Scaffold `feit_model.py` and the test runner

This task creates the empty module and a runnable test script that immediately fails (red), establishing the TDD harness. No model logic yet.

**Files:**
- Create: `feit_model.py`
- Create: `tests/test_feit_model.py`

- [ ] **Step 1: Create `tests/` directory if absent**

```bash
mkdir -p tests
```

- [ ] **Step 2: Write the failing scaffold test**

Create [tests/test_feit_model.py](tests/test_feit_model.py):

```python
"""Runnable smoke tests for feit_model. Plain assert, no pytest.

Run: .venv/bin/python tests/test_feit_model.py
"""
import os
import sys
import torch

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Make FEDformer's series_decomp importable
FED = os.path.join(ROOT, "repos", "FEDformer")
if FED not in sys.path:
    sys.path.insert(0, FED)


def test_module_imports():
    torch.manual_seed(0)
    from feit_model import FEiTransformerForecaster, VolatileFeatures  # noqa: F401
    print("test_module_imports: OK")


if __name__ == "__main__":
    test_module_imports()
    print("\nAll tests passed.")
```

- [ ] **Step 3: Run, confirm `ImportError`**

```bash
.venv/bin/python tests/test_feit_model.py
```

Expected: `ModuleNotFoundError: No module named 'feit_model'` (non-zero exit).

- [ ] **Step 4: Create empty `feit_model.py`**

Create [feit_model.py](feit_model.py):

```python
"""Frequency Enhanced iTransformer (FEiT).

Hybrid that takes iTransformer's variable-as-token framing and front-loads
FEDformer's series decomposition: each variable token is split into a
trend half and a frequency-feature half (Fourier or Wavelets) before the
shared encoder.

Public API:
    VolatileFeatures            — frequency feature extractor for the volatile component
    FEiTransformerForecaster    — the forecasting model

See docs/superpowers/specs/2026-04-27-feitransformer-hybrid-model-design.md
for the full design.
"""
from __future__ import annotations

import os
import sys
from typing import Literal

import numpy as np
import pywt
import torch
import torch.nn as nn

# Vendored FEDformer (for series_decomp)
_FED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repos", "FEDformer")
if _FED not in sys.path:
    sys.path.insert(0, _FED)

from layers.Autoformer_EncDec import series_decomp  # noqa: E402


class VolatileFeatures(nn.Module):
    """Placeholder. Filled in by Task 3 (Fourier) and Task 4 (Wavelets)."""
    pass


class FEiTransformerForecaster(nn.Module):
    """Placeholder. Filled in by Task 5."""
    pass
```

- [ ] **Step 5: Re-run, confirm pass**

```bash
.venv/bin/python tests/test_feit_model.py
```

Expected:
```
test_module_imports: OK

All tests passed.
```

- [ ] **Step 6: Commit**

```bash
git add feit_model.py tests/test_feit_model.py
git commit -m "Scaffold feit_model.py + standalone test runner"
```

---

### Task 3: Implement `VolatileFeatures` (Fourier path)

Implement the Fourier branch end-to-end with shape checks and a determinism property (random `mode_select` must be reproducible across instances).

**Files:**
- Modify: `feit_model.py` (replace `VolatileFeatures` placeholder)
- Modify: `tests/test_feit_model.py` (add three tests)

- [ ] **Step 1: Append failing tests for the Fourier path**

Add the following functions to [tests/test_feit_model.py](tests/test_feit_model.py), and call them from `__main__`:

```python
def test_volatile_fourier_low_shape():
    torch.manual_seed(0)
    from feit_model import VolatileFeatures
    B, N, L, K, D = 4, 5, 60, 16, 32
    vf = VolatileFeatures(version="Fourier", lookback=L, d_model=D,
                          modes=K, mode_select="low", seed=0)
    out = vf(torch.randn(B, N, L))
    assert tuple(out.shape) == (B, N, D), f"got {tuple(out.shape)}"
    print("test_volatile_fourier_low_shape: OK")


def test_volatile_fourier_random_shape():
    torch.manual_seed(0)
    from feit_model import VolatileFeatures
    B, N, L, K, D = 4, 5, 60, 16, 32
    vf = VolatileFeatures(version="Fourier", lookback=L, d_model=D,
                          modes=K, mode_select="random", seed=42)
    out = vf(torch.randn(B, N, L))
    assert tuple(out.shape) == (B, N, D), f"got {tuple(out.shape)}"
    print("test_volatile_fourier_random_shape: OK")


def test_volatile_fourier_random_determinism():
    """Two instances built with the same seed must pick the same modes."""
    from feit_model import VolatileFeatures
    a = VolatileFeatures(version="Fourier", lookback=60, d_model=32,
                         modes=16, mode_select="random", seed=42)
    b = VolatileFeatures(version="Fourier", lookback=60, d_model=32,
                         modes=16, mode_select="random", seed=42)
    assert torch.equal(a.modes_idx, b.modes_idx), \
        f"modes_idx differ: {a.modes_idx.tolist()} vs {b.modes_idx.tolist()}"
    print("test_volatile_fourier_random_determinism: OK")
```

Update `__main__` block:

```python
if __name__ == "__main__":
    test_module_imports()
    test_volatile_fourier_low_shape()
    test_volatile_fourier_random_shape()
    test_volatile_fourier_random_determinism()
    print("\nAll tests passed.")
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv/bin/python tests/test_feit_model.py
```

Expected: `TypeError` (placeholder takes no kwargs) or `AttributeError`. Non-zero exit.

- [ ] **Step 3: Implement the Fourier branch of `VolatileFeatures`**

Replace the `VolatileFeatures` placeholder in [feit_model.py](feit_model.py) with:

```python
class VolatileFeatures(nn.Module):
    """Project a length-L volatile series into a d_model token via FFT or DWT.

    For ``version="Fourier"``: rfft(vol) along the time axis, keep K modes
    (selected at construction by ``mode_select``), flatten real+imag, project
    via ``Linear(2*K, d_model)``.

    For ``version="Wavelets"``: implemented in Task 4.

    Mode selection is fixed at construction. ``mode_select="random"`` uses a
    seeded torch generator so two instances built with the same ``seed``
    yield the same indices.
    """
    def __init__(
        self,
        version: Literal["Fourier", "Wavelets"],
        lookback: int,
        d_model: int,
        # Fourier
        modes: int = 32,
        mode_select: Literal["random", "low"] = "random",
        # Wavelets (filled in Task 4)
        wavelet_base: str = "db4",
        wavelet_level: int = 3,
        seed: int = 0,
    ):
        super().__init__()
        assert version in ("Fourier", "Wavelets"), f"unknown version: {version}"
        self.version  = version
        self.lookback = lookback
        self.d_model  = d_model

        if version == "Fourier":
            n_freq = lookback // 2 + 1
            assert modes <= n_freq, f"modes={modes} exceeds n_freq={n_freq}"
            self.modes = modes
            if mode_select == "low":
                idx = torch.arange(modes)
            elif mode_select == "random":
                gen = torch.Generator().manual_seed(seed)
                idx = torch.randperm(n_freq, generator=gen)[:modes].sort().values
            else:
                raise ValueError(f"unknown mode_select: {mode_select}")
            self.register_buffer("modes_idx", idx, persistent=False)
            self.proj = nn.Linear(2 * modes, d_model)
        else:
            # Wavelets — filled in by Task 4
            raise NotImplementedError("Wavelets path implemented in Task 4")

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """vol: [B, N, L] real-valued volatile series → [B, N, d_model]."""
        if self.version == "Fourier":
            X    = torch.fft.rfft(vol, dim=-1)              # [B, N, n_freq] complex
            X_k  = X.index_select(-1, self.modes_idx)       # [B, N, K] complex
            feat = torch.cat([X_k.real, X_k.imag], dim=-1)  # [B, N, 2K]
            return self.proj(feat)                          # [B, N, d_model]
        raise NotImplementedError(self.version)
```

- [ ] **Step 4: Run, confirm pass**

```bash
.venv/bin/python tests/test_feit_model.py
```

Expected: all three new tests print `OK`, plus the prior `test_module_imports`.

- [ ] **Step 5: Commit**

```bash
git add feit_model.py tests/test_feit_model.py
git commit -m "Implement VolatileFeatures Fourier path with deterministic mode select"
```

---

### Task 4: Implement `VolatileFeatures` (Wavelets path)

**Files:**
- Modify: `feit_model.py` (replace the `NotImplementedError` branch)
- Modify: `tests/test_feit_model.py` (add two tests)

- [ ] **Step 1: Append failing tests**

Add to [tests/test_feit_model.py](tests/test_feit_model.py):

```python
def test_volatile_wavelets_shape():
    torch.manual_seed(0)
    from feit_model import VolatileFeatures
    B, N, L, D = 4, 5, 60, 32
    vf = VolatileFeatures(version="Wavelets", lookback=L, d_model=D,
                          wavelet_base="db4", wavelet_level=3)
    out = vf(torch.randn(B, N, L))
    assert tuple(out.shape) == (B, N, D), f"got {tuple(out.shape)}"
    print("test_volatile_wavelets_shape: OK")


def test_volatile_wavelets_no_grad_through_pywt():
    """pywt is non-differentiable; gradient must still flow through self.proj."""
    torch.manual_seed(0)
    from feit_model import VolatileFeatures
    vf = VolatileFeatures(version="Wavelets", lookback=60, d_model=32,
                          wavelet_base="db4", wavelet_level=3)
    x = torch.randn(2, 3, 60, requires_grad=False)
    out = vf(x)
    out.sum().backward()
    assert vf.proj.weight.grad is not None, "proj weights got no gradient"
    assert torch.isfinite(vf.proj.weight.grad).all(), "non-finite gradients"
    print("test_volatile_wavelets_no_grad_through_pywt: OK")
```

Add the calls to `__main__`:

```python
    test_volatile_wavelets_shape()
    test_volatile_wavelets_no_grad_through_pywt()
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv/bin/python tests/test_feit_model.py
```

Expected: `NotImplementedError: Wavelets path implemented in Task 4`.

- [ ] **Step 3: Implement the Wavelets branch**

In [feit_model.py](feit_model.py), replace the `else: raise NotImplementedError(...)` block in `__init__`, and extend `forward`. Final `VolatileFeatures` class:

```python
class VolatileFeatures(nn.Module):
    """Project a length-L volatile series into a d_model token via FFT or DWT.

    For ``version="Fourier"``: rfft(vol) along the time axis, keep K modes
    (selected at construction by ``mode_select``), flatten real+imag, project
    via ``Linear(2*K, d_model)``.

    For ``version="Wavelets"``: ``pywt.wavedec(vol, wavelet_base, level=wavelet_level)``
    along the time axis, concatenate all coefficient arrays in pywt order
    ``[cA_n, cD_n, cD_(n-1), ..., cD_1]``, project via ``Linear(D_coeffs, d_model)``.
    The DWT is non-differentiable (numpy roundtrip) but the projection is.

    Mode selection is fixed at construction. ``mode_select="random"`` uses a
    seeded torch generator so two instances built with the same ``seed``
    yield the same indices.
    """
    def __init__(
        self,
        version: Literal["Fourier", "Wavelets"],
        lookback: int,
        d_model: int,
        # Fourier
        modes: int = 32,
        mode_select: Literal["random", "low"] = "random",
        # Wavelets
        wavelet_base: str = "db4",
        wavelet_level: int = 3,
        seed: int = 0,
    ):
        super().__init__()
        assert version in ("Fourier", "Wavelets"), f"unknown version: {version}"
        self.version  = version
        self.lookback = lookback
        self.d_model  = d_model

        if version == "Fourier":
            n_freq = lookback // 2 + 1
            assert modes <= n_freq, f"modes={modes} exceeds n_freq={n_freq}"
            self.modes = modes
            if mode_select == "low":
                idx = torch.arange(modes)
            elif mode_select == "random":
                gen = torch.Generator().manual_seed(seed)
                idx = torch.randperm(n_freq, generator=gen)[:modes].sort().values
            else:
                raise ValueError(f"unknown mode_select: {mode_select}")
            self.register_buffer("modes_idx", idx, persistent=False)
            self.proj = nn.Linear(2 * modes, d_model)
        else:
            self.wavelet_base  = wavelet_base
            self.wavelet_level = wavelet_level
            # Determine output coefficient length once via a dry run.
            dummy   = np.zeros((1, lookback), dtype=np.float32)
            coeffs  = pywt.wavedec(dummy, wavelet_base, level=wavelet_level, axis=-1)
            d_coef  = sum(c.shape[-1] for c in coeffs)
            self.d_coef = d_coef
            self.proj   = nn.Linear(d_coef, d_model)

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """vol: [B, N, L] real-valued volatile series → [B, N, d_model]."""
        if self.version == "Fourier":
            X    = torch.fft.rfft(vol, dim=-1)              # [B, N, n_freq] complex
            X_k  = X.index_select(-1, self.modes_idx)       # [B, N, K] complex
            feat = torch.cat([X_k.real, X_k.imag], dim=-1)  # [B, N, 2K]
            return self.proj(feat)                          # [B, N, d_model]

        # Wavelets: numpy roundtrip on CPU; non-differentiable transform.
        with torch.no_grad():
            x_np   = vol.detach().cpu().numpy().astype(np.float32, copy=False)
            coeffs = pywt.wavedec(x_np, self.wavelet_base,
                                  level=self.wavelet_level, axis=-1)
            feat_np = np.concatenate(coeffs, axis=-1)       # [B, N, d_coef]
        feat = torch.from_numpy(feat_np).to(vol.device)
        return self.proj(feat)                              # [B, N, d_model]
```

- [ ] **Step 4: Run, confirm pass**

```bash
.venv/bin/python tests/test_feit_model.py
```

Expected: all five tests now print `OK`.

- [ ] **Step 5: Commit**

```bash
git add feit_model.py tests/test_feit_model.py
git commit -m "Implement VolatileFeatures Wavelets path via pywt.wavedec"
```

---

### Task 5: Implement `FEiTransformerForecaster` with `pool="mean"`

**Files:**
- Modify: `feit_model.py`
- Modify: `tests/test_feit_model.py`

- [ ] **Step 1: Append failing tests**

Add to [tests/test_feit_model.py](tests/test_feit_model.py):

```python
def test_feit_forward_mean_fourier():
    torch.manual_seed(0)
    from feit_model import FEiTransformerForecaster
    B, L, N, H = 4, 60, 5, 3
    m = FEiTransformerForecaster(
        n_features=N, d_model=64, n_heads=4, e_layers=2, dropout=0.1,
        version="Fourier", modes=16, mode_select="low",
        moving_avg=25, n_horizons=H, lookback=L, target_idx=0,
        pool="mean",
    )
    y = m(torch.randn(B, L, N))
    assert tuple(y.shape) == (B, H), f"got {tuple(y.shape)}"
    print("test_feit_forward_mean_fourier: OK")


def test_feit_forward_mean_wavelets():
    torch.manual_seed(0)
    from feit_model import FEiTransformerForecaster
    B, L, N, H = 4, 60, 5, 3
    m = FEiTransformerForecaster(
        n_features=N, d_model=64, n_heads=4, e_layers=2, dropout=0.1,
        version="Wavelets", wavelet_base="db4", wavelet_level=3,
        moving_avg=25, n_horizons=H, lookback=L, target_idx=0,
        pool="mean",
    )
    y = m(torch.randn(B, L, N))
    assert tuple(y.shape) == (B, H), f"got {tuple(y.shape)}"
    print("test_feit_forward_mean_wavelets: OK")


def test_feit_param_count_reasonable():
    """At d_model=64 the FEiT model should be in the same OoM as iTransformer.
    Sanity check, not a strict bound."""
    torch.manual_seed(0)
    from feit_model import FEiTransformerForecaster
    m = FEiTransformerForecaster(
        n_features=145, d_model=64, n_heads=8, e_layers=2, dropout=0.1,
        version="Fourier", modes=32, mode_select="low",
        moving_avg=25, n_horizons=3, lookback=60, target_idx=0,
        pool="mean",
    )
    n_params = sum(p.numel() for p in m.parameters())
    assert 50_000 < n_params < 5_000_000, f"unexpected param count: {n_params:,}"
    print(f"test_feit_param_count_reasonable: OK ({n_params:,} params)")
```

Add corresponding calls to `__main__`.

- [ ] **Step 2: Run, confirm fail**

Expected: `TypeError` (placeholder class), non-zero exit.

- [ ] **Step 3: Implement `FEiTransformerForecaster`**

Replace the `FEiTransformerForecaster` placeholder in [feit_model.py](feit_model.py) with:

```python
class FEiTransformerForecaster(nn.Module):
    """Frequency Enhanced iTransformer.

    Variable-as-token (iTransformer-style) encoder operating on 2N tokens:
    N trend tokens (low-pass filtered) + N volatile tokens (frequency-feature
    encoded). One shared TransformerEncoder; concat target tokens at the head.

    Args:
        n_features:    number of input variables N
        d_model:       embedding width
        n_heads:       attention heads
        e_layers:      encoder depth
        dropout:       dropout rate
        use_norm:      apply RevIN-like per-batch norm before decomposition
        pool:          "sp500_concat" | "mean"
        moving_avg:    kernel size for FEDformer's series_decomp
        version:       "Fourier" | "Wavelets"
        modes:         (Fourier) number of FFT modes to keep
        mode_select:   (Fourier) "random" | "low"
        wavelet_base:  (Wavelets) e.g. "db4"
        wavelet_level: (Wavelets) DWT level
        n_horizons:    output dim H
        lookback:      input series length L
        target_idx:    column index of the target variable for sp500_concat pool
        seed:          for the Fourier random mode selection
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        dropout: float = 0.1,
        use_norm: bool = False,
        pool: Literal["sp500_concat", "mean"] = "sp500_concat",
        moving_avg: int = 25,
        version: Literal["Fourier", "Wavelets"] = "Fourier",
        modes: int = 32,
        mode_select: Literal["random", "low"] = "random",
        wavelet_base: str = "db4",
        wavelet_level: int = 3,
        n_horizons: int = 3,
        lookback: int = 60,
        target_idx: int = 0,
        seed: int = 0,
    ):
        super().__init__()
        assert pool in ("sp500_concat", "mean"), f"unknown pool: {pool}"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_features = n_features
        self.use_norm   = use_norm
        self.pool       = pool
        self.target_idx = target_idx

        # Decomposition
        self.decomp = series_decomp(moving_avg)

        # Embeddings
        self.trend_embed    = nn.Linear(lookback, d_model)
        self.volatile_embed = VolatileFeatures(
            version=version, lookback=lookback, d_model=d_model,
            modes=modes, mode_select=mode_select,
            wavelet_base=wavelet_base, wavelet_level=wavelet_level,
            seed=seed,
        )
        self.token_type_emb = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.token_type_emb, std=0.02)
        self.dropout = nn.Dropout(dropout)

        # Encoder (matches existing iTransformerForecaster config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=e_layers, norm=nn.LayerNorm(d_model)
        )

        # Head — shape depends on pool
        if pool == "sp500_concat":
            self.head = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_horizons),
            )
        else:  # "mean"
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, n_horizons),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, N] → [B, n_horizons]."""
        # RevIN-like normalization (matches existing iTransformerForecaster.use_norm)
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x     = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x     = x / stdev

        # series_decomp returns (seasonal, trend); both shape [B, L, N]
        seasonal, trend = self.decomp(x)

        # Permute to [B, N, L] so each variable's L-step history is the last dim
        trend_t = trend.permute(0, 2, 1)
        vol_t   = seasonal.permute(0, 2, 1)

        # Embed
        trend_tok = self.trend_embed(trend_t)        # [B, N, d_model]
        vol_tok   = self.volatile_embed(vol_t)       # [B, N, d_model]

        # Token-type bias: [0]→trend, [1]→volatile
        trend_tok = trend_tok + self.token_type_emb[0]
        vol_tok   = vol_tok   + self.token_type_emb[1]

        # Stack along token axis → [B, 2N, d_model]
        tokens = torch.cat([trend_tok, vol_tok], dim=1)
        tokens = self.dropout(tokens)

        # Encode
        out = self.encoder(tokens)                   # [B, 2N, d_model]

        # Pool
        if self.pool == "sp500_concat":
            t_tok = out[:, self.target_idx]                       # [B, d_model]
            v_tok = out[:, self.n_features + self.target_idx]     # [B, d_model]
            pooled = torch.cat([t_tok, v_tok], dim=-1)            # [B, 2*d_model]
        else:  # "mean"
            pooled = out.mean(dim=1)                              # [B, d_model]

        return self.head(pooled)                                  # [B, n_horizons]
```

- [ ] **Step 4: Run, confirm pass**

```bash
.venv/bin/python tests/test_feit_model.py
```

Expected: all eight tests so far print `OK`. The param-count test prints the actual count.

- [ ] **Step 5: Commit**

```bash
git add feit_model.py tests/test_feit_model.py
git commit -m "Implement FEiTransformerForecaster with mean pool"
```

---

### Task 6: Verify `pool="sp500_concat"` and gradient flow

`sp500_concat` was already implemented in Task 5; this task adds the explicit tests for it and a backward-pass gradient check.

**Files:**
- Modify: `tests/test_feit_model.py`

- [ ] **Step 1: Append tests**

Add to [tests/test_feit_model.py](tests/test_feit_model.py):

```python
def test_feit_forward_sp500_concat():
    torch.manual_seed(0)
    from feit_model import FEiTransformerForecaster
    B, L, N, H = 4, 60, 5, 3
    m = FEiTransformerForecaster(
        n_features=N, d_model=64, n_heads=4, e_layers=2, dropout=0.0,
        version="Fourier", modes=16, mode_select="low",
        moving_avg=25, n_horizons=H, lookback=L, target_idx=3,
        pool="sp500_concat",
    )
    y = m(torch.randn(B, L, N))
    assert tuple(y.shape) == (B, H), f"got {tuple(y.shape)}"
    print("test_feit_forward_sp500_concat: OK")


def test_feit_backward_no_nan():
    """Loss → backward → all grads finite."""
    torch.manual_seed(0)
    from feit_model import FEiTransformerForecaster
    m = FEiTransformerForecaster(
        n_features=5, d_model=64, n_heads=4, e_layers=2, dropout=0.0,
        version="Fourier", modes=16, mode_select="low",
        moving_avg=25, n_horizons=3, lookback=60, target_idx=3,
        pool="sp500_concat",
    )
    x = torch.randn(4, 60, 5)
    y = m(x)
    loss = y.pow(2).mean()
    loss.backward()
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients in: {bad}"
    print("test_feit_backward_no_nan: OK")


def test_feit_decomp_reconstruction():
    """series_decomp(x) returns (seasonal, trend); seasonal + trend ≈ x."""
    torch.manual_seed(0)
    import sys
    sys.path.insert(0, os.path.join(ROOT, "repos", "FEDformer"))
    from layers.Autoformer_EncDec import series_decomp
    decomp = series_decomp(25)
    x = torch.randn(4, 60, 5)
    seasonal, trend = decomp(x)
    diff = (seasonal + trend - x).abs().max().item()
    assert diff < 1e-5, f"reconstruction error: {diff}"
    print("test_feit_decomp_reconstruction: OK")
```

Add the corresponding three calls to `__main__`.

- [ ] **Step 2: Run, confirm pass**

```bash
.venv/bin/python tests/test_feit_model.py
```

Expected: all eleven tests print `OK`. No code changes were needed in `feit_model.py` for sp500_concat (already implemented in Task 5); these tests close the verification loop.

- [ ] **Step 3: Commit**

```bash
git add tests/test_feit_model.py
git commit -m "Add sp500_concat / backward / decomp-reconstruction tests for FEiT"
```

---

### Task 7: Build `tools/insert_feit_cells.py`

Idempotent script that inserts six new cells (markdown × 3, code × 3) into `research.ipynb` immediately before the `ALL_MODELS = {` cell. The model code is **not** duplicated here — the notebook cell imports from `feit_model`. Walk-forward and HP search code lives inline in the inserted cells (matching the existing pattern for FEDformer and iTransformer).

**Files:**
- Create: `tools/insert_feit_cells.py`

- [ ] **Step 1: Create the helper script**

```bash
mkdir -p tools
```

Create [tools/insert_feit_cells.py](tools/insert_feit_cells.py):

```python
"""One-shot, idempotent insertion of the FEiT section into research.ipynb.

Usage:
    .venv/bin/python tools/insert_feit_cells.py          # inserts (skips if already present)
    .venv/bin/python tools/insert_feit_cells.py --dry    # shows where it would insert

The script also rewrites the existing ALL_MODELS cell to add FEiT entries.
"""
from __future__ import annotations

import argparse
import sys
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
    """Add FEiT entries to ALL_MODELS dict if not already present.

    Returns True if cell was modified.
    """
    idx = find_all_models_idx(nb)
    src = nb.cells[idx].source
    if '"FEiT":' in src:
        return False

    insertion = (
        '    "FEiT":         os.path.join(RESULTS, "feitransformer",        "walk_forward.json"),\n'
        '    "FEiT-simple":  os.path.join(RESULTS, "feitransformer-simple", "walk_forward.json"),\n'
    )
    # Insert immediately before the closing brace of the dict literal.
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
```

- [ ] **Step 2: Dry-run to confirm position**

```bash
.venv/bin/python tools/insert_feit_cells.py --dry
```

Expected output (cell index will be roughly 75; exact number depends on prior edits):
```
Will insert 6 cells before cell index 75 (ALL_MODELS).
ALL_MODELS cell updated with FEiT entries.
(dry-run — no file written)
```

- [ ] **Step 3: Run for real**

```bash
.venv/bin/python tools/insert_feit_cells.py
```

Expected:
```
Will insert 6 cells before cell index 75 (ALL_MODELS).
ALL_MODELS cell updated with FEiT entries.
Wrote /Users/dmytroz/projects/thesis/attention-based-finance-prediction/research.ipynb
```

- [ ] **Step 4: Verify idempotency**

Run the script a second time:

```bash
.venv/bin/python tools/insert_feit_cells.py
```

Expected:
```
FEiT section already present — nothing to insert.
ALL_MODELS already references FEiT — leaving alone.
Wrote /Users/dmytroz/projects/thesis/attention-based-finance-prediction/research.ipynb
```

(The second run still writes the file but with no semantic changes — fine; git diff should be empty.)

```bash
git diff --stat research.ipynb
```

Expected: no changes after the second run.

- [ ] **Step 5: Verify cell count and content**

```bash
.venv/bin/python -c "import nbformat; nb = nbformat.read('research.ipynb', as_version=4); print('total cells:', len(nb.cells)); idx = next(i for i, c in enumerate(nb.cells) if c.cell_type == 'markdown' and 'FEiT' in c.source); print('FEiT header at index:', idx); print('next 6 cell types:', [nb.cells[idx+i].cell_type for i in range(6)])"
```

Expected:
```
total cells: 84
FEiT header at index: 75
next 6 cell types: ['markdown', 'code', 'markdown', 'code', 'markdown', 'code']
```

- [ ] **Step 6: Commit**

```bash
git add tools/insert_feit_cells.py research.ipynb
git commit -m "Insert FEiT section into research.ipynb + ALL_MODELS entries"
```

---

### Task 8: Run notebook smoke cells and validate

This task runs the notebook cells up to and including the new FEiT model cell (the smoke test) headlessly, to confirm the integration works end-to-end before committing GPU time to HP search and walk-forward.

**Files:**
- No edits. Runtime validation only.

- [ ] **Step 1: Execute the notebook headlessly through the FEiT model cell**

```bash
.venv/bin/jupyter nbconvert --to notebook --execute research.ipynb \
    --output research_executed.ipynb --ExecutePreprocessor.timeout=600 \
    --ExecutePreprocessor.allow_errors=False 2>&1 | tail -20
```

If `jupyter` is not installed in the venv, install it first: `.venv/bin/pip install jupyter`.

Expected: completes without errors. If a cell fails, the error is printed.

> **Note:** This runs **all** cells, including the existing HP searches for prior models, which can take a long time. If undesirable, alternative: open the notebook in VS Code / Jupyter Lab and "Run All Above" up to the FEiT model cell only.

- [ ] **Step 2: Inspect FEiT smoke-test output**

```bash
.venv/bin/python - <<'PY'
import nbformat
nb = nbformat.read("research_executed.ipynb", as_version=4)
# Find the FEiT model cell (the code cell immediately after the FEiT header)
for i, c in enumerate(nb.cells):
    if c.cell_type == "markdown" and "Frequency Enhanced iTransformer" in c.source:
        feit_code = nb.cells[i+1]
        for out in feit_code.outputs:
            if "text" in out:
                print(out["text"])
            elif "data" in out and "text/plain" in out["data"]:
                print(out["data"]["text/plain"])
        break
PY
```

Expected output (4 lines, one per `version × pool` combination):
```
SP500_FEAT_IDX (full): <int>    (out of 145 features)
version=Fourier    pool=sp500_concat  output=(4, 3)  params=<int>
version=Fourier    pool=mean          output=(4, 3)  params=<int>
version=Wavelets   pool=sp500_concat  output=(4, 3)  params=<int>
version=Wavelets   pool=mean          output=(4, 3)  params=<int>
```

- [ ] **Step 3: Clean up the executed copy**

```bash
rm research_executed.ipynb
```

- [ ] **Step 4: No commit needed** (no file changes). Mark task complete.

---

### Task 9: Document FEiT in `CLAUDE.md`

Add a brief development-context block so future agents know how the FEiT pieces fit together.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Append a `### FEiT` block under the `## Model-Specific Notes` section**

Use Edit on [CLAUDE.md](CLAUDE.md). Locate the line `### iTransformer` and insert a new section *after* the iTransformer block and *before* `### Complex Gradients (FEDformer FourierBlock)`.

New content to insert:

```markdown
### FEiT (Frequency Enhanced iTransformer) — custom hybrid
- Source: `feit_model.py` (sibling to `research.ipynb`, imported by the notebook cell)
- Tests: `tests/test_feit_model.py` — runnable as `.venv/bin/python tests/test_feit_model.py`
- Cell-insertion script: `tools/insert_feit_cells.py` (idempotent)
- Decomposition once at input; trend + volatile streams concat into 2N tokens
- Volatile branch: FFT-modes (`version="Fourier"`) or `pywt.wavedec` (`version="Wavelets"`)
- HP search filter: `d_model % n_heads == 0`; for Fourier fix `wavelet_level=2`; for Wavelets fix `modes=16` (deduplicates trials, matches existing FEDformer pattern)
- Pool: `sp500_concat` concatenates the trend and volatile target tokens; `mean` averages all 2N
- No FourierBlock complex gradients — `torch.fft.rfft` returns complex but `Linear` weights are real

```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Document FEiT in CLAUDE.md"
```

---

## Out of scope (not implemented by this plan)

- Running the full HP search and walk-forward on FEiT (cells will be ready; user runs when GPU time is allocated).
- Final summary-table re-render and per-horizon comparison plots (existing `summary_table` and `plot_walk_forward` helpers in cell 75 will pick up the new entries automatically once results JSON files exist).
- Per-layer recurrent decomposition; frequency-domain attention inside encoder layers; cross-attention between trend and volatile streams. All deliberately deferred per spec section 6.
