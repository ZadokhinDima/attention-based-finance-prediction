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


class FEiTransformerForecaster(nn.Module):
    """Placeholder. Filled in by Task 5."""
    pass
