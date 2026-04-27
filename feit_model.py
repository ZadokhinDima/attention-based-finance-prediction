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
            # Clamp rather than assert: HP grids may request modes >= n_freq
            # (e.g. modes=32 at L=60 → n_freq=31). Silently use n_freq.
            modes = min(modes, n_freq)
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
    """Frequency Enhanced iTransformer.

    Variable-as-token (iTransformer-style) encoder operating on 2N tokens:
    N trend tokens (low-pass filtered) + N volatile tokens (frequency-feature
    encoded). One shared TransformerEncoder; concat target tokens at the head.
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
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x     = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x     = x / stdev

        # series_decomp returns (seasonal, trend); both shape [B, L, N]
        seasonal, trend = self.decomp(x)

        # Permute to [B, N, L]
        trend_t = trend.permute(0, 2, 1)
        vol_t   = seasonal.permute(0, 2, 1)

        # Embed
        trend_tok = self.trend_embed(trend_t)        # [B, N, d_model]
        vol_tok   = self.volatile_embed(vol_t)       # [B, N, d_model]

        # Token-type bias
        trend_tok = trend_tok + self.token_type_emb[0]
        vol_tok   = vol_tok   + self.token_type_emb[1]

        # Stack along token axis → [B, 2N, d_model]
        tokens = torch.cat([trend_tok, vol_tok], dim=1)
        tokens = self.dropout(tokens)

        out = self.encoder(tokens)                   # [B, 2N, d_model]

        if self.pool == "sp500_concat":
            t_tok = out[:, self.target_idx]
            v_tok = out[:, self.n_features + self.target_idx]
            pooled = torch.cat([t_tok, v_tok], dim=-1)            # [B, 2*d_model]
        else:
            pooled = out.mean(dim=1)                              # [B, d_model]

        return self.head(pooled)                                  # [B, n_horizons]
