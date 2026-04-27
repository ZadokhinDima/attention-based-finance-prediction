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
