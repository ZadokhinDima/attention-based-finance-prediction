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


if __name__ == "__main__":
    test_module_imports()
    test_volatile_fourier_low_shape()
    test_volatile_fourier_random_shape()
    test_volatile_fourier_random_determinism()
    print("\nAll tests passed.")
