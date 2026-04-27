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
    from layers.Autoformer_EncDec import series_decomp
    decomp = series_decomp(25)
    x = torch.randn(4, 60, 5)
    seasonal, trend = decomp(x)
    diff = (seasonal + trend - x).abs().max().item()
    assert diff < 1e-5, f"reconstruction error: {diff}"
    print("test_feit_decomp_reconstruction: OK")


def test_feit_param_count_reasonable():
    """At d_model=64 the FEiT model should be in the same OoM as iTransformer."""
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


if __name__ == "__main__":
    test_module_imports()
    test_volatile_fourier_low_shape()
    test_volatile_fourier_random_shape()
    test_volatile_fourier_random_determinism()
    test_volatile_wavelets_shape()
    test_volatile_wavelets_no_grad_through_pywt()
    test_feit_forward_mean_fourier()
    test_feit_forward_mean_wavelets()
    test_feit_forward_sp500_concat()
    test_feit_backward_no_nan()
    test_feit_decomp_reconstruction()
    test_feit_param_count_reasonable()
    print("\nAll tests passed.")
