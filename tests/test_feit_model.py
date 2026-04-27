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
