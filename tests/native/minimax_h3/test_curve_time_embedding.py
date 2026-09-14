"""Tests for MiniMax H3's curve-form timestep embedding lookup."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

model_mod = load_native_module("minimax_h3.model")
curve_time_embedding = model_mod.curve_time_embedding

_COMFYUI_ROOT = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI")
_COMFYUI_VENV_SITE_PACKAGES = _COMFYUI_ROOT / ".venv" / "lib" / "python3.13" / "site-packages"


def _reference_curve_embedding(table_np, t_vals_np):
    if not _COMFYUI_ROOT.exists() or not _COMFYUI_VENV_SITE_PACKAGES.exists():
        pytest.skip("ComfyUI install not present on this machine")
    sys.path.insert(0, str(_COMFYUI_ROOT))
    sys.path.insert(0, str(_COMFYUI_VENV_SITE_PACKAGES))
    import torch

    table = torch.tensor(table_np, dtype=torch.float32)
    t_vals = torch.tensor(t_vals_np, dtype=torch.float32)
    pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1)).numpy()


def test_matches_reference_interpolation():
    rng = np.random.default_rng(0)
    table_np = rng.normal(size=(1025, 8)).astype(np.float32)
    t_vals_np = np.array([0.0, 0.37, 0.999, 1.0, 0.5], dtype=np.float32)

    expected = _reference_curve_embedding(table_np, t_vals_np)
    out = curve_time_embedding(mx.array(table_np), mx.array(t_vals_np))

    assert np.allclose(np.array(out), expected, atol=1e-5)


def test_exact_grid_point_returns_table_row():
    table = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # t=0.5 on a 3-row table -> pos=1.0 exactly -> row 1
    out = curve_time_embedding(table, mx.array([0.5]))
    assert np.allclose(np.array(out)[0], [3.0, 4.0])


def test_t_equals_one_uses_last_interval_not_out_of_bounds():
    table = mx.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]])
    out = curve_time_embedding(table, mx.array([1.0]))
    # pos = 1.0*(3-1) = 2.0 = last index -> i0 clamped to grid-2=1, frac=1.0
    # -> lerp(table[1], table[2], 1.0) = table[2]
    assert np.allclose(np.array(out)[0], [10.0, 10.0])


def test_out_of_range_t_clamps():
    table = mx.array([[0.0], [5.0]])
    out_low = curve_time_embedding(table, mx.array([-1.0]))
    out_high = curve_time_embedding(table, mx.array([2.0]))
    assert np.allclose(np.array(out_low)[0], [0.0])
    assert np.allclose(np.array(out_high)[0], [5.0])
