"""Tests for MiniMax H3's SwiGLU MLP."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

model_mod = load_native_module("minimax_h3.model")
MLP = model_mod.MLP


def test_mlp_output_shape():
    mlp = MLP(hidden=16, ffn=32)
    x = mx.random.normal((5, 16))
    out = mlp(x)
    assert out.shape == (5, 16)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_mlp_matches_manual_swiglu_gate_first_half():
    # comfy.ops._swiglu_eager: gate, up = x.chunk(2, dim=-1); silu(gate)*up
    # -- gate is explicitly the FIRST half, not the second.
    mlp = MLP(hidden=4, ffn=2)
    x = mx.random.normal((3, 4))

    h = mlp.fc1(x)
    gate_np, up_np = np.array(h)[:, :2], np.array(h)[:, 2:]
    manual_gated = gate_np / (1.0 + np.exp(-gate_np)) * up_np  # silu(gate) * up
    expected = np.array(mlp.fc2(mx.array(manual_gated)))

    out = np.array(mlp(x))
    assert np.abs(out - expected).max() < 1e-5


def test_swapping_gate_and_value_halves_changes_output():
    # Confirms the split order matters (i.e. the test above isn't
    # order-insensitive by coincidence): using the second half as gate
    # instead of the first must give a different result whenever gate != up.
    mlp = MLP(hidden=4, ffn=2)
    x = mx.random.normal((3, 4))
    h = mlp.fc1(x)
    ffn = h.shape[-1] // 2
    gate, up = h[:, :ffn], h[:, ffn:]

    correct = mlp.fc2(nn_silu(gate) * up)
    swapped = mlp.fc2(nn_silu(up) * gate)
    assert float(mx.max(mx.abs(correct - swapped)).item()) > 1e-6


def nn_silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)
