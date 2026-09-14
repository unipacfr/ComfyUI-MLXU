"""Tests for MiniMax H3's AdalnProj and per-segment modulation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

model_mod = load_native_module("minimax_h3.model")
AdalnProj = model_mod.AdalnProj
mod_scale_shift = model_mod.mod_scale_shift
mod_gate = model_mod.mod_gate


def test_adaln_proj_shapes_and_chunk_count():
    # DiTBlock shape: expand=6 (shift/scale/gate x2), modalities=3.
    proj = AdalnProj(t_dim=8, hidden=16, expand=6, modalities=3)
    t_emb = mx.random.normal((2, 8))  # M=2 unique timesteps
    outputs = proj(t_emb)
    assert len(outputs) == 6
    for o in outputs:
        assert o.shape == (2 * 3, 16)  # M*modalities rows


def test_adaln_proj_final_layer_shape():
    # FinalLayer shape: expand=2 (shift/scale), modalities=1.
    proj = AdalnProj(t_dim=8, hidden=16, expand=2, modalities=1)
    t_emb = mx.random.normal((3, 8))
    shift, scale = proj(t_emb)
    assert shift.shape == (3, 16)
    assert scale.shape == (3, 16)


def test_adaln_proj_no_silu_by_default():
    proj = AdalnProj(t_dim=4, hidden=4, expand=1, modalities=1)
    t_emb = mx.array([[1.0, -1.0, 2.0, -2.0]])
    out_no_silu = proj(t_emb)[0]
    expected = np.array(proj.linear(t_emb))
    assert np.allclose(np.array(out_no_silu), expected)


def test_adaln_proj_applies_silu_when_requested():
    proj = AdalnProj(t_dim=4, hidden=4, expand=1, modalities=1, apply_silu=True)
    t_emb = mx.array([[1.0, -1.0, 2.0, -2.0]])
    out = np.array(proj(t_emb)[0])
    expected = np.array(proj.linear(t_emb * mx.sigmoid(t_emb)))
    assert np.allclose(out, expected, atol=1e-5)


def test_mod_scale_shift_applies_per_segment_row():
    h = mx.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])  # 4 rows
    shift = mx.array([[0.0, 0.0], [10.0, 10.0]])  # row 0, row 1
    scale = mx.array([[0.0, 0.0], [1.0, 1.0]])  # row 0 -> x1, row 1 -> x2
    segments = [(0, 2, 0), (2, 4, 1)]  # first 2 rows use mod-row 0, last 2 use mod-row 1

    out = mod_scale_shift(h, shift, scale, segments)
    expected = np.array([[1.0, 1.0], [2.0, 2.0], [3.0 * 2 + 10, 3.0 * 2 + 10], [4.0 * 2 + 10, 4.0 * 2 + 10]])
    assert np.allclose(np.array(out), expected)


def test_mod_gate_accumulates_gated_residual():
    x = mx.array([[1.0], [2.0], [3.0]])
    other = mx.array([[10.0], [20.0], [30.0]])
    gate = mx.array([[0.0], [1.0]])  # row 0 -> zero gate, row 1 -> full gate
    segments = [(0, 1, 0), (1, 3, 1)]

    out = mod_gate(x, gate, other, segments)
    expected = np.array([[1.0 + 10.0 * 0.0], [2.0 + 20.0 * 1.0], [3.0 + 30.0 * 1.0]])
    assert np.allclose(np.array(out), expected)


def test_mod_helpers_preserve_row_order_across_segments():
    h = mx.arange(10).reshape(5, 2).astype(mx.float32)
    shift = mx.zeros((1, 2))
    scale = mx.zeros((1, 2))
    segments = [(0, 2, 0), (2, 3, 0), (3, 5, 0)]
    out = mod_scale_shift(h, shift, scale, segments)
    assert np.array_equal(np.array(out), np.array(h))
