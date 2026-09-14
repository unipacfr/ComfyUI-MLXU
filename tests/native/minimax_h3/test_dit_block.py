"""Tests for MiniMax H3's DiTBlock."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

model_mod = load_native_module("minimax_h3.model")
DiTBlock = model_mod.DiTBlock


def _cos_sin(seq_len, half):
    angles = mx.random.uniform(0, 3.14, (seq_len, half))
    return mx.cos(angles), mx.sin(angles)


def test_dit_block_output_shape_and_finite():
    hidden, heads, head_dim, ffn, t_dim = 16, 2, 8, 32, 8
    block = DiTBlock(hidden, heads, head_dim, ffn, t_dim, eps=1e-5, qk_eps=1e-5, rot_dim=6)
    seq_len = 5
    x = mx.random.normal((seq_len, hidden))
    t_emb = mx.random.normal((2, t_dim))  # 2 unique timesteps
    mod_segments = [(0, 2, 0), (2, 5, 1)]  # (video-like, text-like) rows
    cos, sin = _cos_sin(seq_len, 3)

    out = block(x, t_emb, mod_segments, cos, sin)
    assert out.shape == (seq_len, hidden)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_dit_block_zero_gates_reduce_to_identity():
    # With adaln_proj forced to output zero for every gate, both residual
    # updates vanish and the block must be the identity (independent of
    # whatever attn/mlp compute, since their contribution is gated to zero).
    hidden, heads, head_dim, ffn, t_dim = 8, 1, 8, 16, 4
    block = DiTBlock(hidden, heads, head_dim, ffn, t_dim, eps=1e-5, qk_eps=1e-5, rot_dim=8)
    block.adaln_proj.linear.weight = mx.zeros_like(block.adaln_proj.linear.weight)
    block.adaln_proj.linear.bias = mx.zeros_like(block.adaln_proj.linear.bias)

    seq_len = 4
    x = mx.random.normal((seq_len, hidden))
    t_emb = mx.random.normal((1, t_dim))
    mod_segments = [(0, 4, 0)]
    cos, sin = _cos_sin(seq_len, 4)

    out = block(x, t_emb, mod_segments, cos, sin)
    # scale=0 -> norm(x)*(1+0)+0 = norm(x) still feeds attn/mlp, but gate=0
    # zeroes their contribution entirely, so out must equal x exactly.
    assert bool(mx.array_equal(out, x).item())


def test_dit_block_different_mod_rows_change_output():
    hidden, heads, head_dim, ffn, t_dim = 8, 1, 8, 16, 4
    block = DiTBlock(hidden, heads, head_dim, ffn, t_dim, eps=1e-5, qk_eps=1e-5, rot_dim=8)
    seq_len = 4
    mx.random.seed(2)
    x = mx.random.normal((seq_len, hidden))
    t_emb = mx.random.normal((2, t_dim))
    cos, sin = _cos_sin(seq_len, 4)

    out_row0 = block(x, t_emb, [(0, 4, 0)], cos, sin)
    out_row1 = block(x, t_emb, [(0, 4, 1)], cos, sin)
    assert float(mx.max(mx.abs(out_row0 - out_row1)).item()) > 1e-4
