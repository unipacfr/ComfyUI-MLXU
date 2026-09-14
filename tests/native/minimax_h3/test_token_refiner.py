"""Tests for MiniMax H3's RefinerBlock / TokenRefiner."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

model_mod = load_native_module("minimax_h3.model")
RefinerBlock = model_mod.RefinerBlock
TokenRefiner = model_mod.TokenRefiner


def test_refiner_block_output_shape_and_finite():
    block = RefinerBlock(hidden=16, heads=2, head_dim=8, ffn=32, eps=1e-5, qk_eps=1e-5)
    x = mx.random.normal((5, 16))
    out = block(x)
    assert out.shape == (5, 16)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_refiner_block_is_residual():
    # Zeroing attn/mlp output projections should reduce the block to identity.
    block = RefinerBlock(hidden=8, heads=1, head_dim=8, ffn=16, eps=1e-5, qk_eps=1e-5)
    block.attn.out_proj.weight = mx.zeros_like(block.attn.out_proj.weight)
    block.mlp.fc2.weight = mx.zeros_like(block.mlp.fc2.weight)
    x = mx.random.normal((4, 8))
    out = block(x)
    assert bool(mx.allclose(out, x, atol=1e-5).item())


def test_token_refiner_matches_stacked_blocks_plus_final_norm():
    refiner = TokenRefiner(num_layers=3, hidden=16, heads=2, head_dim=8, ffn=32, eps=1e-5, qk_eps=1e-5, final_eps=1e-6)
    assert len(refiner.blocks) == 3
    x = mx.random.normal((6, 16))

    manual = x
    for block in refiner.blocks:
        manual = block(manual)
    manual = refiner.final_norm(manual)

    out = refiner(x)
    assert bool(mx.array_equal(out, manual).item())
