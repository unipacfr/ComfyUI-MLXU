"""Tests for the Qwen Image 2.1 DiT's base layers (no block/model assembly yet)."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

model_mod = load_native_module("qwen_image21.model")
dit_rope_mod = load_native_module("qwen_image21.dit_rope")

ZeroCenteredRMSNorm = model_mod.ZeroCenteredRMSNorm
TextProjection = model_mod.TextProjection
TimestepProjEmbeddings = model_mod.TimestepProjEmbeddings
SwiGLUFeedForward = model_mod.SwiGLUFeedForward
Attention = model_mod.Attention


def test_zero_centered_rms_norm_matches_manual():
    dim = 16
    norm = ZeroCenteredRMSNorm(dim)
    norm.weight = mx.random.normal((dim,)) * 0.1
    x = mx.random.normal((3, dim))
    out = norm(x)

    w = np.array(norm.weight) + 1.0
    xn = np.array(x).astype(np.float32)
    variance = (xn * xn).mean(axis=-1, keepdims=True)
    expected = xn / np.sqrt(variance + norm.eps) * w
    assert np.allclose(np.array(out), expected, atol=1e-4)


def test_text_projection_shape_and_finite():
    proj = TextProjection(in_dim=32, hidden_size=16)
    x = mx.random.normal((2, 5, 32))
    out = proj(x)
    assert out.shape == (2, 5, 16)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_timestep_proj_embeddings_shape():
    emb = TimestepProjEmbeddings(embedding_dim=16)
    t = mx.array([0.0, 0.5, 1.0])
    out = emb(t)
    assert out.shape == (3, 16)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_swiglu_feed_forward_matches_manual():
    ff = SwiGLUFeedForward(dim=8, hidden_dim=16)
    x = mx.random.normal((3, 8))
    out = ff(x)

    gate_up = np.array(ff.gate_up(x))
    gate, up = gate_up[..., :16], gate_up[..., 16:]
    silu_gate = gate / (1.0 + np.exp(-gate))
    expected = np.array(ff.out(mx.array(silu_gate * up)))
    assert np.allclose(np.array(out), expected, atol=1e-4)


def test_attention_output_shape_and_finite():
    attn = Attention(dim=32, heads=4, dim_head=8)
    x = mx.random.normal((1, 6, 32))
    # dit_rope.apply_rope's freqs contract (verified in Task 2's own tests, and matching
    # flux2's own Attention/joint_attention call site) is a single un-batched table
    # [N, dim_head/2, 2, 2], broadcast across batch and heads by apply_rope itself -- no
    # extra batch/head axis is added here. axes_dim entries must each be even (rope_freqs
    # asserts dim % 2 == 0) and sum to dim_head (2 + 2 + 4 == 8) to match ids' 3 axes.
    ids = mx.arange(6, dtype=mx.float32)[:, None] * mx.ones((1, 3))
    pe = dit_rope_mod.embed_nd(ids, (2, 2, 4), 10000.0)  # [6, 4, 2, 2]
    out = attn(x, pe)
    assert out.shape == (1, 6, 32)
    assert bool(mx.all(mx.isfinite(out)).item())
