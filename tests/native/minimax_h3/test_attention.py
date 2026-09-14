"""Tests for MiniMax H3's Attention block."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

model_mod = load_native_module("minimax_h3.model")
rope_mod = load_native_module("minimax_h3.rope")

RMSNorm = model_mod.RMSNorm
Attention = model_mod.Attention


def test_rms_norm_matches_direct_multiply_convention():
    # Weight centered around 1.0 (real checkpoint convention, see model.py
    # docstring) must scale the normalized vector directly, not as (1+weight).
    norm = RMSNorm(4, eps=1e-6)
    norm.weight = mx.array([2.0, 2.0, 2.0, 2.0])
    x = mx.array([[1.0, 1.0, 1.0, 1.0]])
    out = norm(x)
    # rms(x)=1 exactly here, so normed=x, scaled by weight=2 -> [2,2,2,2]
    assert np.allclose(np.array(out), [[2.0, 2.0, 2.0, 2.0]], atol=1e-5)


def test_attention_output_shape():
    seq_len, hidden, heads, head_dim = 6, 32, 4, 8
    attn = Attention(hidden, heads, head_dim, eps=1e-5, rot_dim=head_dim)
    x = mx.random.normal((seq_len, hidden))
    out = attn(x)
    assert out.shape == (seq_len, hidden)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_attention_without_rope_matches_manual_reimplementation():
    # RefinerBlock's call path: no cos/sin -> plain per-head RMSNorm, no
    # rotation. Cross-checked against an independent manual computation
    # (not reusing Attention's internals) built directly from mx ops.
    seq_len, hidden, heads, head_dim = 5, 16, 2, 8
    attn = Attention(hidden, heads, head_dim, eps=1e-5)
    mx.random.seed(0)
    x = mx.random.normal((seq_len, hidden))

    out = attn(x)

    inner = heads * head_dim
    qkv = attn.qkv_proj(x)
    q, k, v = qkv[:, :inner], qkv[:, inner : 2 * inner], qkv[:, 2 * inner :]
    q = q.reshape(seq_len, heads, head_dim)
    k = k.reshape(seq_len, heads, head_dim)
    v = v.reshape(seq_len, heads, head_dim)
    q = rope_mod.rms_norm(q, attn.q_norm.weight, 1e-5)
    k = rope_mod.rms_norm(k, attn.k_norm.weight, 1e-5)

    qn = np.array(q).transpose(1, 0, 2)  # [heads, S, head_dim]
    kn = np.array(k).transpose(1, 0, 2)
    vn = np.array(v).transpose(1, 0, 2)
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.einsum("hsd,htd->hst", qn, kn) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    manual_out = np.einsum("hst,htd->hsd", weights, vn)  # [heads, S, head_dim]
    manual_out = manual_out.transpose(1, 0, 2).reshape(seq_len, inner)
    expected = np.array(attn.out_proj(mx.array(manual_out)))

    # mx.fast.scaled_dot_product_attention's fused Metal kernel accumulates
    # at reduced precision vs a plain fp32 numpy matmul -- ~1e-3 relative
    # error is expected here (see krea2/model.py's Attention docstring for
    # the same observation on this machine), not a correctness bug.
    assert np.abs(np.array(out) - expected).max() < 2e-3


def test_attention_with_rope_uses_rope_module():
    seq_len, hidden, heads, head_dim, rot_dim = 5, 16, 2, 8, 6
    attn = Attention(hidden, heads, head_dim, eps=1e-5, rot_dim=rot_dim)
    mx.random.seed(1)
    x = mx.random.normal((seq_len, hidden))
    half = rot_dim // 2
    angles = mx.random.uniform(0, 3.14, (seq_len, half))
    cos, sin = mx.cos(angles), mx.sin(angles)

    out = attn(x, cos=cos, sin=sin)
    assert out.shape == (seq_len, hidden)
    assert bool(mx.all(mx.isfinite(out)).item())

    # Changing cos/sin must actually change the output (rope wired in, not
    # silently ignored) -- a real regression this project has hit before
    # with a forgotten wire-up (see the Krea2 Identity Edit pixel-path bug).
    other_angles = mx.random.uniform(0, 3.14, (seq_len, half))
    other_cos, other_sin = mx.cos(other_angles), mx.sin(other_angles)
    out2 = attn(x, cos=other_cos, sin=other_sin)
    assert float(mx.max(mx.abs(out - out2)).item()) > 1e-4
