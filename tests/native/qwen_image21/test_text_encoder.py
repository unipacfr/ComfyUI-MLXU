"""Tests for the Qwen3-VL-8B text-only encoder (Attention/MLP/
TransformerBlock/Qwen3VL8BTextEncoder). Text-only: no vision tower, no
reference-image conditioning -- out of scope for this brick."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

config_mod = load_native_module("qwen_image21.text_encoder_config")
rope_mod = load_native_module("qwen_image21.rope")
text_encoder_mod = load_native_module("qwen_image21.text_encoder")

Qwen3VL8BTextEncoderConfig = config_mod.Qwen3VL8BTextEncoderConfig
Attention = text_encoder_mod.Attention
MLP = text_encoder_mod.MLP
TransformerBlock = text_encoder_mod.TransformerBlock
Qwen3VL8BTextEncoder = text_encoder_mod.Qwen3VL8BTextEncoder


def _tiny_config(**overrides) -> "Qwen3VL8BTextEncoderConfig":
    base = dict(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6,
        rope_theta=5000000.0, dtype="float32",
    )
    base.update(overrides)
    return Qwen3VL8BTextEncoderConfig(**base)


def test_attention_output_shape_and_finite():
    cfg = _tiny_config()
    attn = Attention(cfg)
    seq_len = 5
    x = mx.random.normal((seq_len, cfg.hidden_size))
    cos, sin = rope_mod.qwen3_rope_cos_sin(seq_len, cfg.head_dim, cfg.rope_theta)
    out = attn(x, cos, sin)
    assert out.shape == (seq_len, cfg.hidden_size)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_attention_gqa_matches_manual_repeat():
    cfg = _tiny_config(num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    attn = Attention(cfg)
    seq_len = 4
    mx.random.seed(0)
    x = mx.random.normal((seq_len, cfg.hidden_size))
    cos, sin = rope_mod.qwen3_rope_cos_sin(seq_len, cfg.head_dim, cfg.rope_theta)

    out = attn(x, cos, sin)

    q = attn.q_proj(x).reshape(seq_len, cfg.num_attention_heads, cfg.head_dim)
    k = attn.k_proj(x).reshape(seq_len, cfg.num_key_value_heads, cfg.head_dim)
    v = attn.v_proj(x).reshape(seq_len, cfg.num_key_value_heads, cfg.head_dim)

    cos_b, sin_b = cos[:, None, :], sin[:, None, :]
    q = rope_mod.rms_norm_rope_split_half(q, attn.q_norm.weight, attn.eps, cfg.head_dim, cos_b, sin_b)
    k = rope_mod.rms_norm_rope_split_half(k, attn.k_norm.weight, attn.eps, cfg.head_dim, cos_b, sin_b)

    qn = np.array(q).transpose(1, 0, 2)
    kn = np.array(k).transpose(1, 0, 2)
    vn = np.array(v).transpose(1, 0, 2)
    group = cfg.kv_groups
    kn = np.repeat(kn, group, axis=0)
    vn = np.repeat(vn, group, axis=0)

    scale = 1.0 / np.sqrt(cfg.head_dim)
    scores = np.einsum("hsd,htd->hst", qn, kn) * scale
    causal = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
    scores = scores + causal
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    manual = np.einsum("hst,htd->hsd", weights, vn).transpose(1, 0, 2).reshape(seq_len, -1)
    expected = np.array(attn.o_proj(mx.array(manual)))

    assert np.abs(np.array(out) - expected).max() < 2e-3


def test_mlp_matches_standard_swiglu():
    cfg = _tiny_config()
    mlp = MLP(cfg)
    x = mx.random.normal((3, cfg.hidden_size))
    out = mlp(x)
    gate = np.array(mlp.gate_proj(x))
    up = np.array(mlp.up_proj(x))
    silu_gate = gate / (1.0 + np.exp(-gate))
    expected = np.array(mlp.down_proj(mx.array(silu_gate * up)))
    assert np.allclose(np.array(out), expected, atol=1e-5)


def test_transformer_block_is_residual():
    cfg = _tiny_config()
    block = TransformerBlock(cfg)
    block.self_attn.o_proj.weight = mx.zeros_like(block.self_attn.o_proj.weight)
    block.mlp.down_proj.weight = mx.zeros_like(block.mlp.down_proj.weight)
    x = mx.random.normal((4, cfg.hidden_size))
    cos, sin = rope_mod.qwen3_rope_cos_sin(4, cfg.head_dim, cfg.rope_theta)
    out = block(x, cos, sin)
    assert bool(mx.allclose(out, x, atol=1e-5).item())


def test_encoder_output_shape():
    cfg = _tiny_config()
    encoder = Qwen3VL8BTextEncoder(cfg)
    input_ids = mx.array([1, 2, 3, 4, 5])
    out = encoder(input_ids)
    assert out.shape == (5, cfg.hidden_size)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_encoder_is_causal():
    cfg = _tiny_config()
    encoder = Qwen3VL8BTextEncoder(cfg)
    ids_a = mx.array([1, 2, 3, 4, 5])
    ids_b = mx.array([1, 2, 3, 4, 99])
    out_a = encoder(ids_a)
    out_b = encoder(ids_b)
    assert bool(mx.allclose(out_a[:4], out_b[:4], atol=1e-5).item())
    assert float(mx.max(mx.abs(out_a[4] - out_b[4])).item()) > 1e-4


def test_encoder_has_no_final_norm_attribute():
    # Deliberate divergence from the real checkpoint (which has model.norm.weight
    # and lm_head.weight): this encoder consumes the raw last-block hidden state.
    cfg = _tiny_config()
    encoder = Qwen3VL8BTextEncoder(cfg)
    assert not hasattr(encoder.model, "norm")
    assert not hasattr(encoder, "lm_head")
