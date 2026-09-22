"""Tests for QwenImage21TransformerBlock, LastLayer, block_causal_attention."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

model_mod = load_native_module("qwen_image21.model")
dit_rope_mod = load_native_module("qwen_image21.dit_rope")

block_causal_attention = model_mod.block_causal_attention
QwenImage21TransformerBlock = model_mod.QwenImage21TransformerBlock
LastLayer = model_mod.LastLayer


def _pe(seq_len, axes_dim=(2, 3, 3), theta=10000.0):
    ids = mx.arange(seq_len, dtype=mx.float32)[:, None] * mx.ones((1, 3))
    return dit_rope_mod.embed_nd(ids, axes_dim, theta)


def test_block_causal_attention_text_only_sees_itself_causally():
    # 2 segments: text [0,3) causal, image [3,5) full (sees everything incl. text).
    txt_len, total = 3, 5
    causal = mx.array(np.tril(np.ones((txt_len, txt_len), dtype=bool)))
    segments = [(0, txt_len, causal), (txt_len, total, None)]
    attn_fn = block_causal_attention(segments)

    heads, dim_head = 2, 4
    q = mx.random.normal((1, total, heads, dim_head))
    k = mx.random.normal((1, total, heads, dim_head))
    v = mx.random.normal((1, total, heads, dim_head))
    out = attn_fn(q, k, v, heads)
    assert out.shape == (1, total, heads * dim_head)
    assert bool(mx.all(mx.isfinite(out)).item())

    # Changing a later TEXT token's k/v must not change an earlier text token's output
    # (causal within the text segment), but changing an IMAGE token must not affect text
    # rows at all (text never attends to image).
    k2 = k.at[:, 2].add(10.0)  # perturb last text token
    out2 = attn_fn(q, k2, v, heads)
    assert bool(mx.allclose(out[:, :2], out2[:, :2], atol=1e-4).item())  # rows 0,1 unaffected
    assert not bool(mx.allclose(out[:, 2], out2[:, 2], atol=1e-4).item())  # row 2 sees itself

    k3 = k.at[:, 4].add(10.0)  # perturb an image token
    out3 = attn_fn(q, k3, v, heads)
    assert bool(mx.allclose(out[:, :3], out3[:, :3], atol=1e-4).item())  # text rows unaffected


def test_block_causal_attention_image_sees_prefix_and_itself():
    txt_len, total = 2, 4
    causal = mx.array(np.tril(np.ones((txt_len, txt_len), dtype=bool)))
    segments = [(0, txt_len, causal), (txt_len, total, None)]
    attn_fn = block_causal_attention(segments)
    heads, dim_head = 1, 4
    q = mx.random.normal((1, total, heads, dim_head))
    k = mx.random.normal((1, total, heads, dim_head))
    v = mx.random.normal((1, total, heads, dim_head))
    out = attn_fn(q, k, v, heads)

    k2 = k.at[:, 0].add(10.0)  # perturb a text token
    out2 = attn_fn(q, k2, v, heads)
    assert not bool(mx.allclose(out[:, 2:], out2[:, 2:], atol=1e-4).item())  # image rows see text


def test_transformer_block_output_shape_and_finite():
    dim, heads, head_dim = 32, 4, 8
    block = QwenImage21TransformerBlock(dim, heads, head_dim, mlp_ratio=2)
    seq = 6
    x = mx.random.normal((1, seq, dim))
    pe = _pe(seq, axes_dim=(head_dim // 4, head_dim // 4, head_dim // 2))
    # mod: (scale1, gate1, scale2, gate2, zero), each (prefix_row, target_rows)
    zero = mx.zeros((1, 1, dim))
    def split(t):
        return t[-1:][None], t[:-1][None]
    scale1 = split(mx.random.normal((seq + 1, dim)))
    gate1 = split(mx.random.normal((seq + 1, dim)))
    scale2 = split(mx.random.normal((seq + 1, dim)))
    gate2 = split(mx.random.normal((seq + 1, dim)))
    mod = (scale1, gate1, scale2, gate2, zero)
    attn_fn = block_causal_attention([(0, seq, None)])
    out = block(x, mod, pe, attn_fn, prefix_len=0)
    assert out.shape == x.shape
    assert bool(mx.all(mx.isfinite(out)).item())


def test_last_layer_output_shape():
    dim = 16
    layer = LastLayer(dim)
    x = mx.random.normal((1, 5, dim))
    temb = mx.random.normal((1, dim))
    out = layer(x, temb)
    assert out.shape == x.shape
