"""End-to-end test for QwenImage21Transformer2DModel: reduced random-weight config, checks
shapes flow through and output is NaN/Inf-free (verify-checkpoint step 2)."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

config_mod = load_native_module("qwen_image21.config")
model_mod = load_native_module("qwen_image21.model")

QwenImage21Config = config_mod.QwenImage21Config
QwenImage21Transformer2DModel = model_mod.QwenImage21Transformer2DModel


def _tiny_config(**overrides):
    base = dict(
        in_channels=8, out_channels=8, num_layers=2, attention_head_dim=8,
        num_attention_heads=2, context_in_dim=16, mlp_ratio=2,
        axes_dims_rope=(2, 4, 2), eps=1e-6, dtype="float32",
    )
    base.update(overrides)
    return QwenImage21Config(**base)


def test_forward_shape_and_finite():
    cfg = _tiny_config()
    model = QwenImage21Transformer2DModel(cfg)
    B, H, W = 1, 4, 4
    x = mx.random.normal((B, cfg.in_channels, H, W))
    timestep = mx.array([0.5])
    context = mx.random.normal((B, 6, cfg.context_in_dim))
    out = model(x, timestep, context)
    assert out.shape == (B, cfg.out_channels, H, W)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_different_prompt_changes_output():
    cfg = _tiny_config()
    model = QwenImage21Transformer2DModel(cfg)
    B, H, W = 1, 4, 4
    x = mx.random.normal((B, cfg.in_channels, H, W))
    timestep = mx.array([0.5])
    context_a = mx.random.normal((B, 6, cfg.context_in_dim))
    context_b = mx.random.normal((B, 6, cfg.context_in_dim))
    out_a = model(x, timestep, context_a)
    out_b = model(x, timestep, context_b)
    assert not bool(mx.allclose(out_a, out_b, atol=1e-4).item())


def test_different_timestep_changes_output():
    cfg = _tiny_config()
    model = QwenImage21Transformer2DModel(cfg)
    B, H, W = 1, 4, 4
    x = mx.random.normal((B, cfg.in_channels, H, W))
    context = mx.random.normal((B, 6, cfg.context_in_dim))
    out_a = model(x, mx.array([0.1]), context)
    out_b = model(x, mx.array([0.9]), context)
    assert not bool(mx.allclose(out_a, out_b, atol=1e-4).item())


def test_batch_greater_than_one_forward_shape_and_finite():
    # Regression test: _split_rows used `[None]` (insert axis 0) instead of
    # `[:, None]` (insert axis 1, matching the reference's `.unsqueeze(1)`), which
    # coincidentally produced the same shape as the correct form at B == 1 but broke
    # broadcasting against hidden_states [B, seq, dim] for any B > 1.
    cfg = _tiny_config()
    model = QwenImage21Transformer2DModel(cfg)
    B, H, W = 3, 4, 4
    x = mx.random.normal((B, cfg.in_channels, H, W))
    timestep = mx.random.uniform(shape=(B,))
    context = mx.random.normal((B, 6, cfg.context_in_dim))
    out = model(x, timestep, context)
    assert out.shape == (B, cfg.out_channels, H, W)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_batch_rows_are_independent():
    # Each batch row's output must depend only on its own context/timestep, not on
    # other rows in the batch (a mixed-up broadcast would leak rows into each other).
    cfg = _tiny_config()
    model = QwenImage21Transformer2DModel(cfg)
    H, W = 4, 4
    x = mx.random.normal((2, cfg.in_channels, H, W))
    context = mx.concatenate([
        mx.random.normal((1, 6, cfg.context_in_dim)),
        mx.random.normal((1, 6, cfg.context_in_dim)),
    ], axis=0)
    timestep = mx.array([0.2, 0.2])
    out = model(x, timestep, context)
    assert float(mx.max(mx.abs(out[0] - out[1])).item()) > 1e-3


def test_build_sequence_ids_match_reference_formula():
    # Locks down the position-id math against a hand-computed reference (per the
    # spec design doc's stated T2I reduction: text ids sequential (i,i,i) on all 3
    # RoPE axes; image ids (txt_len constant t-axis, h-centered, w-centered)), with
    # an odd H so the centering formula's `- (H - H // 2)` term is actually exercised
    # (an even H can't distinguish it from a naive `- H // 2`).
    cfg = _tiny_config()
    model = QwenImage21Transformer2DModel(cfg)
    txt_len, H, W = 4, 3, 2
    context = mx.random.normal((1, txt_len, cfg.context_in_dim))
    x = mx.random.normal((1, cfg.in_channels, H, W))
    hidden_states, pe, segments = model._build_sequence(x, context)

    assert segments[0][0] == 0 and segments[0][1] == txt_len
    assert segments[1] == (txt_len, txt_len + H * W, None)

    dit_rope_mod = load_native_module("qwen_image21.dit_rope")
    txt_ids = [[i, i, i] for i in range(txt_len)]
    hh = [h - (H - H // 2) for h in range(H)]
    ww = [w - (W - W // 2) for w in range(W)]
    img_ids = [[float(txt_len), hh[h], ww[w]] for h in range(H) for w in range(W)]
    expected_ids = mx.array(txt_ids + img_ids, dtype=mx.float32)
    expected_pe = dit_rope_mod.embed_nd(expected_ids, cfg.axes_dims_rope, 10000.0)
    assert bool(mx.allclose(pe, expected_pe, atol=1e-5).item())
