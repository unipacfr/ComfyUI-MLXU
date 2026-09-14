"""Tests for MiniMaxH3Config detection.

The real-checkpoint tests build a state dict of tiny placeholder mx.arrays
shaped exactly like the real checkpoints' tensors (read from the real
safetensors/GGUF headers, never the actual multi-GB tensor data) -- this
verifies `detect_minimax_h3_config` against the real architecture without
paying the cost of loading gigabytes of weights just to check shapes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

config_mod = load_native_module("minimax_h3.config")
gguf_reader_mod = load_native_module("gguf.reader")
safetensors_header_mod = load_native_module("safetensors_header")

MiniMaxH3Config = config_mod.MiniMaxH3Config
detect_minimax_h3_config = config_mod.detect_minimax_h3_config


def _curve_form_state_dict() -> dict[str, mx.array]:
    """Minimal state dict covering every key detect_minimax_h3_config reads,
    shaped like the real checkpoint (curve-form adaln, no time_embedder)."""
    return {
        "video_patch_proj.weight": mx.zeros((5376, 96)),
        "audio_patch_proj.weight": mx.zeros((5376, 32)),
        "final_layer.video_out.weight": mx.zeros((96, 5376)),
        "final_layer.audio_out.weight": mx.zeros((32, 5376)),
        "condition_proj.weight": mx.zeros((5376, 5120)),
        "blocks.0.attn.q_norm.weight": mx.zeros((128,)),
        "blocks.0.attn.qkv_proj.weight": mx.zeros((21504, 5376)),
        "blocks.0.mlp.fc1.weight": mx.zeros((28672, 5376)),
        "rope.inv_freq": mx.zeros((16,)),
        "adaln_t_table": mx.zeros((1025, 8)),
        **{f"blocks.{i}.norm1.weight": mx.zeros((5376,)) for i in range(50)},
        **{f"token_refiner.blocks.{i}.norm1.weight": mx.zeros((5376,)) for i in range(2)},
    }


def test_detects_curve_form_config_from_shapes():
    cfg = detect_minimax_h3_config(_curve_form_state_dict())
    assert cfg.num_layers == 50
    assert cfg.token_refiner_num_layers == 2
    assert cfg.hidden_size == 5376
    assert cfg.latents_dim == 24
    assert cfg.audio_latents_dim == 32
    assert cfg.attention_head_dim == 128
    assert cfg.num_attention_heads == 56
    assert cfg.ffn_hidden_size == 14336
    assert cfg.text_dim == 5120
    assert cfg.rope_inv_freq_len == 16
    assert cfg.gate_compress is False
    assert cfg.adaln_curve_grid == 1025
    assert cfg.time_embed_dim == 8
    assert cfg.timestep_input_dim is None


def test_detects_gate_compress_when_present():
    sd = _curve_form_state_dict()
    sd["blocks.0.attn.to_gate_compress.weight"] = mx.zeros((7168, 5376))
    cfg = detect_minimax_h3_config(sd)
    assert cfg.gate_compress is True


def test_detects_time_embedder_variant_when_no_curve_table():
    sd = _curve_form_state_dict()
    del sd["adaln_t_table"]
    sd["time_embedder.proj_in.weight"] = mx.zeros((5376, 256))
    sd["time_embedder.proj_out.weight"] = mx.zeros((2688, 5376))
    cfg = detect_minimax_h3_config(sd)
    assert cfg.adaln_curve_grid is None
    assert cfg.timestep_input_dim == 256
    assert cfg.time_embed_hidden_size == 5376
    assert cfg.time_embed_dim == 2688


def test_missing_required_key_raises():
    sd = _curve_form_state_dict()
    del sd["condition_proj.weight"]
    with pytest.raises(ValueError, match="condition_proj"):
        detect_minimax_h3_config(sd)


def test_missing_time_source_raises():
    sd = _curve_form_state_dict()
    del sd["adaln_t_table"]
    with pytest.raises(ValueError, match="time_embedder"):
        detect_minimax_h3_config(sd)


def test_video_patch_dim_and_attention_inner_dim_properties():
    cfg = MiniMaxH3Config()
    assert cfg.video_patch_dim == 24 * 1 * 2 * 2  # 96
    assert cfg.attention_inner_dim == 56 * 128  # 7168


def test_non_curve_config_without_timestep_fields_raises():
    with pytest.raises(ValueError, match="timestep_input_dim"):
        MiniMaxH3Config(adaln_curve_grid=None)


def test_non_positive_heads_raises():
    with pytest.raises(ValueError, match="num_attention_heads"):
        MiniMaxH3Config(num_attention_heads=0)


# ---------------------------------------------------------------------------
# Real-checkpoint cross-checks: shapes only, read from the real headers
# (safetensors JSON header, GGUF header via our own reader) -- never the
# actual multi-GB tensor payloads.
# ---------------------------------------------------------------------------

_MINIMAX_H3_SAFETENSORS = Path(
    "/Volumes/X10Pro/Images/models/diffusion_models/MiniMax H3/base model/"
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
)
_MINIMAX_H3_GGUF = Path(
    "/Volumes/X10Pro/Images/models/unet/MiniMax H3/minimax_h3_fl2va_pruned-Q5_0.gguf"
)

_real_checkpoint_gate = pytest.mark.skipif(
    os.environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="reads real multi-GB checkpoint headers; set ASDX_FULL_GGUF_TEST=1 to run",
)

_EXPECTED = dict(
    num_layers=50,
    token_refiner_num_layers=2,
    hidden_size=5376,
    latents_dim=24,
    audio_latents_dim=32,
    attention_head_dim=128,
    num_attention_heads=56,
    ffn_hidden_size=14336,
    text_dim=5120,
    rope_inv_freq_len=16,
    gate_compress=False,
    adaln_curve_grid=1025,
    time_embed_dim=8,
)


def _assert_matches_expected(cfg) -> None:
    for field, expected in _EXPECTED.items():
        assert getattr(cfg, field) == expected, f"{field}: {getattr(cfg, field)!r} != {expected!r}"


@_real_checkpoint_gate
def test_real_safetensors_checkpoint_detects_expected_config():
    if not _MINIMAX_H3_SAFETENSORS.exists():
        pytest.skip("no local MiniMax H3 safetensors checkpoint")
    header = safetensors_header_mod.read_safetensors_header(_MINIMAX_H3_SAFETENSORS)
    sd = {name: mx.zeros(entry.shape) for name, entry in header.tensors.items()}
    cfg = detect_minimax_h3_config(sd)
    _assert_matches_expected(cfg)


@_real_checkpoint_gate
def test_real_gguf_checkpoint_detects_expected_config():
    if not _MINIMAX_H3_GGUF.exists():
        pytest.skip("no local MiniMax H3 GGUF checkpoint")
    header = gguf_reader_mod.read_gguf_header(_MINIMAX_H3_GGUF)
    sd = {name: mx.zeros(info.torch_shape) for name, info in header.tensors.items()}
    cfg = detect_minimax_h3_config(sd)
    _assert_matches_expected(cfg)


@_real_checkpoint_gate
def test_safetensors_and_gguf_detect_identical_config():
    if not _MINIMAX_H3_SAFETENSORS.exists() or not _MINIMAX_H3_GGUF.exists():
        pytest.skip("need both real MiniMax H3 checkpoints")
    st_header = safetensors_header_mod.read_safetensors_header(_MINIMAX_H3_SAFETENSORS)
    st_sd = {name: mx.zeros(entry.shape) for name, entry in st_header.tensors.items()}
    st_cfg = detect_minimax_h3_config(st_sd)

    gguf_header = gguf_reader_mod.read_gguf_header(_MINIMAX_H3_GGUF)
    gguf_sd = {name: mx.zeros(info.torch_shape) for name, info in gguf_header.tensors.items()}
    gguf_cfg = detect_minimax_h3_config(gguf_sd)

    assert st_cfg == gguf_cfg
