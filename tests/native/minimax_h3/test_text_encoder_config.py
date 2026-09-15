"""Tests for Qwen3 text encoder config detection."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

config_mod = load_native_module("minimax_h3.text_encoder_config")
gguf_reader_mod = load_native_module("gguf.reader")
safetensors_header_mod = load_native_module("safetensors_header")

Qwen3TextEncoderConfig = config_mod.Qwen3TextEncoderConfig
detect_qwen3_text_encoder_config = config_mod.detect_qwen3_text_encoder_config


def _tiny_state_dict(num_layers=3):
    sd = {
        "model.embed_tokens.weight": mx.zeros((1000, 64)),
        "model.layers.0.self_attn.q_norm.weight": mx.zeros((16,)),
    }
    for i in range(num_layers):
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = mx.zeros((32, 64))  # 2 heads * 16
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = mx.zeros((16, 64))  # 1 kv head * 16
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = mx.zeros((128, 64))
    return sd


def test_detects_config_from_shapes():
    cfg = detect_qwen3_text_encoder_config(_tiny_state_dict(num_layers=3))
    assert cfg.vocab_size == 1000
    assert cfg.hidden_size == 64
    assert cfg.head_dim == 16
    assert cfg.num_attention_heads == 2
    assert cfg.num_key_value_heads == 1
    assert cfg.intermediate_size == 128
    assert cfg.num_hidden_layers == 3


def test_kv_groups_property():
    cfg = Qwen3TextEncoderConfig(num_attention_heads=64, num_key_value_heads=8)
    assert cfg.kv_groups == 8


def test_non_divisible_heads_raises():
    with pytest.raises(ValueError, match="key_value_heads"):
        Qwen3TextEncoderConfig(num_attention_heads=10, num_key_value_heads=3)


def test_missing_required_key_raises():
    sd = _tiny_state_dict()
    del sd["model.embed_tokens.weight"]
    with pytest.raises(ValueError, match="embed_tokens"):
        detect_qwen3_text_encoder_config(sd)


_MINIMAX_H3_TEXT_ENCODER_GGUF = Path(
    "/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
)
_MINIMAX_H3_TEXT_ENCODER_SAFETENSORS = Path(
    "/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
)

_real_checkpoint_gate = pytest.mark.skipif(
    os.environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="reads real multi-GB checkpoint headers; set ASDX_FULL_GGUF_TEST=1 to run",
)

_EXPECTED = dict(
    vocab_size=151936,
    hidden_size=5120,
    intermediate_size=25600,
    num_hidden_layers=50,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
)


def _assert_matches_expected(cfg) -> None:
    for field, expected in _EXPECTED.items():
        assert getattr(cfg, field) == expected, f"{field}: {getattr(cfg, field)!r} != {expected!r}"


@_real_checkpoint_gate
def test_real_gguf_checkpoint_detects_expected_config():
    if not _MINIMAX_H3_TEXT_ENCODER_GGUF.exists():
        pytest.skip("no local MiniMax H3 text encoder GGUF")
    header = gguf_reader_mod.read_gguf_header(_MINIMAX_H3_TEXT_ENCODER_GGUF)
    sd = {name: mx.zeros(info.torch_shape) for name, info in header.tensors.items()}
    cfg = detect_qwen3_text_encoder_config(sd)
    _assert_matches_expected(cfg)


@_real_checkpoint_gate
def test_real_safetensors_checkpoint_detects_expected_config():
    if not _MINIMAX_H3_TEXT_ENCODER_SAFETENSORS.exists():
        pytest.skip("no local MiniMax H3 text encoder safetensors")
    header = safetensors_header_mod.read_safetensors_header(_MINIMAX_H3_TEXT_ENCODER_SAFETENSORS)
    sd = {name: mx.zeros(entry.shape) for name, entry in header.tensors.items()}
    cfg = detect_qwen3_text_encoder_config(sd)
    _assert_matches_expected(cfg)
