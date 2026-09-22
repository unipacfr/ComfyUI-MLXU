"""Tests for Qwen3VL8BTextEncoderConfig and its checkpoint detection."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

config_mod = load_native_module("qwen_image21.text_encoder_config")
Qwen3VL8BTextEncoderConfig = config_mod.Qwen3VL8BTextEncoderConfig
detect_qwen3vl_8b_text_encoder_config = config_mod.detect_qwen3vl_8b_text_encoder_config


def test_default_matches_real_checkpoint_header():
    cfg = Qwen3VL8BTextEncoderConfig()
    assert cfg.vocab_size == 151936
    assert cfg.hidden_size == 4096
    assert cfg.intermediate_size == 12288
    assert cfg.num_hidden_layers == 36
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.kv_groups == 4


def test_rejects_non_divisible_heads():
    with pytest.raises(ValueError, match="kv"):
        Qwen3VL8BTextEncoderConfig(num_attention_heads=33, num_key_value_heads=8)


def test_mlx_dtype_rejects_unknown():
    cfg = Qwen3VL8BTextEncoderConfig(dtype="int8")
    with pytest.raises(ValueError, match="dtype"):
        _ = cfg.mlx_dtype


def _fake_state_dict(num_layers: int = 2, hidden: int = 16, heads: int = 4, kv_heads: int = 2, head_dim: int = 4, inter: int = 32, vocab: int = 10) -> dict:
    sd = {
        "model.embed_tokens.weight": np.zeros((vocab, hidden), dtype=np.float32),
        "model.norm.weight": np.zeros((hidden,), dtype=np.float32),
        "lm_head.weight": np.zeros((vocab, hidden), dtype=np.float32),
    }
    for i in range(num_layers):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = np.zeros((heads * head_dim, hidden), dtype=np.float32)
        sd[p + "self_attn.k_proj.weight"] = np.zeros((kv_heads * head_dim, hidden), dtype=np.float32)
        sd[p + "self_attn.q_norm.weight"] = np.zeros((head_dim,), dtype=np.float32)
        sd[p + "mlp.gate_proj.weight"] = np.zeros((inter, hidden), dtype=np.float32)
    return sd


def test_detect_from_real_shapes():
    sd = _fake_state_dict()
    cfg = detect_qwen3vl_8b_text_encoder_config(sd)
    assert cfg.num_hidden_layers == 2
    assert cfg.hidden_size == 16
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 4
    assert cfg.intermediate_size == 32
    assert cfg.vocab_size == 10


def test_detect_raises_on_missing_keys():
    with pytest.raises(ValueError, match="cannot detect"):
        detect_qwen3vl_8b_text_encoder_config({})
