"""Tests for QwenImage21Config and its checkpoint detection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

config_mod = load_native_module("qwen_image21.config")
QwenImage21Config = config_mod.QwenImage21Config
detect_qwen_image21_config = config_mod.detect_qwen_image21_config


def test_defaults_match_real_checkpoint_header():
    cfg = QwenImage21Config()
    assert cfg.in_channels == 64
    assert cfg.out_channels == 64
    assert cfg.num_layers == 32
    assert cfg.attention_head_dim == 128
    assert cfg.num_attention_heads == 32
    assert cfg.context_in_dim == 4096
    assert cfg.mlp_ratio == 3
    assert cfg.axes_dims_rope == (16, 56, 56)
    assert cfg.inner_dim == 4096


def test_mlx_dtype_rejects_unknown():
    cfg = QwenImage21Config(dtype="int8")
    with pytest.raises(ValueError, match="dtype"):
        _ = cfg.mlx_dtype


def _fake_state_dict(num_layers=2, inner_dim=16, head_dim=4, context_dim=32, in_ch=8):
    heads = inner_dim // head_dim
    sd = {
        "img_in.weight": np.zeros((inner_dim, in_ch), dtype=np.float32),
        "txt_in.in_layer.weight": np.zeros((inner_dim, context_dim), dtype=np.float32),
        "proj_out.weight": np.zeros((in_ch, inner_dim), dtype=np.float32),
    }
    for i in range(num_layers):
        p = f"transformer_blocks.{i}."
        sd[p + "attn.to_q.weight"] = np.zeros((inner_dim, inner_dim), dtype=np.float32)
        sd[p + "attn.norm_q.weight"] = np.zeros((head_dim,), dtype=np.float32)
    return sd


def test_detect_from_real_shapes():
    sd = _fake_state_dict()
    cfg = detect_qwen_image21_config(sd)
    assert cfg.num_layers == 2
    assert cfg.inner_dim == 16
    assert cfg.attention_head_dim == 4
    assert cfg.num_attention_heads == 4
    assert cfg.context_in_dim == 32
    assert cfg.in_channels == 8
    assert cfg.out_channels == 8


def test_detect_raises_on_missing_keys():
    with pytest.raises(ValueError, match="cannot detect"):
        detect_qwen_image21_config({})
