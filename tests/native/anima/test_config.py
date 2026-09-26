from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.anima_module_loader import load_native_module

config_mod = load_native_module("anima.config")
AnimaConfig = config_mod.AnimaConfig
detect_anima_config = config_mod.detect_anima_config


def _state(model_channels=2048, in_ch_plus_mask=17, blocks=28):
    sd = {"x_embedder.proj.1.weight": mx.zeros((model_channels, in_ch_plus_mask * 4))}
    for i in range(blocks):
        sd[f"blocks.{i}.self_attn.q_proj.weight"] = mx.zeros((1, 1))
    return sd


def test_defaults_match_real_checkpoint():
    cfg = AnimaConfig()
    assert (cfg.model_channels, cfg.num_blocks, cfg.num_heads, cfg.head_dim) == (2048, 28, 16, 128)
    assert cfg.in_channels == 16 and cfg.out_channels == 16
    assert cfg.adaln_lora_dim == 256 and cfg.crossattn_emb_channels == 1024
    assert cfg.rope_ratios == (1.0, 4.0, 4.0)
    assert cfg.min_context_len == 512


def test_detect_reads_geometry():
    cfg = detect_anima_config(_state(), dtype="bfloat16")
    assert cfg.model_channels == 2048 and cfg.num_blocks == 28 and cfg.num_heads == 16
    assert cfg.mlx_dtype == mx.bfloat16


def test_detect_rejects_video_variant():
    # in_channels=17 (+mask=18) is Cosmos image-to-video, which uses other RoPE ratios.
    with pytest.raises(ValueError, match="in_channels"):
        detect_anima_config(_state(in_ch_plus_mask=18), dtype="bfloat16")


def test_detect_rejects_unknown_width():
    with pytest.raises(ValueError, match="model_channels"):
        detect_anima_config(_state(model_channels=1024), dtype="bfloat16")
